# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Deploy agent_runtime.py to Amazon Bedrock AgentCore Runtime from a zip.

    python -m examples.stage1_replatform.deploy_runtime \
        --gateway-url https://... --memory-id MigratedAgentMemory-xxxx

CreateAgentRuntime takes an artifact two ways. ``containerConfiguration`` wants an
image in ECR, which means a container build, a registry and an ARM64 builder.
``codeConfiguration`` wants a zip in S3 and a Python version, and that is what this
uses: the only build tool needed is pip, and the agent being deployed is
examples/stage1_replatform/agent_runtime.py **unmodified** — no shim, no sys.path
edit, no restructuring. /var/task is on sys.path, so the repo's package layout
survives the zip and the module's absolute ``examples.…`` imports resolve.

Two things about this path are not obvious and both cost a failed deploy to learn,
so they are handled in code here rather than described in prose:

1. **A plain ``pip install --target`` on macOS produces macOS binaries** and the
   deploy dies with "artifact contains binary files incompatible with Linux
   ARM64". Runtime is aarch64. ``--platform manylinux2014_aarch64
   --only-binary=:all:`` cross-compiles the wheels with pip alone — no container,
   no emulation. _fail_on_wrong_architecture turns the service's generic message
   back into the list of files that caused it.

2. **A ``requirements.txt`` inside the zip is inert.** Nothing installs it;
   dependencies have to be vendored into the zip. What makes this expensive is
   that the symptom lies: a missing module surfaces as "Runtime initialization
   time exceeded ... initialization completes in 30s", a timeout message for an
   ImportError, and only CloudWatch shows the real cause. _explain_invoke_failure
   says so at the point where a reader would otherwise start tuning timeouts.

Re-runnable. The runtime, role and bucket are reused if they exist, and the zip is
cached in build/runtime_zip/ (git-ignored by the existing build/ rule) keyed on a
hash of requirements.txt and every .py under examples/, so a second --stage all
does not re-download 196 MB of wheels.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from typing import Optional, Tuple

import boto3

RUNTIME_NAME = "MigratedAgentRuntime"
ROLE_NAME = "MigratedAgentRuntimeRole"

# Who assumes the execution role. Named once because the create-time failure for
# getting it wrong is indistinguishable from the one for IAM not having propagated
# yet, so the error message for the race has to be able to quote it.
TRUST_PRINCIPAL = "bedrock-agentcore.amazonaws.com"

# The module Runtime imports and calls. Nested rather than at the zip root, which
# is what keeps agent_runtime.py's absolute imports working unchanged.
ENTRY_POINT = "examples/stage1_replatform/agent_runtime.py"

# PYTHON_3_12 matches this repo's .venv. The service also offers 3.10, 3.11, 3.13,
# 3.14 and NODE_22; the vendored wheels are built for whichever is named here, so
# this constant and WHEEL_PYTHON have to agree.
PYTHON_RUNTIME = "PYTHON_3_12"
WHEEL_PYTHON = "3.12"

# Runtime is ARM64. This is the whole cross-compile: pip resolves aarch64 wheels
# on any host, including an Intel or Apple-silicon Mac, with no container.
WHEEL_PLATFORM = "manylinux2014_aarch64"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(REPO_ROOT, "build", "runtime_zip")
ZIP_PATH = os.path.join(CACHE_DIR, "agent.zip")
FINGERPRINT_PATH = os.path.join(CACHE_DIR, "fingerprint.txt")
VENDOR_DIR = os.path.join(CACHE_DIR, "vendor")

# InvokeAgentRuntime rejects a runtimeSessionId shorter than 33 characters.
SESSION_ID_MIN = 33


def _source_fingerprint() -> str:
    """Hash requirements.txt and every .py under examples/.

    Content, not mtime: a fresh clone or a branch switch rewrites mtimes without
    changing what would be deployed, and rebuilding 196 MB of wheels because git
    touched a file is the thing the cache exists to avoid.
    """
    digest = hashlib.sha256()
    with open(os.path.join(REPO_ROOT, "requirements.txt"), "rb") as handle:
        digest.update(handle.read())
    for root, dirs, files in os.walk(os.path.join(REPO_ROOT, "examples")):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(root, name)
                digest.update(os.path.relpath(path, REPO_ROOT).encode())
                with open(path, "rb") as handle:
                    digest.update(handle.read())
    return f"{PYTHON_RUNTIME}:{WHEEL_PLATFORM}:{digest.hexdigest()}"


def _vendor_dependencies() -> None:
    """pip install requirements.txt into the staging tree, cross-compiled.

    --only-binary=:all: is not belt-and-braces with --platform: without it pip is
    allowed to fall back to building an sdist, which it would build for the host
    and not for aarch64, and the resulting zip fails the same way an
    un-cross-compiled one does but with a shorter list of offending files.
    """
    if os.path.isdir(VENDOR_DIR):
        shutil.rmtree(VENDOR_DIR)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--target", VENDOR_DIR,
            "--platform", WHEEL_PLATFORM,
            "--python-version", WHEEL_PYTHON,
            "--only-binary=:all:",
            "-r", os.path.join(REPO_ROOT, "requirements.txt"),
        ],
        check=True,
    )


# ELF header: b"\x7fELF" at 0, and e_machine as a little-endian short at 18.
# 183 is EM_AARCH64, which is the only value Runtime will load.
_ELF_MAGIC = b"\x7fELF"
_EM_AARCH64 = 183


def is_aarch64_elf(path: str) -> bool:
    """Whether this shared object is an ARM64 Linux binary, by its header.

    Read from the file rather than inferred from its name. Filenames are not
    reliable in either direction: pydantic_core ships as
    _pydantic_core.cpython-312-darwin.so, which names its platform, but
    numpy.libs/libscipy_openblas64_*.so and cryptography's _rust.abi3.so name no
    platform at all and are perfectly good aarch64 objects. A name-based check
    reports those two as the cause of an ARM64 failure they had nothing to do
    with, which sends the reader into numpy instead of into their pip flags.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(20)
    except OSError:
        return False
    if not header.startswith(_ELF_MAGIC) or len(header) < 20:
        return False
    return int.from_bytes(header[18:20], "little") == _EM_AARCH64


def _foreign_binaries() -> list:
    """Vendored shared objects Runtime cannot load, worst-first by directory."""
    return sorted(
        os.path.relpath(os.path.join(root, name), VENDOR_DIR)
        for root, _, files in os.walk(VENDOR_DIR)
        for name in files
        if name.endswith(".so") and not is_aarch64_elf(os.path.join(root, name))
    )


def build_zip(force: bool = False) -> Tuple[str, bool]:
    """Vendor the dependencies and zip them with the examples tree.

    Returns (path, rebuilt). __pycache__ is excluded: pip leaves .pyc files behind
    it and including them added 16 MB to the artifact for no effect, since Python
    recompiles from source in the container anyway.
    """
    fingerprint = _source_fingerprint()
    if not force and os.path.exists(ZIP_PATH) and os.path.exists(FINGERPRINT_PATH):
        with open(FINGERPRINT_PATH) as handle:
            if handle.read().strip() == fingerprint:
                size = os.path.getsize(ZIP_PATH) / 1e6
                print(f"Reusing cached artifact {ZIP_PATH} ({size:.1f} MB)")
                return ZIP_PATH, False

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Vendoring {WHEEL_PLATFORM} wheels (this is the slow step)")
    _vendor_dependencies()

    def entries():
        for root, dirs, files in os.walk(VENDOR_DIR):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                if not name.endswith(".pyc"):
                    path = os.path.join(root, name)
                    yield path, os.path.relpath(path, VENDOR_DIR)
        for root, dirs, files in os.walk(os.path.join(REPO_ROOT, "examples")):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                if not name.endswith(".pyc"):
                    path = os.path.join(root, name)
                    yield path, os.path.relpath(path, REPO_ROOT)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in entries():
            archive.write(path, arcname)
    with open(FINGERPRINT_PATH, "w") as handle:
        handle.write(fingerprint)
    print(f"Built {ZIP_PATH} ({os.path.getsize(ZIP_PATH) / 1e6:.1f} MB)")
    return ZIP_PATH, True


def bucket_name(account_id: str) -> str:
    """A deterministic bucket name, so a re-run finds the one it made last time.

    The account id is the suffix because bucket names are global: without it the
    first reader to run this would take the name for everyone else.
    """
    return f"migrated-agent-runtime-{account_id}"


def ensure_bucket(s3, bucket: str, region_name: str) -> None:
    """Create the bucket unless it is already there."""
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"Bucket {bucket} already exists")
        return
    except s3.exceptions.ClientError as error:
        if error.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
            raise
    # us-east-1 is the one region CreateBucket rejects a LocationConstraint for.
    if region_name == "us-east-1":
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region_name},
        )
    print(f"Created bucket {bucket}")


def _trust_policy() -> dict:
    """Runtime assumes this role, so the principal is the service."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": TRUST_PRINCIPAL},
                "Action": "sts:AssumeRole",
            }
        ],
    }


def _execution_policy(bucket: str) -> dict:
    """What the agent needs once it is running, and the S3 read to start at all.

    The S3 read is the one permission this path adds over the container path: the
    service fetches the artifact as this role, so a role without it fails the
    deploy rather than the invocation.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": f"arn:aws:s3:::{bucket}/*",
            },
            {
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": ["bedrock-agentcore:*"],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                "Resource": "*",
            },
        ],
    }


def ensure_role(iam, bucket: str) -> str:
    """Create or update the execution role and return its ARN."""
    trust = json.dumps(_trust_policy())
    try:
        iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=trust,
            Description="Execution role for the AgentCore Runtime zip deploy.",
        )
        print(f"Created role {ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        iam.update_assume_role_policy(RoleName=ROLE_NAME, PolicyDocument=trust)
        print(f"Role {ROLE_NAME} already exists")
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="RuntimeExecution",
        PolicyDocument=json.dumps(_execution_policy(bucket)),
    )
    return iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]


def _find_existing_runtime(control, name: str) -> Optional[str]:
    """Return the agentRuntimeId of a runtime with this name, or None."""
    for page in control.get_paginator("list_agent_runtimes").paginate():
        for summary in page.get("agentRuntimes", []):
            if summary.get("agentRuntimeName") == name:
                return summary["agentRuntimeId"]
    return None


def existing_runtime_id(region_name: str = "us-east-1") -> Optional[str]:
    """Whether a runtime of ours is already in the account, before deploy runs.

    Asked by the walkthrough so that the createdAt -> lastUpdatedAt delta can be
    labelled with which of create and update produced it. The two are not the same
    quantity and reporting one as the other is a wrong number, not a rounding.
    """
    control = boto3.client("bedrock-agentcore-control", region_name=region_name)
    return _find_existing_runtime(control, RUNTIME_NAME)


def provisioning_delta(runtime_id: str, region_name: str = "us-east-1") -> float:
    """The service's own createdAt -> lastUpdatedAt gap, in seconds.

    A second opinion on the time to READY, taken from the control plane's
    timestamps instead of from a client-side stopwatch, so it does not share the
    stopwatch's error: polling interval, clock skew and the latency of the calls
    around it are all in the wall-clock figure and none of them are in this one.

    Only equals provisioning time on a runtime created by the run that reads it.
    After an UpdateAgentRuntime the gap spans everything since the original
    create, which is why the caller passes the create-or-update distinction into
    the note rather than letting the number speak for itself.
    """
    runtime = boto3.client(
        "bedrock-agentcore-control", region_name=region_name
    ).get_agent_runtime(agentRuntimeId=runtime_id)
    return (runtime["lastUpdatedAt"] - runtime["createdAt"]).total_seconds()


def _fail_on_wrong_architecture(reason: str) -> None:
    """Turn the service's ARM64 rejection into the files that caused it.

    The service says the artifact contains incompatible binaries and does not say
    which, and a reader who has just watched a 51 MB upload succeed has no reason
    to suspect pip. Naming the files and the flag is the difference between a
    two-minute fix and an afternoon.
    """
    if "ARM64" not in reason and "arm64" not in reason:
        return
    offenders = _foreign_binaries()
    listed = "\n".join(f"    {name}" for name in offenders[:10]) or "    (none found)"
    raise RuntimeError(
        f"Runtime rejected the artifact: {reason}\n"
        "  Runtime is Linux ARM64 and these vendored binaries are not:\n"
        f"{listed}\n"
        "  Rebuild with --force. The build already passes\n"
        f"    pip install --platform {WHEEL_PLATFORM} --only-binary=:all:\n"
        "  so if this fired, that flag was lost or a dependency has no aarch64 "
        "wheel and pip fell back to building an sdist for this host."
    )


def wait_ready(control, runtime_id: str, timeout: int = 420) -> float:
    """Poll GetAgentRuntime until READY, and return how long that took.

    READY does not mean the same thing here as on the container path. This path
    validates the artifact during creation — a bad binary fails in about ten
    seconds, before any invocation — whereas a container reaches READY without the
    image having been pulled. It still does not mean the app imports cleanly:
    that is only ever proved by an invocation.
    """
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        runtime = control.get_agent_runtime(agentRuntimeId=runtime_id)
        status = runtime["status"]
        if status == "READY":
            return time.monotonic() - started
        if status in ("CREATE_FAILED", "UPDATE_FAILED", "FAILED"):
            reason = str(runtime.get("failureReason", "no failureReason given"))
            _fail_on_wrong_architecture(reason)
            raise RuntimeError(f"Runtime {runtime_id} is {status}: {reason}")
        time.sleep(2)
    raise TimeoutError(f"Runtime {runtime_id} not READY after {timeout}s")


def deploy(
    gateway_url: str,
    memory_id: str,
    actor_id: str = "langgraph",
    region_name: str = "us-east-1",
    force_build: bool = False,
) -> Tuple[str, str, float]:
    """Build, upload and deploy. Returns (runtimeId, runtimeArn, seconds to READY).

    Config reaches the agent through environmentVariables, which is how
    agent_runtime.py:58 and :85 read GATEWAY_URL and AGENTCORE_MEMORY_ID. That is
    the reason this deploy needs no change to the agent: the two things that
    differ between a local run and a hosted one are both environment.
    """
    session = boto3.Session(region_name=region_name)
    account_id = session.client("sts").get_caller_identity()["Account"]
    s3 = session.client("s3")
    iam = session.client("iam")
    control = session.client("bedrock-agentcore-control")

    zip_path, _ = build_zip(force=force_build)
    bucket = bucket_name(account_id)
    ensure_bucket(s3, bucket, region_name)
    role_arn = ensure_role(iam, bucket)

    key = "agent.zip"
    s3.upload_file(zip_path, bucket, key)
    print(f"Uploaded s3://{bucket}/{key}")

    artifact = {
        "codeConfiguration": {
            # prefix is the object key, not a directory prefix.
            "code": {"s3": {"bucket": bucket, "prefix": key}},
            "runtime": PYTHON_RUNTIME,
            "entryPoint": [ENTRY_POINT],
        }
    }
    common = {
        "agentRuntimeArtifact": artifact,
        "roleArn": role_arn,
        "networkConfiguration": {"networkMode": "PUBLIC"},
        "environmentVariables": {
            "GATEWAY_URL": gateway_url,
            "AGENTCORE_MEMORY_ID": memory_id,
            "AGENTCORE_ACTOR_ID": actor_id,
            "AWS_REGION": region_name,
        },
    }

    existing = _find_existing_runtime(control, RUNTIME_NAME)
    if existing is not None:
        # A re-run points the same runtime at the freshly uploaded zip rather than
        # failing on the duplicate name or leaving a stale artifact deployed.
        print(f"Runtime {RUNTIME_NAME} already exists: {existing}; updating")
        control.update_agent_runtime(agentRuntimeId=existing, **common)
        runtime_id = existing
        runtime_arn = control.get_agent_runtime(agentRuntimeId=runtime_id)[
            "agentRuntimeArn"
        ]
    else:
        created = _create_when_the_role_is_visible(control, common)
        runtime_id = created["agentRuntimeId"]
        runtime_arn = created["agentRuntimeArn"]
        print(f"Created runtime {RUNTIME_NAME}: {runtime_id}")

    elapsed = wait_ready(control, runtime_id)
    print(f"Runtime READY in {elapsed:.1f}s")
    return runtime_id, runtime_arn, elapsed


def _create_when_the_role_is_visible(control, common: dict, timeout: int = 120) -> dict:
    """CreateAgentRuntime, retrying while IAM has not yet propagated the role.

    ensure_role returns as soon as CreateRole does, but the role is not yet visible
    to other services, and CreateAgentRuntime validates it by trying to assume it.
    The failure it raises is a ValidationException reading "Role validation failed
    ... verify that the role exists and its trust policy allows assumption by this
    service", which describes a broken trust policy and is in fact a race: the same
    call succeeds a few seconds later against the same unmodified role.

    Retried rather than slept before, because the propagation delay has no
    documented bound and a sleep long enough to be safe is a delay paid on every
    re-run, when the role has existed for hours. Only the create is retried: once a
    runtime exists the role has demonstrably propagated.

    A genuinely wrong trust policy still fails, at the timeout, with the service's
    own message. This is why the retry is narrow — a bare ValidationException retry
    would also swallow a malformed artifact for two minutes.
    """
    deadline = time.monotonic() + timeout
    attempts = 0
    while True:
        try:
            return control.create_agent_runtime(agentRuntimeName=RUNTIME_NAME, **common)
        except control.exceptions.ValidationException as error:
            attempts += 1
            if "Role validation failed" not in str(error):
                raise
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"CreateAgentRuntime still rejected {ROLE_NAME} after {timeout}s "
                    f"and {attempts} attempts. If the role was just created this is "
                    "IAM propagation; if not, check that its trust policy names "
                    f"{TRUST_PRINCIPAL}.\n  {error}"
                ) from error
            if attempts == 1:
                print(f"  role {ROLE_NAME} not visible to the service yet; retrying")
            time.sleep(5)


def _explain_invoke_failure(error: Exception) -> None:
    """Say what the 30s initialization message actually means.

    The service reports a missing dependency as an initialization timeout. A
    reader who believes the message tunes a timeout that was never the problem,
    so the misdirection is called out here rather than left in CloudWatch.
    """
    text = str(error)
    if "initialization" in text.lower() or "30s" in text:
        print(
            "  This message is misleading. An ImportError during module import "
            "surfaces as an initialization timeout, so check CloudWatch for a "
            "missing vendored dependency before touching any timeout: a "
            "requirements.txt inside the zip is not installed."
        )


def invoke(runtime_arn: str, prompt: str, session_id: str, region_name: str) -> Tuple[str, float]:
    """Invoke the runtime once and return (response text, seconds)."""
    client = boto3.client("bedrock-agentcore", region_name=region_name)
    started = time.monotonic()
    try:
        response = client.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn,
            runtimeSessionId=session_id,
            payload=json.dumps({"prompt": prompt}).encode(),
        )
    except Exception as error:
        print(f"  invoke failed after {time.monotonic() - started:.1f}s: {error}")
        _explain_invoke_failure(error)
        raise
    body = response["response"].read().decode()
    return body, time.monotonic() - started


def delete_runtime(runtime_id: str, region_name: str = "us-east-1") -> None:
    """Delete the runtime and wait until GetAgentRuntime says it is gone.

    Deleting the runtime does not delete the CloudWatch log group the service
    created for it, so delete_log_group is a separate step rather than part of
    this one. The name is printed here as well, because the id it is built from
    stops being discoverable the moment this call succeeds.
    """
    control = boto3.client("bedrock-agentcore-control", region_name=region_name)
    try:
        control.delete_agent_runtime(agentRuntimeId=runtime_id)
    except control.exceptions.ResourceNotFoundException:
        print(f"Runtime {runtime_id} already deleted")
        return
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            control.get_agent_runtime(agentRuntimeId=runtime_id)
        except control.exceptions.ResourceNotFoundException:
            print(f"Deleted runtime {runtime_id}")
            print(f"  log group left behind: {log_group_name(runtime_id)}")
            return
        time.sleep(2)
    raise TimeoutError(f"Runtime {runtime_id} still present after 180s")


def log_group_name(runtime_id: str) -> str:
    """Where Runtime writes this runtime's logs, and what survives its deletion."""
    return f"/aws/bedrock-agentcore/runtimes/{runtime_id}-DEFAULT"


def delete_log_group(name: str, region_name: str = "us-east-1") -> None:
    """Delete a log group the service created implicitly.

    Nothing in this walkthrough asked for this group: the first write created it,
    which is exactly why it is easy to leave behind. It is not billed for storage
    it does not hold, but --teardown claims to put the account back the way it was
    found, and a resource that no longer appears in any Delete* call's blast radius
    is how an account accumulates the kind that nobody is auditing.

    Deleted last, after the runtime. In the other order the runtime is still alive
    and its next log line recreates the group.
    """
    logs = boto3.client("logs", region_name=region_name)
    try:
        logs.delete_log_group(logGroupName=name)
        print(f"Deleted log group {name}")
    except logs.exceptions.ResourceNotFoundException:
        # Never written to, so never created. A runtime that reached READY and was
        # then deleted without being invoked leaves no group behind.
        print(f"No log group {name} to delete")


def delete_bucket(bucket: str, region_name: str = "us-east-1") -> None:
    """Empty the artifact bucket and delete it. A non-empty bucket will not go."""
    s3 = boto3.client("s3", region_name=region_name)
    try:
        for page in s3.get_paginator("list_object_versions").paginate(Bucket=bucket):
            targets = [
                {"Key": item["Key"], "VersionId": item["VersionId"]}
                for key in ("Versions", "DeleteMarkers")
                for item in page.get(key, [])
            ]
            if targets:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": targets})
        s3.delete_bucket(Bucket=bucket)
        print(f"Deleted bucket {bucket}")
    except s3.exceptions.NoSuchBucket:
        print(f"Bucket {bucket} already deleted")


def delete_role(region_name: str = "us-east-1") -> None:
    """Delete the execution role, inline policies first."""
    iam = boto3.client("iam", region_name=region_name)
    try:
        for page in iam.get_paginator("list_role_policies").paginate(RoleName=ROLE_NAME):
            for name in page["PolicyNames"]:
                iam.delete_role_policy(RoleName=ROLE_NAME, PolicyName=name)
        iam.delete_role(RoleName=ROLE_NAME)
        print(f"Deleted role {ROLE_NAME}")
    except iam.exceptions.NoSuchEntityException:
        print(f"Role {ROLE_NAME} already deleted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default=os.environ.get("GATEWAY_URL"))
    parser.add_argument("--memory-id", default=os.environ.get("AGENTCORE_MEMORY_ID"))
    parser.add_argument("--actor-id", default="langgraph")
    parser.add_argument(
        "--region", default=os.environ.get("AWS_REGION", "us-east-1")
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Re-vendor the wheels even if the cached artifact still matches.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the zip and stop, creating nothing in AWS.",
    )
    args = parser.parse_args()

    if args.build_only:
        build_zip(force=args.force_build)
        return
    if not args.gateway_url or not args.memory_id:
        parser.error("--gateway-url and --memory-id are required (or set GATEWAY_URL / AGENTCORE_MEMORY_ID)")

    runtime_id, runtime_arn, _ = deploy(
        args.gateway_url,
        args.memory_id,
        args.actor_id,
        args.region,
        force_build=args.force_build,
    )
    session_id = f"deploy-runtime-{runtime_id}".ljust(SESSION_ID_MIN, "0")
    body, seconds = invoke(
        runtime_arn, "Hi, I'm Dana and my order number is 12345.", session_id, args.region
    )
    print(f"invoked in {seconds:.1f}s -> {body[:300]}")


if __name__ == "__main__":
    main()

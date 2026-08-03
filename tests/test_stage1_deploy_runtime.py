# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Verification for the zip deploy to AgentCore Runtime.

Runs offline and makes no AWS calls. pip is not run either: _vendor_dependencies
is replaced by a function that writes a handful of files, so the packaging rules
— what is excluded, where the entry point lands, when the cache is reused — are
checked against a real zipfile built by the real build_zip.

What is worth testing here is not the API plumbing but the three things that cost
a failed deploy to learn, each of which is invisible until the artifact is in the
cloud sixteen seconds later:

1. The pip invocation carries the cross-compile flags. Losing --platform or
   --only-binary is a one-word edit and the resulting zip uploads happily and
   then fails with a message about binary files that names no file.
2. The entry point exists in the archive at the path CreateAgentRuntime is told
   to run. ENTRY_POINT and the archive layout are set in two different places and
   nothing but a deploy connects them.
3. is_aarch64_elf reads headers rather than filenames, in both directions. The
   name-based version of this check reported numpy and cryptography as the cause
   of an ARM64 failure they had no part in.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import io
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from datetime import datetime, timezone
from unittest import mock

from examples.stage1_replatform import deploy_runtime


def elf_header(machine: int) -> bytes:
    """A 20-byte ELF header claiming a given e_machine, padded to length."""
    return b"\x7fELF" + b"\x00" * 14 + machine.to_bytes(2, "little")


class FakeRuntimeControlClient:
    """Enough of bedrock-agentcore-control to exercise the create/update branch."""

    class exceptions:
        class ResourceNotFoundException(Exception):
            pass

        class ValidationException(Exception):
            pass

    def __init__(self, existing=None, statuses=("READY",)):
        self.existing = existing or {}
        self.statuses = list(statuses)
        self.calls = []
        self.failure_reason = ""
        # How many CreateAgentRuntime calls fail on role validation before one
        # works. The live service does this whenever the role was just created.
        self.role_not_propagated = 0
        self.create_error = (
            "Role validation failed for 'arn:aws:iam::1:role/MigratedAgentRuntimeRole'. "
            "Please verify that the role exists and its trust policy allows "
            "assumption by this service"
        )
        # What the control plane reports about its own timing, which is the fourth
        # measurement's only source. Real datetimes, because the code subtracts
        # them and a float would let a wrong subtraction pass.
        self.created_at = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)
        self.last_updated_at = datetime(2026, 8, 2, 12, 0, 16, tzinfo=timezone.utc)

    def get_paginator(self, name):
        assert name == "list_agent_runtimes", name
        summaries = [
            {"agentRuntimeId": rid, "agentRuntimeName": data["name"]}
            for rid, data in self.existing.items()
        ]
        return mock.Mock(paginate=lambda **kw: iter([{"agentRuntimes": summaries}]))

    def create_agent_runtime(self, **kwargs):
        self.calls.append(("create_agent_runtime", kwargs))
        if self.role_not_propagated > 0:
            self.role_not_propagated -= 1
            raise self.exceptions.ValidationException(self.create_error)
        rid = f"{kwargs['agentRuntimeName']}-abc1234567"
        self.existing[rid] = {"name": kwargs["agentRuntimeName"]}
        return {
            "agentRuntimeId": rid,
            "agentRuntimeArn": f"arn:aws:bedrock-agentcore:us-east-1:1:runtime/{rid}",
        }

    def update_agent_runtime(self, **kwargs):
        self.calls.append(("update_agent_runtime", kwargs))
        return {"agentRuntimeId": kwargs["agentRuntimeId"]}

    def get_agent_runtime(self, **kwargs):
        self.calls.append(("get_agent_runtime", kwargs))
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        rid = kwargs["agentRuntimeId"]
        result = {
            "agentRuntimeId": rid,
            "agentRuntimeArn": f"arn:aws:bedrock-agentcore:us-east-1:1:runtime/{rid}",
            "status": status,
            "createdAt": self.created_at,
            "lastUpdatedAt": self.last_updated_at,
        }
        if self.failure_reason:
            result["failureReason"] = self.failure_reason
        return result


class FakeS3Client:
    class exceptions:
        class ClientError(Exception):
            def __init__(self, code):
                super().__init__(code)
                self.response = {"Error": {"Code": code}}

        class NoSuchBucket(Exception):
            pass

    def __init__(self, buckets=()):
        self.buckets = set(buckets)
        self.calls = []

    def head_bucket(self, Bucket):
        self.calls.append(("head_bucket", Bucket))
        if Bucket not in self.buckets:
            raise self.exceptions.ClientError("404")

    def create_bucket(self, **kwargs):
        self.calls.append(("create_bucket", kwargs))
        self.buckets.add(kwargs["Bucket"])

    def upload_file(self, path, bucket, key):
        self.calls.append(("upload_file", path, bucket, key))


class FakeIAMClient:
    class exceptions:
        class EntityAlreadyExistsException(Exception):
            pass

        class NoSuchEntityException(Exception):
            pass

    def __init__(self, existing=False):
        self.existing = existing
        self.calls = []
        self.policies = {}

    def create_role(self, **kwargs):
        self.calls.append(("create_role", kwargs))
        if self.existing:
            raise self.exceptions.EntityAlreadyExistsException()
        self.existing = True

    def update_assume_role_policy(self, **kwargs):
        self.calls.append(("update_assume_role_policy", kwargs))

    def put_role_policy(self, **kwargs):
        self.calls.append(("put_role_policy", kwargs))
        self.policies[kwargs["PolicyName"]] = json.loads(kwargs["PolicyDocument"])

    def get_role(self, RoleName):
        return {"Role": {"Arn": f"arn:aws:iam::123456789012:role/{RoleName}"}}


class PackagingTest(unittest.TestCase):
    """build_zip, against a real zipfile and a faked pip."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vendor = os.path.join(self.tmp, "vendor")
        patches = {
            "CACHE_DIR": self.tmp,
            "ZIP_PATH": os.path.join(self.tmp, "agent.zip"),
            "FINGERPRINT_PATH": os.path.join(self.tmp, "fingerprint.txt"),
            "VENDOR_DIR": self.vendor,
        }
        for name, value in patches.items():
            patcher = mock.patch.object(deploy_runtime, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.vendor_calls = []

    def fake_vendor(self):
        """Stand in for pip: one package, one .pyc and one __pycache__ dir."""
        self.vendor_calls.append(True)
        os.makedirs(os.path.join(self.vendor, "somepkg", "__pycache__"), exist_ok=True)
        with open(os.path.join(self.vendor, "somepkg", "__init__.py"), "w") as handle:
            handle.write("x = 1\n")
        with open(os.path.join(self.vendor, "somepkg", "mod.pyc"), "wb") as handle:
            handle.write(b"\x00")
        with open(
            os.path.join(self.vendor, "somepkg", "__pycache__", "c.cpython-312.pyc"), "wb"
        ) as handle:
            handle.write(b"\x00")

    def build(self, force=False):
        with mock.patch.object(
            deploy_runtime, "_vendor_dependencies", self.fake_vendor
        ), redirect_stdout(io.StringIO()) as out:
            path, rebuilt = deploy_runtime.build_zip(force=force)
        return path, rebuilt, out.getvalue()

    def test_pip_carries_the_cross_compile_flags(self):
        """The one flag combination that makes this work without a container."""
        with mock.patch.object(deploy_runtime.subprocess, "run") as run:
            deploy_runtime._vendor_dependencies()
        argv = run.call_args[0][0]
        self.assertIn("--platform", argv)
        self.assertIn("manylinux2014_aarch64", argv)
        self.assertIn("--only-binary=:all:", argv)
        self.assertIn("--python-version", argv)
        self.assertIn(deploy_runtime.WHEEL_PYTHON, argv)
        self.assertTrue(run.call_args.kwargs["check"])

    def test_wheel_python_matches_the_declared_runtime(self):
        """PYTHON_3_12 and the 3.12 wheels have to be the same 3.12."""
        self.assertEqual(
            deploy_runtime.PYTHON_RUNTIME,
            "PYTHON_" + deploy_runtime.WHEEL_PYTHON.replace(".", "_"),
        )

    def test_archive_excludes_pycache_and_pyc(self):
        """16 MB of .pyc that the container recompiles from source anyway."""
        path, rebuilt, _ = self.build()
        self.assertTrue(rebuilt)
        names = zipfile.ZipFile(path).namelist()
        self.assertTrue(names)
        self.assertEqual([n for n in names if "__pycache__" in n], [])
        self.assertEqual([n for n in names if n.endswith(".pyc")], [])

    def test_archive_contains_the_entry_point_at_the_deployed_path(self):
        """ENTRY_POINT and the archive layout are set in two places."""
        path, _, _ = self.build()
        names = zipfile.ZipFile(path).namelist()
        self.assertIn(deploy_runtime.ENTRY_POINT, names)

    def test_archive_keeps_the_package_layout_the_imports_need(self):
        """agent_runtime.py imports examples.* absolutely, so __init__ must ship."""
        path, _, _ = self.build()
        names = set(zipfile.ZipFile(path).namelist())
        self.assertIn("examples/__init__.py", names)
        self.assertIn("examples/stage1_replatform/__init__.py", names)
        self.assertIn("examples/stage0_langgraph/agent.py", names)

    def test_archive_vendors_dependencies_at_the_root(self):
        """/var/task is on sys.path, so a vendored package sits beside examples/."""
        path, _, _ = self.build()
        names = zipfile.ZipFile(path).namelist()
        self.assertIn("somepkg/__init__.py", names)

    def test_second_build_reuses_the_cache_and_does_not_run_pip(self):
        """The whole point of the cache: --stage all twice, one vendoring."""
        self.build()
        self.assertEqual(len(self.vendor_calls), 1)
        path, rebuilt, output = self.build()
        self.assertFalse(rebuilt)
        self.assertEqual(len(self.vendor_calls), 1)
        self.assertIn("Reusing cached artifact", output)

    def test_force_rebuilds_even_when_the_fingerprint_matches(self):
        self.build()
        _, rebuilt, _ = self.build(force=True)
        self.assertTrue(rebuilt)
        self.assertEqual(len(self.vendor_calls), 2)

    def test_changed_source_invalidates_the_cache(self):
        """Fingerprint is over content, so an edit must force a rebuild."""
        first = deploy_runtime._source_fingerprint()
        target = os.path.join(
            os.path.dirname(os.path.abspath(deploy_runtime.__file__)), "agent_runtime.py"
        )
        with open(target, "rb") as handle:
            original = handle.read()
        try:
            with open(target, "wb") as handle:
                handle.write(original + b"\n# cache probe\n")
            self.assertNotEqual(first, deploy_runtime._source_fingerprint())
        finally:
            with open(target, "wb") as handle:
                handle.write(original)
        self.assertEqual(first, deploy_runtime._source_fingerprint())

    def test_fingerprint_ignores_mtime(self):
        """A fresh clone rewrites mtimes without changing what would deploy.

        The mtime is restored afterwards, and not only for tidiness: zipfile
        stores mtimes and refuses anything before 1980, so a test that leaves an
        epoch timestamp on a source file breaks every later build_zip test with
        an error about timestamps that has nothing to do with what it broke.
        """
        first = deploy_runtime._source_fingerprint()
        target = os.path.join(
            os.path.dirname(os.path.abspath(deploy_runtime.__file__)), "agent_runtime.py"
        )
        stat = os.stat(target)
        self.addCleanup(os.utime, target, (stat.st_atime, stat.st_mtime))
        # 2000-01-01, not 1980-01-01: zip stores LOCAL time, so the epoch that is
        # exactly the 1980 floor in UTC is below it in any western timezone.
        os.utime(target, (946_684_800, 946_684_800))
        self.assertEqual(first, deploy_runtime._source_fingerprint())

    def test_fingerprint_covers_the_platform_and_runtime(self):
        """Retargeting the wheels has to invalidate an artifact built for the old one."""
        self.assertIn(deploy_runtime.WHEEL_PLATFORM, deploy_runtime._source_fingerprint())
        self.assertIn(deploy_runtime.PYTHON_RUNTIME, deploy_runtime._source_fingerprint())


class ArchitectureCheckTest(unittest.TestCase):
    """is_aarch64_elf, in both directions, and the error message built on it."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, name, payload):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def test_accepts_an_aarch64_elf(self):
        self.assertTrue(self.check(self.write("ok.so", elf_header(183))))

    def test_rejects_an_x86_64_elf(self):
        self.assertFalse(self.check(self.write("x86.so", elf_header(62))))

    def test_rejects_a_mach_o_object(self):
        """A plain pip install --target on macOS produces exactly these."""
        self.assertFalse(self.check(self.write("mac.so", b"\xcf\xfa\xed\xfe" + b"\x00" * 16)))

    def test_rejects_a_truncated_header(self):
        self.assertFalse(self.check(self.write("short.so", b"\x7fELF")))

    def test_rejects_a_missing_file(self):
        self.assertFalse(self.check(os.path.join(self.tmp, "absent.so")))

    def check(self, path):
        return deploy_runtime.is_aarch64_elf(path)

    def test_arm64_failure_names_the_files_and_the_flag(self):
        """The service says "binary files" and names none of them."""
        with mock.patch.object(
            deploy_runtime, "_foreign_binaries",
            lambda: ["pydantic_core/_pydantic_core.cpython-312-darwin.so"],
        ):
            with self.assertRaises(RuntimeError) as caught:
                deploy_runtime._fail_on_wrong_architecture(
                    "Your artifact contains binary files incompatible with Linux ARM64"
                )
        message = str(caught.exception)
        self.assertIn("pydantic_core", message)
        self.assertIn("manylinux2014_aarch64", message)
        self.assertIn("--only-binary=:all:", message)

    def test_a_failure_that_is_not_about_architecture_is_left_alone(self):
        """Every other failureReason has to reach the caller unaltered."""
        self.assertIsNone(
            deploy_runtime._fail_on_wrong_architecture("Role cannot be assumed")
        )


class InvokeFailureTest(unittest.TestCase):
    """The misleading timeout message, which is the expensive one."""

    def explain(self, error):
        with redirect_stdout(io.StringIO()) as out:
            deploy_runtime._explain_invoke_failure(RuntimeError(error))
        return out.getvalue()

    def test_initialization_timeout_is_called_misleading(self):
        text = self.explain(
            "Runtime initialization time exceeded. Ensure initialization completes in 30s"
        )
        self.assertIn("misleading", text)
        self.assertIn("CloudWatch", text)
        self.assertIn("requirements.txt", text)

    def test_an_unrelated_error_gets_no_advice(self):
        self.assertEqual(self.explain("AccessDeniedException"), "")


class DeployTest(unittest.TestCase):
    """The create/update branch and the request codeConfiguration is given."""

    def setUp(self):
        self.control = FakeRuntimeControlClient()
        self.s3 = FakeS3Client()
        self.iam = FakeIAMClient()
        self.sts = mock.Mock(
            get_caller_identity=lambda: {"Account": "123456789012"}
        )
        clients = {
            "s3": self.s3, "iam": self.iam,
            "bedrock-agentcore-control": self.control, "sts": self.sts,
        }
        session = mock.Mock(client=lambda name, **kw: clients[name])
        patcher = mock.patch.object(
            deploy_runtime.boto3, "Session", lambda **kw: session
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        build = mock.patch.object(
            deploy_runtime, "build_zip", lambda force=False: ("/tmp/agent.zip", False)
        )
        build.start()
        self.addCleanup(build.stop)
        sleep = mock.patch.object(deploy_runtime.time, "sleep", lambda s: None)
        sleep.start()
        self.addCleanup(sleep.stop)

    def deploy(self):
        with redirect_stdout(io.StringIO()) as out:
            result = deploy_runtime.deploy(
                "https://gateway.example/mcp", "MigratedAgentMemory-abc123"
            )
        return result, out.getvalue()

    def sent(self, operation):
        return [kw for name, kw in self.control.calls if name == operation][0]

    def test_creates_a_runtime_when_none_exists(self):
        (runtime_id, runtime_arn, elapsed), _ = self.deploy()
        self.assertEqual([n for n, _ in self.control.calls][0], "create_agent_runtime")
        self.assertTrue(runtime_id.startswith(deploy_runtime.RUNTIME_NAME))
        self.assertIn("runtime/", runtime_arn)
        self.assertGreaterEqual(elapsed, 0)

    def test_reuses_and_updates_an_existing_runtime(self):
        """A second --stage all must not fail on the duplicate name."""
        self.control.existing = {
            "MigratedAgentRuntime-existing1": {"name": deploy_runtime.RUNTIME_NAME}
        }
        (runtime_id, _, _), output = self.deploy()
        self.assertEqual(runtime_id, "MigratedAgentRuntime-existing1")
        self.assertIn("update_agent_runtime", [n for n, _ in self.control.calls])
        self.assertNotIn("create_agent_runtime", [n for n, _ in self.control.calls])
        self.assertIn("already exists", output)

    def test_a_differently_named_runtime_is_not_mistaken_for_ours(self):
        """The account has other runtimes; only an exact name match is ours."""
        self.control.existing = {"blogTestStrandsAgent-V1": {"name": "blogTestStrandsAgent"}}
        self.deploy()
        self.assertIn("create_agent_runtime", [n for n, _ in self.control.calls])

    def test_the_artifact_is_a_code_configuration(self):
        self.deploy()
        artifact = self.sent("create_agent_runtime")["agentRuntimeArtifact"]
        self.assertIn("codeConfiguration", artifact)
        self.assertNotIn("containerConfiguration", artifact)
        code = artifact["codeConfiguration"]
        self.assertEqual(code["runtime"], deploy_runtime.PYTHON_RUNTIME)
        self.assertEqual(code["entryPoint"], [deploy_runtime.ENTRY_POINT])
        self.assertEqual(
            code["code"]["s3"],
            {"bucket": "migrated-agent-runtime-123456789012", "prefix": "agent.zip"},
        )

    def test_config_reaches_the_agent_as_environment_variables(self):
        """agent_runtime.py reads both from os.environ, which is why it needs no edit."""
        self.deploy()
        env = self.sent("create_agent_runtime")["environmentVariables"]
        self.assertEqual(env["GATEWAY_URL"], "https://gateway.example/mcp")
        self.assertEqual(env["AGENTCORE_MEMORY_ID"], "MigratedAgentMemory-abc123")
        self.assertIn("AGENTCORE_ACTOR_ID", env)

    def test_the_zip_is_uploaded_before_the_runtime_is_created(self):
        """A runtime created against an absent key fails validation."""
        self.deploy()
        self.assertIn(("upload_file", "/tmp/agent.zip", "migrated-agent-runtime-123456789012", "agent.zip"), self.s3.calls)

    def test_create_failed_on_arm64_raises_the_specific_error(self):
        self.control.statuses = ["CREATE_FAILED"]
        self.control.failure_reason = (
            "Your artifact contains binary files incompatible with Linux ARM64"
        )
        with mock.patch.object(
            deploy_runtime, "_foreign_binaries", lambda: ["websockets/speedups.so"]
        ), self.assertRaises(RuntimeError) as caught, redirect_stdout(io.StringIO()):
            deploy_runtime.deploy("https://g", "m-1")
        self.assertIn("websockets/speedups.so", str(caught.exception))

    def test_create_failed_for_another_reason_still_raises(self):
        self.control.statuses = ["CREATE_FAILED"]
        self.control.failure_reason = "Role cannot be assumed"
        with self.assertRaises(RuntimeError) as caught, redirect_stdout(io.StringIO()):
            deploy_runtime.deploy("https://g", "m-1")
        self.assertIn("Role cannot be assumed", str(caught.exception))

    def test_a_role_that_has_not_propagated_yet_is_retried_not_reported(self):
        """The first live run's failure. The role is fine; IAM is just not ready.

        Nothing offline could have predicted this — it is a property of IAM's
        propagation, and the fake only models it because the live service did it
        first. What the test is worth is the other direction: it stops the retry
        being dropped later by someone who reads it as belt-and-braces.
        """
        self.control.role_not_propagated = 2
        (runtime_id, _, _), output = self.deploy()
        creates = [n for n, _ in self.control.calls if n == "create_agent_runtime"]
        self.assertEqual(len(creates), 3)
        self.assertTrue(runtime_id.startswith(deploy_runtime.RUNTIME_NAME))
        self.assertIn("not visible to the service yet", output)

    def test_the_retry_is_announced_once_and_not_per_attempt(self):
        self.control.role_not_propagated = 4
        _, output = self.deploy()
        self.assertEqual(output.count("not visible to the service yet"), 1)

    def test_a_role_that_never_propagates_fails_with_the_trust_principal_named(self):
        """A genuinely wrong trust policy raises the same error as the race.

        So the timeout message has to cover the case the retry cannot fix, or the
        reader waits two minutes and is then told to wait longer.
        """
        self.control.role_not_propagated = 10_000
        with self.assertRaises(RuntimeError) as caught, redirect_stdout(io.StringIO()):
            # timeout=0 rather than a large role_not_propagated: the retry loop is
            # fast enough offline that any finite count of failures is exhausted
            # before a real deadline passes, which would test the opposite thing.
            deploy_runtime._create_when_the_role_is_visible(
                self.control, {"roleArn": "arn:aws:iam::1:role/x"}, timeout=0
            )
        message = str(caught.exception)
        self.assertIn(deploy_runtime.TRUST_PRINCIPAL, message)
        self.assertIn(deploy_runtime.ROLE_NAME, message)

    def test_a_validation_error_that_is_not_about_the_role_is_not_retried(self):
        """A narrow retry. A bad artifact must fail now, not in two minutes."""
        self.control.create_error = "agentRuntimeArtifact.codeConfiguration is invalid"
        self.control.role_not_propagated = 1
        with self.assertRaises(
            FakeRuntimeControlClient.exceptions.ValidationException
        ), redirect_stdout(io.StringIO()):
            deploy_runtime.deploy("https://g", "m-1")
        creates = [n for n, _ in self.control.calls if n == "create_agent_runtime"]
        self.assertEqual(len(creates), 1)

    def test_it_waits_through_creating(self):
        self.control.statuses = ["CREATING", "CREATING", "READY"]
        (_, _, elapsed), _ = self.deploy()
        gets = [n for n, _ in self.control.calls if n == "get_agent_runtime"]
        self.assertGreaterEqual(len(gets), 3)
        self.assertIsInstance(elapsed, float)


class RoleAndBucketTest(unittest.TestCase):
    """The two supporting resources, and the trap in each."""

    def test_bucket_name_is_account_scoped(self):
        """Bucket names are global; without the account id the first reader wins."""
        self.assertEqual(
            deploy_runtime.bucket_name("123456789012"),
            "migrated-agent-runtime-123456789012",
        )

    def test_us_east_1_gets_no_location_constraint(self):
        """CreateBucket rejects a LocationConstraint of us-east-1."""
        s3 = FakeS3Client()
        with redirect_stdout(io.StringIO()):
            deploy_runtime.ensure_bucket(s3, "b", "us-east-1")
        kwargs = [kw for name, kw in s3.calls if name == "create_bucket"][0]
        self.assertNotIn("CreateBucketConfiguration", kwargs)

    def test_another_region_gets_a_location_constraint(self):
        s3 = FakeS3Client()
        with redirect_stdout(io.StringIO()):
            deploy_runtime.ensure_bucket(s3, "b", "eu-west-1")
        kwargs = [kw for name, kw in s3.calls if name == "create_bucket"][0]
        self.assertEqual(
            kwargs["CreateBucketConfiguration"], {"LocationConstraint": "eu-west-1"}
        )

    def test_an_existing_bucket_is_reused(self):
        s3 = FakeS3Client(buckets={"b"})
        with redirect_stdout(io.StringIO()) as out:
            deploy_runtime.ensure_bucket(s3, "b", "us-east-1")
        self.assertEqual([n for n, *_ in s3.calls], ["head_bucket"])
        self.assertIn("already exists", out.getvalue())

    def test_the_role_is_assumed_by_the_service_not_a_user(self):
        iam = FakeIAMClient()
        with redirect_stdout(io.StringIO()):
            deploy_runtime.ensure_role(iam, "some-bucket")
        trust = json.loads(
            [kw for n, kw in iam.calls if n == "create_role"][0][
                "AssumeRolePolicyDocument"
            ]
        )
        self.assertEqual(
            trust["Statement"][0]["Principal"],
            {"Service": "bedrock-agentcore.amazonaws.com"},
        )

    def test_the_role_can_read_the_artifact(self):
        """The service fetches the zip as this role, so a missing s3:GetObject
        fails the deploy rather than the invocation."""
        iam = FakeIAMClient()
        with redirect_stdout(io.StringIO()):
            deploy_runtime.ensure_role(iam, "some-bucket")
        statements = iam.policies["RuntimeExecution"]["Statement"]
        s3_statements = [s for s in statements if "s3:GetObject" in s["Action"]]
        self.assertEqual(len(s3_statements), 1)
        self.assertEqual(s3_statements[0]["Resource"], "arn:aws:s3:::some-bucket/*")

    def test_an_existing_role_is_updated_not_duplicated(self):
        iam = FakeIAMClient(existing=True)
        with redirect_stdout(io.StringIO()) as out:
            deploy_runtime.ensure_role(iam, "some-bucket")
        self.assertIn("update_assume_role_policy", [n for n, _ in iam.calls])
        self.assertIn("already exists", out.getvalue())


class TeardownContractTest(unittest.TestCase):
    """What teardown has to know about, expressed as the names it must delete."""

    def test_log_group_name_matches_what_the_service_creates(self):
        """Measured against a real orphan: the group outlives the runtime, so
        teardown has to name it from the id rather than discover it."""
        self.assertEqual(
            deploy_runtime.log_group_name("blog7bStage1LangGraphAgent-IRSHOsGvpZ"),
            "/aws/bedrock-agentcore/runtimes/"
            "blog7bStage1LangGraphAgent-IRSHOsGvpZ-DEFAULT",
        )

    def test_session_id_minimum_is_respected(self):
        """InvokeAgentRuntime rejects a runtimeSessionId shorter than 33."""
        self.assertGreaterEqual(deploy_runtime.SESSION_ID_MIN, 33)
        padded = f"deploy-runtime-x".ljust(deploy_runtime.SESSION_ID_MIN, "0")
        self.assertGreaterEqual(len(padded), 33)

    def test_deleting_a_log_group_that_was_never_written_to_is_not_an_error(self):
        """A runtime deleted without being invoked leaves no group behind.

        Teardown names the group from the runtime id rather than discovering it, so
        it will always ask. Asking for one that does not exist has to be ordinary,
        or every teardown of an uninvoked runtime ends in a failure list.
        """
        class Logs:
            class exceptions:
                class ResourceNotFoundException(Exception):
                    pass

            def delete_log_group(self, logGroupName):
                raise self.exceptions.ResourceNotFoundException(logGroupName)

        with mock.patch.object(deploy_runtime.boto3, "client", lambda *a, **k: Logs()):
            with redirect_stdout(io.StringIO()) as out:
                deploy_runtime.delete_log_group("/aws/bedrock-agentcore/runtimes/x")
        self.assertIn("No log group", out.getvalue())


class FourthMeasurementTest(unittest.TestCase):
    """The create -> lastUpdated delta, which is the one number with two meanings."""

    def setUp(self):
        self.control = FakeRuntimeControlClient(
            existing={"MigratedAgentRuntime-abc1234567": {"name": "MigratedAgentRuntime"}}
        )
        patcher = mock.patch.object(
            deploy_runtime.boto3, "client", lambda *a, **k: self.control
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_delta_comes_from_the_services_own_timestamps(self):
        """Not from a client stopwatch, which is the point of having it."""
        self.assertEqual(
            deploy_runtime.provisioning_delta("MigratedAgentRuntime-abc1234567"), 16.0
        )

    def test_a_zero_delta_is_reported_rather_than_treated_as_missing(self):
        """createdAt == lastUpdatedAt is a real answer, not an absent one."""
        self.control.last_updated_at = self.control.created_at
        self.assertEqual(
            deploy_runtime.provisioning_delta("MigratedAgentRuntime-abc1234567"), 0.0
        )

    def test_an_existing_runtime_is_reported_before_the_deploy_changes_the_answer(self):
        self.assertEqual(
            deploy_runtime.existing_runtime_id(), "MigratedAgentRuntime-abc1234567"
        )

    def test_no_runtime_of_ours_reads_as_none_not_as_someone_elses(self):
        self.control.existing = {"blogTestStrandsAgent-V1": {"name": "blogTestStrandsAgent"}}
        self.assertIsNone(deploy_runtime.existing_runtime_id())


if __name__ == "__main__":
    unittest.main()

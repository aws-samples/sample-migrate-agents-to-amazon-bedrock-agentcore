# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Register support_tools.cedar with Policy in AgentCore and attach it to the Gateway.

Four resources, in this order, and the order matters because each one needs the
identifier the previous one returned:

    policy engine -> one policy per Cedar rule -> attach the engine to the gateway
        -> tool calls through the gateway are now authorized

The engine is attached to the Gateway rather than to the agent, so nothing in
strands_agent.py changes. The gateway ARN has to be substituted into the rules
before they are created, because Cedar rejects wildcard resources and the ARN
does not exist until the gateway does — the same two-phase order the AgentCore
CLI documents.

Evaluation has no synchronous API to call. The policy engine decides at the
gateway boundary on every tool call, so the way to observe a decision is to make
the call: call_tool_through_gateway does that and reports whether the tool ran or
the request was refused. That is what the walkthrough uses to show the read-only
caller being permitted lookup_order and refused process_return.

Deleting is the other half of this file. A policy engine left attached to a
deleted gateway is a resource nobody is looking at, so teardown deletes the
policies and then the engine.
"""

import argparse
import os
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3

from examples.tools.gateway_mcp_tools import build_mcp_client

POLICY_FILE = Path(__file__).with_name("support_tools.cedar")
POLICY_ENGINE_NAME = "SupportToolsPolicyEngine"

# Substituted into the Cedar text at registration time. Values, not names: the
# rules cannot be created until the gateway they name exists.
GATEWAY_ARN_PLACEHOLDER = "<GATEWAY_ARN>"
READ_ONLY_PLACEHOLDER = "<READ_ONLY_PRINCIPAL_ARN>"
SUPPORT_AGENT_PLACEHOLDER = "<SUPPORT_AGENT_PRINCIPAL_ARN>"

# The privileged identity this module defaults to when invoked as a script with
# no --support-agent-principal. Cedar does not require the principal to exist —
# measured — so nothing creates it, which makes it a usable default and a poor
# demonstration: a permit for a role nobody holds cannot be shown being used.
# run_walkthrough passes two real roles instead; see
# examples/stage2_rebuild/policy/demo_principals.py.
SUPPORT_AGENT_ROLE_NAME = "SupportEscalationRole"

# One AgentCore policy per marked block in the .cedar file.
_POLICY_MARKER = re.compile(r"^// === policy: (\w+) ===$", re.MULTILINE)

# Terminal statuses the create/delete waiters stop on.
_ACTIVE = "ACTIVE"
_CREATING = ("CREATING", "UPDATING")


def read_policy_blocks(path: Path = POLICY_FILE) -> List[Tuple[str, str]]:
    """Split the .cedar file into (policy name, Cedar statement) pairs.

    Text before the first marker is the file's header comment and is dropped.
    Each block keeps its own comments, so the statement stored in AgentCore reads
    the same as the file in the repository.
    """
    text = path.read_text()
    markers = list(_POLICY_MARKER.finditer(text))
    if not markers:
        raise ValueError(f"{path} contains no '// === policy: NAME ===' markers")
    blocks = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        blocks.append((marker.group(1), text[marker.end() : end].strip()))
    return blocks


def render_policies(
    gateway_arn: str,
    read_only_principal_arn: str,
    support_agent_principal_arn: str,
    path: Path = POLICY_FILE,
) -> List[Tuple[str, str]]:
    """Substitute the three placeholders and return the ready-to-create rules.

    Raises rather than shipping a half-substituted policy. A leftover placeholder
    would either fail Cedar validation or, worse, create a rule that matches a
    principal literally named "<READ_ONLY_PRINCIPAL_ARN>"; an empty substitution
    is the quieter version of the same bug, because a permit for the principal
    named "" matches nobody and, under default-deny, refuses every call the rule
    was meant to allow without anything looking broken.
    """
    substitutions = {
        GATEWAY_ARN_PLACEHOLDER: gateway_arn,
        READ_ONLY_PLACEHOLDER: read_only_principal_arn,
        SUPPORT_AGENT_PLACEHOLDER: support_agent_principal_arn,
    }
    empty = [p for p, value in substitutions.items() if not value]
    if empty:
        raise ValueError(f"No value supplied for {empty}")
    rendered = []
    for name, statement in read_policy_blocks(path):
        for placeholder, value in substitutions.items():
            statement = statement.replace(placeholder, value)
        leftover = re.findall(r"<[A-Z_]+>", statement)
        if leftover:
            raise ValueError(f"Policy {name} still contains placeholders: {leftover}")
        rendered.append((name, statement))
    return rendered


def caller_principal(region_name: str = "us-east-1") -> str:
    """The Cedar entity id for the current caller: the read-only rule's principal.

    GetCallerIdentity returns an assumed-role ARN with the session name appended,
    such as arn:aws:sts::123456789012:assumed-role/MyRole/my-session, but the
    Cedar entity id for an assumed role stops at the role name and is stable
    across sessions. Dropping the session name is what makes principal == usable.
    """
    arn = boto3.client("sts", region_name=region_name).get_caller_identity()["Arn"]
    parts = arn.split("/")
    if ":assumed-role/" in arn and len(parts) > 2:
        return "/".join(parts[:2])
    return arn


def support_agent_principal(region_name: str = "us-east-1") -> str:
    """The Cedar entity id for the privileged identity, in the caller's account.

    Same assumed-role shape as caller_principal returns, because that is the id an
    AWS_IAM gateway presents for a role.
    """
    account = boto3.client("sts", region_name=region_name).get_caller_identity()[
        "Account"
    ]
    return f"arn:aws:sts::{account}:assumed-role/{SUPPORT_AGENT_ROLE_NAME}"


def _wait_until_active(get, timeout: int = 300, **kwargs) -> dict:
    """Poll one of the Get* calls until the resource reaches ACTIVE."""
    deadline = time.monotonic() + timeout
    while True:
        resource = get(**kwargs)
        status = resource.get("status")
        if status == _ACTIVE:
            return resource
        if status not in _CREATING:
            raise RuntimeError(
                f"{kwargs} is in unexpected status: {status} "
                f"{resource.get('statusReasons', '')}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{kwargs} not ACTIVE after {timeout}s")
        time.sleep(5)


def _find_engine(control, name: str) -> Optional[str]:
    """Return the policyEngineId of an engine with this name, or None."""
    paginator = control.get_paginator("list_policy_engines")
    for page in paginator.paginate():
        for summary in page.get("policyEngines", []):
            if summary.get("name") == name:
                return summary["policyEngineId"]
    return None


def create_policy_engine(
    name: str = POLICY_ENGINE_NAME,
    region_name: str = "us-east-1",
) -> Tuple[str, str]:
    """Create (or reuse) the policy engine and return (id, ARN).

    One engine holds every rule and can be attached to more than one gateway,
    which is why it is a resource of its own rather than a field on the gateway.
    """
    control = boto3.client("bedrock-agentcore-control", region_name=region_name)
    engine_id = _find_engine(control, name)
    if engine_id is None:
        created = control.create_policy_engine(
            name=name,
            description="Stage 2: Cedar rules for the supportTools gateway target.",
        )
        engine_id = created["policyEngineId"]
        print(f"Created policy engine '{name}': {engine_id}")
    else:
        print(f"Policy engine '{name}' already exists: {engine_id}")
    engine = _wait_until_active(control.get_policy_engine, policyEngineId=engine_id)
    return engine_id, engine["policyEngineArn"]


def create_policies(
    engine_id: str,
    policies: List[Tuple[str, str]],
    region_name: str = "us-east-1",
) -> Dict[str, str]:
    """Create one policy per rendered rule and return {name: policyId}.

    validationMode is left at its FAIL_ON_ANY_FINDINGS default deliberately. The
    engine validates each statement against the Cedar schema it generated from
    the gateway's tool manifest, so a rule naming a tool or an input parameter
    that does not exist is rejected here rather than silently never matching.
    """
    control = boto3.client("bedrock-agentcore-control", region_name=region_name)
    created = {}
    for name, statement in policies:
        response = control.create_policy(
            name=name,
            policyEngineId=engine_id,
            definition={"cedar": {"statement": statement}},
        )
        _wait_until_active(
            control.get_policy,
            policyEngineId=engine_id,
            policyId=response["policyId"],
        )
        created[name] = response["policyId"]
        print(f"Created policy {name}: {response['policyId']}")
    return created


def attach_to_gateway(
    gateway_id: str,
    policy_engine_arn: str,
    mode: str = "ENFORCE",
    region_name: str = "us-east-1",
) -> None:
    """Attach the engine to the gateway, in ENFORCE mode by default.

    UpdateGateway requires name, roleArn and authorizerType alongside the field
    being changed, so this reads the gateway first and echoes them back. Omitting
    any of them is an update that quietly rewrites the gateway's authentication.

    LOG_ONLY is the safer first move on an existing gateway: the engine evaluates
    and logs the decision without acting on it, so a rule that denies more than
    intended shows up in CloudWatch instead of in a customer's failed return.
    """
    control = boto3.client("bedrock-agentcore-control", region_name=region_name)
    gateway = control.get_gateway(gatewayIdentifier=gateway_id)
    kwargs = {
        "gatewayIdentifier": gateway_id,
        "name": gateway["name"],
        "roleArn": gateway["roleArn"],
        "authorizerType": gateway["authorizerType"],
        "protocolType": gateway["protocolType"],
        "policyEngineConfiguration": {"arn": policy_engine_arn, "mode": mode},
    }
    if gateway.get("authorizerConfiguration"):
        kwargs["authorizerConfiguration"] = gateway["authorizerConfiguration"]
    if gateway.get("description"):
        kwargs["description"] = gateway["description"]
    control.update_gateway(**kwargs)
    print(f"Attached policy engine to gateway {gateway_id} in {mode} mode")


def register(
    gateway_id: str,
    gateway_arn: str,
    read_only_principal_arn: str,
    support_agent_principal_arn: str,
    mode: str = "ENFORCE",
    region_name: str = "us-east-1",
) -> Tuple[str, Dict[str, str]]:
    """Do the whole registration and return (policyEngineId, {name: policyId})."""
    engine_id, engine_arn = create_policy_engine(region_name=region_name)
    policies = create_policies(
        engine_id,
        render_policies(
            gateway_arn, read_only_principal_arn, support_agent_principal_arn
        ),
        region_name,
    )
    attach_to_gateway(gateway_id, engine_arn, mode, region_name)
    return engine_id, policies


def call_tool_through_gateway(
    gateway_url: str,
    tool_name: str,
    arguments: dict,
    region_name: str = "us-east-1",
    credentials=None,
) -> Tuple[bool, str]:
    """Call one tool through the gateway and report (allowed, text).

    This is how a policy decision is observed: there is no synchronous Cedar
    authorization API to ask, the engine decides at the gateway boundary, and a
    denied call is a refused tool call. Both shapes a refusal can take are
    treated as a deny — an error result on the MCP response, and an exception
    raised by the client — because which one the gateway uses is a property of
    the service rather than of this code.

    credentials names the principal the request is signed as, and so the
    principal Cedar evaluates. Omit it to call as the current caller.
    """
    client = build_mcp_client(gateway_url, region_name, credentials)
    client.start()
    try:
        result = client.call_tool_sync(
            tool_use_id=f"policy-{uuid.uuid4()}",
            name=tool_name,
            arguments=arguments,
        )
        text = "\n".join(c["text"] for c in result["content"] if "text" in c)
        return result.get("status") != "error", text
    except Exception as error:  # noqa: BLE001 - a refusal must not stop the walkthrough
        return False, f"{type(error).__name__}: {error}"
    finally:
        client.stop(None, None, None)


def delete_policies(
    engine_id: str,
    policy_ids: List[str],
    region_name: str = "us-east-1",
    timeout: int = 120,
) -> None:
    """Delete every policy on the engine, then confirm each one is gone.

    Every policy is attempted even if one fails, and the first failure is
    re-raised at the end: a policy that will not delete must not stop the ones
    that would, or teardown orphans the rest of them.
    """
    control = boto3.client("bedrock-agentcore-control", region_name=region_name)
    first_error = None
    for policy_id in policy_ids:
        try:
            control.delete_policy(policyEngineId=engine_id, policyId=policy_id)
            _wait_until_gone(
                control.get_policy,
                control.exceptions.ResourceNotFoundException,
                timeout,
                policyEngineId=engine_id,
                policyId=policy_id,
            )
            print(f"Deleted policy {policy_id}")
        except control.exceptions.ResourceNotFoundException:
            print(f"Policy {policy_id} already deleted")
        except Exception as error:  # noqa: BLE001 - keep deleting the rest
            print(f"Failed to delete policy {policy_id}: {error}")
            first_error = first_error or error
    if first_error:
        raise first_error


def delete_policy_engine(
    engine_id: str,
    region_name: str = "us-east-1",
    timeout: int = 120,
) -> None:
    """Delete the policy engine and confirm it is gone.

    Retried on ValidationException, which is the failure the gateway deletes in
    this walkthrough already showed: a container resource can refuse to go while
    the service still believes something is attached to it, even after the thing
    attached reports itself deleted.
    """
    control = boto3.client("bedrock-agentcore-control", region_name=region_name)
    deadline = time.monotonic() + timeout
    while True:
        try:
            control.delete_policy_engine(policyEngineId=engine_id)
            break
        except control.exceptions.ResourceNotFoundException:
            print(f"Policy engine {engine_id} already deleted")
            return
        except control.exceptions.ValidationException:
            if time.monotonic() >= deadline:
                raise
            time.sleep(5)
    _wait_until_gone(
        control.get_policy_engine,
        control.exceptions.ResourceNotFoundException,
        timeout,
        policyEngineId=engine_id,
    )
    print(f"Deleted policy engine {engine_id}")


def _wait_until_gone(get, not_found, timeout: int, **kwargs) -> None:
    """Poll a Get* call until it raises ResourceNotFoundException.

    Deletion is asynchronous throughout this API, and a list call returning
    nothing is not proof: the gateway teardown measured DeleteGateway failing
    with ValidationException while ListGatewayTargets already returned []. Only a
    Get raising not-found is evidence.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            get(**kwargs)
        except not_found:
            return
        time.sleep(2)
    raise TimeoutError(f"{kwargs} still present after {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gateway-id",
        default=os.environ.get("GATEWAY_ID"),
        help="Gateway to attach the engine to (or set GATEWAY_ID).",
    )
    parser.add_argument(
        "--gateway-arn",
        default=os.environ.get("GATEWAY_ARN"),
        help="Gateway ARN named by the Cedar resource clause (or set GATEWAY_ARN).",
    )
    parser.add_argument(
        "--read-only-principal",
        default=os.environ.get("POLICY_READ_ONLY_PRINCIPAL"),
        help="Cedar principal allowed lookup_order (default: the current caller).",
    )
    parser.add_argument(
        "--support-agent-principal",
        default=os.environ.get("POLICY_SUPPORT_AGENT_PRINCIPAL"),
        help=(
            "Cedar principal also allowed process_return "
            f"(default: the {SUPPORT_AGENT_ROLE_NAME} role in this account)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("ENFORCE", "LOG_ONLY"),
        default="ENFORCE",
        help="Enforcement mode for the attachment (default: ENFORCE).",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1).",
    )
    args = parser.parse_args()

    if not args.gateway_id or not args.gateway_arn:
        parser.error("--gateway-id and --gateway-arn are both required.")

    register(
        args.gateway_id,
        args.gateway_arn,
        args.read_only_principal or caller_principal(args.region),
        args.support_agent_principal or support_agent_principal(args.region),
        args.mode,
        args.region,
    )


if __name__ == "__main__":
    main()

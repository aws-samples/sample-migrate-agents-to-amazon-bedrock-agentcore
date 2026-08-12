# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Two IAM roles that differ in nothing IAM can see, for the Cedar proof.

A demonstration that a tool call was refused is worth nothing on its own,
because the ordinary reason a call fails is that the caller lacked the IAM
permission to make it. To attribute a refusal to Cedar rather than to IAM, two
callers have to be identical to IAM and different only to the policy engine.

So both roles get the same trust policy and the same inline policy, written from
one dict, and assert_identical_iam reads them back from the service and compares
them. It compares what IAM stores rather than what was sent, because that is
what the gateway will evaluate against, and a proof that rests on an unchecked
assumption is not one.

The roles are ephemeral. delete_demo_roles removes them, and the walkthrough's
teardown calls it.
"""

import json
import time
from typing import Dict, Tuple

import boto3

# Two names, one purpose each, and no permission difference between them.
READ_ONLY_ROLE_NAME = "MigratedAgentReadOnlyCaller"
SUPPORT_ROLE_NAME = "MigratedAgentSupportAgent"

POLICY_NAME = "GatewayInvokeOnly"


def _trust_policy(account_id: str) -> dict:
    """Let IAM principals in this account assume the role.

    Trusting the account root is not "anyone": it delegates the decision to IAM
    in this account, so a caller still needs sts:AssumeRole on the role in its
    own identity policy. It is used here because the walkthrough does not know
    which role the reader is running as.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": f"arn:aws:iam::{account_id}:root"},
                "Action": "sts:AssumeRole",
            }
        ],
    }


def _gateway_policy(gateway_arn: str) -> dict:
    """The only permission either role gets: this gateway, nothing else.

    Deliberately not scoped per tool. Scoping by tool here is exactly the job
    being handed to Cedar, and doing it in IAM as well would make the denial
    ambiguous — the point is that IAM permits both tools to both roles and the
    gateway still refuses one of the four calls.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "bedrock-agentcore:*",
                "Resource": gateway_arn,
            }
        ],
    }


def create_demo_roles(
    gateway_arn: str, region_name: str = "us-east-1"
) -> Dict[str, str]:
    """Create (or update) both roles and return {role name: role ARN}."""
    iam = boto3.client("iam", region_name=region_name)
    account_id = boto3.client("sts", region_name=region_name).get_caller_identity()[
        "Account"
    ]
    trust = json.dumps(_trust_policy(account_id))
    permissions = json.dumps(_gateway_policy(gateway_arn))

    arns = {}
    for name in (READ_ONLY_ROLE_NAME, SUPPORT_ROLE_NAME):
        try:
            iam.create_role(
                RoleName=name,
                AssumeRolePolicyDocument=trust,
                Description="Ephemeral principal for the AgentCore Policy walkthrough.",
            )
            print(f"Created role {name}")
        except iam.exceptions.EntityAlreadyExistsException:
            iam.update_assume_role_policy(
                RoleName=name, PolicyDocument=trust
            )
            print(f"Role {name} already exists")
        # Same document to both, from the same string.
        iam.put_role_policy(
            RoleName=name, PolicyName=POLICY_NAME, PolicyDocument=permissions
        )
        arns[name] = iam.get_role(RoleName=name)["Role"]["Arn"]
    return arns


def assert_identical_iam(region_name: str = "us-east-1") -> Tuple[dict, dict]:
    """Read both roles back from IAM and fail unless their permissions match.

    Returns the two policy documents so a caller can print them. Raises rather
    than warning: if the two roles differ in IAM, every conclusion drawn from
    the four calls that follow is unsound, and continuing would produce output
    that looks like a proof and is not.
    """
    iam = boto3.client("iam", region_name=region_name)
    documents = []
    for name in (READ_ONLY_ROLE_NAME, SUPPORT_ROLE_NAME):
        attached = iam.list_attached_role_policies(RoleName=name)["AttachedPolicies"]
        if attached:
            raise RuntimeError(
                f"{name} has managed policies attached ({attached}), so the two "
                "roles are not IAM-identical."
            )
        inline = iam.list_role_policies(RoleName=name)["PolicyNames"]
        if inline != [POLICY_NAME]:
            raise RuntimeError(f"{name} has unexpected inline policies: {inline}")
        documents.append(
            iam.get_role_policy(RoleName=name, PolicyName=POLICY_NAME)["PolicyDocument"]
        )

    read_only, support = documents
    if read_only != support:
        raise RuntimeError(
            "The two demo roles do not have identical IAM permissions:\n"
            f"  {READ_ONLY_ROLE_NAME}: {json.dumps(read_only, sort_keys=True)}\n"
            f"  {SUPPORT_ROLE_NAME}: {json.dumps(support, sort_keys=True)}"
        )
    print(
        "IAM is identical on both roles: "
        f"{json.dumps(read_only, sort_keys=True, separators=(',', ':'))}"
    )
    return read_only, support


def cedar_principal(role_arn: str) -> str:
    """The Cedar entity id for a role, given its IAM ARN.

    An AWS_IAM gateway presents an assumed role as
    arn:aws:sts::<account>:assumed-role/<role name>, with no session name, which
    is a different ARN from the arn:aws:iam::<account>:role/<role name> that IAM
    returns. Writing the IAM form into a Cedar rule produces a permit that
    matches nobody, and under default-deny that reads as the feature working.
    """
    account_id, role_name = role_arn.split(":")[4], role_arn.split("/")[-1]
    return f"arn:aws:sts::{account_id}:assumed-role/{role_name}"


def assume(
    role_arn: str,
    session_name: str,
    region_name: str = "us-east-1",
    timeout: int = 60,
):
    """Assume one of the roles and return credentials build_mcp_client can sign with.

    Retried on AccessDenied, which is what a role that was created seconds ago
    returns while its trust policy propagates. IAM is eventually consistent and
    the first AssumeRole after CreateRole regularly fails; treating that as a
    real authorization failure would make this walkthrough flaky in a way that
    has nothing to do with what it is demonstrating.
    """
    sts = boto3.Session(region_name=region_name).client("sts")
    deadline = time.monotonic() + timeout
    while True:
        try:
            return sts.assume_role(RoleArn=role_arn, RoleSessionName=session_name)[
                "Credentials"
            ]
        except sts.exceptions.ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code != "AccessDenied" or time.monotonic() >= deadline:
                raise
            # nosemgrep: python.lang.best-practice.sleep.arbitrary-sleep -- deadline-bounded retry of STS AssumeRole during IAM propagation; monotonic deadline stops retries, no boto3 waiter exists for propagation
            time.sleep(3)


def delete_demo_roles(region_name: str = "us-east-1") -> None:
    """Delete both roles, inline policy first. Every role is attempted."""
    iam = boto3.client("iam", region_name=region_name)
    first_error = None
    for name in (READ_ONLY_ROLE_NAME, SUPPORT_ROLE_NAME):
        try:
            for policy_name in iam.list_role_policies(RoleName=name)["PolicyNames"]:
                iam.delete_role_policy(RoleName=name, PolicyName=policy_name)
            iam.delete_role(RoleName=name)
            print(f"Deleted role {name}")
        except iam.exceptions.NoSuchEntityException:
            print(f"Role {name} already deleted")
        except Exception as error:  # noqa: BLE001 - the other role still matters
            print(f"Failed to delete role {name}: {error}")
            first_error = first_error or error
    if first_error:
        raise first_error

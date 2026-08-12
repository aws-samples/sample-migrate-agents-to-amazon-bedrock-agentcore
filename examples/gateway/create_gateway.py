# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Create an Amazon Bedrock AgentCore Gateway (MCP) with an AWS_IAM authorizer.

The AWS_IAM authorizer requires no external identity provider: callers sign
requests with SigV4 using their AWS credentials. To use a JWT authorizer
instead, set authorizerType="CUSTOM_JWT" and pass authorizerConfiguration with
customJWTAuthorizer.discoveryUrl pointing at your identity provider's OpenID
configuration endpoint.

This script is idempotent: if a gateway with the same name already exists, it
returns that gateway instead of creating a duplicate.
"""

import argparse
import os
import time
from typing import Optional, Tuple

import boto3


def _find_existing_gateway(client, name: str) -> Optional[str]:
    """Return the gatewayId of an existing gateway with this name, or None."""
    paginator = client.get_paginator("list_gateways")
    for page in paginator.paginate():
        for summary in page.get("items", []):
            if summary.get("name") == name:
                return summary["gatewayId"]
    return None


def _wait_until_ready(client, gateway_id: str, timeout: int = 120) -> dict:
    """Poll get_gateway until the gateway leaves CREATING, then return it.

    A newly created gateway starts in CREATING; registering a target before it
    reaches READY fails with a ValidationException, so callers must wait.
    """
    deadline = time.monotonic() + timeout
    while True:
        gateway = client.get_gateway(gatewayIdentifier=gateway_id)
        status = gateway.get("status")
        if status == "READY":
            return gateway
        if status not in ("CREATING", "UPDATING"):
            raise RuntimeError(f"Gateway {gateway_id} is in unexpected status: {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")
        # nosemgrep: python.lang.best-practice.sleep.arbitrary-sleep -- deadline-bounded poll of GetGateway readiness; loop raises TimeoutError past deadline, no boto3 waiter exists for AgentCore gateways
        time.sleep(2)


def existing_gateway_id(name: str, region_name: str = "us-east-1") -> Optional[str]:
    """Whether a gateway of this name is already there, before create_gateway runs.

    Asked by the walkthrough so the number it prints beside CreateGateway can say
    whether a gateway was provisioned or looked up. Provisioning takes seconds and
    a lookup takes a fraction of one, so the two are different quantities under
    one label.
    """
    client = boto3.client("bedrock-agentcore-control", region_name=region_name)
    return _find_existing_gateway(client, name)


def create_gateway(
    name: str,
    role_arn: str,
    region_name: str = "us-east-1",
) -> Tuple[str, str]:
    """Create (or reuse) an MCP gateway and return (gatewayId, gatewayUrl)."""
    client = boto3.client("bedrock-agentcore-control", region_name=region_name)

    existing_id = _find_existing_gateway(client, name)
    if existing_id is not None:
        gateway = client.get_gateway(gatewayIdentifier=existing_id)
        print(f"Gateway '{name}' already exists: {gateway['gatewayId']}")
    else:
        created = client.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType="MCP",
            authorizerType="AWS_IAM",
        )
        gateway = _wait_until_ready(client, created["gatewayId"])
        print(f"Created gateway '{name}': {gateway['gatewayId']}")

    print(f"Gateway URL: {gateway['gatewayUrl']}")
    return gateway["gatewayId"], gateway["gatewayUrl"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=os.environ.get("GATEWAY_NAME", "MigratedAgentGateway"),
        help="Gateway name (default: MigratedAgentGateway).",
    )
    parser.add_argument(
        "--role-arn",
        default=os.environ.get("GATEWAY_ROLE_ARN"),
        help="Execution role ARN for the gateway (or set GATEWAY_ROLE_ARN).",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1).",
    )
    args = parser.parse_args()

    if not args.role_arn:
        parser.error("--role-arn is required (or set GATEWAY_ROLE_ARN).")

    create_gateway(args.name, args.role_arn, args.region)


if __name__ == "__main__":
    main()

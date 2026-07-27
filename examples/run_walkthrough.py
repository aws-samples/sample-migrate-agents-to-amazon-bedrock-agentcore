# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Run the full migration walkthrough as one continuous sequence.

Each stage consumes the previous stage's output:

    create gateway -> register target -> create memory
        -> build agent with memory + gateway tools -> invoke across turns

The gateway id and URL flow into target registration and the MCP client, the
memory id flows into the agent's session manager, and the same session/actor ids
are reused across two agent instances to show conversation state surviving.

Pass --teardown to delete the gateway target, gateway, and memory afterward.
"""

import argparse
import os
import time
import uuid

import boto3
from bedrock_agentcore.memory import MemoryClient

from examples.gateway.create_gateway import create_gateway
from examples.gateway.register_target import register_target
from examples.memory.configure_memory import configure_memory
from examples.rebuild.strands_agent import build_agent
from examples.tools.gateway_mcp_tools import build_mcp_client


def _wait_for_target_deletion(
    control, gateway_id: str, target_id: str, timeout: int = 60
) -> None:
    """Poll until the gateway target is fully deleted or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            control.get_gateway_target(
                gatewayIdentifier=gateway_id, targetId=target_id
            )
        except control.exceptions.ResourceNotFoundException:
            return
        time.sleep(2)


def teardown(
    gateway_id: str,
    target_id: str,
    memory_id: str,
    region_name: str,
) -> None:
    """Delete the resources created during the walkthrough."""
    control = boto3.client("bedrock-agentcore-control", region_name=region_name)
    if target_id:
        control.delete_gateway_target(gatewayIdentifier=gateway_id, targetId=target_id)
        # Target deletion is asynchronous; the gateway cannot be deleted while a
        # target is still attached, so wait for the target to disappear first.
        _wait_for_target_deletion(control, gateway_id, target_id)
        print(f"Deleted target {target_id}")
    if gateway_id:
        control.delete_gateway(gatewayIdentifier=gateway_id)
        print(f"Deleted gateway {gateway_id}")
    if memory_id:
        MemoryClient(region_name=region_name).delete_memory(memory_id)
        print(f"Deleted memory {memory_id}")


def run(args: argparse.Namespace) -> None:
    region = args.region
    session_id = f"walkthrough-{uuid.uuid4().hex[:8]}"
    actor_id = f"customer-{uuid.uuid4().hex[:8]}"

    gateway_id, gateway_url = create_gateway(args.gateway_name, args.role_arn, region)
    target_id = register_target(gateway_id, args.lambda_arn, args.target_name, region)
    memory_id = configure_memory(region)

    try:
        mcp_client = build_mcp_client(gateway_url, region)
        with mcp_client:
            gateway_tools = mcp_client.list_tools_sync()

            # Turn 1: establish context that the agent should remember.
            agent = build_agent(
                memory_id=memory_id,
                session_id=session_id,
                actor_id=actor_id,
                extra_tools=gateway_tools,
                region_name=region,
            )
            first = agent("Hi, my name is Dana and my order id is 12345.")
            print(f"Turn 1: {first.message}")

            # Turn 2: a fresh agent on the same session/actor resumes the
            # conversation from AgentCore Memory, so it still knows the order id.
            resumed_agent = build_agent(
                memory_id=memory_id,
                session_id=session_id,
                actor_id=actor_id,
                extra_tools=gateway_tools,
                region_name=region,
            )
            second = resumed_agent("What's the status of my order?")
            print(f"Turn 2: {second.message}")
    finally:
        if args.teardown:
            teardown(gateway_id, target_id, memory_id, region)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role-arn",
        default=os.environ.get("GATEWAY_ROLE_ARN"),
        help="Execution role ARN for the gateway (or set GATEWAY_ROLE_ARN).",
    )
    parser.add_argument(
        "--lambda-arn",
        default=os.environ.get("TARGET_LAMBDA_ARN"),
        help="ARN of the Lambda backing the tools (or set TARGET_LAMBDA_ARN).",
    )
    parser.add_argument(
        "--gateway-name",
        default=os.environ.get("GATEWAY_NAME", "MigratedAgentGateway"),
        help="Gateway name (default: MigratedAgentGateway).",
    )
    parser.add_argument(
        "--target-name",
        default=os.environ.get("TARGET_NAME", "supportTools"),
        help="Target name (default: supportTools).",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1).",
    )
    parser.add_argument(
        "--teardown",
        action="store_true",
        help="Delete the gateway target, gateway, and memory after running.",
    )
    args = parser.parse_args()

    if not args.role_arn:
        parser.error("--role-arn is required (or set GATEWAY_ROLE_ARN).")
    if not args.lambda_arn:
        parser.error("--lambda-arn is required (or set TARGET_LAMBDA_ARN).")

    run(args)


if __name__ == "__main__":
    main()

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Register a Lambda target on an Amazon Bedrock AgentCore Gateway.

The target exposes the customer-support tools (lookup_order, process_return)
that the migrated agent already uses. The gateway prefixes each discovered tool
name with the target name and a triple underscore, e.g. the target below
publishes "supportTools___lookup_order" and "supportTools___process_return".

credentialProviderConfigurations uses GATEWAY_IAM_ROLE, so the gateway invokes
the Lambda with its own execution role and no external credentials are needed.
"""

import argparse
import os

import boto3

# Tool definitions published to MCP clients through the gateway. These match the
# lookup_order / process_return tools in examples/rebuild/strands_agent.py.
TOOL_SCHEMA = [
    {
        "name": "lookup_order",
        "description": "Look up order status by order ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order identifier to look up.",
                },
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "process_return",
        "description": "Initiate a return for an order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order identifier to return.",
                },
                "reason": {
                    "type": "string",
                    "description": "The reason for the return.",
                },
            },
            "required": ["order_id", "reason"],
        },
    },
]


def register_target(
    gateway_id: str,
    lambda_arn: str,
    target_name: str = "supportTools",
    region_name: str = "us-east-1",
) -> str:
    """Register a Lambda MCP target on the gateway and return its targetId."""
    client = boto3.client("bedrock-agentcore-control", region_name=region_name)

    response = client.create_gateway_target(
        gatewayIdentifier=gateway_id,
        name=target_name,
        targetConfiguration={
            "mcp": {
                "lambda": {
                    "lambdaArn": lambda_arn,
                    "toolSchema": {"inlinePayload": TOOL_SCHEMA},
                }
            }
        },
        credentialProviderConfigurations=[
            {"credentialProviderType": "GATEWAY_IAM_ROLE"}
        ],
    )

    target_id = response["targetId"]
    print(f"Registered target '{target_name}' on gateway {gateway_id}: {target_id}")
    print(
        "Published tools: "
        f"{target_name}___lookup_order, {target_name}___process_return"
    )
    return target_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gateway_id",
        nargs="?",
        default=os.environ.get("GATEWAY_ID"),
        help="Gateway ID to register the target on (or set GATEWAY_ID).",
    )
    parser.add_argument(
        "--lambda-arn",
        default=os.environ.get("TARGET_LAMBDA_ARN"),
        help="ARN of the Lambda backing the tools (or set TARGET_LAMBDA_ARN).",
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
    args = parser.parse_args()

    if not args.gateway_id:
        parser.error("gateway_id is required (or set GATEWAY_ID).")
    if not args.lambda_arn:
        parser.error("--lambda-arn is required (or set TARGET_LAMBDA_ARN).")

    register_target(args.gateway_id, args.lambda_arn, args.target_name, args.region)


if __name__ == "__main__":
    main()

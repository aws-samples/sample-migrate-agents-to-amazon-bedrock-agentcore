# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Connect to existing APIs as MCP tools via Amazon Bedrock AgentCore Gateway.

The gateway created by examples/gateway/create_gateway.py uses the AWS_IAM
authorizer, so every MCP request must be signed with SigV4. This module wraps
the caller's AWS credentials in an httpx.Auth that signs each outbound request
for the "bedrock-agentcore" service before it reaches the gateway.
"""

import argparse
import os

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.tools.mcp import MCPClient

# SigV4 signing name for the AgentCore data plane (gateway) endpoint.
SERVICE = "bedrock-agentcore"

# Signature headers copied onto each outbound request after signing.
_SIGNED_HEADERS = (
    "Authorization",
    "X-Amz-Date",
    "X-Amz-Security-Token",
    "X-Amz-Content-SHA256",
)


class SigV4HTTPXAuth(httpx.Auth):
    """httpx.Auth that signs each request with SigV4 for the AWS_IAM authorizer."""

    # Ensure httpx materializes the request body before auth_flow runs, so the
    # payload hash SigV4 computes matches what is sent.
    requires_request_body = True

    def __init__(self, credentials, service: str, region: str):
        self._signer = SigV4Auth(credentials, service, region)

    def auth_flow(self, request: httpx.Request):
        aws_request = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers={"Content-Type": request.headers.get("Content-Type", "application/json")},
        )
        self._signer.add_auth(aws_request)
        for header in _SIGNED_HEADERS:
            if header in aws_request.headers:
                request.headers[header] = aws_request.headers[header]
        yield request


def build_mcp_client(
    gateway_url: str,
    region_name: str = "us-east-1",
    credentials=None,
) -> MCPClient:
    """Build an MCPClient that signs requests to an AWS_IAM gateway with SigV4.

    credentials defaults to the ambient ones, which is what an agent running in
    Runtime or on a laptop wants. Pass them explicitly to call the gateway as a
    different principal — assumed-role credentials from sts.assume_role, for
    instance. Since the gateway authorizes on the signing identity, which
    principal signs is the whole input to a policy decision, and taking it from
    the process environment alone means it can only be changed by launching
    another process.
    """
    if credentials is None:
        credentials = boto3.Session().get_credentials()
    auth = SigV4HTTPXAuth(credentials, SERVICE, region_name)
    return MCPClient(lambda: streamablehttp_client(gateway_url, auth=auth))

    # For a CUSTOM_JWT gateway, drop the SigV4 auth and pass a bearer token
    # from your identity provider instead:
    #   headers = {"Authorization": f"Bearer {token}"}
    #   return MCPClient(lambda: streamablehttp_client(gateway_url, headers=headers))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gateway_url",
        nargs="?",
        default=os.environ.get("GATEWAY_URL"),
        help="Gateway MCP URL (or set GATEWAY_URL).",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1).",
    )
    args = parser.parse_args()

    if not args.gateway_url:
        parser.error("gateway_url is required (or set GATEWAY_URL).")

    mcp_client = build_mcp_client(args.gateway_url, args.region)

    # Agent automatically discovers available tools
    with mcp_client:
        tools = mcp_client.list_tools_sync()
        agent = Agent(
            model="us.anthropic.claude-sonnet-5",
            system_prompt="You are a customer support assistant...",
            tools=tools,
        )
        result = agent("Hello, how can you help me?")
        print(result.message)


if __name__ == "__main__":
    main()

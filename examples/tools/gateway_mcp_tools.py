# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Connect to existing APIs as MCP tools via Amazon Bedrock AgentCore Gateway."""

from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.tools.mcp import MCPClient


def main():
    # Connect to AgentCore Gateway via Streamable HTTP (MCP 1.27+)
    mcp_client = MCPClient(
        lambda: streamablehttp_client(
            "https://your-gateway-id.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
        )
    )

    # Agent automatically discovers available tools
    with mcp_client:
        tools = mcp_client.list_tools_sync()
        agent = Agent(
            model="us.anthropic.claude-sonnet-4-6",
            system_prompt="You are a customer support assistant...",
            tools=tools,
        )
        result = agent("Hello, how can you help me?")
        print(result.message)


if __name__ == "__main__":
    main()

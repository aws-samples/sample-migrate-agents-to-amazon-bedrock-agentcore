# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Connect to existing APIs as MCP tools via Amazon Bedrock AgentCore Gateway."""

from mcp.client.sse import sse_client
from strands import Agent
from strands.tools.mcp import MCPClient


def main():
    # Connect to AgentCore Gateway via SSE
    mcp_client = MCPClient(
        lambda: sse_client("https://your-gateway-endpoint.amazonaws.com/default/sse")
    )

    # Agent automatically discovers available tools
    with mcp_client:
        tools = mcp_client.list_tools_sync()
        agent = Agent(
            model="us.anthropic.claude-sonnet-4-20250514-v1:0",
            system_prompt="You are a customer support assistant...",
            tools=tools,
        )
        result = agent("Hello, how can you help me?")
        print(result.message)


if __name__ == "__main__":
    main()

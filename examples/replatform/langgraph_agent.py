# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Replatform a LangGraph agent on Amazon Bedrock AgentCore Runtime.

Usage: Replace 'from my_agent import graph' with your existing LangGraph
graph module, then deploy to AgentCore Runtime.
"""

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_core.messages import HumanMessage
from my_agent import graph  # your existing LangGraph graph

app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload):
    user_input = payload.get("prompt")
    stream = graph.astream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="values",
    )
    async for chunk in stream:
        yield chunk


if __name__ == "__main__":
    app.run()

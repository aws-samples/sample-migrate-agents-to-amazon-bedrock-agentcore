# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Replatform a LangGraph agent on Amazon Bedrock AgentCore Runtime.

Usage: Replace 'from my_agent import graph' with your existing LangGraph
graph module, then deploy to AgentCore Runtime.
"""

from bedrock_agentcore import BedrockAgentCoreApp
from my_agent import graph  # your existing LangGraph graph

app = BedrockAgentCoreApp()


@app.entrypoint
def agent_invocation(payload, context):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": payload.get("prompt", "")}]}
    )
    return {"result": result["messages"][-1].content}


if __name__ == "__main__":
    app.run()

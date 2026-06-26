# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Replatform a LangGraph agent on Amazon Bedrock AgentCore Runtime.

This example includes a minimal LangGraph agent inline for testing.
Replace the 'graph' definition with your own LangGraph graph.
"""

from bedrock_agentcore import BedrockAgentCoreApp
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# --- Replace this section with your existing LangGraph agent ---
class State(TypedDict):
    messages: list


def chatbot(state: State):
    """Minimal demo node — replace with your real agent logic."""
    user_msg = state["messages"][-1]["content"] if state["messages"] else ""
    return {"messages": state["messages"] + [{"role": "assistant", "content": f"Echo: {user_msg}"}]}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()
# --- End of replaceable section ---


app = BedrockAgentCoreApp()


@app.entrypoint
def agent_invocation(payload, context):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": payload.get("prompt", "")}]}
    )
    return {"result": result["messages"][-1]["content"]}


if __name__ == "__main__":
    app.run()

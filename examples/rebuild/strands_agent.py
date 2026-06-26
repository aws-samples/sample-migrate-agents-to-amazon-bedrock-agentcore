# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Rebuild a customer support agent using Strands Agents SDK on Amazon Bedrock AgentCore Runtime."""

import requests
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent
from strands.tools import tool

app = BedrockAgentCoreApp()


@tool
def lookup_order(order_id: str) -> dict:
    """Look up order status by order ID."""
    response = requests.get(f"https://api.example.com/orders/{order_id}", timeout=30)
    response.raise_for_status()
    return response.json()


@tool
def process_return(order_id: str, reason: str) -> dict:
    """Initiate a return for an order."""
    response = requests.post(
        "https://api.example.com/returns",
        json={"order_id": order_id, "reason": reason},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


agent = Agent(
    model="us.anthropic.claude-sonnet-4-6",
    system_prompt="You are a customer support assistant for ExampleCorp.",
    tools=[lookup_order, process_return],
)


@app.entrypoint
def agent_invocation(payload, context):
    result = agent(payload.get("prompt", ""))
    return {"result": result.message}


if __name__ == "__main__":
    app.run()

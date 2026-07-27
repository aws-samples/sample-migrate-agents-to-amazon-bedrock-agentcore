# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Rebuild a customer support agent using Strands Agents SDK on Amazon Bedrock AgentCore Runtime."""

import os
from typing import Optional, Sequence

import requests
from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)
from strands import Agent
from strands.tools import tool

MODEL_ID = "us.anthropic.claude-sonnet-5"
SYSTEM_PROMPT = "You are a customer support assistant for ExampleCorp."

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


def build_agent(
    memory_id: Optional[str] = None,
    session_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    extra_tools: Optional[Sequence] = None,
    region_name: str = "us-east-1",
) -> Agent:
    """Construct the customer support agent, optionally backed by AgentCore Memory.

    When ``memory_id`` is provided, conversation turns are persisted to and
    restored from AgentCore Memory via an AgentCoreMemorySessionManager, so a new
    agent built with the same ``session_id`` and ``actor_id`` resumes prior state.

    The local ``@tool`` stubs represent the replatform state, before the tools
    moved to Gateway. When ``extra_tools`` supplies gateway-discovered MCP tools,
    each one supersedes the local stub of the same name so the agent calls the
    gateway version rather than the placeholder endpoint. Local stubs the gateway
    does not provide are still registered.
    """
    tools = [lookup_order, process_return]
    if extra_tools:
        gateway_tools = list(extra_tools)
        # Gateway tool names carry a "<targetName>___<toolName>" prefix, e.g.
        # "supportTools___lookup_order"; the suffix identifies the superseded stub.
        superseded = {t.tool_name.split("___")[-1] for t in gateway_tools}
        tools = [t for t in tools if t.tool_name not in superseded] + gateway_tools

    kwargs = {"model": MODEL_ID, "system_prompt": SYSTEM_PROMPT, "tools": tools}

    if memory_id:
        if not (session_id and actor_id):
            raise ValueError("memory_id requires both session_id and actor_id.")
        config = AgentCoreMemoryConfig(
            memory_id=memory_id, session_id=session_id, actor_id=actor_id
        )
        kwargs["session_manager"] = AgentCoreMemorySessionManager(
            config, region_name=region_name
        )

    return Agent(**kwargs)


# Runtime agent. Set AGENTCORE_MEMORY_ID (plus AGENTCORE_SESSION_ID /
# AGENTCORE_ACTOR_ID) to back the deployed agent with AgentCore Memory.
agent = build_agent(
    memory_id=os.environ.get("AGENTCORE_MEMORY_ID"),
    session_id=os.environ.get("AGENTCORE_SESSION_ID"),
    actor_id=os.environ.get("AGENTCORE_ACTOR_ID"),
)


@app.entrypoint
def agent_invocation(payload, context):
    result = agent(payload.get("prompt", ""))
    return {"result": result.message}


if __name__ == "__main__":
    app.run()

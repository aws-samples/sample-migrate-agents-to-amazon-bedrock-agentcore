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

from examples.stage2_rebuild.guardrail import guarded_model

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


@tool
def search_faq(query: str) -> dict:
    """Search the support knowledge base for a policy or FAQ answer."""
    response = requests.get(
        "https://api.example.com/faq/search", params={"q": query}, timeout=30
    )
    response.raise_for_status()
    return response.json()


def build_agent(
    memory_id: Optional[str] = None,
    session_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    extra_tools: Optional[Sequence] = None,
    region_name: str = "us-east-1",
    model=None,
) -> Agent:
    """Construct the customer support agent, optionally backed by AgentCore Memory.

    When ``memory_id`` is provided, conversation turns are persisted to and
    restored from AgentCore Memory via an AgentCoreMemorySessionManager, so a new
    agent built with the same ``session_id`` and ``actor_id`` resumes prior state.

    The local ``@tool`` stubs represent the replatform state, before the tools
    moved to Gateway. When ``extra_tools`` supplies gateway-discovered MCP tools,
    each one supersedes the local stub of the same name so the agent calls the
    gateway version rather than the placeholder endpoint. Local stubs the gateway
    does not provide are still registered — which is why search_faq is here.
    Only lookup_order and process_return moved to the gateway; search_faq stayed
    local through every stage, and the agent has to keep offering all three or
    the rebuild has quietly dropped a capability stages 0 and 1 had.

    ``model`` overrides the bare model id, and the only reason it exists is
    guardrail.guarded_model(): a BedrockModel carrying guardrail_id and
    guardrail_version. Left as None the agent behaves exactly as it did before
    the guardrail existed.
    """
    tools = [lookup_order, process_return, search_faq]
    if extra_tools:
        gateway_tools = list(extra_tools)
        # Gateway tool names carry a "<targetName>___<toolName>" prefix, e.g.
        # "supportTools___lookup_order"; the suffix identifies the superseded stub.
        superseded = {t.tool_name.split("___")[-1] for t in gateway_tools}
        tools = [t for t in tools if t.tool_name not in superseded] + gateway_tools

    kwargs = {
        "model": model or MODEL_ID,
        "system_prompt": SYSTEM_PROMPT,
        "tools": tools,
    }

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
# AGENTCORE_ACTOR_ID) to back the deployed agent with AgentCore Memory, and
# BEDROCK_GUARDRAIL_ID (plus BEDROCK_GUARDRAIL_VERSION, default "1") to run its
# model calls behind the guardrail. Both are configuration: unset, this is the
# same agent it was before either existed.
_guardrail_id = os.environ.get("BEDROCK_GUARDRAIL_ID")
agent = build_agent(
    memory_id=os.environ.get("AGENTCORE_MEMORY_ID"),
    session_id=os.environ.get("AGENTCORE_SESSION_ID"),
    actor_id=os.environ.get("AGENTCORE_ACTOR_ID"),
    model=(
        guarded_model(
            _guardrail_id,
            MODEL_ID,
            guardrail_version=os.environ.get("BEDROCK_GUARDRAIL_VERSION", "1"),
        )
        if _guardrail_id
        else None
    ),
)


@app.entrypoint
def agent_invocation(payload, context):
    result = agent(payload.get("prompt", ""))
    return {"result": result.message}


if __name__ == "__main__":
    app.run()

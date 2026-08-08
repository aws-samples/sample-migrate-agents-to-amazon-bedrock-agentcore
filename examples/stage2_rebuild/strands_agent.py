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

TIMEOUT_SECONDS = 30

app = BedrockAgentCoreApp()


def _base_url() -> str:
    # Read at call time, not import time, so a caller can point the tools at a
    # local stub after this module is imported.
    return os.environ.get("ORDERS_API_BASE", "https://api.example.com").rstrip("/")


def _request(method: str, path: str, **kwargs) -> dict:
    # Stage 0's error contract (stage0_langgraph/tools.py), kept through the
    # rebuild: a failed call returns an {"error": ...} payload the model can
    # read and react to. Strands would catch a raise rather than crash, but the
    # model would see a traceback string instead of the payload the same
    # backend failure produces in stages 0 and 1 — search_faq never moves to
    # the gateway, so this local body is the one serving FAQ answers in
    # production, not a placeholder.
    try:
        response = requests.request(
            method, f"{_base_url()}{path}", timeout=TIMEOUT_SECONDS, **kwargs
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": f"{method} {path} failed: {exc}"}


@tool
def lookup_order(order_id: str) -> dict:
    """Look up order status by order ID."""
    return _request("GET", f"/orders/{order_id}")


@tool
def process_return(order_id: str, reason: str) -> dict:
    """Initiate a return for an order."""
    return _request("POST", "/returns", json={"order_id": order_id, "reason": reason})


@tool
def search_faq(query: str) -> dict:
    """Search the support knowledge base for a policy or FAQ answer."""
    return _request("GET", "/faq/search", params={"q": query})


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
    does not provide are still registered — which is why search_faq is here.
    Only lookup_order and process_return moved to the gateway; search_faq stayed
    local through every stage, and the agent has to keep offering all three or
    the rebuild has quietly dropped a capability stages 0 and 1 had.
    """
    tools = [lookup_order, process_return, search_faq]
    if extra_tools:
        gateway_tools = list(extra_tools)
        # Gateway tool names carry a "<targetName>___<toolName>" prefix, e.g.
        # "supportTools___lookup_order"; the suffix identifies the superseded stub.
        superseded = {t.tool_name.split("___")[-1] for t in gateway_tools}
        tools = [t for t in tools if t.tool_name not in superseded] + gateway_tools

    kwargs = {
        "model": MODEL_ID,
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


_agents = {}


def support_agent(session_id: str) -> Agent:
    """Build the agent for one session, and hold it for the life of the process.

    Stage 1's pattern (stage1_replatform/agent_runtime.py:65), for the same two
    reasons and one more. Not built at import time, so this module can be
    imported offline — the environment Runtime provides has AGENTCORE_MEMORY_ID
    and AGENTCORE_ACTOR_ID set and no session id, and building at import time
    with that combination raised ValueError before the entrypoint could supply
    the session id it actually has. Built once rather than per invocation,
    because rebuilding constructs a new model client each turn.

    The extra reason is the memory key. A session manager is pinned to one
    session_id at construction, so one process-wide agent would file every
    caller's turns under whichever session happened to arrive first. Stage 1
    passes thread_id at invoke time and needs no such cache; Strands does not,
    so the cache is keyed on the session.
    """
    if session_id not in _agents:
        # Set AGENTCORE_MEMORY_ID (plus AGENTCORE_ACTOR_ID) to back the agent with
        # AgentCore Memory. It is configuration: unset, this is the same agent it
        # was before memory existed.
        _agents[session_id] = build_agent(
            memory_id=os.environ.get("AGENTCORE_MEMORY_ID"),
            session_id=session_id,
            actor_id=os.environ.get("AGENTCORE_ACTOR_ID"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _agents[session_id]


@app.entrypoint
def agent_invocation(payload, context):
    """What Runtime calls. The agent inside it is build_agent's, unchanged.

    RequestContext.session_id is Optional and is None when the container is
    invoked without one, so a fixed fallback keeps local runs on one session —
    the same fallback stage 1 uses for thread_id.
    """
    agent = support_agent(context.session_id or "local-session")
    result = agent(payload.get("prompt", ""))
    return {"result": result.message}


if __name__ == "__main__":
    app.run()

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Diff A: host the stage-0 graph on Amazon Bedrock AgentCore Runtime.

    python -m examples.stage1_replatform.agent_runtime      # serves on :8080

This is the whole of the Runtime diff. Every node function, route_intent,
SupportState and all three prompts are imported from stage 0 rather than
rewritten — the graph that runs here is the graph that ran in
examples/stage0_langgraph/run_local.py, constructed by the same build_graph call.

What changes is who runs it. Stage 0 ran in a process you started and holds its
conversation state in that process's memory. Here the loop is wrapped in
BedrockAgentCoreApp, the entrypoint receives a RequestContext whose session_id is
stable across container instances, and that session_id is what the checkpointer
keys on.

The model call is the one thing that does not change, because it was already
going to Amazon Bedrock.
"""

import os

from bedrock_agentcore import BedrockAgentCoreApp
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from examples.stage0_langgraph.agent import build_graph
from examples.stage0_langgraph.tools import SUPPORT_TOOLS
from examples.stage1_replatform.langchain_mcp_tools import merge_tools, to_langchain_tools
from examples.tools.gateway_mcp_tools import build_mcp_client

MODEL_ID = "us.anthropic.claude-sonnet-5"

app = BedrockAgentCoreApp()

_graph = None

# Module scope on purpose. The MCP session's background thread has to outlive any
# single invocation, because tools built inside a `with mcp_client:` block raise
# MCPClientInitializationError once the block exits.
_mcp_client = None


def gateway_tools():
    """Discover the Gateway's tools, and let them supersede the local stubs.

    lookup_order and process_return now come from Gateway. search_faq is not
    published there, so merge_tools keeps the local one — the partial migration
    the post argues for, expressed in one function call.
    """
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = build_mcp_client(
            os.environ["GATEWAY_URL"],
            os.environ.get("AWS_REGION", "us-east-1"),
        )
        _mcp_client.start()  # not `with`: held for the process lifetime
    tools = to_langchain_tools(_mcp_client, _mcp_client.list_tools_sync())
    return merge_tools(SUPPORT_TOOLS, tools)


def support_graph():
    """Build the graph once and hold it for the life of the process.

    Not built at import time so that this module can be imported offline, but
    built once rather than per invocation: a per-request graph would rebuild the
    model client and, once Gateway arrives in diff B, would tear down the MCP
    session that the tools need to stay alive.
    """
    global _graph
    if _graph is None:
        llm = ChatBedrockConverse(
            model=MODEL_ID,
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        # Still MemorySaver, and still in-process: diff C-1 replaces this line.
        _graph = build_graph(
            llm=llm, tools=gateway_tools(), checkpointer=MemorySaver()
        )
    return _graph


@app.entrypoint
def agent_invocation(payload, context):
    """What Runtime calls. The graph invocation inside it is stage 0's, unchanged."""
    state = support_graph().invoke(
        {"messages": [HumanMessage(payload.get("prompt", ""))]},
        # RequestContext.session_id is Optional, and is None when the container is
        # invoked without one; a fixed fallback keeps local runs on one thread.
        config={"configurable": {"thread_id": context.session_id or "local-session"}},
    )
    return {"result": state["messages"][-1].text}


if __name__ == "__main__":
    # Deploy with an ARM64 build: Runtime rejects x86 images.
    app.run()

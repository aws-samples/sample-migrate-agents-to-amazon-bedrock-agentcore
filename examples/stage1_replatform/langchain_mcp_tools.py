# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Diff B: Amazon Bedrock AgentCore Gateway MCP tools as LangChain tools.

MCPClient.list_tools_sync() returns Strands MCPAgentTool objects. LangGraph's
ToolNode and llm.bind_tools() want LangChain BaseTool objects. They share no
interface, so something has to convert, and that something is this file rather
than a new dependency.

Three constraints, each a verified failure mode rather than a precaution:

1. Wrap before binding. An MCP inputSchema is a bare JSON schema with no name
   envelope, and that is the one input form ChatBedrockConverse.bind_tools
   rejects — langchain_core/utils/function_calling.py:457 raises
   ValueError: Unsupported function, because every dict branch there needs an
   OpenAI-function shape, a toolSpec, a name, or a title. A StructuredTool whose
   args_schema is that same raw dict binds fine, and the schema reaches the
   Bedrock payload intact. So the conversion must happen before bind_tools sees
   anything, which is why this returns StructuredTool objects.
2. Return error text, never raise. ToolNode re-raises anything that is not a
   ToolInvocationError (langgraph/prebuilt/tool_node.py:383-389), so a raised
   MCP error kills the graph run. An error string is something the model reads.
3. The client outlives the tools' construction. call_tool_sync raises
   MCPClientInitializationError on an inactive session, so tools built inside a
   `with mcp_client:` block are dead the moment the block exits. The caller must
   hold the client open for the process lifetime.
"""

import uuid

from langchain_core.tools import StructuredTool


def to_langchain_tools(mcp_client, agent_tools):
    """Wrap Strands MCPAgentTools as LangChain tools for ToolNode and bind_tools.

    mcp_client must already be started and must stay started for as long as any
    returned tool can be called.
    """
    return [_to_langchain_tool(mcp_client, t) for t in agent_tools]


def _to_langchain_tool(mcp_client, agent_tool):
    # One helper per tool, so each closure gets its own mcp_tool binding. An
    # inline loop would share the last one.
    mcp_tool = agent_tool.mcp_tool  # raw mcp.types.Tool

    def call(**kwargs):
        result = mcp_client.call_tool_sync(
            tool_use_id=f"lg-{uuid.uuid4()}",
            name=mcp_tool.name,  # the original server-side name
            arguments=kwargs,
        )
        # Constraint 2: text out, not an exception. Non-text content blocks are
        # dropped, which these two tools never return.
        return "\n".join(c["text"] for c in result["content"] if "text" in c)

    return StructuredTool.from_function(
        func=call,  # sync: ToolNode's sync path needs a sync func
        name=agent_tool.tool_name,  # e.g. "supportTools___lookup_order"
        description=mcp_tool.description or mcp_tool.name,
        args_schema=mcp_tool.inputSchema,  # JSON-schema dict, stored as-is
    )


def merge_tools(local_tools, gateway_tools):
    """Let a gateway tool supersede the local stub of the same name.

    Gateway publishes tools as "<targetName>___<name>", so the suffix after the
    last "___" is what collides with a local tool. This is the LangGraph
    restatement of the guard at examples/rebuild/strands_agent.py:63-69, which
    reads .tool_name; LangChain tools expose .name instead, so the logic has to
    be re-expressed rather than copied.

    Local tools the gateway does not publish are kept. search_faq is that case,
    and it is why a partial migration is expressible here at all.
    """
    superseded = {t.name.split("___")[-1] for t in gateway_tools}
    return [t for t in local_tools if t.name not in superseded] + list(gateway_tools)

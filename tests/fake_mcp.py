# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Stand-ins for the Strands MCP objects an AgentCore Gateway hands back.

Only the three surfaces the adapter touches are implemented: MCPAgentTool's
.mcp_tool and .tool_name, MCPClient.start()/list_tools_sync(), and
call_tool_sync() returning an MCPToolResult-shaped dict.

The tool list is built from the real TOOL_SCHEMA in
examples/gateway/register_target.py rather than a copy of it, so the schemas the
adapter is asked to convert are the ones the deployed gateway target actually
publishes. Names carry the "supportTools___" prefix Gateway applies server-side.
"""

from examples.gateway.register_target import TOOL_SCHEMA

TARGET = "supportTools"


class FakeMCPTool:
    """The raw mcp.types.Tool fields the adapter reads."""

    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.inputSchema = input_schema  # noqa: N815 - the MCP field name


class FakeMCPAgentTool:
    """What MCPClient.list_tools_sync() yields: not a LangChain BaseTool."""

    def __init__(self, name, description, input_schema):
        self.mcp_tool = FakeMCPTool(name, description, input_schema)

    @property
    def tool_name(self):
        return self.mcp_tool.name


def gateway_tool_list():
    """The two tools the walkthrough's Lambda target publishes, as discovered."""
    return [
        FakeMCPAgentTool(
            f"{TARGET}___{spec['name']}", spec["description"], spec["inputSchema"]
        )
        for spec in TOOL_SCHEMA
    ]


class FakeMCPClient:
    """A started-and-held MCP session. Records every call it is handed."""

    def __init__(self, tools=None, results=None):
        self._tools = gateway_tool_list() if tools is None else tools
        self._results = results or {}
        self.start_calls = 0
        self.calls = []

    def start(self):
        self.start_calls += 1
        return self

    def list_tools_sync(self):
        return list(self._tools)

    def call_tool_sync(self, tool_use_id, name, arguments=None):
        self.calls.append({"tool_use_id": tool_use_id, "name": name, "arguments": arguments})
        if name in self._results:
            return self._results[name]
        return {
            "status": "success",
            "content": [{"text": f"{name} ok: {arguments}"}],
        }

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Stage-1 verification for diffs A and B, per GOAL 4 of the build plan.

Runs offline and makes no AWS calls of any kind. MCP is faked, so is the chat
model where a model reply is needed; everything else is real — the real compiled
stage-0 graph, the real ToolNode, the real StructuredTool conversion, the real
local @tool functions over real HTTP against the local stub, and, for the binding
assertions, the real ChatBedrockConverse. Constructing that model creates a boto3
client but calls nothing, and bind_tools is pure local conversion.

The load-bearing pair is
test_bind_tools_rejects_a_raw_mcp_input_schema and
test_bind_tools_accepts_the_adapter_output: together they are the reason the
adapter exists at all rather than passing inputSchema dicts straight through.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import os
import unittest

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from examples.stage0_langgraph.agent import build_graph
from examples.stage0_langgraph.local_api import running_stub
from examples.stage0_langgraph.tools import SUPPORT_TOOLS
from examples.stage1_replatform import agent_runtime
from examples.stage1_replatform.langchain_mcp_tools import (
    merge_tools,
    to_langchain_tools,
)
from tests.fake_chat_model import FakeChatModel
from tests.fake_mcp import FakeMCPClient, gateway_tool_list

MODEL_ID = "us.anthropic.claude-sonnet-5"


def a_model():
    """The real Bedrock chat model. Constructed, never invoked."""
    return ChatBedrockConverse(model=MODEL_ID, region_name="us-east-1")


def bound_schema(bound, name):
    """Dig the JSON schema for one tool out of what bind_tools produced."""
    for entry in bound.kwargs["tools"]:
        function = entry.get("function", entry)
        if function.get("name") == name:
            return function["parameters"]
    raise AssertionError(f"{name} is not in the bound payload")


class GatewayFakeChatModel(FakeChatModel):
    """FakeChatModel, but it calls tools by the name they are actually bound as.

    The scripted model picks tools by their bare name. Once Gateway is in front of
    them the bound name is "supportTools___lookup_order", so this remaps the
    chosen name onto whichever bound tool ends with it. Nothing else changes,
    which is the point: the same script drives stage 0 and stage 1.
    """

    def bind_tools(self, tools):
        bound = super().bind_tools(tools)
        return _RemappingBoundModel(bound, {t.name.split("___")[-1]: t.name for t in tools})


class _RemappingBoundModel:
    def __init__(self, bound, names):
        self._bound = bound
        self._names = names

    def invoke(self, messages):
        message = self._bound.invoke(messages)
        if not message.tool_calls:
            return message
        return AIMessage(
            content=message.content,
            tool_calls=[
                {**call, "name": self._names.get(call["name"], call["name"])}
                for call in message.tool_calls
            ],
        )


class FakeContext:
    """The RequestContext shape Runtime hands the entrypoint."""

    def __init__(self, session_id=None):
        self.session_id = session_id


class AdapterBindingTest(unittest.TestCase):
    """Diff B constraint 3: wrap before binding, or bind_tools raises."""

    def test_bind_tools_rejects_a_raw_mcp_input_schema(self):
        # The negative control for the whole adapter. An MCP inputSchema is a bare
        # JSON schema with no name envelope, and that is the one form
        # ChatBedrockConverse rejects.
        schema = gateway_tool_list()[0].mcp_tool.inputSchema

        with self.assertRaises(ValueError) as caught:
            a_model().bind_tools([schema])

        self.assertIn("Unsupported function", str(caught.exception))

    def test_bind_tools_accepts_the_adapter_output(self):
        tools = to_langchain_tools(FakeMCPClient(), gateway_tool_list())

        bound = a_model().bind_tools(tools)  # must not raise

        names = [e.get("function", e).get("name") for e in bound.kwargs["tools"]]
        self.assertEqual(
            sorted(names),
            ["supportTools___lookup_order", "supportTools___process_return"],
        )

    def test_the_mcp_schema_survives_into_the_bedrock_payload(self):
        agent_tools = gateway_tool_list()
        tools = to_langchain_tools(FakeMCPClient(), agent_tools)

        bound = a_model().bind_tools(tools)

        original = agent_tools[1].mcp_tool.inputSchema  # process_return
        arrived = bound_schema(bound, "supportTools___process_return")
        self.assertEqual(arrived, original)
        self.assertEqual(sorted(arrived["required"]), ["order_id", "reason"])
        self.assertEqual(
            arrived["properties"]["reason"]["description"],
            original["properties"]["reason"]["description"],
        )

    def test_args_schema_stays_a_dict_rather_than_being_coerced(self):
        # StructuredTool storing the dict as-is is what makes the chain work; a
        # coercion to a pydantic model would silently change the published schema.
        tool = to_langchain_tools(FakeMCPClient(), gateway_tool_list())[0]
        self.assertIsInstance(tool.args_schema, dict)
        self.assertEqual(tool.args_schema, gateway_tool_list()[0].mcp_tool.inputSchema)


class AdapterCallTest(unittest.TestCase):
    """The adapter's other two constraints: sync invocation, and error text."""

    def test_a_wrapped_tool_is_sync_invocable_and_sends_the_original_name(self):
        client = FakeMCPClient()
        tool = to_langchain_tools(client, gateway_tool_list())[0]

        result = tool.invoke({"order_id": "12345"})

        self.assertIn("lookup_order ok", result)
        self.assertEqual(len(client.calls), 1)
        # The tool is bound as the prefixed name but called by the name the
        # server knows, which here are the same string; asserted so a future
        # name_override cannot silently break the call.
        self.assertEqual(client.calls[0]["name"], "supportTools___lookup_order")
        self.assertEqual(client.calls[0]["arguments"], {"order_id": "12345"})

    def test_an_mcp_error_result_comes_back_as_text_and_does_not_raise(self):
        client = FakeMCPClient(
            results={
                "supportTools___lookup_order": {
                    "status": "error",
                    "content": [{"text": "Order 99999 not found"}],
                }
            }
        )
        tool = to_langchain_tools(client, gateway_tool_list())[0]

        # Raising here would kill the graph run: ToolNode re-raises anything that
        # is not a ToolInvocationError.
        self.assertEqual(tool.invoke({"order_id": "99999"}), "Order 99999 not found")

    def test_non_text_content_blocks_are_dropped_rather_than_crashing(self):
        client = FakeMCPClient(
            results={
                "supportTools___lookup_order": {
                    "status": "success",
                    "content": [{"image": {"bytes": b"x"}}, {"text": "shipped"}],
                }
            }
        )
        tool = to_langchain_tools(client, gateway_tool_list())[0]
        self.assertEqual(tool.invoke({"order_id": "12345"}), "shipped")

    def test_each_wrapped_tool_closes_over_its_own_schema(self):
        client = FakeMCPClient()
        lookup, process = to_langchain_tools(client, gateway_tool_list())

        lookup.invoke({"order_id": "12345"})
        process.invoke({"order_id": "12345", "reason": "damaged"})

        self.assertEqual(
            [c["name"] for c in client.calls],
            ["supportTools___lookup_order", "supportTools___process_return"],
        )


class MergeToolsTest(unittest.TestCase):
    """The E2E Failure 3 guard, restated on .name instead of .tool_name."""

    def test_gateway_tools_supersede_the_local_stubs_of_the_same_name(self):
        gateway = to_langchain_tools(FakeMCPClient(), gateway_tool_list())

        merged = merge_tools(SUPPORT_TOOLS, gateway)

        self.assertEqual(
            sorted(t.name for t in merged),
            ["search_faq", "supportTools___lookup_order", "supportTools___process_return"],
        )
        # Exactly one lookup_order variant survives, and it is the gateway one.
        variants = [t for t in merged if t.name.endswith("lookup_order")]
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].name, "supportTools___lookup_order")

    def test_search_faq_survives_as_the_tool_gateway_does_not_publish(self):
        gateway = to_langchain_tools(FakeMCPClient(), gateway_tool_list())
        merged = merge_tools(SUPPORT_TOOLS, gateway)
        self.assertIn("search_faq", [t.name for t in merged])

    def test_an_empty_gateway_list_supersedes_nothing(self):
        merged = merge_tools(SUPPORT_TOOLS, [])
        self.assertEqual([t.name for t in merged], [t.name for t in SUPPORT_TOOLS])


class GatewayBackedGraphTest(unittest.TestCase):
    """The stage-0 graph, unchanged, running on gateway tools."""

    def setUp(self):
        stub = running_stub()
        self.base_url = stub.__enter__()
        self.addCleanup(stub.__exit__, None, None, None)
        previous = os.environ.get("ORDERS_API_BASE")
        os.environ["ORDERS_API_BASE"] = self.base_url
        self.addCleanup(self._restore, previous)

        self.client = FakeMCPClient()
        self.llm = GatewayFakeChatModel()
        self.tools = merge_tools(
            SUPPORT_TOOLS, to_langchain_tools(self.client, gateway_tool_list())
        )
        self.graph = build_graph(
            llm=self.llm, tools=self.tools, checkpointer=MemorySaver()
        )

    def _restore(self, previous):
        if previous is None:
            os.environ.pop("ORDERS_API_BASE", None)
        else:
            os.environ["ORDERS_API_BASE"] = previous

    def ask(self, prompt, thread_id):
        return self.graph.invoke(
            {"messages": [HumanMessage(prompt)]},
            config={"configurable": {"thread_id": thread_id}},
        )

    def tool_messages(self, state):
        return [m for m in state["messages"] if isinstance(m, ToolMessage)]

    def test_an_order_prompt_now_calls_the_gateway_tool_and_not_a_local_stub(self):
        state = self.ask("Where is my order 12345?", "t-gw-order")

        self.assertEqual(state["intent"], "assist")
        calls = self.tool_messages(state)
        self.assertEqual([m.name for m in calls], ["supportTools___lookup_order"])
        # Went to the gateway, and nothing reached ORDERS_API_BASE for this tool.
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0]["arguments"], {"order_id": "12345"})

    def test_search_faq_still_reaches_the_local_http_backend(self):
        state = self.ask("What is your return policy?", "t-gw-faq")

        calls = self.tool_messages(state)
        self.assertEqual([m.name for m in calls], ["search_faq"])
        self.assertIn("FAQ-RETURNS-001", calls[0].content)
        self.assertEqual(self.client.calls, [], "search_faq must not go to Gateway")

    def test_the_escalate_branch_is_untouched_by_the_tool_swap(self):
        state = self.ask("This is unacceptable, I want a human.", "t-gw-angry")

        self.assertEqual(state["intent"], "escalate")
        self.assertTrue(state["escalated"])
        self.assertEqual(self.tool_messages(state), [])
        self.assertEqual(self.client.calls, [])

    def test_recall_across_two_turns_still_works_with_gateway_tools(self):
        thread = "t-gw-recall"
        self.ask("Hi, I'm Dana and my order number is 12345.", thread)
        self.ask("Has it shipped yet?", thread)

        # Turn 2 never repeats 12345, so the gateway call carrying it proves the
        # checkpointer replayed turn 1.
        self.assertEqual(self.client.calls[-1]["arguments"], {"order_id": "12345"})


class RuntimeEntrypointTest(unittest.TestCase):
    """Diff A: the entrypoint shape, and session_id becoming thread_id."""

    def setUp(self):
        self.addCleanup(self._reset_module_state)
        agent_runtime._graph = build_graph(
            llm=FakeChatModel(), tools=SUPPORT_TOOLS, checkpointer=MemorySaver()
        )
        stub = running_stub()
        base_url = stub.__enter__()
        self.addCleanup(stub.__exit__, None, None, None)
        previous = os.environ.get("ORDERS_API_BASE")
        os.environ["ORDERS_API_BASE"] = base_url
        self.addCleanup(self._restore, previous)

    def _reset_module_state(self):
        agent_runtime._graph = None
        agent_runtime._mcp_client = None

    def _restore(self, previous):
        if previous is None:
            os.environ.pop("ORDERS_API_BASE", None)
        else:
            os.environ["ORDERS_API_BASE"] = previous

    def test_the_entrypoint_returns_a_result_dict(self):
        out = agent_runtime.agent_invocation(
            {"prompt": "What is your return policy?"}, FakeContext("sess-1")
        )
        self.assertEqual(list(out), ["result"])
        self.assertIsInstance(out["result"], str)
        self.assertIn("30 days", out["result"])

    def test_the_session_id_is_the_thread_id(self):
        agent_runtime.agent_invocation(
            {"prompt": "Hi, I'm Dana and my order number is 12345."},
            FakeContext("sess-recall"),
        )
        same = agent_runtime.agent_invocation(
            {"prompt": "Has it shipped yet?"}, FakeContext("sess-recall")
        )
        other = agent_runtime.agent_invocation(
            {"prompt": "Has it shipped yet?"}, FakeContext("sess-different")
        )

        self.assertIn("shipped", same["result"])
        self.assertIn("What is your order number?", other["result"])

    def test_the_config_carries_the_actor_id_beside_the_thread_id(self):
        # AgentCoreMemorySaver resolves the event stream from thread_id and
        # actor_id together and raises InvalidConfigError without either, so the
        # entrypoint has to send both. The MemorySaver the other tests here run on
        # would ignore a missing actor_id, which is why this one records the config
        # instead of asserting on the answer.
        captured = {}

        class RecordingGraph:
            def invoke(self, state, config):
                captured.update(config["configurable"])
                return {"messages": [AIMessage("recorded")]}

        agent_runtime._graph = RecordingGraph()
        previous = os.environ.get("AGENTCORE_ACTOR_ID")
        os.environ["AGENTCORE_ACTOR_ID"] = "customer-7"
        self.addCleanup(self._restore_actor_id, previous)

        agent_runtime.agent_invocation({"prompt": "hi"}, FakeContext("sess-actor"))

        self.assertEqual(captured, {"thread_id": "sess-actor", "actor_id": "customer-7"})

    @staticmethod
    def _restore_actor_id(previous):
        if previous is None:
            os.environ.pop("AGENTCORE_ACTOR_ID", None)
        else:
            os.environ["AGENTCORE_ACTOR_ID"] = previous

    def test_a_missing_session_id_falls_back_rather_than_crashing(self):
        # RequestContext.session_id is Optional and is None on a bare local call.
        out = agent_runtime.agent_invocation(
            {"prompt": "What is your return policy?"}, FakeContext(None)
        )
        self.assertIn("30 days", out["result"])

    def test_an_empty_payload_does_not_raise(self):
        out = agent_runtime.agent_invocation({}, FakeContext("sess-empty"))
        self.assertIsInstance(out["result"], str)


class GatewayClientLifecycleTest(unittest.TestCase):
    """Diff B constraint 1: one client, started once, held for the process."""

    def setUp(self):
        self.addCleanup(self._reset)
        self.addCleanup(setattr, agent_runtime, "build_mcp_client", agent_runtime.build_mcp_client)
        previous = os.environ.get("GATEWAY_URL")
        os.environ["GATEWAY_URL"] = "https://example-gateway.invalid/mcp"
        self.addCleanup(self._restore, previous)

        self.client = FakeMCPClient()
        self.built = []

        def fake_build(url, region):
            self.built.append((url, region))
            return self.client

        agent_runtime.build_mcp_client = fake_build

    def _reset(self):
        agent_runtime._graph = None
        agent_runtime._mcp_client = None

    def _restore(self, previous):
        if previous is None:
            os.environ.pop("GATEWAY_URL", None)
        else:
            os.environ["GATEWAY_URL"] = previous

    def test_the_client_is_built_and_started_once_across_invocations(self):
        first = agent_runtime.gateway_tools()
        second = agent_runtime.gateway_tools()

        self.assertEqual(len(self.built), 1, "one client for the whole process")
        self.assertEqual(self.client.start_calls, 1)
        self.assertEqual(
            [t.name for t in first], [t.name for t in second]
        )

    def test_the_wired_tool_list_is_the_merged_one(self):
        names = sorted(t.name for t in agent_runtime.gateway_tools())
        self.assertEqual(
            names,
            ["search_faq", "supportTools___lookup_order", "supportTools___process_return"],
        )
        self.assertEqual(self.built[0][0], "https://example-gateway.invalid/mcp")


if __name__ == "__main__":
    unittest.main()

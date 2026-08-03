# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Stage-0 verification, per GOAL 2 of the build plan.

Runs offline. The chat model is faked; everything else is the real thing — the
real compiled graph, the real ToolNode, the real @tool functions over real HTTP
against the local stub, and the real MemorySaver. No AgentCore calls.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import os
import unittest

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from examples.stage0_langgraph.agent import build_graph
from examples.stage0_langgraph.local_api import running_stub
from examples.stage0_langgraph.tools import SUPPORT_TOOLS
from tests.fake_chat_model import FakeChatModel


def tool_messages(state, name=None):
    return [
        m
        for m in state["messages"]
        if isinstance(m, ToolMessage) and (name is None or m.name == name)
    ]


class Stage0AgentTest(unittest.TestCase):
    """Each test gets its own stub, its own model and its own checkpointer."""

    def setUp(self):
        stub = running_stub()
        self.base_url = stub.__enter__()
        self.addCleanup(stub.__exit__, None, None, None)

        previous = os.environ.get("ORDERS_API_BASE")
        os.environ["ORDERS_API_BASE"] = self.base_url
        self.addCleanup(self._restore_base_url, previous)

        self.llm = FakeChatModel()
        self.graph = build_graph(
            llm=self.llm, tools=SUPPORT_TOOLS, checkpointer=MemorySaver()
        )

    def _restore_base_url(self, previous):
        if previous is None:
            os.environ.pop("ORDERS_API_BASE", None)
        else:
            os.environ["ORDERS_API_BASE"] = previous

    def ask(self, prompt, thread_id):
        return self.graph.invoke(
            {"messages": [HumanMessage(prompt)]},
            config={"configurable": {"thread_id": thread_id}},
        )

    # GOAL 2 assertion 1
    def test_order_prompt_takes_the_tools_path_and_calls_lookup_order_once(self):
        state = self.ask("Where is my order 12345?", "t-order")

        self.assertEqual(state["intent"], "assist")
        self.assertFalse(state.get("escalated", False))

        calls = tool_messages(state, "lookup_order")
        self.assertEqual(len(calls), 1, "lookup_order should be called exactly once")
        self.assertIn("1Z999AA10123456784", calls[0].content)

        # assist ran twice: once to request the tool, once to answer from its result.
        self.assertEqual(len(self.llm.assist_calls), 2)
        self.assertEqual(
            [c["name"] for c in self.llm.tool_calls_made], ["lookup_order"]
        )

    # GOAL 2 assertion 2
    def test_angry_prompt_takes_the_escalate_edge_and_calls_no_tool(self):
        state = self.ask(
            "This is unacceptable, I want to speak to a human.", "t-angry"
        )

        self.assertEqual(state["intent"], "escalate")
        self.assertTrue(state["escalated"])
        self.assertEqual(tool_messages(state), [])
        self.assertEqual(self.llm.tool_calls_made, [])
        self.assertEqual(
            self.llm.assist_calls, [], "the assist node must not run on this edge"
        )
        self.assertIn("human support specialist", state["messages"][-1].text)

    # GOAL 2 assertion 2, the other direction: the router is a real branch, not a
    # constant. The same graph must reach both successors.
    def test_router_reaches_both_successors(self):
        escalated = self.ask("This is ridiculous.", "t-both-a")
        assisted = self.ask("Where is my order 12345?", "t-both-b")

        self.assertEqual(escalated["intent"], "escalate")
        self.assertEqual(assisted["intent"], "assist")

    # GOAL 2: search_faq is reachable. It is the tool that never moves to Gateway,
    # so stage 1 must still be able to reach it.
    def test_search_faq_is_reachable(self):
        state = self.ask("What is your return policy?", "t-faq")

        calls = tool_messages(state, "search_faq")
        self.assertEqual(len(calls), 1)
        self.assertIn("FAQ-RETURNS-001", calls[0].content)
        self.assertIn("30 days", calls[0].content)

    def test_search_faq_is_bound_alongside_the_two_gateway_bound_tools(self):
        self.ask("Where is my order 12345?", "t-bound")
        self.assertEqual(
            sorted(t.name for t in self.llm.bound_tools),
            ["lookup_order", "process_return", "search_faq"],
        )

    # GOAL 2 assertion 3: the local twin of the Dana / 12345 recall proof.
    def test_two_turns_on_one_thread_id_recall_a_turn_1_fact(self):
        thread = "t-recall"
        self.ask("Hi, I'm Dana and my order number is 12345.", thread)
        turn_two = self.ask("Has it shipped yet?", thread)

        # Turn 2 never says "12345", so a lookup_order carrying it can only have
        # come from state the checkpointer replayed.
        self.assertEqual(
            self.llm.tool_calls_made[-1],
            {"name": "lookup_order", "args": {"order_id": "12345"}, "id": "call_2"},
        )
        self.assertIn("shipped", tool_messages(turn_two, "lookup_order")[-1].content)

        # Turn 1's facts are visible to the turn-2 model call and in final state.
        turn_two_prompt = " ".join(m.text for m in self.llm.assist_calls[-1])
        self.assertIn("Dana", turn_two_prompt)
        self.assertIn("Dana", " ".join(m.text for m in turn_two["messages"]))

    def test_a_fresh_thread_id_recalls_nothing(self):
        """Negative control: without the shared thread_id the recall disappears."""
        self.ask("Hi, I'm Dana and my order number is 12345.", "t-control-1")
        turn_two = self.ask("Has it shipped yet?", "t-control-2")

        self.assertEqual(tool_messages(turn_two, "lookup_order"), [])
        self.assertNotIn("Dana", " ".join(m.text for m in turn_two["messages"]))
        self.assertIn("What is your order number?", turn_two["messages"][-1].text)


if __name__ == "__main__":
    unittest.main()

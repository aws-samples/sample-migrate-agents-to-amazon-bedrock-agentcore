# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Offline verification for AgentCoreMemorySaver, diff C-1 of GOAL 4.

The AgentCore Memory data plane is faked; the saver, LangGraph's serde, the
CheckpointTuple contract and the compiled stage-0 graph are all real. No AWS
calls.

Three of these tests exist because of specific verified service behaviour rather
than general caution. The ordering pair asserts that selection is imposed
client-side on uuid6 ids, by running the identical assertions against a fake that
returns events newest-first and one that returns them oldest-first, and by writing
checkpoints in an order that does not match their ids — ListEvents has no sort
parameter, so trusting the returned order is not an option. The paging test
asserts a thread longer than one 100-event page is read completely and costs the
ceil(N/100) calls the read-cost comment claims.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import json
import os
import unittest
from unittest import mock

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.base import empty_checkpoint

from examples.stage0_langgraph.agent import build_graph
from examples.stage0_langgraph.local_api import running_stub
from examples.stage0_langgraph.tools import SUPPORT_TOOLS
from examples.stage1_replatform import agentcore_memory_saver
from examples.stage1_replatform.agentcore_memory_saver import (
    MAX_EVENTS,
    AgentCoreMemorySaver,
)
from tests.fake_chat_model import FakeChatModel
from tests.fake_memory import API_PAGE_SIZE, FakeMemoryDataPlane

MEMORY_ID = "mem-test-0001"


def a_checkpoint(checkpoint_id, **channel_values):
    """A Checkpoint dict with a chosen id, so tests control the sort key."""
    checkpoint = empty_checkpoint()
    checkpoint["id"] = checkpoint_id
    checkpoint["channel_values"] = channel_values
    return checkpoint


def first_write_value(saver, event):
    """The first channel value inside a stored writes event, decoded.

    Values go through serde and base64 on the way in, so a test that wants to see
    what an event actually holds has to come back out the same way.
    """
    record = json.loads(event["payload"][0]["blob"])
    return agentcore_memory_saver._dec(saver.serde, record["writes"][0][1])


def a_config(thread_id, checkpoint_id=None, checkpoint_ns=""):
    configurable = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


class SaverTestCase(unittest.TestCase):
    """Builds a saver over a fake data plane, patched at the import site."""

    newest_first = True

    def setUp(self):
        self.plane = FakeMemoryDataPlane(newest_first=self.newest_first)
        patcher = mock.patch.object(
            agentcore_memory_saver, "MemoryClient", return_value=self.plane
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.saver = AgentCoreMemorySaver(MEMORY_ID, actor_id="actor-1")

    def put(self, thread_id, checkpoint, step=0, parent_id=None, checkpoint_ns=""):
        return self.saver.put(
            a_config(thread_id, parent_id, checkpoint_ns),
            checkpoint,
            {"step": step, "source": "loop"},
            {},
        )


class RoundTripTest(SaverTestCase):
    def test_put_then_get_tuple_returns_the_checkpoint(self):
        checkpoint = a_checkpoint("1f18e122-0001-6868-bffe-000000000001", greeting="hi")

        returned = self.put("t-round", checkpoint, step=3)
        tuple_ = self.saver.get_tuple(a_config("t-round"))

        self.assertEqual(returned, a_config("t-round", checkpoint["id"]))
        self.assertEqual(tuple_.checkpoint, checkpoint)
        self.assertEqual(tuple_.checkpoint["channel_values"], {"greeting": "hi"})
        # metadata["step"] is read unguarded by the loop, so losing it breaks resume.
        self.assertEqual(tuple_.metadata["step"], 3)
        self.assertEqual(tuple_.config, a_config("t-round", checkpoint["id"]))
        self.assertIsNone(tuple_.parent_config)
        self.assertEqual(tuple_.pending_writes, [])

    def test_langchain_messages_survive_the_round_trip(self):
        # The real channel_values hold message objects, not strings; this is what
        # serde.dumps_typed is for, and base64 is how JSON carries the result.
        messages = [HumanMessage("I'm Dana, order 12345"), ToolMessage("shipped", tool_call_id="c1")]
        checkpoint = a_checkpoint("1f18e122-0002-6868-bffe-000000000001", messages=messages)

        self.put("t-msgs", checkpoint)
        restored = self.saver.get_tuple(a_config("t-msgs")).checkpoint["channel_values"]

        self.assertEqual([type(m) for m in restored["messages"]], [HumanMessage, ToolMessage])
        self.assertEqual(restored["messages"][0].text, "I'm Dana, order 12345")

    def test_an_empty_thread_returns_none(self):
        # The loop then synthesises an empty checkpoint with step -2. Returning
        # anything else here breaks the first invocation.
        self.assertIsNone(self.saver.get_tuple(a_config("t-never-written")))

    def test_parent_linkage_comes_from_the_incoming_config(self):
        first = a_checkpoint("1f18e122-0003-6868-bffe-000000000001")
        second = a_checkpoint("1f18e122-0003-6868-bffe-000000000002")
        self.put("t-parent", first, step=0)
        self.put("t-parent", second, step=1, parent_id=first["id"])

        tuple_ = self.saver.get_tuple(a_config("t-parent"))

        self.assertEqual(tuple_.checkpoint["id"], second["id"])
        self.assertEqual(tuple_.parent_config, a_config("t-parent", first["id"]))

    def test_a_named_checkpoint_id_is_fetched_rather_than_the_latest(self):
        older = a_checkpoint("1f18e122-0004-6868-bffe-000000000001")
        newer = a_checkpoint("1f18e122-0004-6868-bffe-000000000009")
        self.put("t-named", older)
        self.put("t-named", newer)

        tuple_ = self.saver.get_tuple(a_config("t-named", older["id"]))
        self.assertEqual(tuple_.checkpoint["id"], older["id"])

    def test_an_unknown_checkpoint_id_returns_none(self):
        self.put("t-unknown", a_checkpoint("1f18e122-0005-6868-bffe-000000000001"))
        self.assertIsNone(
            self.saver.get_tuple(a_config("t-unknown", "1f18e122-ffff-6868-bffe-ffffffffffff"))
        )

    def test_threads_and_namespaces_are_isolated(self):
        self.put("t-a", a_checkpoint("1f18e122-0006-6868-bffe-00000000000a"))
        self.put("t-b", a_checkpoint("1f18e122-0006-6868-bffe-00000000000b"))
        self.put("t-a", a_checkpoint("1f18e122-0006-6868-bffe-00000000000c"), checkpoint_ns="sub")

        self.assertEqual(
            self.saver.get_tuple(a_config("t-a")).checkpoint["id"],
            "1f18e122-0006-6868-bffe-00000000000a",
        )
        self.assertEqual(
            self.saver.get_tuple(a_config("t-a", checkpoint_ns="sub")).checkpoint["id"],
            "1f18e122-0006-6868-bffe-00000000000c",
        )
        self.assertEqual(
            self.saver.get_tuple(a_config("t-b")).checkpoint["id"],
            "1f18e122-0006-6868-bffe-00000000000b",
        )

    def test_foreign_events_in_the_same_stream_are_skipped(self):
        # A memory resource's event stream is shared. Conversational events carry
        # no "blob" key, and another writer's blob carries no matching "ns".
        self.put("t-shared", a_checkpoint("1f18e122-0007-6868-bffe-000000000001"))
        stream = self.plane.events[(MEMORY_ID, "actor-1", "t-shared")]
        stream.append({"payload": [{"conversational": {"content": "hi", "role": "USER"}}]})
        stream.append({"payload": [{"blob": '{"kind": "something-else"}'}]})

        tuple_ = self.saver.get_tuple(a_config("t-shared"))
        self.assertEqual(tuple_.checkpoint["id"], "1f18e122-0007-6868-bffe-000000000001")


class OrderingIsImposedClientSideTest(SaverTestCase):
    """ListEvents has no sort parameter, so the saver must not trust the order."""

    # Ids chosen so that the highest is written NEITHER first nor last. Any
    # implementation that takes events[0], events[-1], the first appended or the
    # last appended picks the wrong one.
    IDS = [
        "1f18e122-0100-6868-bffe-000000000005",
        "1f18e122-0100-6868-bffe-000000000009",  # the real latest
        "1f18e122-0100-6868-bffe-000000000002",
    ]
    LATEST = "1f18e122-0100-6868-bffe-000000000009"

    def write_all(self, thread_id="t-order"):
        for step, checkpoint_id in enumerate(self.IDS):
            self.put(thread_id, a_checkpoint(checkpoint_id, step=step), step=step)
        return thread_id

    def test_the_max_uuid6_id_wins_regardless_of_write_order(self):
        thread = self.write_all()
        self.assertEqual(self.saver.get_tuple(a_config(thread)).checkpoint["id"], self.LATEST)

    def test_the_same_answer_when_the_service_returns_the_other_order(self):
        # Same assertions against a data plane that hands back oldest-first. The
        # service promises neither, so both must give the identical result.
        self.plane.newest_first = False
        thread = self.write_all("t-order-flipped")
        self.assertEqual(self.saver.get_tuple(a_config(thread)).checkpoint["id"], self.LATEST)

    def test_the_returned_order_really_does_disagree_with_the_answer(self):
        # Guards the test above from being vacuous: if the fake happened to return
        # the latest checkpoint first, "max() works" would prove nothing.
        thread = self.write_all("t-order-proof")
        events = self.plane.list_events(MEMORY_ID, "actor-1", thread, max_results=MAX_EVENTS)
        first_returned = events[0]["metadata"]["checkpoint_id"]["stringValue"]
        self.assertNotEqual(first_returned, self.LATEST)
        self.assertEqual(
            self.saver.get_tuple(a_config(thread)).checkpoint["id"], self.LATEST
        )


class PagingTest(SaverTestCase):
    def test_a_thread_longer_than_one_page_is_read_completely(self):
        thread = "t-paged"
        total = API_PAGE_SIZE * 2 + 7  # 207 events: three pages
        for n in range(total):
            self.put(thread, a_checkpoint(f"1f18e122-0200-6868-bffe-{n:012d}"), step=n)

        self.plane.page_reads = 0
        tuple_ = self.saver.get_tuple(a_config(thread))

        # The last id is the max id here, and it only exists on page 3.
        self.assertEqual(
            tuple_.checkpoint["id"], f"1f18e122-0200-6868-bffe-{total - 1:012d}"
        )
        # The read-cost comment in the saver claims ceil(min(N, 10000)/100) calls.
        self.assertEqual(self.plane.page_reads, 3)
        self.assertEqual(self.plane.list_calls[-1]["max_results"], MAX_EVENTS)

    def test_the_first_page_alone_would_have_given_the_wrong_answer(self):
        # Not vacuous: with newest-first ordering the max id is on page 1, so this
        # asserts the reverse case, where the answer is only reachable by paging.
        thread = "t-paged-proof"
        for n in range(API_PAGE_SIZE + 5):
            self.put(thread, a_checkpoint(f"1f18e122-0201-6868-bffe-{n:012d}"), step=n)

        page_one = self.plane.list_events(
            MEMORY_ID, "actor-1", thread, max_results=API_PAGE_SIZE
        )
        ids_on_page_one = {e["metadata"]["checkpoint_id"]["stringValue"] for e in page_one}
        latest = self.saver.get_tuple(a_config(thread)).checkpoint["id"]
        self.assertEqual(latest, f"1f18e122-0201-6868-bffe-{API_PAGE_SIZE + 4:012d}")
        self.assertIn(latest, ids_on_page_one)  # newest-first: page 1 has it

        # Oldest-first: the same latest checkpoint is now only on the last page.
        self.plane.newest_first = False
        page_one = self.plane.list_events(
            MEMORY_ID, "actor-1", thread, max_results=API_PAGE_SIZE
        )
        ids_on_page_one = {e["metadata"]["checkpoint_id"]["stringValue"] for e in page_one}
        self.assertNotIn(latest, ids_on_page_one)
        self.assertEqual(self.saver.get_tuple(a_config(thread)).checkpoint["id"], latest)


class PendingWritesTest(SaverTestCase):
    def setUp(self):
        super().setUp()
        self.checkpoint = a_checkpoint("1f18e122-0300-6868-bffe-000000000001")
        self.put("t-writes", self.checkpoint)
        self.config = a_config("t-writes", self.checkpoint["id"])

    def test_writes_come_back_attached_to_their_checkpoint(self):
        self.saver.put_writes(self.config, [("messages", "hello")], "task-a")

        pending = self.saver.get_tuple(a_config("t-writes")).pending_writes
        self.assertEqual(pending, [("task-a", "messages", "hello")])

    def test_writes_for_another_checkpoint_are_not_attached(self):
        later = a_checkpoint("1f18e122-0300-6868-bffe-000000000002")
        self.put("t-writes", later, parent_id=self.checkpoint["id"])
        self.saver.put_writes(self.config, [("messages", "old")], "task-a")

        pending = self.saver.get_tuple(a_config("t-writes")).pending_writes
        self.assertEqual(pending, [], "these writes belong to the previous checkpoint")

    def test_a_repeated_non_negative_index_keeps_the_first_write(self):
        self.saver.put_writes(self.config, [("messages", "first")], "task-a")
        self.saver.put_writes(self.config, [("messages", "second")], "task-a")

        pending = self.saver.get_tuple(a_config("t-writes")).pending_writes
        self.assertEqual(pending, [("task-a", "messages", "first")])

    def test_a_special_channel_is_overwritten_rather_than_kept(self):
        # __error__ maps to index -1 in WRITES_IDX_MAP, and negative indices
        # overwrite. This is the half of the dedup contract the previous test
        # would otherwise leave unproven.
        self.saver.put_writes(self.config, [("__error__", "first")], "task-a")
        self.saver.put_writes(self.config, [("__error__", "second")], "task-a")

        pending = self.saver.get_tuple(a_config("t-writes")).pending_writes
        self.assertEqual(pending, [("task-a", "__error__", "second")])

    def test_writes_from_different_tasks_are_kept_separately(self):
        self.saver.put_writes(self.config, [("messages", "from-a")], "task-a")
        self.saver.put_writes(self.config, [("messages", "from-b")], "task-b")

        pending = self.saver.get_tuple(a_config("t-writes")).pending_writes
        self.assertEqual(
            sorted(pending), [("task-a", "messages", "from-a"), ("task-b", "messages", "from-b")]
        )

    def test_the_dedup_answer_does_not_come_from_the_returned_order(self):
        # Guards both dedup tests above from being vacuous. Under newest-first the
        # second write is returned first, so reducing in the order ListEvents gave
        # would keep "second" for a first-wins channel; only the eventId sort gets
        # "first". Under oldest-first the returned order happens to be creation
        # order, and nothing distinguishes the two implementations — which is
        # exactly why this pair of orderings is both run.
        self.saver.put_writes(self.config, [("messages", "first")], "task-a")
        self.saver.put_writes(self.config, [("messages", "second")], "task-a")

        events = self.plane.list_events(MEMORY_ID, "actor-1", "t-writes", max_results=MAX_EVENTS)
        writes = [
            event
            for event in events
            if event["metadata"]["kind"]["stringValue"] == "writes"
        ]
        naive = "second" if self.newest_first else "first"
        self.assertEqual(first_write_value(self.saver, writes[0]), naive)

        pending = self.saver.get_tuple(a_config("t-writes")).pending_writes
        self.assertEqual(pending, [("task-a", "messages", "first")])

    def test_unpadded_event_ids_still_order_correctly(self):
        # EventId's pattern is [0-9]+#[a-fA-F0-9]+. The observed ids are
        # zero-padded, but nothing in the model says they must be, so ordering has
        # to survive comparing 999 against 1000 — where a string comparison gets
        # the answer backwards. The clock is set to straddle that boundary
        # deliberately; without it this test passes against a string sort too.
        self.plane.zero_pad = False
        self.plane.clock = 999
        self.saver.put_writes(self.config, [("messages", "first")], "task-a")
        self.saver.put_writes(self.config, [("messages", "second")], "task-a")

        pending = self.saver.get_tuple(a_config("t-writes")).pending_writes
        self.assertEqual(pending, [("task-a", "messages", "first")])


class PendingWritesOldestFirstTest(PendingWritesTest):
    """Every write assertion again, against a store that returns oldest-first.

    Both orderings have to give the same answer, because the service commits to
    neither. Running the whole case twice is the cheapest way to say that.
    """

    newest_first = False


class CrossInstanceRecallTest(SaverTestCase):
    """The actual claim: durable cross-instance thread state for sync invoke.

    Turn 2 runs on a graph built from a second saver instance over the same store,
    which is the offline twin of two container replicas sharing one session_id.
    """

    def setUp(self):
        super().setUp()
        stub = running_stub()
        base_url = stub.__enter__()
        self.addCleanup(stub.__exit__, None, None, None)
        previous = os.environ.get("ORDERS_API_BASE")
        os.environ["ORDERS_API_BASE"] = base_url
        self.addCleanup(self._restore, previous)

    def _restore(self, previous):
        if previous is None:
            os.environ.pop("ORDERS_API_BASE", None)
        else:
            os.environ["ORDERS_API_BASE"] = previous

    def an_instance(self):
        """A fresh agent instance, sharing only the store."""
        llm = FakeChatModel()
        saver = AgentCoreMemorySaver(MEMORY_ID, actor_id="actor-1")
        return llm, build_graph(llm=llm, tools=SUPPORT_TOOLS, checkpointer=saver)

    def ask(self, graph, prompt, thread_id):
        return graph.invoke(
            {"messages": [HumanMessage(prompt)]},
            config={"configurable": {"thread_id": thread_id}},
        )

    def test_a_second_instance_resumes_the_first_instances_thread(self):
        thread = "session-cross-instance"
        _, first = self.an_instance()
        self.ask(first, "Hi, I'm Dana and my order number is 12345.", thread)

        llm, second = self.an_instance()
        turn_two = self.ask(second, "Has it shipped yet?", thread)

        # Turn 2 never repeats 12345, so a lookup_order carrying it can only have
        # come from state the saver persisted on the other instance.
        self.assertEqual(
            llm.tool_calls_made[-1]["args"], {"order_id": "12345"}
        )
        self.assertIn("Dana", " ".join(m.text for m in turn_two["messages"]))

    def test_a_different_thread_id_on_the_same_store_recalls_nothing(self):
        _, first = self.an_instance()
        self.ask(first, "Hi, I'm Dana and my order number is 12345.", "session-x")

        llm, second = self.an_instance()
        turn_two = self.ask(second, "Has it shipped yet?", "session-y")

        self.assertEqual(llm.tool_calls_made, [])
        self.assertIn("What is your order number?", turn_two["messages"][-1].text)

    def test_the_thread_id_is_the_session_id_the_events_were_written_under(self):
        thread = "session-keying"
        _, graph = self.an_instance()
        self.ask(graph, "Where is my order 12345?", thread)

        self.assertIn((MEMORY_ID, "actor-1", thread), self.plane.events)
        kinds = [
            event["metadata"]["kind"]["stringValue"]
            for event in self.plane.events[(MEMORY_ID, "actor-1", thread)]
        ]
        self.assertIn("checkpoint", kinds)
        self.assertIn("writes", kinds)


if __name__ == "__main__":
    unittest.main()

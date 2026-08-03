# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Diff C-1: LangGraph checkpoints as events in Amazon Bedrock AgentCore Memory.

bedrock_agentcore.memory.integrations ships exactly one integration, strands.
There is no LangGraph checkpointer in the SDK, so the AgentCoreMemoryConfig and
AgentCoreMemorySessionManager wiring that works for a Strands agent cannot be
reused here. What is reusable is the storage pattern underneath it: everything is
an event in one (memory_id, actor_id, session_id) stream, and an event payload can
carry an arbitrary JSON blob.

So a checkpoint becomes a blob event. thread_id becomes session_id, which is what
Runtime's RequestContext already hands the entrypoint, and that is the whole
reason this is worth doing: the key is stable across container instances, so two
replicas of the agent see one conversation.

Read the class docstring before assuming this is a general-purpose checkpoint
backend. It is not.
"""

import base64
import json

from bedrock_agentcore.memory import MemoryClient
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    CheckpointTuple,
)

# MemoryClient.list_events hardcodes the API maxResults to 100, follows nextToken
# internally, and treats its own max_results as a total cap across pages. So one
# get_tuple costs ceil(min(N, MAX_EVENTS) / 100) ListEvents calls, where N is
# every event in the thread. Every superstep appends one more, and there is no
# trim primitive: the cost of reading a thread grows linearly with its length. Live
# counts match: 1 ListEvents call at N=11, and 2 at N=121.
MAX_EVENTS = 10000

# ListEvents has no sort parameter. Its input members are exactly memoryId,
# sessionId, actorId, includePayloads, filter, maxResults and nextToken; the
# service model is silent on ordering and so are the public docs, while the SDK
# contradicts itself — the list_events docstring says chronological, whereas
# get_last_k_turns and the Strands session manager both behave newest-first. A live
# read of a 121-event stream in us-east-1 came back newest-first, so the docstring
# at bedrock_agentcore/memory/client.py:841 is wrong today; that is one region on
# one day, not a contract, which is the point.
# Therefore this class never trusts the returned order. Every selection below is
# made client-side on checkpoint_id, which is a uuid6 and so sorts by time as a
# string. Past MAX_EVENTS the scan silently truncates, and with no ordering
# contract, *which* checkpoints vanish is undefined rather than "the oldest".
#
# Two other limits worth knowing before this holds anything you care about.
# Metadata tags are attached to each event below, but they cannot be filtered on:
# only indexed keys are usable in metadata filters, examples/memory/
# configure_memory.py declares none, and declaring one later is a one-way door
# because indexed keys cannot be removed. And MemoryDocument is a sensitive shape
# with no documented size constraint. A blob event of 4 MiB (4194304 bytes) was
# written and read back byte-identical in us-east-1, as were 10 KB, 100 KB, 500 KB
# and 1 MiB, so the ceiling is above 4 MiB rather than at some small value — but it
# is still undocumented, so treat 4 MiB as the largest size measured, not as a
# supported limit. A checkpoint that large is a problem of its own anyway: the whole
# blob is rewritten every superstep and re-read on every get_tuple.


def _event_order(event_id):
    """Creation order for a service-assigned eventId.

    An eventId is `[0-9]+#[a-fA-F0-9]+`, a timestamp prefix and a hex suffix
    (`0000001756147154000#ffa53e54`). The observed values are zero-padded, but the
    model's pattern does not promise that, so the prefix is compared as an integer
    rather than as a string. The suffix is not a sequence, so events stamped in the
    same millisecond remain unordered relative to each other.
    """
    try:
        return int(event_id.split("#")[0])
    except (AttributeError, ValueError):
        return -1  # no id: sort first rather than crash the read


def _enc(serde, obj):
    """Serialise with LangGraph's own serde, then base64 so JSON can carry it."""
    type_name, payload = serde.dumps_typed(obj)
    return [type_name, base64.b64encode(payload).decode()]


def _dec(serde, pair):
    return serde.loads_typed((pair[0], base64.b64decode(pair[1])))


class AgentCoreMemorySaver(BaseCheckpointSaver):
    """Durable cross-instance thread state for synchronous graph.invoke.

    That claim is the whole claim. This is not parity with a checkpoint backend
    such as InMemorySaver or a SQL saver. It implements the three methods the
    synchronous invoke path calls — get_tuple, put and put_writes — faithfully,
    which is enough for a conversation to resume on a different container than the
    one that started it.

    What is deliberately absent: list(), and therefore get_state_history and time
    travel; delete_thread; forks and branches; and every async variant, since the
    entrypoint is sync graph.invoke. Those are reconstructable client-side but none
    of them is implemented or demonstrated here, so do not read their absence as
    "works, untested".

    Storage layout: one blob event per checkpoint (kind=checkpoint) and one per
    put_writes call (kind=writes), all in the (actor_id, thread_id) event stream.
    The whole checkpoint is stored as one snapshot with channel_values intact;
    new_versions is ignored, which the contract permits and which trades write size
    for not multiplying the read scan per channel.

    Note also that these events are subject to the memory resource's
    eventExpiryDuration. Checkpoints are TTL'd, so "resume this thread whenever"
    is only true inside that window.
    """

    def __init__(self, memory_id, *, actor_id="langgraph", region_name=None):
        super().__init__()
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.client = MemoryClient(region_name=region_name)

    def _records(self, thread_id, checkpoint_ns):
        """Every record this saver wrote for one thread and namespace.

        Yields (event_id, record). The event id is the service-assigned one, and
        it is paired with the record because the returned order is unspecified and
        the caller needs a key to order by. Callers order what they need.
        """
        events = self.client.list_events(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=thread_id,
            max_results=MAX_EVENTS,
        )
        for event in events:
            for item in event.get("payload", []):
                if "blob" not in item:
                    continue  # not ours: conversational events share this stream
                record = json.loads(item["blob"])
                if record.get("ns") == checkpoint_ns:
                    yield event.get("eventId", ""), record

    def get_tuple(self, config):
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")

        records = list(self._records(thread_id, checkpoint_ns))
        checkpoints = {r["id"]: r for _, r in records if r["kind"] == "checkpoint"}
        if not checkpoints:
            return None  # first invocation; the loop synthesises an empty checkpoint

        wanted = configurable.get("checkpoint_id")
        # max() over uuid6 ids, not events[0]: see the ordering note at the top.
        record = checkpoints.get(wanted) if wanted else checkpoints[max(checkpoints)]
        if record is None:
            return None

        # Reduce the writes events client-side, because the store has no overwrite
        # primitive. Non-negative indices are first-wins; the negative special
        # channels (__error__, __interrupt__, ...) overwrite. Both halves of that
        # contract are about *write* order, so the records are sorted by eventId
        # first rather than reduced in whatever order ListEvents happened to
        # return them. See _event_order for what that buys and what it does not.
        writes = {}
        for _, record_ in sorted(records, key=lambda pair: _event_order(pair[0])):
            if record_["kind"] != "writes" or record_["checkpoint_id"] != record["id"]:
                continue
            for idx, (channel, value) in enumerate(record_["writes"]):
                key = (record_["task_id"], WRITES_IDX_MAP.get(channel, idx))
                if key[1] >= 0 and key in writes:
                    continue
                writes[key] = (record_["task_id"], channel, _dec(self.serde, value))

        parent_id = record.get("parent_id")
        return CheckpointTuple(
            config=_config(thread_id, checkpoint_ns, record["id"]),
            # metadata must survive the round trip: the loop reads
            # metadata["step"] unguarded, so dropping it breaks resume.
            checkpoint=_dec(self.serde, record["checkpoint"]),
            metadata=_dec(self.serde, record["metadata"]),
            parent_config=(
                _config(thread_id, checkpoint_ns, parent_id) if parent_id else None
            ),
            pending_writes=list(writes.values()),
        )

    def put(self, config, checkpoint, metadata, new_versions):
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        record = {
            "kind": "checkpoint",
            "ns": checkpoint_ns,
            "id": checkpoint["id"],
            # Parent linkage is implicit: the incoming config carries the previous
            # checkpoint's id.
            "parent_id": configurable.get("checkpoint_id"),
            "checkpoint": _enc(self.serde, checkpoint),
            "metadata": _enc(self.serde, dict(metadata)),
        }
        self._write(thread_id, record, "checkpoint", checkpoint["id"])
        return _config(thread_id, checkpoint_ns, checkpoint["id"])

    def put_writes(self, config, writes, task_id, task_path=""):
        configurable = config["configurable"]
        checkpoint_id = configurable["checkpoint_id"]
        record = {
            "kind": "writes",
            "ns": configurable.get("checkpoint_ns", ""),
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "writes": [[channel, _enc(self.serde, value)] for channel, value in writes],
        }
        self._write(configurable["thread_id"], record, "writes", checkpoint_id)

    def _write(self, thread_id, record, kind, checkpoint_id):
        # The metadata tags are for a human reading the event stream. They are not
        # queryable on a memory that declares no indexed keys.
        self.client.create_blob_event(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=thread_id,
            blob_data=json.dumps(record),
            metadata={
                "kind": {"stringValue": kind},
                "checkpoint_id": {"stringValue": checkpoint_id},
            },
        )


def _config(thread_id, checkpoint_ns, checkpoint_id):
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }

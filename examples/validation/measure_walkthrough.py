# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The numbers this article prints, measured by the run that prints them.

Every figure quoted about AgentCore's behaviour here is taken while the
walkthrough runs, not typed in from a notebook. Run the walkthrough and the last
thing it prints is this table; the values will differ from the ones in the
article, because they are latencies against a live service, and that is the
point of measuring them rather than quoting them.

A row prefixed ``!`` is how long something took to *fail*. The table is printed
even when the run breaks, so those rows are the ones that say where it got to,
and they are marked because the names read like claims.

Two of them are not timings and are the ones worth reading:

``memory events`` is how many events the thread holds after each turn. Every
superstep appends one event per channel it wrote plus one for the checkpoint
itself, and there is no trim primitive, so this number is the read cost of the
next turn: AgentCoreMemorySaver.get_tuple pages the whole stream at 100 events a
call and, unless it is built with a ``limit``, does not stop early.

``messages rehydrated`` is how many messages a freshly built graph recovered
from AgentCore Memory *before* invoking anything. It is the cross-instance claim
reduced to an integer: nothing in the process held those messages.

This module used to carry a NOT_MEASURED_HERE table: four Runtime numbers the
article printed that the walkthrough could not produce, each with the reason. The
reason was the same one four times over — they need a deployed runtime, and
deploying one meant a container build, a registry and an ARM64 builder, which is a
different article. CreateAgentRuntime's ``codeConfiguration`` needs none of those.
It takes a zip in S3, pip builds it, and the walkthrough now deploys and invokes
the agent, so all four are measured here and the table is gone rather than
maintained. See examples/stage1_replatform/deploy_runtime.py.
"""

import time
from contextlib import contextmanager

from bedrock_agentcore.memory import MemoryClient

# A cap on the count below, not a limit anything enforces. MemoryClient.list_events
# hardcodes the API maxResults to 100, follows nextToken internally, and treats its
# own max_results as a total across pages, so this is how far the two functions here
# will page before they stop counting. It is set high because the number it bounds
# is the one being measured: a thread that has quietly passed the cap would report a
# flat count and read like a thread that had stopped growing.
EVENT_SCAN_CAP = 10000

# Sizes the payload probe writes, in bytes. The largest is 4 MiB. The ladder
# stops there rather than binary-searching for a ceiling, so what it establishes
# is a floor: blob events of at least this size round-trip intact. Treat it as
# the largest size measured and not as a supported limit, which is the honest
# reading and also the useful one, because a checkpoint this large is a bad idea
# under any ceiling — a channel is written out in full on every superstep that
# touches it, and the whole stream is read back on every get_tuple.
PAYLOAD_LADDER = (10_240, 102_400, 512_000, 1_048_576, 4_194_304)

class Measurements:
    """Collect named measurements in the order they are taken, then print them."""

    def __init__(self):
        self.taken = []

    def record(
        self, name: str, value, unit: str = "", note: str = "", failed: bool = False
    ) -> None:
        self.taken.append((name, value, unit, note, failed))

    @contextmanager
    def timing(self, name: str, note: str = ""):
        """Time a block and record it in seconds, marked if the block raised.

        The elapsed time is recorded either way, because how long something took
        to fail is worth knowing. What it must not do is record a failure as a
        measurement of success: these names are claims — "CreateAgentRuntime ->
        READY" — and a run that raised inside the block printed exactly that
        claim with a number beside it, which is a measurement table asserting
        the opposite of what happened.
        """
        started = time.monotonic()
        failure = ""
        try:
            yield
        except BaseException as error:  # recorded, then re-raised untouched
            failure = type(error).__name__
            raise
        finally:
            if failure != "":
                reason = f"time to FAIL with {failure}, not to reach this state"
                note = f"{reason}; {note}" if note else reason
            self.record(name, round(time.monotonic() - started, 1), "s", note, bool(failure))

    def report(self) -> None:
        print("\n=== measurements, from this run ===")
        width = max((len(name) for name, *_ in self.taken), default=0)
        for name, value, unit, note, failed in self.taken:
            suffix = f"  ({note})" if note else ""
            gutter = "! " if failed else "  "
            print(f"{gutter}{name:<{width}}  {value}{unit}{suffix}")


def count_events(
    memory_id: str, actor_id: str, session_id: str, region_name: str
) -> int:
    """How many events the thread holds right now.

    This is the quantity the saver's read cost is a function of, so counting it
    is not bookkeeping: one get_tuple costs ceil(N / 100) ListEvents calls, and N
    only ever goes up.
    """
    return len(
        MemoryClient(region_name=region_name).list_events(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            max_results=EVENT_SCAN_CAP,
        )
    )


def count_rehydrated_messages(saver, thread_id: str, actor_id: str) -> int:
    """How many messages a new saver recovers for this thread before any invoke.

    Called on a saver built after the previous turn's graph was discarded, so a
    non-zero answer here is state that came back from AgentCore Memory rather
    than from anything still in this process.

    ``actor_id`` is required, not incidental: the saver resolves the event stream
    from thread_id and actor_id together and raises without either.
    """
    tuple_ = saver.get_tuple(
        {"configurable": {"thread_id": thread_id, "actor_id": actor_id}}
    )
    if tuple_ is None:
        return 0
    return len(tuple_.checkpoint.get("channel_values", {}).get("messages", []))


def probe_payload_floor(
    memory_id: str, actor_id: str, session_id: str, region_name: str
) -> int:
    """Write each size in PAYLOAD_LADDER as a blob event and read it back.

    Returns the largest size that round-tripped byte-identical. Writes into a
    session of its own so that the event counts measured on the conversation
    thread are the conversation's, not the probe's.

    Byte-identical is checked rather than assumed. A store that silently
    truncated a large payload would still return an event, and a checkpoint that
    comes back short is a resumed conversation that has quietly lost its middle.
    """
    client = MemoryClient(region_name=region_name)
    largest = 0
    for size in PAYLOAD_LADDER:
        blob = "x" * size
        client.create_blob_event(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            blob_data=blob,
        )
        events = client.list_events(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            max_results=EVENT_SCAN_CAP,
            include_payload=True,
        )
        read_back = {
            item["blob"]
            for event in events
            for item in event.get("payload", [])
            if "blob" in item
        }
        if blob not in read_back:
            print(f"  payload {size} B did NOT read back identical")
            break
        print(f"  payload {size} B round-tripped byte-identical")
        largest = size
    return largest

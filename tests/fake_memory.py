# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""A fake AgentCore Memory data plane, faithful to the parts that constrain us.

Two behaviours are deliberately hostile, because they are what the real service
does and what the saver has to survive:

1. list_events returns events in an unspecified order. FakeMemoryDataPlane
   reverses the append order by default, so any code that trusts events[0] to be
   the newest — or the oldest — gets the wrong answer. There is no sort parameter
   to ask for an order, so this is not pessimism, it is the contract.
2. list_events pages. The real MemoryClient wrapper asks the API for 100 events at
   a time, follows nextToken internally, and treats its own max_results as a total
   cap before returning a bare list. This fake reproduces the paging so a thread
   longer than one page is exercised, and counts the pages so a test can assert
   the read cost.

3. eventIds have the real shape — a timestamp prefix and a hex suffix, stamped from
   a counter shared by every stream. That prefix is the only handle on creation
   order the service gives, so the saver's write-dedup depends on it and the fake
   has to hand back something faithful rather than a bare index.

Everything else is the minimum the saver touches: create_blob_event wrapping
blob_data as payload [{"blob": ...}], and metadata passed through.
"""

API_PAGE_SIZE = 100

# The shape of a real service-assigned eventId: a timestamp prefix and a hex
# suffix, as in the SDK's own example 0000001756147154000#ffa53e54. Events are
# stamped one millisecond apart here so that creation order is recoverable, which
# is the only property the saver relies on.
FIRST_EVENT_MILLIS = 1756147154000


class FakeMemoryDataPlane:
    """Stands in for bedrock_agentcore.memory.MemoryClient.

    Only create_blob_event and list_events exist, because they are the only two
    operations AgentCoreMemorySaver calls.
    """

    def __init__(self, *, region_name=None, newest_first=True, zero_pad=True):
        self.region_name = region_name
        # Which way round the unspecified order comes back. Flip it in a test and
        # every assertion must still hold.
        self.newest_first = newest_first
        # The observed eventIds are zero-padded, but the model's pattern only says
        # [0-9]+, so a test can turn the padding off.
        self.zero_pad = zero_pad
        self.events = {}  # (memory_id, actor_id, session_id) -> list of events
        # The stamp for the next event, shared by every stream. Settable so a test
        # can start it just below a digit-count boundary.
        self.clock = FIRST_EVENT_MILLIS
        self.page_reads = 0
        self.list_calls = []

    def _next_event_id(self):
        millis = self.clock
        self.clock += 1
        return f"{millis:019d}#deadbeef" if self.zero_pad else f"{millis}#deadbeef"

    def create_blob_event(
        self,
        memory_id,
        actor_id,
        session_id,
        blob_data,
        event_timestamp=None,
        branch=None,
        metadata=None,
    ):
        stream = self.events.setdefault((memory_id, actor_id, session_id), [])
        event = {
            "eventId": self._next_event_id(),
            "payload": [{"blob": blob_data}],
            "metadata": metadata or {},
        }
        stream.append(event)
        return event

    def list_events(
        self, memory_id, actor_id, session_id, max_results=100, **kwargs
    ):
        self.list_calls.append(
            {"session_id": session_id, "actor_id": actor_id, "max_results": max_results}
        )
        stream = list(self.events.get((memory_id, actor_id, session_id), []))
        if self.newest_first:
            stream.reverse()

        # Page exactly the way the real wrapper does, and stop at the total cap.
        collected = []
        for start in range(0, len(stream), API_PAGE_SIZE):
            if len(collected) >= max_results:
                break
            self.page_reads += 1
            collected.extend(stream[start : start + API_PAGE_SIZE])
        return collected[:max_results]


class FakeMemoryControlPlane:
    """The control-plane half: create, list and get, with the name rule enforced.

    Separate from FakeMemoryDataPlane because the two are used by different code
    for different reasons — the saver only ever reads and writes events, while
    configure_memory only ever creates the resource. What this one has to be
    faithful about is the constraint that broke a live re-run: CreateMemory
    rejects a name that already exists, and ListMemories summaries carry no name
    to match on, only an id with the service's suffix on it.
    """

    def __init__(self, *, region_name=None, existing=()):
        self.region_name = region_name
        self.memories = {}  # id -> the Memory shape GetMemory returns
        self.calls = []
        for name, expiry in existing:
            self._add(name, expiry)
        # configure_memory reaches through for GetMemory, which the SDK wrapper
        # does not expose. Same object, so a test asserts against one place.
        self.gmcp_client = self

    def _add(self, name, event_expiry_days):
        memory_id = f"{name}-{len(self.memories) + 1:010d}"
        self.memories[memory_id] = {
            "id": memory_id,
            "name": name,
            "status": "ACTIVE",
            "eventExpiryDuration": event_expiry_days,
        }
        return memory_id

    def create_memory_and_wait(self, name, event_expiry_days=None, **kwargs):
        self.calls.append(("create_memory_and_wait", {"name": name, **kwargs}))
        for memory in self.memories.values():
            if memory["name"] == name:
                raise RuntimeError(
                    "Validation failed during CreateMemory: "
                    f"Memory with name {name} already exists"
                )
        return {"id": self._add(name, event_expiry_days)}

    def list_memories(self, max_results=100):
        self.calls.append(("list_memories", {"max_results": max_results}))
        # No name field, which is why callers match on the id prefix.
        return [
            {"id": memory["id"], "status": memory["status"], "arn": f"arn:{memory['id']}"}
            for memory in list(self.memories.values())[:max_results]
        ]

    def get_memory(self, memoryId):  # noqa: N803 - the API's own parameter name
        self.calls.append(("get_memory", {"memoryId": memoryId}))
        return {"memory": dict(self.memories[memoryId])}

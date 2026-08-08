# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""A fake AgentCore Memory control plane, faithful to the parts that constrain us.

This covers the operations configure_memory calls to bring the resource up. The
data plane is not faked here: checkpoint reads and writes go through
langgraph-checkpoint-aws, which ships its own tests for them.
"""


class FakeMemoryControlPlane:
    """Create, list and get, with the name rule enforced.

    What this has to be faithful about is the constraint that broke a live re-run:
    CreateMemory rejects a name that already exists, and ListMemories summaries
    carry no name to match on, only an id with the service's suffix on it.
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

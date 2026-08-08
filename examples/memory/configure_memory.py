# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Configure Amazon Bedrock AgentCore Memory: the raw event stream both stages ride.

event_expiry_days is passed explicitly on purpose. MemoryClient.create_memory
defaults it to 90 (bedrock_agentcore/memory/client.py:160), so a caller that omits
it silently accepts a 90-day retention it never chose; the service accepts 3 to
365. It is the retention of the raw event stream, which is also where
AgentCoreMemorySaver writes its checkpoints, so this number is how long a
conversation can be resumed, not just how long the transcript is kept.

Idempotent, like create_gateway.py and register_target.py: a memory of this name
is reused rather than recreated, because CreateMemory rejects a duplicate name.
Reused untouched, unlike the gateway target — a memory resource holds
conversations, and the walkthrough gives every run a fresh actor and session id,
so nothing measured against a reused resource inherits an older run's events. The
one thing reuse cannot promise is the retention it prints, so on reuse the number
is read back off the resource instead of quoted from EVENT_EXPIRY_DAYS.
"""

import argparse
import os
from typing import Optional

from bedrock_agentcore.memory import MemoryClient

# 30 days, chosen rather than inherited. Raise it if a thread has to be resumable
# for longer than that.
EVENT_EXPIRY_DAYS = 30

# The service appends a suffix, so the memory id comes back as
# MigratedAgentMemory-XXXXXXXXXX. Named here rather than inline in the call
# because teardown has to find this resource by name in an account it did not
# create it in, and a name spelled in two places is a name that drifts.
MEMORY_NAME = "MigratedAgentMemory"


def _find_existing_memory(client, name: str = MEMORY_NAME) -> Optional[str]:
    """Return the id of an existing memory of this name, or None.

    Matched on the id prefix rather than on the name, because ListMemories
    summaries carry arn, createdAt, id and status and no name at all. The suffix
    is the service's; the prefix and the separator are ours.
    """
    for summary in client.list_memories():
        if summary["id"].startswith(name + "-"):
            return summary["id"]
    return None


def existing_memory_id(region_name: str = "us-east-1") -> Optional[str]:
    """Whether our memory is already there, before configure_memory runs.

    Memory is the slowest thing this walkthrough provisions — minutes, not
    seconds — so a reused one reported under a CreateMemory label is the most
    misleading number of the set.
    """
    return _find_existing_memory(MemoryClient(region_name=region_name))


def configure_memory(
    region_name: str = "us-east-1",
    event_expiry_days: int = EVENT_EXPIRY_DAYS,
) -> str:
    """Create (or reuse) the migrated agent's memory and return its memory id."""
    client = MemoryClient(region_name=region_name)

    existing_id = _find_existing_memory(client)
    if existing_id is not None:
        # Printed from the resource, not from the argument: this run did not set
        # the retention and an older run may have chosen a different one.
        memory = client.gmcp_client.get_memory(memoryId=existing_id)["memory"]
        days = memory.get("eventExpiryDuration", "unknown")
        print(f"Memory '{MEMORY_NAME}' already exists: {existing_id}")
        print(f"  reused as it stands, events expire after {days} days")
        return existing_id

    memory = client.create_memory_and_wait(
        name=MEMORY_NAME,
        description="Memory for migrated customer support agent",
        event_expiry_days=event_expiry_days,
        # No long-term strategies, deliberately. Each one the service offers
        # (summary, preference, semantic) runs LLM extraction against every
        # event — an ongoing per-event cost — and nothing in either stage reads
        # the records a strategy would produce: the stage-1 checkpointer and
        # the stage-2 session manager both ride the raw event stream, and no
        # retrieval_config is ever set. create_memory_and_wait requires the
        # argument, so the empty list is spelled out rather than omitted.
        strategies=[],
    )
    memory_id = memory["id"]
    print(f"Memory created: {memory_id} (events expire after {event_expiry_days} days)")
    return memory_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1).",
    )
    parser.add_argument(
        "--event-expiry-days",
        type=int,
        default=int(os.environ.get("EVENT_EXPIRY_DAYS", EVENT_EXPIRY_DAYS)),
        help=f"Event retention in days, 3 to 365 (default: {EVENT_EXPIRY_DAYS}).",
    )
    args = parser.parse_args()

    configure_memory(args.region, args.event_expiry_days)


if __name__ == "__main__":
    main()

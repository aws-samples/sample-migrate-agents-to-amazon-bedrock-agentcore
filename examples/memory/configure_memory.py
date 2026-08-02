# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Configure Amazon Bedrock AgentCore Memory with summary, preference, and semantic strategies.

event_expiry_days is passed explicitly on purpose. MemoryClient.create_memory
defaults it to 90 (bedrock_agentcore/memory/client.py:160), so a caller that omits
it silently accepts a 90-day retention it never chose; the service accepts 3 to
365. It is the retention of the raw event stream, which is also where
examples/stage1_replatform/agentcore_memory_saver.py writes its checkpoints, so
this number is how long a conversation can be resumed, not just how long the
transcript is kept.
"""

import argparse
import os

from bedrock_agentcore.memory import MemoryClient

# 30 days, chosen rather than inherited. Raise it if a thread has to be resumable
# for longer than that.
EVENT_EXPIRY_DAYS = 30


def configure_memory(
    region_name: str = "us-east-1",
    event_expiry_days: int = EVENT_EXPIRY_DAYS,
) -> str:
    """Create the migrated agent's memory resource and return its memory id."""
    client = MemoryClient(region_name=region_name)
    memory = client.create_memory_and_wait(
        name="MigratedAgentMemory",
        description="Memory for migrated customer support agent",
        event_expiry_days=event_expiry_days,
        strategies=[
            {
                "summaryMemoryStrategy": {
                    "name": "SessionSummarizer",
                    "namespaceTemplates": ["/summaries/{actorId}/{sessionId}/"],
                }
            },
            {
                "userPreferenceMemoryStrategy": {
                    "name": "PreferenceLearner",
                    "namespaceTemplates": ["/preferences/{actorId}/"],
                }
            },
            {
                "semanticMemoryStrategy": {
                    "name": "FactExtractor",
                    "namespaceTemplates": ["/facts/{actorId}/"],
                }
            },
        ],
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

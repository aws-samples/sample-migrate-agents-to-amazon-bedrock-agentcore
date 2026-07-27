# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Configure Amazon Bedrock AgentCore Memory with summary, preference, and semantic strategies."""

from bedrock_agentcore.memory import MemoryClient


def configure_memory(region_name: str = "us-east-1") -> str:
    """Create the migrated agent's memory resource and return its memory id."""
    client = MemoryClient(region_name=region_name)
    memory = client.create_memory_and_wait(
        name="MigratedAgentMemory",
        description="Memory for migrated customer support agent",
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
    print(f"Memory created: {memory_id}")
    return memory_id


def main() -> None:
    configure_memory()


if __name__ == "__main__":
    main()

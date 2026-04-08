# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Configure Amazon Bedrock AgentCore Memory with summary, preference, and semantic strategies."""

from bedrock_agentcore.memory import MemoryClient

client = MemoryClient(region_name="us-east-1")
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

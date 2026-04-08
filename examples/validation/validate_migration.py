# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Validate agent behavior after migration to Amazon Bedrock AgentCore.

This is a template. Replace the commented agent invocation with your
actual agent call to run validation against your migrated agent.
"""

test_cases = [
    {
        "input": "What's the status of order #12345?",
        "expected_tools": ["lookup_order"],
        "expected_behavior": "Returns order status without processing any changes",
    },
    {
        "input": "I want to return order #12345, it arrived damaged",
        "expected_tools": ["lookup_order", "process_return"],
        "expected_behavior": "Looks up order first, then processes return",
    },
]


def run_validation(agent=None):
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['input']}")
        print(f"  Expected tools: {test['expected_tools']}")
        print(f"  Expected behavior: {test['expected_behavior']}")
        if agent:
            response = agent(test["input"])
            print(f"  Result: {response.message}")
        else:
            print("  [SKIPPED] No agent provided. Pass your agent to run_validation().")
        print()


if __name__ == "__main__":
    # Replace None with your agent instance:
    # from examples.rebuild.strands_agent import agent
    run_validation(agent=None)

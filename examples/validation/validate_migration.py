# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Validate agent behavior after migration to Amazon Bedrock AgentCore."""

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

for test in test_cases:
    # Replace with your agent invocation
    # response = agent(test["input"])
    # Compare tool call sequence against expected_tools
    # Verify response addresses the customer's request
    print(f"Test: {test['input']}")
    print(f"  Expected tools: {test['expected_tools']}")
    print(f"  Expected behavior: {test['expected_behavior']}")

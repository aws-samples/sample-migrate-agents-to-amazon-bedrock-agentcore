# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Inventory existing agent tools before migration to Amazon Bedrock AgentCore."""

existing_tools = {
    "lookup_order": {
        "description": "Look up order status by order ID",
        "parameters": {"order_id": "string"},
        "api_endpoint": "https://api.example.com/orders/{order_id}",
    },
    "process_return": {
        "description": "Initiate a return for an order",
        "parameters": {"order_id": "string", "reason": "string"},
        "api_endpoint": "https://api.example.com/returns",
    },
    "search_faq": {
        "description": "Search knowledge base for FAQ answers",
        "parameters": {"query": "string"},
        "api_endpoint": "https://api.example.com/faq/search",
    },
}

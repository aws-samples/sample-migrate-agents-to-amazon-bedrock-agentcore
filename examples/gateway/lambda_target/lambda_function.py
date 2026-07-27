# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Customer-support tools backing an Amazon Bedrock AgentCore Gateway target.

AgentCore Gateway invokes this Lambda once per tool call. The tool arguments
arrive as the raw event dict, and the tool name (prefixed with the gateway
target name, e.g. "supportTools___lookup_order") is provided on
context.client_context.custom['bedrockAgentCoreToolName']. The Lambda's return
value becomes the MCP tool result content.

Implements lookup_order and process_return to match the toolSchema declared in
examples/gateway/register_target.py and the scenario in
examples/rebuild/strands_agent.py.
"""


def _tool_name(context) -> str:
    client_context = getattr(context, "client_context", None)
    custom = getattr(client_context, "custom", None) or {}
    name = custom.get("bedrockAgentCoreToolName", "")
    # Strip the gateway target prefix, e.g. "supportTools___lookup_order".
    return name.rsplit("___", 1)[-1]


def lookup_order(order_id: str) -> dict:
    return {
        "order_id": order_id,
        "status": "shipped",
        "carrier": "UPS",
        "tracking_number": "1Z999AA10123456784",
        "estimated_delivery": "2026-08-02",
        "items": [
            {"sku": "EC-1042", "name": "Wireless Keyboard", "quantity": 1},
        ],
    }


def process_return(order_id: str, reason: str) -> dict:
    return {
        "order_id": order_id,
        "return_id": "RET-12345",
        "status": "return_initiated",
        "reason": reason,
        "refund_amount": "49.99",
        "instructions": "A prepaid shipping label has been emailed to you.",
    }


def handler(event, context):
    tool = _tool_name(context)
    args = event if isinstance(event, dict) else {}

    if tool == "lookup_order":
        return lookup_order(args.get("order_id", ""))
    if tool == "process_return":
        return process_return(args.get("order_id", ""), args.get("reason", ""))

    return {"error": f"Unknown tool: {tool or '(none)'}"}

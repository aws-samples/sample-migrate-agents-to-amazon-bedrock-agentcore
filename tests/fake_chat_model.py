# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""A scripted stand-in for ChatBedrockConverse, so the stage tests run offline.

It implements only the two surfaces the graph actually uses: invoke() for the
router node, and bind_tools().invoke() for the assist node. It records every
message list it was handed, which is how the memory test asserts what the model
could see on turn 2.

The tool it chooses depends on keywords in the customer's latest message, but the
order ID it passes is recovered from the whole conversation. That is deliberate:
on turn 2 the customer does not repeat the order number, so the only way a tool
call can carry it is if the checkpointer replayed turn 1. Faking the model does
not fake the recall.
"""

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

ANGRY_WORDS = (
    "angry",
    "furious",
    "ridiculous",
    "unacceptable",
    "cancel my",
    "lawyer",
    "legal action",
    "speak to a human",
    "speak to a manager",
)
FAQ_WORDS = ("policy", "faq", "how long", "return window", "allowed")
RETURN_WORDS = ("return this", "send it back", "refund", "start a return")

_ORDER_ID = re.compile(r"\b\d{5}\b")


def _latest_human(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text.lower()
    return ""


class FakeChatModel:
    """The unbound model. The router node calls this one."""

    def __init__(self):
        self.router_calls = []
        self.assist_calls = []
        self.tool_calls_made = []
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return _BoundFakeChatModel(self)

    def invoke(self, messages):
        self.router_calls.append(list(messages))
        latest = _latest_human(messages)
        intent = "escalate" if any(w in latest for w in ANGRY_WORDS) else "assist"
        return AIMessage(intent)


class _BoundFakeChatModel:
    """What bind_tools returns. The assist node calls this one."""

    def __init__(self, parent: FakeChatModel):
        self._parent = parent

    def invoke(self, messages):
        messages = list(messages)
        self._parent.assist_calls.append(messages)

        # Second pass: a tool has already answered, so write the final reply.
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(f"Here is what I found: {messages[-1].content}")

        latest = _latest_human(messages)
        history = " ".join(m.text for m in messages)
        found = _ORDER_ID.search(history)

        if any(w in latest for w in FAQ_WORDS):
            call = {"name": "search_faq", "args": {"query": latest}}
        elif not found:
            # No order ID anywhere in the conversation, so no tool can be called.
            return AIMessage("What is your order number?")
        elif any(w in latest for w in RETURN_WORDS):
            call = {
                "name": "process_return",
                "args": {"order_id": found.group(), "reason": latest},
            }
        else:
            call = {"name": "lookup_order", "args": {"order_id": found.group()}}

        self._parent.tool_calls_made.append(call)
        call["id"] = f"call_{len(self._parent.tool_calls_made)}"
        return AIMessage(content="", tool_calls=[call])

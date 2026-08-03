# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The stage-0 support agent: a compiled StateGraph with a model-driven tool loop.

    START
      └─> classify_intent                  model call, returns state["intent"]
            ├─(intent == "escalate")─> escalate ─> END
            └─(intent == "assist")──> assist     model call with tools bound
                                        ├─(tool calls present)─> tools
                                        └─(else)──────────────> END
                  tools ─────────────────────────> assist       the cycle

Four nodes, two conditional edges, one cycle. route_intent is the hand-written
business branch: it is hosted unchanged by every later stage, and it is the thing
no managed service takes over for you.

build_graph parameterises the three things a migration replaces — the model, the
tools, and the checkpointer. A reader whose graph hardcodes those will add this
signature as part of the migration; that is a real cost of the move, not
something this sample gets for free.
"""

from functools import partial
from typing import Annotated, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from examples.stage0_langgraph.prompts import (
    ESCALATION_MESSAGE,
    ROUTER_PROMPT,
    SYSTEM_PROMPT,
)


class SupportState(TypedDict):
    """Conversation state. add_messages appends rather than replacing."""

    messages: Annotated[list, add_messages]
    intent: str
    escalated: bool


def classify_intent(state: SupportState, llm) -> dict:
    """Ask the model for one word, and accept only the two words we asked for.

    The customer's turns only. ROUTER_PROMPT classifies "the customer's latest
    message", so the assistant's replies and the tool traffic are not input to the
    decision — and passing them costs something: this llm has no tools bound, and
    Converse rejects a request carrying toolResult blocks with no toolConfig, so
    the client rewrites those blocks as text and warns while doing it.
    """
    customer_turns = [m for m in state["messages"] if m.type == "human"]
    response = llm.invoke([SystemMessage(ROUTER_PROMPT), *customer_turns])
    answer = response.text.strip().lower()
    return {"intent": answer if answer in ("escalate", "assist") else "assist"}


def route_intent(state: SupportState) -> Literal["escalate", "assist"]:
    """The hand-tuned business branch. Stage 1 does not change this function."""
    return "escalate" if state["intent"] == "escalate" else "assist"


def escalate(state: SupportState) -> dict:
    """Hand off to a human. No model call, so this node is free and deterministic."""
    return {"messages": [AIMessage(ESCALATION_MESSAGE)], "escalated": True}


def assist(state: SupportState, llm, tools: Sequence) -> dict:
    """Answer the customer, calling tools when the model asks for them."""
    response = llm.bind_tools(tools).invoke(
        [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
    )
    return {"messages": [response]}


def build_graph(llm, tools: Sequence, checkpointer: Optional[object] = None):
    """Compile the support graph.

    Pass a checkpointer to hold conversation state across invocations; callers
    then supply config={"configurable": {"thread_id": ...}} at invoke time.
    """
    builder = StateGraph(SupportState)
    builder.add_node("classify_intent", partial(classify_intent, llm=llm))
    builder.add_node("escalate", escalate)
    builder.add_node("assist", partial(assist, llm=llm, tools=tools))
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_intent,
        {"escalate": "escalate", "assist": "assist"},
    )
    builder.add_edge("escalate", END)
    builder.add_conditional_edges("assist", tools_condition)
    builder.add_edge("tools", "assist")

    return builder.compile(checkpointer=checkpointer)

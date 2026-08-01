# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Run the stage-0 agent locally against Amazon Bedrock.

    python -m examples.stage0_langgraph.run_local

This is the whole of stage 0's operating story: you start the tool backend, you
construct the model, you hold the conversation state in your own process, and
nothing outside this process knows the agent exists. It makes no Amazon Bedrock
AgentCore calls — Bedrock InvokeModel is the only AWS dependency.

Needs AWS credentials and Bedrock model access for MODEL_ID. The offline
assertions live in tests/test_stage0_agent.py and need neither.
"""

import os

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from examples.stage0_langgraph.agent import build_graph
from examples.stage0_langgraph.local_api import running_stub
from examples.stage0_langgraph.tools import SUPPORT_TOOLS

MODEL_ID = "us.anthropic.claude-sonnet-5"
SESSION_ID = "stage0-demo-session"


def ask(graph, prompt: str, thread_id: str) -> dict:
    print(f"\n[{thread_id}] customer: {prompt}")
    state = graph.invoke(
        {"messages": [HumanMessage(prompt)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    for message in state["messages"]:
        if message.type == "tool":
            print(f"  tool {message.name} -> {message.content[:100]}")
    print(f"  intent={state['intent']} escalated={state.get('escalated', False)}")
    print(f"  agent: {state['messages'][-1].text}")
    return state


def main() -> None:
    with running_stub() as base_url:
        os.environ["ORDERS_API_BASE"] = base_url
        print(f"orders API stub: {base_url}")

        # No temperature is set: this model does not accept one. langchain-aws
        # warns and drops the value (bedrock_converse.py:1191), so the router
        # node earns its reproducibility by accepting only two tokens and
        # defaulting to "assist", not by pinning temperature to 0.
        llm = ChatBedrockConverse(
            model=MODEL_ID,
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        # MemorySaver is in-process: it dies with this process and cannot be
        # shared across two replicas. That is the burden stage 1 sheds.
        graph = build_graph(
            llm=llm, tools=SUPPORT_TOOLS, checkpointer=MemorySaver()
        )

        # Two turns on one thread_id. Turn 2 never repeats the order number, so an
        # answer that knows it came from the checkpointer.
        ask(graph, "Hi, I'm Dana and my order number is 12345.", SESSION_ID)
        ask(graph, "Has it shipped yet, and who is carrying it?", SESSION_ID)

        # The branch demo: this one never reaches the model's tool loop.
        ask(
            graph,
            "This is unacceptable. I want to speak to a human right now.",
            f"{SESSION_ID}-escalation",
        )


if __name__ == "__main__":
    main()

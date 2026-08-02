# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Run the migration one stage at a time, or every stage in order.

    python -m examples.run_walkthrough --stage 0
    python -m examples.run_walkthrough --stage 1 --role-arn ... --lambda-arn ...
    python -m examples.run_walkthrough --stage all --role-arn ... --lambda-arn ... --teardown

Stage 0 is the self-hosted starting point and touches no Amazon Bedrock AgentCore
API at all: the graph runs in this process, its tools call a local HTTP stub, and
its conversation state lives in an in-process checkpointer that dies with the
process.

Stage 1 keeps that graph and replaces what surrounds it. It creates the gateway,
registers the Lambda target on it, creates the memory resource, then builds the
same compiled graph with the Gateway's tools and the AgentCore Memory
checkpointer:

    create gateway -> register target -> create memory
        -> build graph with Gateway tools + AgentCore Memory -> invoke twice

The gateway id and URL flow into target registration and the MCP client, the
memory id flows into the checkpointer, and one thread id is reused across two
separately constructed graphs so that the second invocation resumes state no
object in this process was holding.

Stage 2 is the hardening stage. It is a declared no-op here.

The Strands rebuild path is a different migration path rather than a later stage,
so it keeps its own entry point at examples/rebuild/strands_agent.py.

Pass --teardown to delete the gateway target, the gateway and the memory
afterward. The Lambda and the two IAM roles come from
examples/gateway/lambda_target/deploy.sh and are not deleted here.
"""

import argparse
import os
import time
import uuid

import boto3
from bedrock_agentcore.memory import MemoryClient
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from examples.gateway.create_gateway import create_gateway
from examples.gateway.register_target import register_target
from examples.memory.configure_memory import EVENT_EXPIRY_DAYS, configure_memory
from examples.stage0_langgraph.agent import build_graph
from examples.stage0_langgraph.local_api import running_stub
from examples.stage0_langgraph.tools import SUPPORT_TOOLS
from examples.stage1_replatform.agentcore_memory_saver import AgentCoreMemorySaver
from examples.stage1_replatform.langchain_mcp_tools import merge_tools, to_langchain_tools
from examples.tools.gateway_mcp_tools import build_mcp_client

MODEL_ID = "us.anthropic.claude-sonnet-5"

# Stages that need a gateway, a target and a memory resource, and therefore need
# --role-arn and --lambda-arn.
AGENTCORE_STAGES = ("1", "all")


def _wait_for_target_deletion(
    control, gateway_id: str, target_id: str, timeout: int = 60
) -> None:
    """Poll until the gateway target is fully deleted or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            control.get_gateway_target(
                gatewayIdentifier=gateway_id, targetId=target_id
            )
        except control.exceptions.ResourceNotFoundException:
            return
        time.sleep(2)


def teardown(
    gateway_id: str,
    target_id: str,
    memory_id: str,
    region_name: str,
) -> None:
    """Delete the resources created during the walkthrough."""
    control = boto3.client("bedrock-agentcore-control", region_name=region_name)
    if target_id:
        control.delete_gateway_target(gatewayIdentifier=gateway_id, targetId=target_id)
        # Target deletion is asynchronous; the gateway cannot be deleted while a
        # target is still attached, so wait for the target to disappear first.
        _wait_for_target_deletion(control, gateway_id, target_id)
        print(f"Deleted target {target_id}")
    if gateway_id:
        control.delete_gateway(gatewayIdentifier=gateway_id)
        print(f"Deleted gateway {gateway_id}")
    if memory_id:
        MemoryClient(region_name=region_name).delete_memory(memory_id)
        print(f"Deleted memory {memory_id}")


def _ask(graph, prompt: str, thread_id: str) -> dict:
    """Invoke the graph on one thread and print what the run did."""
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


def _llm(region_name: str) -> ChatBedrockConverse:
    """The one thing a migration does not change: the model call already went to Bedrock."""
    return ChatBedrockConverse(model=MODEL_ID, region_name=region_name)


def run_stage0(region_name: str, session_id: str) -> None:
    """Stage 0: local tools, in-process state, no AgentCore calls."""
    print("\n=== stage 0: self-hosted LangGraph ===")
    with running_stub() as base_url:
        os.environ["ORDERS_API_BASE"] = base_url
        print(f"orders API stub: {base_url}")
        graph = build_graph(
            llm=_llm(region_name), tools=SUPPORT_TOOLS, checkpointer=MemorySaver()
        )
        _ask(graph, "Hi, I'm Dana and my order number is 12345.", session_id)
        # Turn 2 never repeats the order number, so an answer that names the
        # carrier came from the checkpointer.
        _ask(graph, "Has it shipped yet, and who is carrying it?", session_id)
        _ask(
            graph,
            "This is unacceptable. I want to speak to a human right now.",
            f"{session_id}-escalation",
        )


def run_stage1(
    gateway_url: str,
    memory_id: str,
    actor_id: str,
    session_id: str,
    region_name: str,
) -> None:
    """Stage 1: Gateway tools and AgentCore Memory behind the same graph."""
    print("\n=== stage 1: Gateway tools + AgentCore Memory ===")
    # search_faq never moves to Gateway, so the local backend is still needed.
    with running_stub() as base_url:
        os.environ["ORDERS_API_BASE"] = base_url

        # Started rather than entered: tools built inside a `with mcp_client:`
        # block raise MCPClientInitializationError once the block exits.
        mcp_client = build_mcp_client(gateway_url, region_name)
        mcp_client.start()
        try:
            tools = merge_tools(
                SUPPORT_TOOLS, to_langchain_tools(mcp_client, mcp_client.list_tools_sync())
            )
            print(f"tools: {[t.name for t in tools]}")

            def graph():
                """A fresh graph, and so a fresh checkpointer, per invocation."""
                return build_graph(
                    llm=_llm(region_name),
                    tools=tools,
                    checkpointer=AgentCoreMemorySaver(
                        memory_id, actor_id=actor_id, region_name=region_name
                    ),
                )

            _ask(graph(), "Hi, I'm Dana and my order number is 12345.", session_id)
            # A second graph on the same thread id. Runtime would supply that id
            # as RequestContext.session_id on a different container; here two
            # separate objects stand in for two replicas, and the state they
            # share is in AgentCore Memory rather than in either of them.
            _ask(graph(), "Has it shipped yet, and who is carrying it?", session_id)
            _ask(
                graph(),
                "This is unacceptable. I want to speak to a human right now.",
                f"{session_id}-escalation",
            )
        finally:
            mcp_client.stop(None, None, None)


def run_stage2() -> None:
    """Stage 2: hardening. Nothing here yet, and saying so beats implying otherwise."""
    print("\n=== stage 2: hardening ===")
    print("No-op: guardrails and policy are not implemented in this repository.")


def run(args: argparse.Namespace) -> None:
    region = args.region
    stages = ("0", "1", "2") if args.stage == "all" else (args.stage,)
    session_id = f"walkthrough-{uuid.uuid4().hex[:8]}"
    actor_id = f"customer-{uuid.uuid4().hex[:8]}"

    gateway_id = target_id = memory_id = ""
    gateway_url = ""
    if args.stage in AGENTCORE_STAGES:
        gateway_id, gateway_url = create_gateway(
            args.gateway_name, args.role_arn, region
        )
        target_id = register_target(
            gateway_id, args.lambda_arn, args.target_name, region
        )
        memory_id = configure_memory(region, args.event_expiry_days)

    try:
        if "0" in stages:
            run_stage0(region, session_id)
        if "1" in stages:
            run_stage1(gateway_url, memory_id, actor_id, session_id, region)
        if "2" in stages:
            run_stage2()
    finally:
        if args.teardown:
            teardown(gateway_id, target_id, memory_id, region)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("0", "1", "2", "all"),
        default="all",
        help="Which stage to run (default: all).",
    )
    parser.add_argument(
        "--role-arn",
        default=os.environ.get("GATEWAY_ROLE_ARN"),
        help="Execution role ARN for the gateway (or set GATEWAY_ROLE_ARN).",
    )
    parser.add_argument(
        "--lambda-arn",
        default=os.environ.get("TARGET_LAMBDA_ARN"),
        help="ARN of the Lambda backing the tools (or set TARGET_LAMBDA_ARN).",
    )
    parser.add_argument(
        "--gateway-name",
        default=os.environ.get("GATEWAY_NAME", "MigratedAgentGateway"),
        help="Gateway name (default: MigratedAgentGateway).",
    )
    parser.add_argument(
        "--target-name",
        default=os.environ.get("TARGET_NAME", "supportTools"),
        help="Target name (default: supportTools).",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1).",
    )
    parser.add_argument(
        "--event-expiry-days",
        type=int,
        default=int(os.environ.get("EVENT_EXPIRY_DAYS", EVENT_EXPIRY_DAYS)),
        help=(
            "Event retention on the memory resource, 3 to 365 "
            f"(default: {EVENT_EXPIRY_DAYS}). Checkpoints expire with the events."
        ),
    )
    parser.add_argument(
        "--teardown",
        action="store_true",
        help="Delete the gateway target, gateway, and memory after running.",
    )
    args = parser.parse_args()

    # Stage 0 and stage 2 create nothing, so they need neither ARN.
    if args.stage in AGENTCORE_STAGES:
        if not args.role_arn:
            parser.error("--role-arn is required for stage 1 (or set GATEWAY_ROLE_ARN).")
        if not args.lambda_arn:
            parser.error(
                "--lambda-arn is required for stage 1 (or set TARGET_LAMBDA_ARN)."
            )

    run(args)


if __name__ == "__main__":
    main()

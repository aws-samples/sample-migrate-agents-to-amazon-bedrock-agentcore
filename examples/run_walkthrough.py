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

Stage 2 rebuilds the agent on the Strands Agents SDK and hardens it. The graph and
its hand-written router are gone, replaced by a model-driven loop over the same
Gateway tools and the same memory resource, and one security feature is layered
on: Cedar rules in Policy in AgentCore on the tool calls.

    create two IAM roles that differ in nothing IAM can see
        -> register Cedar rules naming both and attach them in ENFORCE mode
        -> build the Strands agent, signing as the read-only role, and invoke it
        -> call both tools as both roles and check all four decisions

Stage 2 reuses the gateway, target and memory stage 1 creates, so it needs the
same two ARNs. It has its own entry point at
examples/stage2_rebuild/strands_agent.py for deploying it to Runtime.

Pass --teardown to delete what ran: the gateway target, the gateway, the memory,
the Cedar policies, their engine and the two demo roles. Every step is attempted
even if an earlier one failed, because a delete that skips the rest of the list is
how resources get orphaned. The Lambda and the two roles it needs come from
examples/gateway/lambda_target/deploy.sh and are not deleted here.

--teardown with no --stage is the recovery path, and it is the one to reach for
when a run has already died:

    python -m examples.run_walkthrough --teardown --dry-run
    python -m examples.run_walkthrough --teardown

It finds the gateway, target, memory, policy engine, policies and demo roles by
the names this walkthrough gives them and deletes those, so it works from a
different process than the one that created them. It needs neither ARN.
"""

import argparse
import os
import time
import uuid
from contextlib import contextmanager
from typing import Sequence

import boto3
from bedrock_agentcore.memory import MemoryClient
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from examples.gateway.create_gateway import create_gateway
from examples.gateway.register_target import register_target
from examples.memory.configure_memory import (
    EVENT_EXPIRY_DAYS,
    MEMORY_NAME,
    configure_memory,
)
from examples.stage0_langgraph.agent import build_graph
from examples.stage0_langgraph.local_api import running_stub
from examples.stage0_langgraph.tools import SUPPORT_TOOLS
from examples.stage1_replatform.agentcore_memory_saver import AgentCoreMemorySaver
from examples.stage1_replatform.langchain_mcp_tools import merge_tools, to_langchain_tools
from examples.stage2_rebuild.policy.attach_policy import (
    POLICY_ENGINE_NAME,
    call_tool_through_gateway,
    delete_policies,
    delete_policy_engine,
    register,
)
from examples.stage2_rebuild.policy.demo_principals import (
    READ_ONLY_ROLE_NAME,
    SUPPORT_ROLE_NAME,
    assert_identical_iam,
    assume,
    cedar_principal,
    create_demo_roles,
    delete_demo_roles,
)
from examples.stage2_rebuild.strands_agent import build_agent
from examples.tools.gateway_mcp_tools import build_mcp_client
from examples.validation.measure_walkthrough import (
    Measurements,
    count_events,
    count_rehydrated_messages,
    probe_payload_floor,
)

MODEL_ID = "us.anthropic.claude-sonnet-5"

# Stages that need a gateway, a target and a memory resource, and therefore need
# --role-arn and --lambda-arn. Stage 2 rebuilds the agent on top of the same three
# resources rather than creating its own.
AGENTCORE_STAGES = ("1", "2", "all")

# The order the demo calls name. Stage 2's Cedar rules scope tools to callers, not
# orders, so one order id is enough: the same caller asking for the same order is
# allowed to look it up and refused a return on it.
DEMO_ORDER_ID = "12345"


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


def _delete_target(control, gateway_id: str, target_id: str) -> None:
    """Delete the gateway target and wait for it to disappear."""
    try:
        control.delete_gateway_target(gatewayIdentifier=gateway_id, targetId=target_id)
    except control.exceptions.ResourceNotFoundException:
        print(f"Target {target_id} already deleted")
        return
    # Target deletion is asynchronous; the gateway cannot be deleted while a
    # target is still attached, so wait for the target to disappear first.
    _wait_for_target_deletion(control, gateway_id, target_id)
    print(f"Deleted target {target_id}")


def _delete_gateway(control, gateway_id: str, timeout: int = 120) -> None:
    """Delete the gateway, retrying while the service still sees a target on it.

    Measured, not assumed: DeleteGateway failed with "Gateway ... has targets
    associated with it" while ListGatewayTargets already returned [] for the same
    gateway. So the absence of a target in a list call is not proof the gateway can
    go, and the delete has to be retried; completion is then confirmed by
    GetGateway raising ResourceNotFoundException rather than by the delete
    returning.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            control.delete_gateway(gatewayIdentifier=gateway_id)
            break
        except control.exceptions.ResourceNotFoundException:
            print(f"Gateway {gateway_id} already deleted")
            return
        except control.exceptions.ValidationException:
            if time.monotonic() >= deadline:
                raise
            time.sleep(5)

    while time.monotonic() < deadline:
        try:
            control.get_gateway(gatewayIdentifier=gateway_id)
        except control.exceptions.ResourceNotFoundException:
            print(f"Deleted gateway {gateway_id}")
            return
        time.sleep(2)
    raise TimeoutError(f"Gateway {gateway_id} still present after {timeout}s")


def _delete_memory(memory_id: str, region_name: str) -> None:
    """Delete the memory resource and wait until GetMemory says it is gone.

    DeleteMemory is asynchronous and the memory is still listed seconds after the
    call returns, so delete_memory_and_wait is used rather than delete_memory: it
    polls GetMemory until ResourceNotFoundException.
    """
    MemoryClient(region_name=region_name).delete_memory_and_wait(memory_id)
    print(f"Deleted memory {memory_id}")


def teardown(
    gateway_id: str,
    target_id: str,
    memory_id: str,
    region_name: str,
    policy_engine_id: str = "",
    policy_ids: Sequence[str] = (),
    demo_roles: bool = False,
) -> None:
    """Delete everything the walkthrough created, in dependency order.

    Target, then gateway, then memory, then the Cedar policies, their engine, and
    last the two demo IAM roles. The first two are ordered by the service — a
    gateway with a target attached will not delete — and the engine goes after the
    gateway that referenced it. IAM has no such ordering; the roles go last only
    so that a failure to delete a role cannot leave a billable resource behind it.

    Every step is attempted even when an earlier one raises, and the failures are
    collected and re-raised together at the end. A teardown that stops at its first
    error is how a run orphans a gateway behind a target that would not delete, and
    it leaves whatever came after that step in the list behind too.
    """
    control = boto3.client("bedrock-agentcore-control", region_name=region_name)
    steps = []
    if target_id:
        steps.append(("target", lambda: _delete_target(control, gateway_id, target_id)))
    if gateway_id:
        steps.append(("gateway", lambda: _delete_gateway(control, gateway_id)))
    if memory_id:
        steps.append(("memory", lambda: _delete_memory(memory_id, region_name)))
    if policy_ids:
        steps.append(
            (
                "policies",
                lambda: delete_policies(policy_engine_id, list(policy_ids), region_name),
            )
        )
    if policy_engine_id:
        steps.append(
            (
                "policy engine",
                lambda: delete_policy_engine(policy_engine_id, region_name),
            )
        )
    if demo_roles:
        steps.append(("demo IAM roles", lambda: delete_demo_roles(region_name)))

    failures = []
    for label, delete in steps:
        try:
            delete()
        except Exception as error:  # noqa: BLE001 - the next resource still matters
            print(f"Failed to delete {label}: {type(error).__name__}: {error}")
            failures.append(f"{label} ({type(error).__name__})")
    if failures:
        raise RuntimeError(
            "Teardown did not finish; check these by hand: " + ", ".join(failures)
        )


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


def run_stage0(
    region_name: str, session_id: str, measured: Measurements = None
) -> None:
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
        # carrier came from the checkpointer. Timed because it is the baseline the
        # same turn against Gateway and AgentCore Memory is compared with: local
        # HTTP stub, in-process state, nothing on the network but the model call.
        with _optional_timing(measured, "stage 0 turn, local tools + in-process state"):
            _ask(graph, "Has it shipped yet, and who is carrying it?", session_id)
        _ask(
            graph,
            "This is unacceptable. I want to speak to a human right now.",
            f"{session_id}-escalation",
        )


@contextmanager
def _optional_timing(measured, name: str):
    """Time a block when measuring, and do nothing when not."""
    if measured is None:
        yield
        return
    with measured.timing(name):
        yield


def run_stage1(
    gateway_url: str,
    memory_id: str,
    actor_id: str,
    session_id: str,
    region_name: str,
    measured: Measurements = None,
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
            if measured is not None:
                measured.record(
                    "memory events after turn 1",
                    count_events(memory_id, actor_id, session_id, region_name),
                )
                # Counted on a saver built now, so anything it finds came back
                # from the service. Taken before turn 2 rather than after,
                # because after the invoke it would include what turn 2 wrote.
                measured.record(
                    "messages rehydrated before turn 2",
                    count_rehydrated_messages(
                        AgentCoreMemorySaver(
                            memory_id, actor_id=actor_id, region_name=region_name
                        ),
                        session_id,
                    ),
                )
            # A second graph on the same thread id. Runtime would supply that id
            # as RequestContext.session_id on a different container; here two
            # separate objects stand in for two replicas, and the state they
            # share is in AgentCore Memory rather than in either of them.
            with _optional_timing(
                measured, "stage 1 turn, Gateway tools + AgentCore Memory"
            ):
                _ask(graph(), "Has it shipped yet, and who is carrying it?", session_id)
            if measured is not None:
                measured.record(
                    "memory events after turn 2",
                    count_events(memory_id, actor_id, session_id, region_name),
                )
            _ask(
                graph(),
                "This is unacceptable. I want to speak to a human right now.",
                f"{session_id}-escalation",
            )
        finally:
            mcp_client.stop(None, None, None)


def _agent_text(message: dict) -> str:
    """Join the text blocks of a Strands message into the reply itself.

    AgentResult.message is the raw Bedrock Converse message: a dict with a role
    and a list of content blocks, only some of which carry text. str() on it
    prints the whole dict, metadata and token counts included, which is not the
    agent's answer and is not what the graph stages print for the same turn.
    """
    return "\n".join(
        block["text"] for block in message.get("content", []) if "text" in block
    )


def _ask_strands(agent, prompt: str) -> str:
    """Invoke the Strands agent once and print what came back.

    There is no state dict to inspect and no intent field to print, because there
    is no router: what the agent did is whatever the model decided to do. That
    absence is the stage-2 diff.
    """
    print(f"\ncustomer: {prompt}")
    result = agent(prompt)
    text = _agent_text(result.message)
    print(f"  agent: {text}")
    return text


def _report_decision(caller: str, tool_name: str, allowed: bool, text: str) -> None:
    """Print one gateway policy decision.

    A permitted call returns the tool's own payload, which is a JSON blob the
    reader does not need in full, so it is cut short. A refusal is the opposite:
    the text *is* the finding, and the part that says why sits at the end of it.
    Truncating a denial at a fixed width drops the reason and leaves a reader
    looking at a refusal with no explanation attached.
    """
    print(f"\n{caller} -> {tool_name}({DEMO_ORDER_ID}) allowed={allowed}")
    print(f"  {text[:120] if allowed else text}")


# What the Cedar rules in support_tools.cedar say, expressed as the four calls
# that test them: (caller, tool) -> expected decision. Written out rather than
# derived from the rules, because a table derived from the thing it is checking
# agrees with it by construction.
EXPECTED_DECISIONS = {
    (READ_ONLY_ROLE_NAME, "supportTools___lookup_order"): True,
    (READ_ONLY_ROLE_NAME, "supportTools___process_return"): False,
    (SUPPORT_ROLE_NAME, "supportTools___lookup_order"): True,
    (SUPPORT_ROLE_NAME, "supportTools___process_return"): True,
}

_PROOF_CALLS = (
    ("supportTools___lookup_order", {"order_id": DEMO_ORDER_ID}),
    (
        "supportTools___process_return",
        {"order_id": DEMO_ORDER_ID, "reason": "damaged in transit"},
    ),
)


def prove_policy_enforcement(
    gateway_url: str, role_arns: dict, region_name: str
) -> dict:
    """Call both tools as both principals and check all four decisions.

    Two callers times two tools is the smallest experiment that attributes a
    refusal to Cedar. One caller refused one tool proves nothing on its own: the
    ordinary explanation for a failed call is a missing IAM permission, and a
    single denial cannot tell that apart from an authorization decision. Holding
    IAM identical across the two rows and the tool identical down the two columns
    leaves the Cedar rules as the only thing that differs between an allow and a
    deny.

    Raises if any of the four decisions is not the one the rules describe. A
    walkthrough that prints "denied" where it should print "allowed" and still
    exits 0 teaches the reader that the feature works when what they just watched
    was it failing.
    """
    print("\n=== policy enforcement, two principals x two tools ===")
    decisions = {}
    for role_name, role_arn in role_arns.items():
        credentials = assume(role_arn, f"proof-{uuid.uuid4().hex[:8]}", region_name)
        for tool_name, arguments in _PROOF_CALLS:
            allowed, text = call_tool_through_gateway(
                gateway_url, tool_name, arguments, region_name, credentials
            )
            decisions[(role_name, tool_name)] = allowed
            _report_decision(role_name, tool_name, allowed, text)

    print("\n  caller                          lookup_order  process_return")
    for role_name in (READ_ONLY_ROLE_NAME, SUPPORT_ROLE_NAME):
        row = [
            decisions.get((role_name, tool_name)) for tool_name, _ in _PROOF_CALLS
        ]
        print(
            f"  {role_name:<30}  {str(row[0]):<12}  {row[1]}"
        )

    wrong = {
        key: (expected, decisions.get(key))
        for key, expected in EXPECTED_DECISIONS.items()
        if decisions.get(key) is not expected
    }
    if wrong:
        raise RuntimeError(
            "Policy enforcement did not match support_tools.cedar. "
            "(caller, tool): expected -> observed: "
            + "; ".join(
                f"{key}: {expected} -> {observed}"
                for key, (expected, observed) in wrong.items()
            )
        )
    print(
        "\nAll four decisions match support_tools.cedar, and IAM is identical on "
        "both callers, so the one refusal is Cedar's."
    )
    return decisions


def run_stage2(
    gateway_id: str,
    gateway_url: str,
    gateway_arn: str,
    memory_id: str,
    actor_id: str,
    session_id: str,
    region_name: str,
    created: dict,
) -> None:
    """Stage 2: the Strands rebuild, with Cedar on its tools.

    ``created`` is filled in as each resource comes up rather than returned at the
    end, so that a failure part way through still leaves teardown able to delete
    what exists. A policy engine created and then lost to an exception is a
    resource nobody knows to look for.
    """
    print("\n=== stage 2: Strands rebuild + Policy ===")

    # Both Cedar principals are real roles this walkthrough creates, and neither
    # is the identity running it. That is what makes the denial legible: the two
    # roles are identical to IAM, so the only difference between the caller that
    # may start a return and the caller that may not is a Cedar rule.
    role_arns = create_demo_roles(gateway_arn, region_name)
    created["demo_roles"] = True
    assert_identical_iam(region_name)

    read_only = cedar_principal(role_arns[READ_ONLY_ROLE_NAME])
    privileged = cedar_principal(role_arns[SUPPORT_ROLE_NAME])
    print(f"policy read-only principal: {read_only}")
    print(f"policy support-agent principal: {privileged}")
    engine_id, policies = register(
        gateway_id, gateway_arn, read_only, privileged, "ENFORCE", region_name
    )
    created["policy_engine_id"] = engine_id
    created["policy_ids"] = list(policies.values())

    # The agent signs its gateway calls as the read-only role, so it holds exactly
    # the tools the rules give that principal: it can look an order up and cannot
    # start a return. Its model calls still go out as the ambient identity, which
    # is the split Runtime would give it too — an execution role for the gateway,
    # separate from whatever the container is allowed to invoke.
    read_only_credentials = assume(
        role_arns[READ_ONLY_ROLE_NAME], f"agent-{uuid.uuid4().hex[:8]}", region_name
    )
    mcp_client = build_mcp_client(gateway_url, region_name, read_only_credentials)
    mcp_client.start()
    try:
        agent = build_agent(
            memory_id=memory_id,
            session_id=session_id,
            actor_id=actor_id,
            extra_tools=mcp_client.list_tools_sync(),
            region_name=region_name,
        )
        # get_all_tools_config() is keyed by tool name, so the keys are the names.
        print(f"tools: {sorted(agent.tool_registry.get_all_tools_config())}")

        _ask_strands(agent, f"Hi, I'm Dana. Where is my order {DEMO_ORDER_ID}?")
    finally:
        mcp_client.stop(None, None, None)

    # Cedar decides at the gateway and there is no synchronous API to ask, so the
    # way to read a decision is to make the call.
    prove_policy_enforcement(gateway_url, role_arns, region_name)


def discover_resources(
    gateway_name: str,
    memory_name: str = MEMORY_NAME,
    engine_name: str = POLICY_ENGINE_NAME,
    region_name: str = "us-east-1",
) -> dict:
    """Find what a previous run of this walkthrough left in the account, by name.

    A ledger built in this process only covers resources this process created, so
    it cannot help the reader whose run died in another terminal an hour ago. The
    names are all deterministic — the gateway and the engine are named exactly,
    and the service suffixes the memory — so the account can be asked instead of
    remembered.

    Returns the same keys teardown takes, so a discovered ledger and a live one
    are deleted by the same code path.
    """
    control = boto3.client("bedrock-agentcore-control", region_name=region_name)
    found = {"gateway_id": "", "target_id": "", "memory_id": ""}

    for page in control.get_paginator("list_gateways").paginate():
        for summary in page.get("items", []):
            if summary.get("name") == gateway_name:
                found["gateway_id"] = summary["gatewayId"]
    if found["gateway_id"]:
        for page in control.get_paginator("list_gateway_targets").paginate(
            gatewayIdentifier=found["gateway_id"]
        ):
            for summary in page.get("items", []):
                found["target_id"] = summary["targetId"]

    for page in control.get_paginator("list_memories").paginate():
        for summary in page.get("memories", []):
            # The suffix is the service's; the prefix and the separator are ours.
            if summary["id"].startswith(memory_name + "-"):
                found["memory_id"] = summary["id"]

    for page in control.get_paginator("list_policy_engines").paginate():
        for summary in page.get("policyEngines", []):
            if summary.get("name") == engine_name:
                engine_id = summary["policyEngineId"]
                found["policy_engine_id"] = engine_id
                policy_ids = []
                for policies in control.get_paginator("list_policies").paginate(
                    policyEngineId=engine_id
                ):
                    policy_ids += [
                        policy["policyId"] for policy in policies.get("policies", [])
                    ]
                found["policy_ids"] = policy_ids

    # The two Cedar principals are IAM roles, so they are not free of consequence
    # if left behind: a role that can call a deleted gateway is a permission
    # nobody is auditing. Recovery has to find them too, or the flag that claims
    # to return the account to baseline leaves two roles in it.
    iam = boto3.client("iam", region_name=region_name)
    for role_name in (READ_ONLY_ROLE_NAME, SUPPORT_ROLE_NAME):
        try:
            iam.get_role(RoleName=role_name)
            found["demo_roles"] = True
        except iam.exceptions.NoSuchEntityException:
            pass
    return found


def _print_ledger(found: dict) -> bool:
    """Print what teardown would delete. Returns whether anything was found."""
    labels = (
        ("gateway_id", "gateway"),
        ("target_id", "gateway target"),
        ("memory_id", "memory"),
        ("policy_engine_id", "policy engine"),
    )
    anything = False
    for key, label in labels:
        if found.get(key):
            print(f"  {label}: {found[key]}")
            anything = True
    for policy_id in found.get("policy_ids", ()):
        print(f"  policy: {policy_id}")
        anything = True
    if found.get("demo_roles"):
        print(f"  IAM roles: {READ_ONLY_ROLE_NAME}, {SUPPORT_ROLE_NAME}")
        anything = True
    if not anything:
        print("  nothing found")
    return anything


def run(args: argparse.Namespace) -> None:
    region = args.region
    stages = ("0", "1", "2") if args.stage == "all" else (args.stage,)
    session_id = f"walkthrough-{uuid.uuid4().hex[:8]}"
    actor_id = f"customer-{uuid.uuid4().hex[:8]}"

    gateway_url = gateway_arn = ""
    # Every resource lands here the moment it exists, and teardown deletes
    # whatever is in it. Creation is inside the try for the same reason: a
    # gateway created and then lost to a failure in the next call is a resource
    # nobody knows to look for, and --teardown that skips it because the run
    # never reached the try block is the flag not doing what it says.
    created = {}
    measured = None if args.no_measure else Measurements()
    try:
        if args.stage in AGENTCORE_STAGES:
            # Timed from the CreateGateway call to the first READY from
            # GetGateway, which is what create_gateway waits for. On a re-run it
            # reuses the existing gateway and this number is a GetGateway, so it
            # is labelled with which of the two happened.
            with _optional_timing(measured, "gateway CreateGateway -> READY"):
                gateway_id, gateway_url = create_gateway(
                    args.gateway_name, args.role_arn, region
                )
            created["gateway_id"] = gateway_id
            with _optional_timing(measured, "target registration"):
                created["target_id"] = register_target(
                    gateway_id, args.lambda_arn, args.target_name, region
                )
            with _optional_timing(measured, "memory CreateMemory -> ACTIVE"):
                created["memory_id"] = configure_memory(
                    region, args.event_expiry_days
                )
            if "2" in stages:
                # Cedar names the gateway by ARN, which create_gateway does not
                # return.
                gateway_arn = boto3.client(
                    "bedrock-agentcore-control", region_name=region
                ).get_gateway(gatewayIdentifier=gateway_id)["gatewayArn"]

        if "0" in stages:
            run_stage0(region, session_id, measured)
        if "1" in stages:
            run_stage1(
                gateway_url,
                created["memory_id"],
                actor_id,
                session_id,
                region,
                measured,
            )
        if "2" in stages:
            run_stage2(
                created["gateway_id"],
                gateway_url,
                gateway_arn,
                created["memory_id"],
                actor_id,
                f"{session_id}-strands",
                region,
                created,
            )
        if measured is not None and created.get("memory_id"):
            print("\n=== payload floor probe ===")
            measured.record(
                "largest blob event round-tripped",
                probe_payload_floor(
                    created["memory_id"],
                    actor_id,
                    # Its own session, so the conversation's event counts stay
                    # the conversation's.
                    f"{session_id}-payload-probe",
                    region,
                ),
                " B",
            )
    finally:
        # Reported from the finally rather than after the stages, so that a run
        # that fails part way through still prints what it measured before it
        # broke. The failure is the interesting part of such a run and the
        # numbers taken up to it are what say where it got to.
        if measured is not None and measured.taken:
            measured.report()
        if args.teardown:
            teardown(
                created.get("gateway_id", ""),
                created.get("target_id", ""),
                created.get("memory_id", ""),
                region,
                policy_engine_id=created.get("policy_engine_id", ""),
                policy_ids=created.get("policy_ids", ()),
                demo_roles=created.get("demo_roles", False),
            )


def run_teardown_only(args: argparse.Namespace) -> None:
    """Delete what an earlier run left behind, without running anything first."""
    found = discover_resources(args.gateway_name, region_name=args.region)
    print("Found in this account:")
    anything = _print_ledger(found)
    if not anything or args.dry_run:
        if args.dry_run:
            print("--dry-run: nothing deleted")
        return
    print()
    teardown(
        found.get("gateway_id", ""),
        found.get("target_id", ""),
        found.get("memory_id", ""),
        args.region,
        policy_engine_id=found.get("policy_engine_id", ""),
        policy_ids=found.get("policy_ids", ()),
        demo_roles=found.get("demo_roles", False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("0", "1", "2", "all"),
        # No default, so that --teardown on its own is a teardown rather than a
        # whole second walkthrough followed by one. run() still treats a missing
        # --stage as "all".
        default=None,
        help="Which stage to run (default: all, unless --teardown is the only flag).",
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
        help=(
            "Delete the gateway target, gateway, memory and Cedar policies after "
            "running. On its own, with no --stage, it deletes what an earlier run "
            "left in the account instead of running anything."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --teardown and no --stage, list what would be deleted and stop.",
    )
    parser.add_argument(
        "--no-measure",
        action="store_true",
        help=(
            "Skip the measurements table and the payload probe. The probe writes "
            "up to one 4 MiB event."
        ),
    )
    args = parser.parse_args()

    # --teardown with no --stage is the recovery path: an earlier run died, its
    # identifiers went with the process, and the resources are still billing.
    if args.teardown and args.stage is None:
        run_teardown_only(args)
        return
    if args.stage is None:
        args.stage = "all"

    # Stage 0 creates nothing, so it needs neither ARN. Stage 2 needs both because
    # it reuses the gateway, target and memory that stage 1 stands up.
    if args.stage in AGENTCORE_STAGES:
        if not args.role_arn:
            parser.error(
                f"--role-arn is required for stage {args.stage} "
                "(or set GATEWAY_ROLE_ARN)."
            )
        if not args.lambda_arn:
            parser.error(
                f"--lambda-arn is required for stage {args.stage} "
                "(or set TARGET_LAMBDA_ARN)."
            )

    run(args)


if __name__ == "__main__":
    main()

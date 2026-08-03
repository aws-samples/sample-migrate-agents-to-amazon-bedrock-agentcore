# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""What each principal can SEE on the gateway, not just what it can call.

    python -m examples.validation.verify_policy_visibility --gateway-url URL

Run it after run_walkthrough.py --stage all has attached the policy engine and
while the two demo roles still exist, so there are three distinct principals to
ask as.

WHY THIS IS SEPARATE FROM THE PERMIT/DENY MATRIX

run_walkthrough's prove_policy_enforcement makes four tools/call requests and
checks the two rows differ. That establishes enforcement. It does not establish
what this does, which was measured and is easy to state wrongly: the policy engine
filters ``tools/list`` per principal as well. The manifest a gateway publishes is
not a property of the gateway. It is a property of the gateway and the caller
together.

Measured against one live gateway, one moment, three signing identities:

    ambient caller, holds IAM, holds no Cedar permit   ->  []
    MigratedAgentReadOnlyCaller, permitted lookup_order ->  1 tool
    MigratedAgentSupportAgent, permitted both           ->  2 tools

The first row is the one to notice. That caller was an account administrator. It
could call UpdateGateway. Its tool list was empty, because Cedar's default-deny
applies to discovery too and nothing permitted it.

THE PART THAT IS A HAZARD RATHER THAN A FEATURE

examples/stage2_rebuild/strands_agent.py:76-82 merges local @tool stubs with
gateway-discovered tools by superseding on name: a local stub is dropped when the
gateway publishes a tool whose suffix matches. That is correct against a fixed
manifest. Against a per-principal manifest it means the tighter the Cedar rule,
the MORE local stubs survive — a principal denied process_return is never shown
process_return, so nothing supersedes the local process_return, so the agent
registers it and can call it in-process, where the gateway and the policy engine
see nothing.

So the printed registry below is not decoration. For the read-only principal it
is expected to contain a local ``process_return`` alongside the gateway's
``supportTools___lookup_order``, and that combination is the whole finding. The
authorization boundary is the gateway. Tools that never traverse it are outside
it, and Cedar's own filtering is what leaves them there.

This module reports. It does not fail the run on that combination, because
whether stage 2 should keep unsuperseded stubs is a decision documented at
strands_agent.py:70-74 and belongs to whoever owns that decision, not to a
validation script.
"""

import argparse
import os

import boto3

from examples.stage0_langgraph.tools import SUPPORT_TOOLS
from examples.stage2_rebuild.policy import demo_principals
from examples.stage2_rebuild.strands_agent import build_agent
from examples.tools.gateway_mcp_tools import build_mcp_client


def visible_tools(gateway_url: str, region_name: str, credentials=None) -> list:
    """The tool names this principal can discover, sorted.

    A principal with no permit gets an empty list rather than an error, so an
    empty result here is a policy decision and not a broken connection. The
    handshake itself succeeded — that is what makes it a decision.
    """
    client = build_mcp_client(gateway_url, region_name, credentials)
    client.start()
    try:
        return sorted(client.list_tools_sync(), key=lambda t: t.tool_name)
    finally:
        client.stop(None, None, None)


def registry_under(gateway_url: str, region_name: str, credentials=None) -> list:
    """The tool names stage 2's agent would register, as this principal.

    Builds the agent the way run_stage2 does, without a memory id: this asks what
    the merge produces, and a session manager would add AgentCore Memory calls
    these roles are not permitted to make.
    """
    tools = visible_tools(gateway_url, region_name, credentials)
    agent = build_agent(extra_tools=tools, region_name=region_name)
    return sorted(agent.tool_registry.get_all_tools_config())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gateway-url",
        default=os.environ.get("GATEWAY_URL"),
        help="Gateway MCP URL (or set GATEWAY_URL).",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1).",
    )
    args = parser.parse_args()

    if not args.gateway_url:
        parser.error("--gateway-url is required (or set GATEWAY_URL).")

    iam = boto3.client("iam", region_name=args.region)
    principals = [("ambient caller", None)]
    for role_name in (
        demo_principals.READ_ONLY_ROLE_NAME,
        demo_principals.SUPPORT_ROLE_NAME,
    ):
        try:
            role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
        except iam.exceptions.NoSuchEntityException:
            print(f"  {role_name} does not exist; run --stage all first. Skipping.")
            continue
        principals.append(
            (
                role_name,
                demo_principals.assume(role_arn, "visibility-probe", args.region),
            )
        )

    local = sorted(tool.name for tool in SUPPORT_TOOLS)
    print("Tool visibility per signing principal")
    print("=" * 72)
    print(f"\nstage 0/1 local tools, for comparison: {local}\n")

    for caller, credentials in principals:
        published = [t.tool_name for t in visible_tools(args.gateway_url, args.region, credentials)]
        registered = registry_under(args.gateway_url, args.region, credentials)
        surviving = [name for name in registered if "___" not in name]
        print(f"  {caller}")
        print(f"    gateway publishes to it : {published}")
        print(f"    stage 2 would register  : {registered}")
        print(f"    of which local, in-process, outside the gateway: {surviving}")
        print()

    print("READING THIS")
    print("  An empty publishes-to-it line is Cedar's default-deny applied to")
    print("  discovery, not a connection failure: the MCP handshake succeeded.")
    print("  A local name in the last line that also exists on the gateway is a")
    print("  stub the merge kept because this principal was not shown the gateway")
    print("  version. It is reachable in-process and the policy engine never sees")
    print("  it. See strands_agent.py:76-82.")


if __name__ == "__main__":
    main()

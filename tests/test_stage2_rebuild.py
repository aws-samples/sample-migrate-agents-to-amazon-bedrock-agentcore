# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Stage-2 verification: the Strands rebuild and its Cedar rules.

Runs offline and makes no AWS calls. The AgentCore control plane is faked
(tests/fake_control_plane.py), the MCP session is faked (tests/fake_mcp.py), and
the clocks inside the waiters are faked so a poll loop finishes instantly and a
timeout is reachable. What is real: the shipped support_tools.cedar text, the real
Strands Agent (it constructs a boto3 client and calls nothing), the real
MCPAgentTool wrapper, and the real botocore service models, used to validate every
request shape this code would send.

One limit worth stating plainly, because it bounds what passing here proves:
there is no synchronous Cedar authorization API. The deny is asserted by
evaluating the shipped rules with the Cedar subset evaluator in
tests/fake_cedar.py, so what is verified is that the rules mean what stage 2
claims — not that the Gateway enforces them. That needs the live run.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import contextlib
import importlib
import io
import os
import unittest
import unittest.mock

import botocore.session
import httpx
import mcp.types as mcp_types
from botocore.credentials import Credentials
from botocore.validate import ParamValidator
from strands.tools.mcp.mcp_agent_tool import MCPAgentTool

from examples import run_walkthrough
from examples.stage0_langgraph.tools import SUPPORT_TOOLS
from examples.stage2_rebuild.policy import attach_policy
from examples.stage2_rebuild.policy.attach_policy import (
    POLICY_ENGINE_NAME,
    caller_principal,
    delete_policies,
    delete_policy_engine,
    read_policy_blocks,
    register,
    render_policies,
    support_agent_principal,
)
from examples.stage2_rebuild.policy import demo_principals
from examples.stage2_rebuild import strands_agent
from examples.stage2_rebuild.strands_agent import build_agent
from examples.tools import gateway_mcp_tools
from examples.validation.measure_walkthrough import Measurements
from tests import fake_cedar
from tests.fake_cedar import Entity
from tests.fake_control_plane import (
    ClientError as FakeClientError,
    FakeAgentCoreControlClient,
    FakeBoto3,
    FakeClock,
    FakeIAMClient,
    FakeSTSClient,
    ValidationException,
)
from tests.fake_mcp import FakeMCPClient, gateway_tool_list

GATEWAY_ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/gw-abc123"
OTHER_GATEWAY_ARN = (
    "arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/gw-someone-else"
)
READ_ONLY = "arn:aws:sts::123456789012:assumed-role/SupportAgentRole"
SUPPORT_AGENT = "arn:aws:sts::123456789012:assumed-role/SupportEscalationRole"
STRANGER = "arn:aws:sts::123456789012:assumed-role/SomeOtherRole"
LOOKUP = "supportTools___lookup_order"
PROCESS_RETURN = "supportTools___process_return"


def valid_request(service, operation, params):
    """Validate params against the real botocore model. No AWS call is made."""
    model = botocore.session.get_session().get_service_model(service)
    shape = model.operation_model(operation).input_shape
    report = ParamValidator().validate(params, shape)
    if report.has_errors():
        raise AssertionError(report.generate_report())
    return True


class FakeContext:
    """The one field of Runtime's RequestContext the entrypoint reads."""

    def __init__(self, session_id):
        self.session_id = session_id


def real_gateway_tools(client=None):
    """The tools list_tools_sync() returns live: real MCPAgentTools, faked session.

    Strands' tool registry drops an object it does not recognise with a log line
    and no exception, so a hand-rolled stand-in would register nothing and every
    tool assertion below would pass against an empty registry. The wrapper has to
    be the real one.
    """
    client = client or FakeMCPClient()
    return [
        MCPAgentTool(
            mcp_types.Tool(
                name=tool.mcp_tool.name,
                description=tool.mcp_tool.description,
                inputSchema=tool.mcp_tool.inputSchema,
            ),
            client,
        )
        for tool in gateway_tool_list()
    ]


class BuildAgentTest(unittest.TestCase):
    """build_agent's contract, on the two points a caller can get wrong."""

    def test_the_gateway_tools_supersede_the_local_stubs(self):
        agent = build_agent(extra_tools=real_gateway_tools())

        registered = sorted(agent.tool_registry.get_all_tools_config())
        self.assertEqual(registered, ["search_faq", LOOKUP, PROCESS_RETURN])
        # Stage 1's supersede rule still holds: no bare lookup_order stub left.
        self.assertNotIn("lookup_order", registered)
        self.assertNotIn("process_return", registered)

    def test_memory_still_requires_both_ids(self):
        # Stage 1's assertion, restated on the moved module.
        with self.assertRaises(ValueError):
            build_agent(memory_id="mem-1")


class PartialVisibilityTest(unittest.TestCase):
    """What the merge does when the gateway publishes only SOME of its tools.

    Not hypothetical. Measured live: with a policy engine attached in ENFORCE
    mode, the gateway filters tools/list per principal, so the manifest a caller
    receives is a function of who is asking. build_agent supersedes on the names
    it was handed, which means a principal denied a tool is never shown it, so
    nothing supersedes the local stub of it.

    These tests pin that behaviour rather than assert it is correct. If the
    supersede rule is ever changed to use the target's registered manifest
    instead of the principal's visible one — the fix named in
    examples/validation/verify_policy_visibility.py — they fail by name, which is
    the point: the decision at strands_agent.py:70-74 should not be reversed
    silently in either direction.
    """

    def test_a_denied_tool_leaves_its_local_stub_registered(self):
        # The read-only principal's manifest: lookup_order only.
        visible = [t for t in real_gateway_tools() if t.tool_name == LOOKUP]

        registered = sorted(
            build_agent(extra_tools=visible).tool_registry.get_all_tools_config()
        )

        self.assertEqual(registered, ["process_return", "search_faq", LOOKUP])
        # The finding, as an assertion: the tool Cedar refused is still callable
        # in-process, under its bare name, on a path the gateway never sees.
        self.assertIn("process_return", registered)
        self.assertNotIn(PROCESS_RETURN, registered)

    def test_an_empty_manifest_leaves_every_local_stub_registered(self):
        # What an administrator holding no Cedar permit was measured to see.
        registered = sorted(
            build_agent(extra_tools=[]).tool_registry.get_all_tools_config()
        )
        self.assertEqual(registered, sorted(t.name for t in SUPPORT_TOOLS))


class PolicyVisibilityReportTest(unittest.TestCase):
    """The promoted visibility probe, with the MCP session faked."""

    def setUp(self):
        self.module = importlib.import_module(
            "examples.validation.verify_policy_visibility"
        )
        self.stopped = 0

    def _patch_client(self, tools):
        """Hand the probe a client that publishes exactly `tools`."""
        outer = self

        class Recording(FakeMCPClient):
            def stop(self, *args):
                outer.stopped += 1

        client = Recording(tools=tools)
        unittest.mock.patch.object(
            self.module, "build_mcp_client", lambda *a, **k: client
        ).start()
        self.addCleanup(unittest.mock.patch.stopall)
        return client

    def test_visible_tools_reports_what_the_principal_was_published(self):
        self._patch_client([t for t in real_gateway_tools() if t.tool_name == LOOKUP])

        names = [t.tool_name for t in self.module.visible_tools("https://gw", "us-east-1")]

        self.assertEqual(names, [LOOKUP])

    def test_an_empty_manifest_is_reported_rather_than_raised(self):
        # A principal with no permit gets [], not an error. Raising here would
        # read as a broken connection and hide the policy decision.
        self._patch_client([])
        self.assertEqual(self.module.visible_tools("https://gw", "us-east-1"), [])

    def test_the_session_is_stopped_even_though_nothing_failed(self):
        self._patch_client([])
        self.module.visible_tools("https://gw", "us-east-1")
        self.assertEqual(self.stopped, 1)

    def test_registry_under_a_partial_manifest_shows_the_surviving_stub(self):
        self._patch_client([t for t in real_gateway_tools() if t.tool_name == LOOKUP])

        registered = self.module.registry_under("https://gw", "us-east-1")

        self.assertEqual(registered, ["process_return", "search_faq", LOOKUP])


class StageParityTest(unittest.TestCase):
    """Stage 2 is a rebuild of the same agent, so it cannot offer less."""

    def test_stage2_registers_no_fewer_tools_than_stage1(self):
        # The defect this catches: stage 2 was rebuilt with lookup_order and
        # process_return only, so search_faq — a local tool that never moved to the
        # gateway — was silently dropped and the "rebuilt agent" answered fewer
        # questions than the one it replaced. Counted against SUPPORT_TOOLS rather
        # than hard-coded to three, so adding a fourth tool to stages 0/1 without
        # rebuilding stage 2 fails here too.
        stage1_names = {t.name for t in SUPPORT_TOOLS}
        stage2_names = set(build_agent().tool_registry.get_all_tools_config())
        self.assertGreaterEqual(len(stage2_names), len(stage1_names))
        self.assertEqual(stage2_names, stage1_names)

    def test_the_gateway_tools_do_not_reduce_the_count(self):
        # Superseding replaces a stub, it does not remove a capability: the prefixed
        # gateway names count, so the total still matches stage 1's.
        agent = build_agent(extra_tools=real_gateway_tools())
        self.assertEqual(
            len(agent.tool_registry.get_all_tools_config()), len(SUPPORT_TOOLS)
        )


class RuntimeEntrypointTest(unittest.TestCase):
    """Importing the Runtime module, and where the session id comes from."""

    def setUp(self):
        strands_agent._agents.clear()
        self.addCleanup(strands_agent._agents.clear)

    def test_the_module_imports_in_the_environment_runtime_provides(self):
        # The defect this catches: the agent was built at import time from
        # AGENTCORE_MEMORY_ID / AGENTCORE_ACTOR_ID / AGENTCORE_SESSION_ID, and
        # agent_runtime.py:85-87 sets the first two and never the third. Importing
        # the module in the environment it is deployed into raised ValueError, so
        # the container could not start. Nothing is built until an invocation.
        with unittest.mock.patch.dict(
            os.environ,
            {"AGENTCORE_MEMORY_ID": "mem-1", "AGENTCORE_ACTOR_ID": "customer-1"},
        ):
            reloaded = importlib.reload(strands_agent)
        self.addCleanup(importlib.reload, strands_agent)
        self.assertFalse(reloaded._agents)
        self.assertTrue(callable(reloaded.support_agent))

    def test_the_entrypoint_uses_the_callers_session_id(self):
        # The other half of the same defect: the entrypoint ignored
        # context.session_id and used whatever AGENTCORE_SESSION_ID held, so every
        # caller shared one memory session.
        captured = []
        self._patch_build(captured)
        strands_agent.agent_invocation({"prompt": "hi"}, FakeContext("session-abc"))
        self.assertEqual([kwargs["session_id"] for kwargs in captured], ["session-abc"])

    def test_two_callers_do_not_share_one_agent(self):
        # A Strands session manager is pinned to one session_id at construction, so
        # a single process-wide agent would file both callers' turns under the first
        # session id to arrive.
        captured = []
        self._patch_build(captured)
        strands_agent.agent_invocation({"prompt": "hi"}, FakeContext("session-a"))
        strands_agent.agent_invocation({"prompt": "hi"}, FakeContext("session-b"))
        strands_agent.agent_invocation({"prompt": "hi"}, FakeContext("session-a"))
        # Three invocations, two sessions, two agents: cached per session, not
        # rebuilt per turn.
        self.assertEqual(
            [kwargs["session_id"] for kwargs in captured], ["session-a", "session-b"]
        )

    def test_a_missing_session_id_falls_back_rather_than_failing(self):
        # RequestContext.session_id is Optional and is None on a local invoke.
        captured = []
        self._patch_build(captured)
        strands_agent.agent_invocation({"prompt": "hi"}, FakeContext(None))
        self.assertEqual(captured[0]["session_id"], "local-session")

    def _patch_build(self, captured):
        """Record what support_agent asks build_agent for, and return a stub agent."""
        def fake_build_agent(**kwargs):
            captured.append(kwargs)
            return lambda prompt: type("Result", (), {"message": f"echo: {prompt}"})()

        patcher = unittest.mock.patch.object(
            strands_agent, "build_agent", fake_build_agent
        )
        patcher.start()
        self.addCleanup(patcher.stop)


class CedarRuleTest(unittest.TestCase):
    """What the shipped rules decide, evaluated as Cedar decides them."""

    def setUp(self):
        quiet(self)
        self.statements = fake_cedar.parse(
            "\n".join(
                statement
                for _, statement in render_policies(
                    GATEWAY_ARN, READ_ONLY, SUPPORT_AGENT
                )
            )
        )

    def allowed(self, principal, action, gateway_arn=GATEWAY_ARN):
        return fake_cedar.is_authorized(
            self.statements,
            Entity("AgentCore::IamEntity", principal),
            action,
            Entity("AgentCore::Gateway", gateway_arn),
        )

    def test_process_return_is_denied_for_the_read_only_caller(self):
        # The stage-2 claim, and the one no prompt can make reliable.
        self.assertFalse(self.allowed(READ_ONLY, PROCESS_RETURN))

    def test_lookup_order_still_succeeds_for_the_denied_caller(self):
        # The deny has to be specific to the tool, or stage 2 has broken stage 1's
        # agent rather than secured it.
        self.assertTrue(self.allowed(READ_ONLY, LOOKUP))

    def test_the_support_agent_is_permitted_both_tools(self):
        # One permit with an action list, so the privileged identity does not lose
        # the read it already had when it gained the write.
        self.assertTrue(self.allowed(SUPPORT_AGENT, PROCESS_RETURN))
        self.assertTrue(self.allowed(SUPPORT_AGENT, LOOKUP))

    def test_a_caller_named_by_neither_rule_is_denied_both_tools(self):
        self.assertFalse(self.allowed(STRANGER, LOOKUP))
        self.assertFalse(self.allowed(STRANGER, PROCESS_RETURN))

    def test_an_unknown_tool_is_denied_by_default(self):
        self.assertFalse(self.allowed(SUPPORT_AGENT, "supportTools___refund_order"))

    def test_the_rules_only_apply_to_the_gateway_they_name(self):
        # Cedar rejects a wildcard resource, so the ARN substitution is the thing
        # that keeps these rules off another team's gateway.
        self.assertFalse(self.allowed(READ_ONLY, LOOKUP, gateway_arn=OTHER_GATEWAY_ARN))
        self.assertFalse(
            self.allowed(SUPPORT_AGENT, PROCESS_RETURN, gateway_arn=OTHER_GATEWAY_ARN)
        )

    def test_an_oauth_principal_does_not_match_the_iam_rules(self):
        # The gateway is AWS_IAM. A rule written for IamEntity must not silently
        # authorize a JWT principal if the authorizer is ever changed.
        self.assertFalse(
            fake_cedar.is_authorized(
                self.statements,
                Entity("AgentCore::OAuthUser", READ_ONLY),
                LOOKUP,
                Entity("AgentCore::Gateway", GATEWAY_ARN),
            )
        )


class CedarEvaluatorFidelityTest(unittest.TestCase):
    """The evaluator's own semantics, since the deny assertions rest on them."""

    def test_no_statements_is_a_deny(self):
        self.assertFalse(
            fake_cedar.is_authorized(
                [], Entity("AgentCore::IamEntity", READ_ONLY), LOOKUP,
                Entity("AgentCore::Gateway", GATEWAY_ARN),
            )
        )

    def test_a_matching_forbid_beats_a_matching_permit(self):
        statements = fake_cedar.parse(
            f'permit(principal is AgentCore::IamEntity, '
            f'action == AgentCore::Action::"{LOOKUP}", '
            f'resource == AgentCore::Gateway::"{GATEWAY_ARN}");\n'
            f'forbid(principal == AgentCore::IamEntity::"{STRANGER}", '
            f'action == AgentCore::Action::"{LOOKUP}", '
            f'resource == AgentCore::Gateway::"{GATEWAY_ARN}");'
        )
        gateway = Entity("AgentCore::Gateway", GATEWAY_ARN)
        self.assertTrue(
            fake_cedar.is_authorized(
                statements, Entity("AgentCore::IamEntity", READ_ONLY), LOOKUP, gateway
            )
        )
        self.assertFalse(
            fake_cedar.is_authorized(
                statements, Entity("AgentCore::IamEntity", STRANGER), LOOKUP, gateway
            )
        )

    def test_cedar_outside_the_subset_raises_rather_than_being_ignored(self):
        # The failure mode that matters: a rule this cannot evaluate must break the
        # test rather than quietly evaluate to "permitted". A when clause is in this
        # list because the shipped rules have none: if one is ever added, the rule
        # stops being evaluable here and has to be proven live instead.
        for text in (
            'permit(principal, action, resource);',
            f'permit(principal in AgentCore::Group::"support", '
            f'action == AgentCore::Action::"{LOOKUP}", resource);',
            f'permit(principal is AgentCore::IamEntity, '
            f'action == AgentCore::Action::"{LOOKUP}", resource) '
            f'unless {{ context.input.order_id == "1" }};',
            f'permit(principal == AgentCore::IamEntity::"{READ_ONLY}", '
            f'action == AgentCore::Action::"{LOOKUP}", '
            f'resource == AgentCore::Gateway::"{GATEWAY_ARN}") '
            f'when {{ context.input.order_id != "" }};',
            f'permit(principal is AgentCore::IamEntity, '
            f'action in [AgentCore::Action::"{LOOKUP}", "not-an-action"], resource);',
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    fake_cedar.parse(text)

    def test_an_empty_policy_file_raises(self):
        with self.assertRaises(ValueError):
            fake_cedar.parse("// only a comment\n")


class PolicyRenderingTest(unittest.TestCase):
    """The .cedar file's block markers, and the substitution guards."""

    def test_each_marked_block_becomes_one_named_policy(self):
        names = [name for name, _ in read_policy_blocks()]
        self.assertEqual(
            names, ["LookupOrderForReadOnlyCaller", "LookupAndReturnForSupportAgent"]
        )
        # AgentCore requires [A-Za-z][A-Za-z0-9_]* and at most 48 characters.
        for name in names:
            self.assertRegex(name, r"^[A-Za-z][A-Za-z0-9_]*$")
            self.assertLessEqual(len(name), 48)

    def test_rendering_substitutes_every_placeholder(self):
        for name, statement in render_policies(GATEWAY_ARN, READ_ONLY, SUPPORT_AGENT):
            with self.subTest(policy=name):
                self.assertNotRegex(statement, r"<[A-Z_]+>")
                self.assertIn(GATEWAY_ARN, statement)
                self.assertGreaterEqual(len(statement), 35)  # the API's minimum
                self.assertLessEqual(len(statement), 10_000)

    def test_an_empty_substitution_is_refused(self):
        # The quiet bug this guard exists for: an empty principal renders a permit
        # for the caller named "", which matches nobody, so every call the rule was
        # meant to allow is denied and nothing looks broken.
        for args in (
            ("", READ_ONLY, SUPPORT_AGENT),
            (GATEWAY_ARN, "", SUPPORT_AGENT),
            (GATEWAY_ARN, READ_ONLY, ""),
        ):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    render_policies(*args)

    def test_each_rule_names_one_principal_and_carries_no_condition(self):
        # The whole control is the principal ARN: measured live, a permit naming a
        # specific principal passes FAIL_ON_ANY_FINDINGS with no when clause, while
        # a permit naming a principal *type* is refused as overly permissive. A
        # condition reappearing here would be a rule fake_cedar can no longer
        # evaluate, so this is the assertion that keeps the file inside the subset.
        rules = dict(render_policies(GATEWAY_ARN, READ_ONLY, SUPPORT_AGENT))
        for name, statement in rules.items():
            with self.subTest(policy=name):
                # Each block keeps its own comments, and those comments discuss the
                # absence of a when clause, so the assertion has to read the Cedar.
                cedar = "\n".join(
                    line for line in statement.splitlines()
                    if not line.lstrip().startswith("//")
                )
                self.assertNotIn("when", cedar)
                self.assertNotIn("context.input", cedar)
                self.assertIn("principal == AgentCore::IamEntity::", cedar)
        self.assertIn(READ_ONLY, rules["LookupOrderForReadOnlyCaller"])
        self.assertNotIn(SUPPORT_AGENT, rules["LookupOrderForReadOnlyCaller"])
        self.assertIn(SUPPORT_AGENT, rules["LookupAndReturnForSupportAgent"])
        self.assertNotIn(READ_ONLY, rules["LookupAndReturnForSupportAgent"])


class PolicyRegistrationTest(unittest.TestCase):
    """attach_policy against the faked control plane."""

    def setUp(self):
        quiet(self)
        self.control = FakeAgentCoreControlClient(active_after=1)
        self.sts = FakeSTSClient()
        self.boto3 = patch_boto3(
            self,
            attach_policy,
            **{"bedrock-agentcore-control": self.control, "sts": self.sts},
        )
        self.clock = patch_clock(self, attach_policy)

    def operations(self):
        return [name for name, _ in self.control.calls]

    def created_policies(self):
        return [params for name, params in self.control.calls if name == "create_policy"]

    def test_register_creates_one_policy_per_rule_and_attaches_the_engine(self):
        engine_id, policies = register(
            "gw-abc123", GATEWAY_ARN, READ_ONLY, SUPPORT_AGENT, "ENFORCE", "us-east-1"
        )

        self.assertEqual(
            sorted(policies),
            ["LookupAndReturnForSupportAgent", "LookupOrderForReadOnlyCaller"],
        )
        self.assertTrue(engine_id.startswith("policy-engine-"))
        self.assertEqual(self.operations().count("create_policy_engine"), 1)
        self.assertEqual(self.operations().count("create_policy"), 2)
        self.assertEqual(self.operations().count("update_gateway"), 1)
        # The engine has to exist before the rules that name the gateway ARN, and
        # the attach has to come last.
        self.assertLess(
            self.operations().index("create_policy_engine"),
            self.operations().index("create_policy"),
        )
        self.assertLess(
            self.operations().index("create_policy"),
            self.operations().index("update_gateway"),
        )

    def test_the_rendered_cedar_is_what_gets_created(self):
        register("gw-abc123", GATEWAY_ARN, READ_ONLY, SUPPORT_AGENT, "ENFORCE", "us-east-1")

        statements = {
            params["name"]: params["definition"]["cedar"]["statement"]
            for params in self.created_policies()
        }
        self.assertIn(GATEWAY_ARN, statements["LookupOrderForReadOnlyCaller"])
        self.assertIn(READ_ONLY, statements["LookupOrderForReadOnlyCaller"])
        self.assertIn(SUPPORT_AGENT, statements["LookupAndReturnForSupportAgent"])
        # The privileged rule is the only one naming process_return; that is where
        # the read-only caller's deny comes from.
        self.assertNotIn(PROCESS_RETURN, statements["LookupOrderForReadOnlyCaller"])
        self.assertIn(PROCESS_RETURN, statements["LookupAndReturnForSupportAgent"])

    def test_create_policy_requests_are_valid_requests(self):
        register("gw-abc123", GATEWAY_ARN, READ_ONLY, SUPPORT_AGENT, "ENFORCE", "us-east-1")
        for params in self.created_policies():
            with self.subTest(policy=params["name"]):
                self.assertTrue(valid_request("bedrock-agentcore-control", "CreatePolicy", params))

    def test_the_attach_echoes_the_gateways_authentication_back(self):
        # UpdateGateway is a full replacement. Omitting authorizerType or roleArn
        # is an update that silently rewrites who may call the gateway.
        register("gw-abc123", GATEWAY_ARN, READ_ONLY, SUPPORT_AGENT, "ENFORCE", "us-east-1")

        update = [p for name, p in self.control.calls if name == "update_gateway"][0]
        self.assertEqual(update["authorizerType"], "AWS_IAM")
        self.assertEqual(update["roleArn"], "arn:aws:iam::123456789012:role/GatewayExecutionRole")
        self.assertEqual(update["name"], "MigratedAgentGateway")
        self.assertEqual(update["policyEngineConfiguration"]["mode"], "ENFORCE")
        self.assertTrue(valid_request("bedrock-agentcore-control", "UpdateGateway", update))

    def test_log_only_is_available_for_a_first_attach(self):
        register("gw-abc123", GATEWAY_ARN, READ_ONLY, SUPPORT_AGENT, "LOG_ONLY", "us-east-1")
        update = [p for name, p in self.control.calls if name == "update_gateway"][0]
        self.assertEqual(update["policyEngineConfiguration"]["mode"], "LOG_ONLY")

    def test_an_existing_engine_is_reused_rather_than_duplicated(self):
        first, _ = register(
            "gw-abc123", GATEWAY_ARN, READ_ONLY, SUPPORT_AGENT, "ENFORCE", "us-east-1"
        )
        self.control.calls.clear()

        second, _ = register(
            "gw-abc123", GATEWAY_ARN, READ_ONLY, SUPPORT_AGENT, "ENFORCE", "us-east-1"
        )

        self.assertEqual(second, first)
        self.assertNotIn("create_policy_engine", self.operations())
        self.assertEqual(self.control.engines[first]["name"], POLICY_ENGINE_NAME)

    def test_the_engine_is_active_before_any_rule_is_created_on_it(self):
        # CreatePolicy against an engine still CREATING is a request that fails at
        # step 10 and nowhere earlier, so the waiter has to run before the first
        # rule rather than being implied by the policy waits after it.
        self.control.active_after = 2

        register("gw-abc123", GATEWAY_ARN, READ_ONLY, SUPPORT_AGENT, "ENFORCE", "us-east-1")

        operations = self.operations()
        before_first_rule = operations[: operations.index("create_policy")]
        self.assertGreaterEqual(before_first_rule.count("get_policy_engine"), 3)
        self.assertTrue(self.clock.slept)

    def test_an_engine_stuck_creating_times_out_rather_than_being_attached(self):
        self.control.active_after = 10_000
        with self.assertRaises(TimeoutError):
            register("gw-abc123", GATEWAY_ARN, READ_ONLY, SUPPORT_AGENT, "ENFORCE", "us-east-1")
        self.assertNotIn("update_gateway", self.operations())

    def test_the_caller_principal_drops_the_session_name(self):
        # A session name changes per assume-role call, so a principal == rule
        # written with one attached would stop matching on the next run.
        self.assertEqual(caller_principal("us-east-1"), READ_ONLY)

    def test_a_plain_role_or_user_arn_is_left_alone(self):
        self.sts.arn = "arn:aws:iam::123456789012:user/dana"
        self.assertEqual(caller_principal("us-east-1"), "arn:aws:iam::123456789012:user/dana")

    def test_the_support_agent_principal_is_a_role_the_caller_is_not(self):
        # The two principals have to differ or the deny cannot be observed: if the
        # walkthrough ran as the privileged role, its process_return call would be
        # permitted and the demo would prove nothing.
        self.assertEqual(support_agent_principal("us-east-1"), SUPPORT_AGENT)
        self.assertNotEqual(support_agent_principal("us-east-1"), caller_principal("us-east-1"))


class PolicyDeletionTest(unittest.TestCase):
    """Deleting the Cedar resources, including when one of them will not go."""

    def setUp(self):
        quiet(self)
        self.control = FakeAgentCoreControlClient(delete_lag=2, active_after=0)
        patch_boto3(self, attach_policy, **{"bedrock-agentcore-control": self.control})
        self.clock = patch_clock(self, attach_policy)
        self.engine_id = self.control.create_policy_engine(name=POLICY_ENGINE_NAME)[
            "policyEngineId"
        ]
        self.policy_ids = [
            self.control.create_policy(name=f"P{index}", policyEngineId=self.engine_id)[
                "policyId"
            ]
            for index in range(2)
        ]
        self.control.calls.clear()

    def test_every_policy_is_deleted_and_confirmed_gone(self):
        delete_policies(self.engine_id, self.policy_ids, "us-east-1")

        deleted = [p["policyId"] for name, p in self.control.calls if name == "delete_policy"]
        self.assertEqual(deleted, self.policy_ids)
        # delete_lag=2 means get_policy still answered twice after the delete, so
        # returning means the code polled instead of trusting the delete.
        self.assertGreaterEqual(
            len([1 for name, _ in self.control.calls if name == "get_policy"]), 4
        )

    def test_one_stuck_policy_does_not_strand_the_others(self):
        stuck = self.policy_ids[0]
        original = self.control.delete_policy
        attempted = []

        def sometimes_fails(**kwargs):
            attempted.append(kwargs["policyId"])
            if kwargs["policyId"] == stuck:
                raise ValidationException("policy is referenced")
            return original(**kwargs)

        self.control.delete_policy = sometimes_fails

        with self.assertRaises(ValidationException):
            delete_policies(self.engine_id, self.policy_ids, "us-east-1")

        # The second policy was still attempted, and deleted, before the raise.
        self.assertEqual(attempted, self.policy_ids)
        self.assertIn(self.policy_ids[1], self.control.deleted)
        self.assertNotIn(stuck, self.control.deleted)

    def test_an_already_deleted_policy_is_not_an_error(self):
        delete_policies(self.engine_id, self.policy_ids, "us-east-1")
        delete_policies(self.engine_id, self.policy_ids, "us-east-1")  # must not raise

    def test_the_engine_delete_is_retried_on_validation_exception(self):
        self.control.engine_validation_failures = 2

        delete_policy_engine(self.engine_id, "us-east-1")

        attempts = [1 for name, _ in self.control.calls if name == "delete_policy_engine"]
        self.assertEqual(len(attempts), 3)
        self.assertIn(self.engine_id, self.control.deleted)

    def test_an_engine_that_never_deletes_times_out(self):
        self.control.engine_validation_failures = 10_000
        with self.assertRaises(ValidationException):
            delete_policy_engine(self.engine_id, "us-east-1", timeout=30)


class DemoPrincipalsTest(unittest.TestCase):
    """The two roles the Cedar proof rests on, and what makes the proof sound."""

    def setUp(self):
        quiet(self)
        self.iam = FakeIAMClient()
        self.sts = FakeSTSClient()
        patch_boto3(self, demo_principals, iam=self.iam, sts=self.sts)
        self.clock = patch_clock(self, demo_principals)

    def create(self):
        return demo_principals.create_demo_roles(GATEWAY_ARN, "us-east-1")

    def test_both_roles_get_the_same_permissions(self):
        self.create()
        documents = demo_principals.assert_identical_iam("us-east-1")
        self.assertEqual(*documents)
        # And the permission is the gateway, not the tools: scoping tools in IAM
        # would give a denial two possible explanations.
        statement = documents[0]["Statement"][0]
        self.assertEqual(statement["Resource"], GATEWAY_ARN)
        self.assertEqual(statement["Action"], "bedrock-agentcore:*")

    def test_a_difference_between_the_two_roles_is_refused(self):
        # The whole proof is that IAM cannot explain the denial. If the two roles
        # drift, the four calls that follow mean nothing, so this has to raise
        # rather than warn.
        self.create()
        self.iam.inline[demo_principals.SUPPORT_ROLE_NAME][
            demo_principals.POLICY_NAME
        ]["Statement"][0]["Action"] = "bedrock-agentcore:InvokeGateway"

        with self.assertRaises(RuntimeError) as caught:
            demo_principals.assert_identical_iam("us-east-1")
        self.assertIn("identical", str(caught.exception))

    def test_a_managed_policy_on_one_role_is_refused(self):
        # Comparing inline documents alone would miss it: a managed policy is a
        # permission difference the inline comparison cannot see.
        self.create()
        self.iam.attached[demo_principals.READ_ONLY_ROLE_NAME] = [
            {"PolicyName": "PowerUserAccess", "PolicyArn": "arn:aws:iam::aws:policy/x"}
        ]

        with self.assertRaises(RuntimeError) as caught:
            demo_principals.assert_identical_iam("us-east-1")
        self.assertIn("managed policies", str(caught.exception))

    def test_the_cedar_principal_is_the_assumed_role_form(self):
        # The single highest-consequence conversion in stage 2. IAM returns
        # arn:aws:iam::...:role/Name; an AWS_IAM gateway presents
        # arn:aws:sts::...:assumed-role/Name. A rule written with the IAM form
        # matches nobody, and default-deny makes that look like enforcement
        # working rather than a typo.
        principal = demo_principals.cedar_principal(
            f"arn:aws:iam::123456789012:role/{demo_principals.READ_ONLY_ROLE_NAME}"
        )
        self.assertEqual(
            principal,
            "arn:aws:sts::123456789012:assumed-role/MigratedAgentReadOnlyCaller",
        )
        self.assertNotIn(":iam:", principal)

    def test_creating_twice_is_not_an_error(self):
        self.create()
        arns = self.create()
        self.assertEqual(len(arns), 2)
        demo_principals.assert_identical_iam("us-east-1")

    def test_assume_retries_the_access_denied_a_new_role_returns(self):
        self.create()
        self.sts.access_denied_times = 3
        credentials = demo_principals.assume(
            self.iam.roles[demo_principals.SUPPORT_ROLE_NAME]["Arn"],
            "proof-1",
            "us-east-1",
        )
        self.assertIn("AccessKeyId", credentials)
        self.assertTrue(self.clock.slept)

    def test_assume_gives_up_rather_than_retrying_forever(self):
        self.create()
        self.sts.access_denied_times = 10_000
        with self.assertRaises(FakeClientError):
            demo_principals.assume("arn:aws:iam::1:role/x", "proof-1", "us-east-1", 30)

    def test_the_inline_policy_goes_before_the_role_it_is_on(self):
        # IAM refuses to delete a role that still has a policy attached, so the
        # order here is the service's and not a preference.
        self.create()
        demo_principals.delete_demo_roles("us-east-1")

        self.assertEqual(self.iam.roles, {})
        deletes = [name for name, _ in self.iam.calls if name.startswith("delete_")]
        self.assertEqual(
            deletes,
            ["delete_role_policy", "delete_role", "delete_role_policy", "delete_role"],
        )

    def test_a_role_that_will_not_delete_does_not_strand_the_other(self):
        self.create()
        first = demo_principals.READ_ONLY_ROLE_NAME
        self.iam.inline[first]["Stuck"] = {"Version": "2012-10-17"}

        def refuse(**kwargs):
            if kwargs["PolicyName"] == "Stuck":
                raise FakeClientError("DeleteConflict", "policy is not deletable")
            return None

        original = self.iam.delete_role_policy
        self.iam.delete_role_policy = lambda **kwargs: refuse(**kwargs) or original(
            **kwargs
        )

        with self.assertRaises(FakeClientError):
            demo_principals.delete_demo_roles("us-east-1")

        self.assertIn(first, self.iam.roles)
        self.assertNotIn(demo_principals.SUPPORT_ROLE_NAME, self.iam.roles)

    def test_deleting_roles_that_are_already_gone_is_not_an_error(self):
        self.create()
        demo_principals.delete_demo_roles("us-east-1")
        demo_principals.delete_demo_roles("us-east-1")  # must not raise


class GatewaySigningTest(unittest.TestCase):
    """Which identity a gateway request is signed as, since that is the whole input
    to a policy decision.

    Nothing is mocked below build_mcp_client: the real SigV4Auth signs a real
    httpx.Request, and the assertions read the headers that would have gone out.
    Only the MCP transport is replaced, because opening one needs a server.
    """

    ASSUME_ROLE_RESPONSE = {
        "AccessKeyId": "ASIAREADONLYEXAMPLE",
        "SecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "SessionToken": "FwoGZXIvYXdzEXAMPLETOKEN",
    }
    URL = "https://gw-abc123.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"

    def setUp(self):
        self.captured = {}

        def fake_transport(gateway_url, auth=None):
            self.captured["url"] = gateway_url
            self.captured["auth"] = auth

        def fake_mcp_client(factory):
            # MCPClient does not call the factory until start(); calling it here
            # is what lets the auth it was built with be inspected.
            factory()
            return "mcp-client"

        for name, value in (
            ("streamablehttp_client", fake_transport),
            ("MCPClient", fake_mcp_client),
        ):
            self.addCleanup(
                setattr, gateway_mcp_tools, name, getattr(gateway_mcp_tools, name)
            )
            setattr(gateway_mcp_tools, name, value)

    def sign(self, credentials):
        gateway_mcp_tools.build_mcp_client(self.URL, "us-east-1", credentials)
        request = httpx.Request(
            "POST",
            self.URL,
            content=b'{"jsonrpc":"2.0","method":"tools/call"}',
            headers={"Content-Type": "application/json"},
        )
        list(self.captured["auth"].auth_flow(request))
        return request.headers

    def test_an_assume_role_response_dict_signs_as_that_role(self):
        # Measured live before this was handled: passing the STS response straight
        # through failed with "'dict' object has no attribute 'token'" from inside
        # SigV4Auth, four frames below the call that supplied it and after the MCP
        # client had already opened a connection. The API returns a JSON document
        # and Session.get_credentials() returns an object; both are the obvious
        # thing to hand this function.
        headers = self.sign(self.ASSUME_ROLE_RESPONSE)

        self.assertIn(
            self.ASSUME_ROLE_RESPONSE["AccessKeyId"], headers["Authorization"]
        )
        self.assertEqual(
            headers["X-Amz-Security-Token"],
            self.ASSUME_ROLE_RESPONSE["SessionToken"],
        )

    def test_a_botocore_credentials_object_still_works(self):
        headers = self.sign(
            Credentials("AKIAAMBIENTEXAMPLE", "secret", "ambient-token")
        )
        self.assertIn("AKIAAMBIENTEXAMPLE", headers["Authorization"])
        self.assertEqual(headers["X-Amz-Security-Token"], "ambient-token")

    def test_two_principals_produce_two_different_signatures(self):
        # The check that the proof in prove_policy_enforcement is not a tautology:
        # if both rows signed identically, all four decisions would agree by
        # construction and the matrix would prove nothing.
        first = self.sign(self.ASSUME_ROLE_RESPONSE)["Authorization"]
        second = self.sign(
            {**self.ASSUME_ROLE_RESPONSE, "AccessKeyId": "ASIASUPPORTEXAMPLE"}
        )["Authorization"]
        self.assertNotEqual(first, second)

    def test_the_signature_covers_the_body(self):
        # requires_request_body is set so httpx materializes the body before
        # auth_flow runs. If it did not, SigV4 would hash an empty payload and the
        # gateway would reject every signature.
        self.assertTrue(gateway_mcp_tools.SigV4HTTPXAuth.requires_request_body)
        signed = self.sign(self.ASSUME_ROLE_RESPONSE)["Authorization"]
        credentials = dict(self.ASSUME_ROLE_RESPONSE)
        auth = gateway_mcp_tools.SigV4HTTPXAuth(
            Credentials(
                credentials["AccessKeyId"],
                credentials["SecretAccessKey"],
                credentials["SessionToken"],
            ),
            gateway_mcp_tools.SERVICE,
            "us-east-1",
        )
        other = httpx.Request(
            "POST",
            self.URL,
            content=b'{"jsonrpc":"2.0","method":"tools/list"}',
            headers={"Content-Type": "application/json"},
        )
        list(auth.auth_flow(other))
        self.assertNotEqual(signed, other.headers["Authorization"])

    def test_omitting_credentials_signs_as_the_ambient_identity(self):
        # The path every caller other than the policy proof takes, including an
        # agent running in Runtime, where the ambient identity is the execution
        # role and there is nothing to pass in.
        ambient = Credentials("AKIARUNTIMEEXAMPLE", "secret", "runtime-token")

        class Boto3:
            Session = staticmethod(
                lambda: type("S", (), {"get_credentials": lambda self: ambient})()
            )

        self.addCleanup(setattr, gateway_mcp_tools, "boto3", gateway_mcp_tools.boto3)
        gateway_mcp_tools.boto3 = Boto3
        headers = self.sign(None)

        self.assertEqual(self.captured["url"], self.URL)
        self.assertIn("AKIARUNTIMEEXAMPLE", headers["Authorization"])


class ObservingADecisionTest(unittest.TestCase):
    """call_tool_through_gateway returns a decision, including when it cannot connect."""

    def setUp(self):
        quiet(self)
        self.started = []
        self.stopped = []

    def build(self, on_start=None, result=None, on_stop=None):
        test = self

        class Client:
            def start(self):
                test.started.append(True)
                if on_start:
                    raise on_start

            def call_tool_sync(self, **kwargs):
                return result

            def stop(self, *args):
                test.stopped.append(True)
                if on_stop:
                    raise on_stop

        self.addCleanup(
            setattr, attach_policy, "build_mcp_client", attach_policy.build_mcp_client
        )
        attach_policy.build_mcp_client = lambda *args, **kwargs: Client()

    def call(self):
        return attach_policy.call_tool_through_gateway(
            "https://gw.example/mcp", "supportTools___lookup_order", {"order_id": "1"}
        )

    def test_a_refusal_at_connect_time_is_a_denial_not_a_crash(self):
        # Measured live: the client raised MCPClientInitializationError from
        # start(), which sat outside the try, so the exception escaped a function
        # whose whole contract is to return a decision — and stopped the
        # walkthrough on the call it exists to report on.
        self.build(on_start=RuntimeError("the client initialization failed"))

        allowed, text = self.call()

        self.assertFalse(allowed)
        self.assertIn("the client initialization failed", text)

    def test_an_error_result_is_a_denial(self):
        self.build(
            result={
                "status": "error",
                "content": [{"text": "Tool Execution Denied: policy enforcement"}],
            }
        )
        allowed, text = self.call()
        self.assertFalse(allowed)
        self.assertIn("Denied", text)

    def test_a_successful_call_returns_the_tool_output(self):
        self.build(result={"status": "success", "content": [{"text": '{"ok": true}'}]})
        allowed, text = self.call()
        self.assertTrue(allowed)
        self.assertEqual(text, '{"ok": true}')
        self.assertEqual(len(self.stopped), 1)

    def test_a_client_that_will_not_stop_does_not_replace_the_decision(self):
        # stop() on a client that failed to start raises over the top of the real
        # failure, which is how a denial gets reported as a shutdown error.
        self.build(
            on_start=RuntimeError("connect refused"),
            on_stop=RuntimeError("cannot stop what never started"),
        )
        allowed, text = self.call()
        self.assertFalse(allowed)
        self.assertIn("connect refused", text)


class PolicyEnforcementProofTest(unittest.TestCase):
    """Two callers times two tools: the check that a denial is Cedar's."""

    ROLE_ARNS = {
        run_walkthrough.READ_ONLY_ROLE_NAME: (
            "arn:aws:iam::123456789012:role/MigratedAgentReadOnlyCaller"
        ),
        run_walkthrough.SUPPORT_ROLE_NAME: (
            "arn:aws:iam::123456789012:role/MigratedAgentSupportAgent"
        ),
    }
    DENIAL = (
        "Tool execution failed: Tool Execution Denied: Tool call not allowed due to "
        "policy enforcement [No policy applies to the request (denied by default).]"
    )

    def setUp(self):
        self.output = io.StringIO()
        captured = contextlib.redirect_stdout(self.output)
        captured.__enter__()
        self.addCleanup(captured.__exit__, None, None, None)
        self.calls = []
        self.decisions = dict(run_walkthrough.EXPECTED_DECISIONS)
        self.replace(run_walkthrough, "assume", self._assume)
        self.replace(run_walkthrough, "call_tool_through_gateway", self._call)

    def replace(self, module, name, value):
        self.addCleanup(setattr, module, name, getattr(module, name))
        setattr(module, name, value)

    def _assume(self, role_arn, session_name, region_name, *args):
        return {"AccessKeyId": f"ASIA-{role_arn.split('/')[-1]}"}

    def _call(self, gateway_url, tool_name, arguments, region_name, credentials=None):
        # The caller is recovered from the credentials rather than passed in, so a
        # proof that signed both rows with the same identity fails here.
        role = credentials["AccessKeyId"].removeprefix("ASIA-")
        self.calls.append((role, tool_name))
        allowed = self.decisions[(role, tool_name)]
        return allowed, '{"order_id": "12345"}' if allowed else self.DENIAL

    def prove(self):
        return run_walkthrough.prove_policy_enforcement(
            "https://gateway.example/mcp", self.ROLE_ARNS, "us-east-1"
        )

    def test_both_tools_are_called_as_both_principals(self):
        self.prove()
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(set(self.calls), set(run_walkthrough.EXPECTED_DECISIONS))

    def test_each_row_is_signed_as_its_own_role(self):
        # The defect this exists for: signing every call with the ambient
        # credentials makes all four decisions identical and the matrix a
        # tautology.
        self.prove()
        for role_name in self.ROLE_ARNS:
            self.assertEqual(
                sum(1 for role, _ in self.calls if role == role_name), 2
            )

    def test_the_expected_matrix_passes_and_says_so(self):
        self.prove()
        printed = self.output.getvalue()
        self.assertIn("All four decisions match", printed)
        self.assertIn("Cedar", printed)

    def test_a_permitted_call_that_is_denied_fails_the_run(self):
        self.decisions[
            (run_walkthrough.READ_ONLY_ROLE_NAME, "supportTools___lookup_order")
        ] = False

        with self.assertRaises(RuntimeError) as caught:
            self.prove()
        self.assertIn("support_tools.cedar", str(caught.exception))
        self.assertIn("True -> False", str(caught.exception))

    def test_a_denied_call_that_is_permitted_fails_the_run(self):
        # The failure that matters most: enforcement silently off. Cedar in
        # LOG_ONLY mode, or an engine that never attached, allows all four calls
        # and every one of them looks like a success.
        self.decisions[
            (run_walkthrough.READ_ONLY_ROLE_NAME, "supportTools___process_return")
        ] = True

        with self.assertRaises(RuntimeError) as caught:
            self.prove()
        self.assertIn("False -> True", str(caught.exception))

    def test_the_denial_is_printed_whole(self):
        # The reason sits at the end of the message, so a fixed-width truncation
        # leaves a reader looking at a refusal with no explanation attached.
        self.prove()
        self.assertIn(self.DENIAL, self.output.getvalue())


class TeardownTest(unittest.TestCase):
    """The walkthrough's teardown: correct order, and nothing skipped."""

    def setUp(self):
        quiet(self)
        self.control = FakeAgentCoreControlClient(delete_lag=1)
        self.memory = []
        self.attempted = []
        patch_boto3(
            self, run_walkthrough, **{"bedrock-agentcore-control": self.control}
        )
        patch_clock(self, run_walkthrough)
        self.replace(run_walkthrough, "MemoryClient", self._memory_client)
        self.replace(run_walkthrough, "delete_policies", self._recorder("policies"))
        self.replace(
            run_walkthrough, "delete_policy_engine", self._recorder("policy engine")
        )
        self.replace(
            run_walkthrough, "delete_demo_roles", self._recorder("demo IAM roles")
        )
        # The target and gateway deletes stay real, because their retry and
        # confirm behaviour is part of what is under test here; they are only
        # instrumented so that one ordered list covers all five steps.
        self._instrument("delete_gateway_target", "target")
        self._instrument("delete_gateway", "gateway")

    def replace(self, module, name, value):
        self.addCleanup(setattr, module, name, getattr(module, name))
        setattr(module, name, value)

    def _instrument(self, operation, label):
        """Record that a step was attempted, before it can raise, then call through."""
        original = getattr(self.control, operation)

        def recording(**kwargs):
            # A retried delete is still one attempted step.
            if label not in self.attempted:
                self.attempted.append(label)
            return original(**kwargs)

        setattr(self.control, operation, recording)

    def _memory_client(self, region_name=None):
        test = self

        class Client:
            def delete_memory_and_wait(self, memory_id):
                test.attempted.append("memory")
                test.memory.append(memory_id)

        return Client()

    def _recorder(self, label, error=None):
        def record(*args, **kwargs):
            self.attempted.append(label)
            if error:
                raise error

        return record

    def full_teardown(self, **overrides):
        kwargs = {
            "gateway_id": "gw-abc123",
            "target_id": "tgt-abc123",
            "memory_id": "mem-abc123",
            "region_name": "us-east-1",
            "policy_engine_id": "policy-engine-1",
            "policy_ids": ["policy-1", "policy-2"],
        }
        kwargs.update(overrides)
        return run_walkthrough.teardown(**kwargs)

    def test_every_resource_stage_2_creates_is_deleted_in_dependency_order(self):
        self.full_teardown()

        self.assertEqual(
            self.attempted,
            ["target", "gateway", "memory", "policies", "policy engine"],
        )
        self.assertEqual(self.memory, ["mem-abc123"])
        self.assertTrue(self.control.gateway_deleted)

    def test_a_failing_gateway_delete_does_not_skip_the_rest(self):
        # Exactly the run-2 failure R11 is about: a fail-fast teardown orphans
        # everything after the first error, and nothing later in the list is
        # cheaper to leave behind than the gateway that would not go.
        self.control.validation_failures = 10_000

        with self.assertRaises(RuntimeError) as caught:
            self.full_teardown()

        self.assertEqual(
            self.attempted,
            ["target", "gateway", "memory", "policies", "policy engine"],
        )
        self.assertIn("gateway", str(caught.exception))
        self.assertIn("ValidationException", str(caught.exception))

    def test_several_failures_are_all_reported_not_just_the_first(self):
        self.control.validation_failures = 10_000
        self.replace(
            run_walkthrough,
            "delete_policies",
            self._recorder("policies", error=ValidationException("policy is referenced")),
        )

        with self.assertRaises(RuntimeError) as caught:
            self.full_teardown()

        message = str(caught.exception)
        self.assertIn("gateway", message)
        self.assertIn("policies", message)

    def test_a_stage_1_teardown_attempts_nothing_stage_2_would_have_created(self):
        # --stage 1 --teardown must behave exactly as it did before stage 2.
        run_walkthrough.teardown("gw-abc123", "tgt-abc123", "mem-abc123", "us-east-1")

        self.assertEqual(self.attempted, ["target", "gateway", "memory"])

    def _patch_runtime_deletes(self):
        """Record the runtime deletes instead of making them."""
        for name, label in (
            ("delete_runtime", "runtime"),
            ("delete_bucket", "artifact bucket"),
            ("delete_role", "runtime execution role"),
            ("delete_log_group", "log group"),
        ):
            self.replace(
                run_walkthrough.deploy_runtime, name, self._recorder(label)
            )

    RUNTIME_KWARGS = {
        "runtime_id": "MigratedAgentRuntime-abc1234567",
        "zip_bucket": "migrated-agent-runtime-123456789012",
        "runtime_role": "MigratedAgentRuntimeRole",
        "log_groups": ["/aws/bedrock-agentcore/runtimes/MigratedAgentRuntime-abc1234567-DEFAULT"],
    }

    def test_the_runtime_goes_first_and_its_leftovers_go_last(self):
        """The runtime holds the zip and assumes the role, so both outlive it.

        The log group is last for the opposite reason: it survives the runtime, and
        deleting it while the runtime is still alive just means the next log line
        creates it again.
        """
        self._patch_runtime_deletes()
        self.full_teardown(**self.RUNTIME_KWARGS)

        self.assertEqual(
            self.attempted,
            [
                "runtime", "target", "gateway", "memory", "policies", "policy engine",
                "artifact bucket", "runtime execution role", "log group",
            ],
        )

    def test_a_failing_runtime_delete_does_not_orphan_the_bucket_or_the_log_group(self):
        """The leak this whole class of resource is prone to, as a test.

        A fail-fast teardown here leaves behind a bucket holding 50 MB, an IAM role
        that can read it, and a log group nothing will ever look at again.
        """
        self._patch_runtime_deletes()
        self.replace(
            run_walkthrough.deploy_runtime,
            "delete_runtime",
            self._recorder("runtime", error=RuntimeError("still present after 180s")),
        )

        with self.assertRaises(RuntimeError) as caught:
            self.full_teardown(**self.RUNTIME_KWARGS)

        for label in ("artifact bucket", "runtime execution role", "log group"):
            self.assertIn(label, self.attempted)
        self.assertIn("runtime", str(caught.exception))

    def test_every_discovered_log_group_is_deleted_not_just_the_first(self):
        """Discovery matches on a prefix, so it can return more than one.

        Two runs that both left a group behind is exactly the state the account was
        found in, and a teardown that deletes one of them reports success.
        """
        self._patch_runtime_deletes()
        groups = [
            "/aws/bedrock-agentcore/runtimes/MigratedAgentRuntime-aaaaaaaaaa-DEFAULT",
            "/aws/bedrock-agentcore/runtimes/MigratedAgentRuntime-bbbbbbbbbb-DEFAULT",
        ]
        deleted = []
        self.replace(
            run_walkthrough.deploy_runtime,
            "delete_log_group",
            lambda name, region: deleted.append(name),
        )
        self.full_teardown(**{**self.RUNTIME_KWARGS, "log_groups": groups})

        self.assertEqual(deleted, groups)

    def test_the_gateway_delete_is_retried_and_then_confirmed_gone(self):
        # Measured live: DeleteGateway raised ValidationException while
        # ListGatewayTargets already returned [].
        self.control.validation_failures = 2

        self.full_teardown()

        attempts = [1 for name, _ in self.control.calls if name == "delete_gateway"]
        self.assertEqual(len(attempts), 3)
        self.assertTrue(self.control.gateway_deleted)
        self.assertIn("get_gateway", [name for name, _ in self.control.calls])

    def test_an_already_deleted_target_and_gateway_are_not_errors(self):
        self.full_teardown()
        self.attempted.clear()
        self.full_teardown()  # must not raise

        self.assertEqual(
            self.attempted,
            ["target", "gateway", "memory", "policies", "policy engine"],
        )

    def test_a_gateway_that_never_disappears_times_out(self):
        self.control.delete_lag = 10_000
        with self.assertRaises(RuntimeError) as caught:
            self.full_teardown()
        self.assertIn("TimeoutError", str(caught.exception))

    def test_the_two_cedar_principals_are_deleted_last(self):
        # Two IAM roles are not billable and are still not free to leave behind:
        # a role that can call a gateway is a permission nobody is auditing. Last
        # because a role that will not delete must not strand a resource that
        # costs money.
        self.full_teardown(demo_roles=True)

        self.assertEqual(
            self.attempted,
            [
                "target",
                "gateway",
                "memory",
                "policies",
                "policy engine",
                "demo IAM roles",
            ],
        )

    def test_roles_are_attempted_even_when_everything_before_them_failed(self):
        self.control.validation_failures = 10_000
        self.replace(
            run_walkthrough,
            "delete_policies",
            self._recorder("policies", error=ValidationException("still referenced")),
        )

        with self.assertRaises(RuntimeError):
            self.full_teardown(demo_roles=True)

        self.assertIn("demo IAM roles", self.attempted)

    def test_a_run_that_created_no_roles_does_not_touch_iam(self):
        self.full_teardown()
        self.assertNotIn("demo IAM roles", self.attempted)


class MeasurementTableTest(unittest.TestCase):
    """The table has to say which numbers came from something that worked.

    Untested until a live run printed "CreateAgentRuntime -> READY  43.6s" for a
    create that raised. The timing recorded from a finally block is right — the
    number is real and losing it on failure would be worse — but the name is a
    claim, and nothing marked the row.
    """

    def table(self, measured) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            measured.report()
        return out.getvalue()

    def test_a_block_that_raised_is_recorded_and_marked(self):
        measured = Measurements()
        with self.assertRaises(ValueError):
            with measured.timing("CreateAgentRuntime -> READY"):
                raise ValueError("role validation failed")

        (name, value, unit, note, failed), = measured.taken
        self.assertEqual(name, "CreateAgentRuntime -> READY")
        self.assertIsInstance(value, float)
        self.assertEqual(unit, "s")
        self.assertTrue(failed)
        self.assertIn("ValueError", note)

        table = self.table(measured)
        row, = [line for line in table.splitlines() if "CreateAgentRuntime" in line]
        self.assertTrue(row.startswith("! "), table)
        self.assertIn("time to FAIL", row)

    def test_the_exception_is_re_raised_untouched(self):
        original = ValueError("role validation failed")
        measured = Measurements()
        with self.assertRaises(ValueError) as caught:
            with measured.timing("CreateAgentRuntime -> READY"):
                raise original
        self.assertIs(caught.exception, original)

    def test_a_caller_note_survives_beside_the_failure_reason(self):
        measured = Measurements()
        with self.assertRaises(RuntimeError):
            with measured.timing("provisioning", note="from this create"):
                raise RuntimeError("boom")
        note = measured.taken[0][3]
        self.assertIn("RuntimeError", note)
        self.assertIn("from this create", note)

    def test_a_block_that_returned_is_not_marked(self):
        measured = Measurements()
        with measured.timing("CreateAgentRuntime -> READY", note="from this create"):
            pass
        self.assertEqual(measured.taken[0][3], "from this create")
        self.assertFalse(measured.taken[0][4])

        table = self.table(measured)
        self.assertNotIn("!", table)
        self.assertNotIn("FAIL", table)

    def test_a_recorded_value_that_is_not_a_timing_is_never_marked(self):
        measured = Measurements()
        measured.record("largest blob event round-tripped", 4_194_304, " B")
        self.assertFalse(measured.taken[0][4])
        self.assertNotIn("!", self.table(measured))

    def test_keyboardinterrupt_is_marked_rather_than_swallowed(self):
        """BaseException, not Exception: a run cancelled mid-wait is not a READY.

        A ^C during a 3-minute memory wait is the likeliest way this table ever
        sees an unfinished block, and it does not inherit from Exception.
        """
        measured = Measurements()
        with self.assertRaises(KeyboardInterrupt):
            with measured.timing("memory -> ACTIVE"):
                raise KeyboardInterrupt
        self.assertTrue(measured.taken[0][4])
        self.assertIn("KeyboardInterrupt", measured.taken[0][3])


def quiet(test):
    """Swallow the progress printing the create and delete paths do.

    The functions under test print what they created and deleted, which is the
    right behaviour for a walkthrough and only noise in a test run. Captured
    rather than disabled, so the print calls are still executed.
    """
    captured = contextlib.redirect_stdout(io.StringIO())
    captured.__enter__()
    test.addCleanup(captured.__exit__, None, None, None)


def patch_boto3(test, module, **clients):
    """Swap the module's boto3 for one that hands back fakes. Restored on cleanup."""
    fake = FakeBoto3(**clients)
    test.addCleanup(setattr, module, "boto3", module.boto3)
    module.boto3 = fake
    return fake


def patch_clock(test, module):
    """Swap the module's time for a clock that only advances when slept on."""
    clock = FakeClock()
    test.addCleanup(setattr, module, "time", module.time)
    module.time = clock
    return clock


if __name__ == "__main__":
    unittest.main()

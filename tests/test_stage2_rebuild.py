# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Stage-2 verification: the Strands rebuild, its Guardrail, and its Cedar rules.

Runs offline and makes no AWS calls. The guardrail and AgentCore control planes
are faked (tests/fake_control_plane.py), the MCP session is faked
(tests/fake_mcp.py), and the clocks inside the waiters are faked so a poll loop
finishes instantly and a timeout is reachable. What is real: the shipped
support_tools.cedar text, the real BedrockModel and the real Strands Agent (both
construct boto3 clients and call nothing), the real MCPAgentTool wrapper, and the
real botocore service models, used to validate every request shape this code
would send.

Two limits worth stating plainly, because they bound what passing here proves:

1. There is no synchronous Cedar authorization API. The deny is asserted by
   evaluating the shipped rules with the Cedar subset evaluator in
   tests/fake_cedar.py, so what is verified is that the rules mean what stage 2
   claims — not that the Gateway enforces them. That needs the live run.
2. A guardrail's filtering behaviour is a model-side decision. What is verified
   here is that the PII rule is configured, that the request carrying it is a
   valid CreateGuardrail request, and that the guardrail reaches the model the
   agent invokes. Whether a given card number is anonymised is only observable
   live.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import contextlib
import io
import unittest

import botocore.session
import mcp.types as mcp_types
from botocore.validate import ParamValidator
from strands.tools.mcp.mcp_agent_tool import MCPAgentTool

from examples import run_walkthrough
from examples.stage0_langgraph.tools import SUPPORT_TOOLS
from examples.stage2_rebuild import guardrail as guardrail_module
from examples.stage2_rebuild.guardrail import (
    BLOCKED_INPUT_MESSAGE,
    GUARDRAIL_NAME,
    PII_ENTITIES,
    create_guardrail,
    delete_guardrail,
    guarded_model,
)
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
from examples.stage2_rebuild.strands_agent import MODEL_ID, build_agent
from tests import fake_cedar
from tests.fake_cedar import Entity
from tests.fake_control_plane import (
    FakeAgentCoreControlClient,
    FakeBedrockClient,
    FakeBoto3,
    FakeClock,
    FakeSTSClient,
    ResourceNotFoundException,
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


class GuardrailConfigTest(unittest.TestCase):
    """The PII rule is configured, and configured as a valid request."""

    def test_the_create_request_is_a_valid_create_guardrail_request(self):
        # Validated against the real bedrock service model, so a misspelled key or
        # a missing required field fails here rather than at step 10.
        bedrock = FakeBedrockClient()
        self._create_with(bedrock)
        operation, params = bedrock.calls[0]
        self.assertEqual(operation, "create_guardrail")
        self.assertTrue(valid_request("bedrock", "CreateGuardrail", params))

    def test_the_pii_rule_reaches_the_create_request(self):
        bedrock = FakeBedrockClient()
        self._create_with(bedrock)
        params = bedrock.calls[0][1]
        configured = params["sensitiveInformationPolicyConfig"]["piiEntitiesConfig"]
        self.assertEqual(configured, PII_ENTITIES)

    def test_no_topic_policy_is_configured(self):
        # Measured live: a DENY topic written as "an order that is not the caller's"
        # matched every prompt naming an order number, including the customer's own,
        # because a topic classifier reads the sentence and cannot see who is asking.
        # Who may call which tool is Cedar's job, at the gateway. A topic policy
        # reappearing here would put that decision back in the wrong place.
        bedrock = FakeBedrockClient()
        self._create_with(bedrock)
        params = bedrock.calls[0][1]
        self.assertNotIn("topicPolicyConfig", params)
        self.assertFalse(hasattr(guardrail_module, "DENIED_TOPIC"))

    def test_the_refusal_text_is_configured_rather_than_left_to_the_model(self):
        # CreateGuardrail requires both messages whatever the policies are, and an
        # unset one would leave a refusal in the model's own words — the thing a
        # guardrail exists to stop being a prompt-engineering problem.
        bedrock = FakeBedrockClient()
        self._create_with(bedrock)
        params = bedrock.calls[0][1]
        self.assertEqual(params["blockedInputMessaging"], BLOCKED_INPUT_MESSAGE)
        self.assertTrue(params["blockedOutputsMessaging"])

    def test_card_numbers_are_anonymised_rather_than_blocked(self):
        # BLOCK would refuse the whole turn, so a customer who pastes a card number
        # into a return reason would lose the return as well as the number.
        self.assertTrue(PII_ENTITIES)
        self.assertEqual({e["action"] for e in PII_ENTITIES}, {"ANONYMIZE"})
        self.assertIn(
            "CREDIT_DEBIT_CARD_NUMBER", {e["type"] for e in PII_ENTITIES}
        )

    def _create_with(self, bedrock):
        quiet(self)
        patch_boto3(self, guardrail_module, bedrock=bedrock)
        patch_clock(self, guardrail_module)
        return create_guardrail(region_name="us-east-1")


class GuardrailCreateTest(unittest.TestCase):
    """Idempotency, versioning, and the two waits around them."""

    def setUp(self):
        quiet(self)
        self.bedrock = FakeBedrockClient(ready_after=2)
        patch_boto3(self, guardrail_module, bedrock=self.bedrock)
        self.clock = patch_clock(self, guardrail_module)

    def operations(self):
        return [name for name, _ in self.bedrock.calls]

    def test_a_new_guardrail_is_created_versioned_and_waited_for(self):
        guardrail_id, version = create_guardrail(region_name="us-east-1")

        self.assertIn("create_guardrail", self.operations())
        self.assertIn("create_guardrail_version", self.operations())
        self.assertEqual(version, "1")
        # ready_after=2 means the first two get_guardrail calls said CREATING, so
        # returning at all means the waiter polled rather than assuming.
        self.assertGreaterEqual(self.operations().count("get_guardrail"), 4)
        self.assertTrue(self.clock.slept)
        self.assertTrue(guardrail_id.startswith("gr-"))

    def test_an_existing_guardrail_is_reused_rather_than_duplicated(self):
        existing = self.bedrock.seed(GUARDRAIL_NAME, versions=("1", "2"))

        guardrail_id, version = create_guardrail(region_name="us-east-1")

        self.assertEqual(guardrail_id, existing)
        self.assertNotIn("create_guardrail", self.operations())
        self.assertNotIn("create_guardrail_version", self.operations())
        # The highest numbered version, not the first one listed.
        self.assertEqual(version, "2")

    def test_a_draft_only_guardrail_gets_a_numbered_version(self):
        # ListGuardrails returns one row per version, so this is a state that
        # really occurs: created once, never versioned.
        self.bedrock.seed(GUARDRAIL_NAME, versions=())

        _, version = create_guardrail(region_name="us-east-1")

        self.assertEqual(version, "1")
        self.assertIn("create_guardrail_version", self.operations())

    def test_the_returned_version_is_never_draft(self):
        # DRAFT is mutable: an agent pinned to it can have its filtering changed
        # without a deployment.
        for versions in ((), ("1",), ("1", "2", "3")):
            with self.subTest(versions=versions):
                bedrock = FakeBedrockClient(ready_after=0)
                bedrock.seed(GUARDRAIL_NAME, versions=versions)
                patch_boto3(self, guardrail_module, bedrock=bedrock)
                _, version = create_guardrail(region_name="us-east-1")
                self.assertNotEqual(version, "DRAFT")
                self.assertTrue(version.isdigit())

    def test_another_guardrails_name_is_not_mistaken_for_this_one(self):
        self.bedrock.seed("some-other-teams-guardrail", versions=("7",))

        guardrail_id, version = create_guardrail(region_name="us-east-1")

        self.assertIn("create_guardrail", self.operations())
        self.assertEqual(version, "1")
        self.assertNotEqual(guardrail_id, "gr-000000000001")

    def test_a_failed_guardrail_raises_instead_of_being_used(self):
        guardrail_id = self.bedrock.seed(GUARDRAIL_NAME, versions=())
        self.bedrock.guardrails[guardrail_id]["config"] = {}

        def failed(**kwargs):
            return {"guardrailId": guardrail_id, "status": "FAILED"}

        self.bedrock.get_guardrail = failed
        with self.assertRaises(RuntimeError) as caught:
            create_guardrail(region_name="us-east-1")
        self.assertIn("FAILED", str(caught.exception))

    def test_a_guardrail_stuck_creating_times_out(self):
        guardrail_id = self.bedrock.seed(GUARDRAIL_NAME, versions=())
        self.bedrock.get_guardrail = lambda **kwargs: {
            "guardrailId": guardrail_id,
            "status": "CREATING",
        }
        with self.assertRaises(TimeoutError):
            create_guardrail(region_name="us-east-1")


class GuardedModelTest(unittest.TestCase):
    """The guardrail has to reach the model the agent actually invokes."""

    def test_guarded_model_carries_the_id_and_the_pinned_version(self):
        model = guarded_model("gr-1", MODEL_ID, "3", "us-west-2")
        config = model.get_config()
        self.assertEqual(config["model_id"], MODEL_ID)
        self.assertEqual(config["guardrail_id"], "gr-1")
        self.assertEqual(config["guardrail_version"], "3")

    def test_the_agent_invokes_the_guarded_model(self):
        agent = build_agent(model=guarded_model("gr-1", MODEL_ID, "1"))
        # Not "a guarded model was constructed": the model on the agent is the one
        # carrying the guardrail, so every turn goes through it.
        self.assertEqual(agent.model.get_config()["guardrail_id"], "gr-1")

    def test_an_unguarded_agent_is_the_negative_control(self):
        agent = build_agent()
        self.assertIsNone(agent.model.get_config().get("guardrail_id"))
        self.assertEqual(agent.model.get_config()["model_id"], MODEL_ID)

    def test_the_guardrail_and_the_gateway_tools_hold_at_the_same_time(self):
        agent = build_agent(model=guarded_model("gr-1", MODEL_ID), extra_tools=real_gateway_tools())

        self.assertEqual(agent.model.get_config()["guardrail_id"], "gr-1")
        registered = sorted(agent.tool_registry.get_all_tools_config())
        self.assertEqual(registered, ["search_faq", LOOKUP, PROCESS_RETURN])
        # Stage 1's supersede rule still holds: no bare lookup_order stub left.
        self.assertNotIn("lookup_order", registered)
        self.assertNotIn("process_return", registered)

    def test_memory_still_requires_both_ids(self):
        # Stage 1's assertion, restated on the moved module: the guardrail did not
        # change the memory contract.
        with self.assertRaises(ValueError):
            build_agent(memory_id="mem-1")


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


class GuardrailDeletionTest(unittest.TestCase):
    """R11: the billable resource has to actually go."""

    def setUp(self):
        quiet(self)
        self.bedrock = FakeBedrockClient(ready_after=0, delete_lag=2)
        patch_boto3(self, guardrail_module, bedrock=self.bedrock)
        self.clock = patch_clock(self, guardrail_module)
        self.guardrail_id = self.bedrock.seed(GUARDRAIL_NAME)
        self.bedrock.calls.clear()

    def test_the_delete_is_confirmed_by_a_get_rather_than_by_the_call_returning(self):
        delete_guardrail(self.guardrail_id, "us-east-1")

        operations = [name for name, _ in self.bedrock.calls]
        self.assertEqual(operations.count("delete_guardrail"), 1)
        # delete_lag=2: two get_guardrail calls still succeeded after the delete.
        self.assertGreaterEqual(operations.count("get_guardrail"), 3)
        self.assertIn(self.guardrail_id, self.bedrock.deleted)

    def test_the_delete_names_no_version_so_every_version_goes(self):
        delete_guardrail(self.guardrail_id, "us-east-1")
        params = [p for name, p in self.bedrock.calls if name == "delete_guardrail"][0]
        self.assertEqual(params, {"guardrailIdentifier": self.guardrail_id})
        self.assertTrue(valid_request("bedrock", "DeleteGuardrail", params))

    def test_an_in_use_guardrail_is_retried_rather_than_abandoned(self):
        self.bedrock.in_use_failures = 2

        delete_guardrail(self.guardrail_id, "us-east-1")

        attempts = [1 for name, _ in self.bedrock.calls if name == "delete_guardrail"]
        self.assertEqual(len(attempts), 3)
        self.assertIn(self.guardrail_id, self.bedrock.deleted)

    def test_an_already_deleted_guardrail_is_not_an_error(self):
        delete_guardrail(self.guardrail_id, "us-east-1")
        delete_guardrail(self.guardrail_id, "us-east-1")  # must not raise

    def test_a_guardrail_that_never_goes_raises_rather_than_reporting_success(self):
        # Silence here would be the R11 failure: a billable resource reported gone.
        self.bedrock.delete_lag = 10_000
        with self.assertRaises(TimeoutError):
            delete_guardrail(self.guardrail_id, "us-east-1", timeout=30)


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
        self.replace(run_walkthrough, "delete_guardrail", self._recorder("guardrail"))
        self.replace(run_walkthrough, "delete_policies", self._recorder("policies"))
        self.replace(
            run_walkthrough, "delete_policy_engine", self._recorder("policy engine")
        )
        # The target and gateway deletes stay real, because their retry and
        # confirm behaviour is part of what is under test here; they are only
        # instrumented so that one ordered list covers all six steps.
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
            "guardrail_id": "gr-1",
            "policy_engine_id": "policy-engine-1",
            "policy_ids": ["policy-1", "policy-2"],
        }
        kwargs.update(overrides)
        return run_walkthrough.teardown(**kwargs)

    def test_every_resource_stage_2_creates_is_deleted_in_dependency_order(self):
        self.full_teardown()

        self.assertEqual(
            self.attempted,
            ["target", "gateway", "memory", "guardrail", "policies", "policy engine"],
        )
        self.assertEqual(self.memory, ["mem-abc123"])
        self.assertTrue(self.control.gateway_deleted)

    def test_a_failing_gateway_delete_does_not_skip_the_guardrail(self):
        # Exactly the run-2 failure R11 is about: a fail-fast teardown orphans
        # everything after the first error, and the guardrail is billable.
        self.control.validation_failures = 10_000

        with self.assertRaises(RuntimeError) as caught:
            self.full_teardown()

        self.assertEqual(
            self.attempted,
            ["target", "gateway", "memory", "guardrail", "policies", "policy engine"],
        )
        self.assertIn("gateway", str(caught.exception))
        self.assertIn("ValidationException", str(caught.exception))

    def test_a_failing_guardrail_delete_still_reports_and_still_deletes_the_rest(self):
        self.replace(
            run_walkthrough,
            "delete_guardrail",
            self._recorder("guardrail", error=RuntimeError("guardrail is in use")),
        )

        with self.assertRaises(RuntimeError) as caught:
            self.full_teardown()

        self.assertIn("guardrail", str(caught.exception))
        self.assertEqual(self.attempted[-2:], ["policies", "policy engine"])

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

    def test_resources_that_were_never_created_are_not_deleted(self):
        # A stage 2 that died after the guardrail has no policy ids, and a delete
        # against an empty id would fail and mask the real failure.
        self.full_teardown(policy_ids=(), policy_engine_id="")

        self.assertEqual(self.attempted, ["target", "gateway", "memory", "guardrail"])

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
            ["target", "gateway", "memory", "guardrail", "policies", "policy engine"],
        )

    def test_a_gateway_that_never_disappears_times_out(self):
        self.control.delete_lag = 10_000
        with self.assertRaises(RuntimeError) as caught:
            self.full_teardown()
        self.assertIn("TimeoutError", str(caught.exception))


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

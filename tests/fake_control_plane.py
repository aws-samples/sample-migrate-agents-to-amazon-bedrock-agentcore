# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Fake bedrock and bedrock-agentcore-control clients for the stage-2 tests.

No AWS calls. Every request is recorded so a test can assert what would have been
sent, and the behaviours that are faked are the ones measured live during the
walkthrough rather than the happy path:

1. Creation is asynchronous. A new guardrail reports CREATING, and a new policy
   engine reports CREATING, for a settable number of Get calls before going
   READY / ACTIVE. Code that skips the waiter sees a resource it cannot use yet.
2. Deletion is asynchronous, and worse, a Get can still succeed after the delete
   returned. ``delete_lag`` controls how many Get calls succeed before
   ResourceNotFoundException, so a teardown that trusts the delete call rather
   than polling fails here.
3. DeleteGateway raised ValidationException while ListGatewayTargets already
   returned []. ``validation_failures`` reproduces that: the first N deletes
   raise ValidationException whatever the target list says.
4. ListGuardrails returns one entry per version, so a guardrail that exists only
   as DRAFT is a real state the create path has to handle.

FakeClock replaces the module-level ``time`` in the code under test, so waiter
loops finish instantly and a timeout is reachable without waiting for one.
"""


class FakeClock:
    """A monotonic clock that only moves when something sleeps on it."""

    def __init__(self, start=1_000.0):
        self.now = start
        self.slept = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


class ResourceNotFoundException(Exception):
    pass


class ResourceInUseException(Exception):
    pass


class ConflictException(Exception):
    pass


class ValidationException(Exception):
    pass


class _Exceptions:
    """The client.exceptions namespace botocore generates per service."""

    ResourceNotFoundException = ResourceNotFoundException
    ResourceInUseException = ResourceInUseException
    ConflictException = ConflictException
    ValidationException = ValidationException


class _Paginator:
    """Enough of a botocore paginator to page a list operation."""

    def __init__(self, key, items, page_size=2):
        self.key = key
        self.items = items
        self.page_size = page_size

    def paginate(self, **kwargs):
        items = self.items() if callable(self.items) else self.items
        if not items:
            yield {self.key: []}
            return
        for start in range(0, len(items), self.page_size):
            yield {self.key: items[start : start + self.page_size]}


class FakeBedrockClient:
    """The guardrail control plane: create, version, get, list, delete."""

    exceptions = _Exceptions

    def __init__(self, ready_after=1, delete_lag=1, in_use_failures=0):
        # Guardrail id -> {"name", "config", "versions": [str], "status_gets"}
        self.guardrails = {}
        self.ready_after = ready_after
        self.delete_lag = delete_lag
        self.in_use_failures = in_use_failures
        self.calls = []
        self.deleted = []
        self._next_id = 0
        self._get_counts = {}
        self._delete_gets = {}

    # -- helpers a test uses to set the starting state -------------------------

    def seed(self, name, versions=("1",), guardrail_id=None):
        """Pretend a guardrail already exists, optionally DRAFT-only."""
        guardrail_id = guardrail_id or self._make_id()
        self.guardrails[guardrail_id] = {
            "name": name,
            "config": {},
            "versions": ["DRAFT", *versions],
        }
        self._get_counts[guardrail_id] = self.ready_after
        return guardrail_id

    def _make_id(self):
        self._next_id += 1
        return f"gr-{self._next_id:012d}"

    # -- the API --------------------------------------------------------------

    def create_guardrail(self, **kwargs):
        self.calls.append(("create_guardrail", kwargs))
        guardrail_id = self._make_id()
        self.guardrails[guardrail_id] = {
            "name": kwargs["name"],
            "config": kwargs,
            "versions": ["DRAFT"],
        }
        self._get_counts[guardrail_id] = 0
        return {"guardrailId": guardrail_id, "version": "DRAFT"}

    def create_guardrail_version(self, **kwargs):
        self.calls.append(("create_guardrail_version", kwargs))
        guardrail = self._require(kwargs["guardrailIdentifier"])
        numbered = [v for v in guardrail["versions"] if v.isdigit()]
        version = str(max((int(v) for v in numbered), default=0) + 1)
        guardrail["versions"].append(version)
        # Versioning puts the guardrail back into a non-READY status.
        self._get_counts[kwargs["guardrailIdentifier"]] = 0
        return {"version": version}

    def get_guardrail(self, **kwargs):
        self.calls.append(("get_guardrail", kwargs))
        guardrail_id = kwargs["guardrailIdentifier"]
        if guardrail_id in self.deleted:
            remaining = self._delete_gets.get(guardrail_id, 0)
            if remaining <= 0:
                raise ResourceNotFoundException(f"{guardrail_id} not found")
            self._delete_gets[guardrail_id] = remaining - 1
            return {"guardrailId": guardrail_id, "status": "DELETING"}
        guardrail = self._require(guardrail_id)
        seen = self._get_counts[guardrail_id]
        self._get_counts[guardrail_id] = seen + 1
        status = "READY" if seen >= self.ready_after else "CREATING"
        return {"guardrailId": guardrail_id, "status": status, **guardrail["config"]}

    def get_paginator(self, name):
        if name != "list_guardrails":
            raise ValueError(f"Unfaked paginator: {name}")
        return _Paginator("guardrails", self._guardrail_summaries)

    def _guardrail_summaries(self):
        # One entry per version, which is what makes DRAFT-only detectable.
        return [
            {"id": guardrail_id, "name": guardrail["name"], "version": version}
            for guardrail_id, guardrail in self.guardrails.items()
            if guardrail_id not in self.deleted
            for version in guardrail["versions"]
        ]

    def delete_guardrail(self, **kwargs):
        self.calls.append(("delete_guardrail", kwargs))
        guardrail_id = kwargs["guardrailIdentifier"]
        if guardrail_id in self.deleted:
            raise ResourceNotFoundException(f"{guardrail_id} not found")
        if guardrail_id not in self.guardrails:
            raise ResourceNotFoundException(f"{guardrail_id} not found")
        if self.in_use_failures > 0:
            self.in_use_failures -= 1
            raise ResourceInUseException(f"{guardrail_id} is in use")
        self.deleted.append(guardrail_id)
        self._delete_gets[guardrail_id] = self.delete_lag
        return {}

    def _require(self, guardrail_id):
        if guardrail_id not in self.guardrails or guardrail_id in self.deleted:
            raise ResourceNotFoundException(f"{guardrail_id} not found")
        return self.guardrails[guardrail_id]


class FakeAgentCoreControlClient:
    """The AgentCore control plane: gateways, targets, policy engines, policies."""

    exceptions = _Exceptions

    def __init__(
        self,
        gateway=None,
        active_after=1,
        delete_lag=1,
        validation_failures=0,
        engine_validation_failures=0,
    ):
        self.gateway = gateway or {
            "gatewayId": "gw-abc123",
            "gatewayArn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/gw-abc123",
            "name": "MigratedAgentGateway",
            "roleArn": "arn:aws:iam::123456789012:role/GatewayExecutionRole",
            "authorizerType": "AWS_IAM",
            "protocolType": "MCP",
        }
        self.engines = {}
        self.policies = {}
        self.targets = {"tgt-abc123": {"gatewayId": self.gateway["gatewayId"]}}
        self.active_after = active_after
        self.delete_lag = delete_lag
        self.validation_failures = validation_failures
        self.engine_validation_failures = engine_validation_failures
        self.calls = []
        self.deleted = []
        self._next_id = 0
        self._get_counts = {}
        self._delete_gets = {}
        self.gateway_deleted = False

    def _make_id(self, prefix):
        self._next_id += 1
        return f"{prefix}-{self._next_id:012d}"

    # -- gateway and target ---------------------------------------------------

    def get_gateway(self, **kwargs):
        self.calls.append(("get_gateway", kwargs))
        if self.gateway_deleted:
            remaining = self._delete_gets.get("gateway", 0)
            if remaining <= 0:
                raise ResourceNotFoundException("gateway not found")
            self._delete_gets["gateway"] = remaining - 1
            return {**self.gateway, "status": "DELETING"}
        return {**self.gateway, "status": "READY"}

    def update_gateway(self, **kwargs):
        self.calls.append(("update_gateway", kwargs))
        self.gateway = {**self.gateway, **kwargs}
        return {**self.gateway}

    def delete_gateway(self, **kwargs):
        self.calls.append(("delete_gateway", kwargs))
        if self.gateway_deleted:
            raise ResourceNotFoundException("gateway not found")
        if self.validation_failures > 0:
            self.validation_failures -= 1
            # Measured live: this is raised while ListGatewayTargets returns [].
            raise ValidationException(
                f"Gateway {kwargs['gatewayIdentifier']} has targets associated with it"
            )
        self.gateway_deleted = True
        self._delete_gets["gateway"] = self.delete_lag
        return {}

    def get_gateway_target(self, **kwargs):
        self.calls.append(("get_gateway_target", kwargs))
        target_id = kwargs["targetId"]
        if target_id in self.deleted:
            remaining = self._delete_gets.get(target_id, 0)
            if remaining <= 0:
                raise ResourceNotFoundException(f"{target_id} not found")
            self._delete_gets[target_id] = remaining - 1
            return {"targetId": target_id, "status": "DELETING"}
        if target_id not in self.targets:
            raise ResourceNotFoundException(f"{target_id} not found")
        return {"targetId": target_id, "status": "READY"}

    def delete_gateway_target(self, **kwargs):
        self.calls.append(("delete_gateway_target", kwargs))
        target_id = kwargs["targetId"]
        if target_id in self.deleted or target_id not in self.targets:
            raise ResourceNotFoundException(f"{target_id} not found")
        self.deleted.append(target_id)
        self._delete_gets[target_id] = self.delete_lag
        return {}

    # -- policy engine and policies -------------------------------------------

    def create_policy_engine(self, **kwargs):
        self.calls.append(("create_policy_engine", kwargs))
        engine_id = self._make_id("policy-engine")
        self.engines[engine_id] = {
            "name": kwargs["name"],
            "policyEngineArn": (
                "arn:aws:bedrock-agentcore:us-east-1:123456789012:"
                f"policy-engine/{engine_id}"
            ),
        }
        self._get_counts[engine_id] = 0
        return {"policyEngineId": engine_id, **self.engines[engine_id]}

    def get_policy_engine(self, **kwargs):
        self.calls.append(("get_policy_engine", kwargs))
        engine_id = kwargs["policyEngineId"]
        if engine_id in self.deleted:
            remaining = self._delete_gets.get(engine_id, 0)
            if remaining <= 0:
                raise ResourceNotFoundException(f"{engine_id} not found")
            self._delete_gets[engine_id] = remaining - 1
            return {"policyEngineId": engine_id, "status": "DELETING"}
        if engine_id not in self.engines:
            raise ResourceNotFoundException(f"{engine_id} not found")
        seen = self._get_counts[engine_id]
        self._get_counts[engine_id] = seen + 1
        status = "ACTIVE" if seen >= self.active_after else "CREATING"
        return {"policyEngineId": engine_id, "status": status, **self.engines[engine_id]}

    def get_paginator(self, name):
        if name != "list_policy_engines":
            raise ValueError(f"Unfaked paginator: {name}")
        return _Paginator("policyEngines", self._engine_summaries)

    def _engine_summaries(self):
        return [
            {"policyEngineId": engine_id, "name": engine["name"]}
            for engine_id, engine in self.engines.items()
            if engine_id not in self.deleted
        ]

    def create_policy(self, **kwargs):
        self.calls.append(("create_policy", kwargs))
        policy_id = self._make_id("policy")
        self.policies[policy_id] = kwargs
        self._get_counts[policy_id] = 0
        return {"policyId": policy_id}

    def get_policy(self, **kwargs):
        self.calls.append(("get_policy", kwargs))
        policy_id = kwargs["policyId"]
        if policy_id in self.deleted:
            remaining = self._delete_gets.get(policy_id, 0)
            if remaining <= 0:
                raise ResourceNotFoundException(f"{policy_id} not found")
            self._delete_gets[policy_id] = remaining - 1
            return {"policyId": policy_id, "status": "DELETING"}
        if policy_id not in self.policies:
            raise ResourceNotFoundException(f"{policy_id} not found")
        seen = self._get_counts[policy_id]
        self._get_counts[policy_id] = seen + 1
        status = "ACTIVE" if seen >= self.active_after else "CREATING"
        return {"policyId": policy_id, "status": status}

    def delete_policy(self, **kwargs):
        self.calls.append(("delete_policy", kwargs))
        policy_id = kwargs["policyId"]
        if policy_id in self.deleted or policy_id not in self.policies:
            raise ResourceNotFoundException(f"{policy_id} not found")
        self.deleted.append(policy_id)
        self._delete_gets[policy_id] = self.delete_lag
        return {}

    def delete_policy_engine(self, **kwargs):
        self.calls.append(("delete_policy_engine", kwargs))
        engine_id = kwargs["policyEngineId"]
        if engine_id in self.deleted:
            raise ResourceNotFoundException(f"{engine_id} not found")
        if engine_id not in self.engines:
            raise ResourceNotFoundException(f"{engine_id} not found")
        if self.engine_validation_failures > 0:
            self.engine_validation_failures -= 1
            raise ValidationException(f"{engine_id} still has policies attached")
        self.deleted.append(engine_id)
        self._delete_gets[engine_id] = self.delete_lag
        return {}


class FakeSTSClient:
    """GetCallerIdentity only, returning an assumed-role ARN with a session name."""

    def __init__(self, arn=None):
        self.arn = arn or (
            "arn:aws:sts::123456789012:assumed-role/SupportAgentRole/session-1"
        )

    def get_caller_identity(self):
        return {"Arn": self.arn, "Account": "123456789012", "UserId": "AROAEXAMPLE"}


class FakeBoto3:
    """A stand-in for the ``boto3`` module the code under test calls client() on."""

    def __init__(self, **clients):
        self.clients = clients
        self.requested = []

    def client(self, service_name, region_name=None, **kwargs):
        self.requested.append((service_name, region_name))
        if service_name not in self.clients:
            raise AssertionError(f"Unexpected AWS client requested: {service_name}")
        return self.clients[service_name]

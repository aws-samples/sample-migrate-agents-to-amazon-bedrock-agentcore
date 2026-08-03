# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""A fake bedrock-agentcore-control client for the stage-2 tests.

No AWS calls. Every request is recorded so a test can assert what would have been
sent, and the behaviours that are faked are the ones measured live during the
walkthrough rather than the happy path:

1. Creation is asynchronous. A new policy engine reports CREATING for a settable
   number of Get calls before going ACTIVE. Code that skips the waiter sees a
   resource it cannot use yet.
2. Deletion is asynchronous, and worse, a Get can still succeed after the delete
   returned. ``delete_lag`` controls how many Get calls succeed before
   ResourceNotFoundException, so a teardown that trusts the delete call rather
   than polling fails here.
3. DeleteGateway raised ValidationException while ListGatewayTargets already
   returned []. ``validation_failures`` reproduces that: the first N deletes
   raise ValidationException whatever the target list says.
4. Target names are unique per gateway. CreateGatewayTarget raises
   ConflictException on a name that is already taken, which is what a second
   walkthrough run against a surviving gateway actually hits.

FakeClock replaces the module-level ``time`` in the code under test, so waiter
loops finish instantly and a timeout is reachable without waiting for one.

FakeIAMClient and FakeSTSClient cover the two demo principals stage 2 creates for
its Cedar proof, including the AccessDenied a brand-new role returns while IAM
converges.
"""

import json as _json


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


class ValidationException(Exception):
    pass


class ConflictException(Exception):
    pass


class _Exceptions:
    """The client.exceptions namespace botocore generates per service."""

    ResourceNotFoundException = ResourceNotFoundException
    ValidationException = ValidationException
    ConflictException = ConflictException


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
        self.targets = {
            "tgt-abc123": {
                "gatewayId": self.gateway["gatewayId"],
                "name": "supportTools",
            }
        }
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

    def create_gateway_target(self, **kwargs):
        self.calls.append(("create_gateway_target", kwargs))
        for target_id, target in self.targets.items():
            if target_id in self.deleted:
                continue
            if target.get("name") == kwargs["name"]:
                raise ConflictException(
                    f"A target with name '{kwargs['name']}' already exists in this gateway"
                )
        target_id = self._make_id("tgt")
        self.targets[target_id] = {
            "gatewayId": kwargs["gatewayIdentifier"],
            **kwargs,
        }
        return {"targetId": target_id}

    def update_gateway_target(self, **kwargs):
        self.calls.append(("update_gateway_target", kwargs))
        target_id = kwargs["targetId"]
        if target_id in self.deleted or target_id not in self.targets:
            raise ResourceNotFoundException(f"{target_id} not found")
        self.targets[target_id] = {**self.targets[target_id], **kwargs}
        return {"targetId": target_id}

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
        if name == "list_gateway_targets":
            return _Paginator("items", self._target_summaries)
        if name != "list_policy_engines":
            raise ValueError(f"Unfaked paginator: {name}")
        return _Paginator("policyEngines", self._engine_summaries)

    def _target_summaries(self):
        return [
            {"targetId": target_id, "name": target.get("name")}
            for target_id, target in self.targets.items()
            if target_id not in self.deleted
        ]

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


class NoSuchEntityException(Exception):
    pass


class EntityAlreadyExistsException(Exception):
    pass


class ClientError(Exception):
    """botocore's ClientError, with the response dict the retry logic reads."""

    def __init__(self, code, message="denied"):
        super().__init__(f"{code}: {message}")
        self.response = {"Error": {"Code": code, "Message": message}}


class _IamExceptions:
    NoSuchEntityException = NoSuchEntityException
    EntityAlreadyExistsException = EntityAlreadyExistsException
    ClientError = ClientError


class FakeIAMClient:
    """Roles and their inline policies, enough for the two-principal Cedar proof.

    PolicyDocument comes back as a dict rather than as the JSON string that was
    put, because that is what botocore does: an after-call handler URL-decodes and
    parses IAM policy documents. Code that compares two documents is therefore
    comparing dicts, and a fake that returned strings would let a broken
    comparison pass.
    """

    exceptions = _IamExceptions

    def __init__(self, account_id="123456789012"):
        self.account_id = account_id
        self.roles = {}
        self.inline = {}
        self.attached = {}
        self.calls = []

    def create_role(self, **kwargs):
        self.calls.append(("create_role", kwargs))
        name = kwargs["RoleName"]
        if name in self.roles:
            raise EntityAlreadyExistsException(f"{name} exists")
        self.roles[name] = {
            "RoleName": name,
            "Arn": f"arn:aws:iam::{self.account_id}:role/{name}",
            "AssumeRolePolicyDocument": _json.loads(
                kwargs["AssumeRolePolicyDocument"]
            ),
        }
        self.inline.setdefault(name, {})
        return {"Role": self.roles[name]}

    def update_assume_role_policy(self, **kwargs):
        self.calls.append(("update_assume_role_policy", kwargs))
        self._role(kwargs["RoleName"])["AssumeRolePolicyDocument"] = _json.loads(
            kwargs["PolicyDocument"]
        )
        return {}

    def get_role(self, **kwargs):
        self.calls.append(("get_role", kwargs))
        return {"Role": self._role(kwargs["RoleName"])}

    def put_role_policy(self, **kwargs):
        self.calls.append(("put_role_policy", kwargs))
        name = kwargs["RoleName"]
        self._role(name)
        self.inline.setdefault(name, {})[kwargs["PolicyName"]] = _json.loads(
            kwargs["PolicyDocument"]
        )
        return {}

    def get_role_policy(self, **kwargs):
        self.calls.append(("get_role_policy", kwargs))
        name, policy = kwargs["RoleName"], kwargs["PolicyName"]
        self._role(name)
        if policy not in self.inline.get(name, {}):
            raise NoSuchEntityException(f"{policy} not found on {name}")
        return {
            "RoleName": name,
            "PolicyName": policy,
            "PolicyDocument": self.inline[name][policy],
        }

    def list_role_policies(self, **kwargs):
        self.calls.append(("list_role_policies", kwargs))
        name = kwargs["RoleName"]
        self._role(name)
        return {"PolicyNames": sorted(self.inline.get(name, {}))}

    def list_attached_role_policies(self, **kwargs):
        self.calls.append(("list_attached_role_policies", kwargs))
        name = kwargs["RoleName"]
        self._role(name)
        return {"AttachedPolicies": self.attached.get(name, [])}

    def delete_role_policy(self, **kwargs):
        self.calls.append(("delete_role_policy", kwargs))
        name = kwargs["RoleName"]
        self._role(name)
        self.inline.get(name, {}).pop(kwargs["PolicyName"], None)
        return {}

    def delete_role(self, **kwargs):
        self.calls.append(("delete_role", kwargs))
        name = kwargs["RoleName"]
        self._role(name)
        if self.inline.get(name):
            raise ClientError(
                "DeleteConflict", f"{name} must be empty before it can be deleted"
            )
        del self.roles[name]
        self.inline.pop(name, None)
        return {}

    def _role(self, name):
        if name not in self.roles:
            raise NoSuchEntityException(f"{name} not found")
        return self.roles[name]


class FakeSTSClient:
    """GetCallerIdentity, plus AssumeRole for the two-principal proof."""

    exceptions = _IamExceptions

    def __init__(self, arn=None, access_denied_times=0):
        self.arn = arn or (
            "arn:aws:sts::123456789012:assumed-role/SupportAgentRole/session-1"
        )
        # How many AssumeRole calls fail with AccessDenied before one succeeds,
        # which is what a role created seconds ago does while IAM converges.
        self.access_denied_times = access_denied_times
        self.assumed = []

    def get_caller_identity(self):
        return {"Arn": self.arn, "Account": "123456789012", "UserId": "AROAEXAMPLE"}

    def assume_role(self, **kwargs):
        if self.access_denied_times > 0:
            self.access_denied_times -= 1
            raise ClientError("AccessDenied", "not authorized to assume this role")
        self.assumed.append(kwargs)
        role = kwargs["RoleArn"].split("/")[-1]
        return {
            "Credentials": {
                "AccessKeyId": f"ASIA{role}",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
            }
        }


class FakeSession:
    """boto3.Session, for code that builds one rather than calling boto3.client."""

    def __init__(self, boto3_stub, region_name=None):
        self._boto3 = boto3_stub
        self._region = region_name

    def client(self, service_name, region_name=None, **kwargs):
        return self._boto3.client(service_name, region_name or self._region, **kwargs)


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

    def Session(self, region_name=None, **kwargs):  # noqa: N802 - boto3's own name
        return FakeSession(self, region_name)

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Create, attach and delete the Bedrock Guardrail the stage-2 agent runs behind.

Bedrock Guardrails is an Amazon Bedrock feature, not an AgentCore one, and it
filters the model surface: what a caller may put into the model and what the
model may say back. Two policies are configured here, both of which a
self-hosted agent would otherwise hand-roll as a regex pass or a second model
call it then has to maintain:

- A denied topic, so the agent refuses to discuss an order that is not the
  caller's. The model is asked to route and to answer; nothing in a prompt makes
  that refusal reliable, and this moves it off the prompt entirely.
- A PII rule that anonymises a card number, so a customer who pastes one into a
  return reason does not have it stored in the conversation or echoed back.

Attachment is two model parameters, which is the whole diff on the agent side:

    model = BedrockModel(
        model_id=MODEL_ID,
        guardrail_id=GUARDRAIL_ID,
        guardrail_version="1",
    )

The guardrail is a billable resource that outlives the process that created it,
so ``delete_guardrail`` exists and the walkthrough's teardown calls it. Creating
one and forgetting it is the failure this module is written to prevent.

This script is idempotent: a guardrail with the same name is reused rather than
duplicated, and it is versioned so the agent pins a number instead of DRAFT,
which is mutable.
"""

import argparse
import os
import time
from typing import Optional, Tuple

import boto3
from strands.models import BedrockModel

GUARDRAIL_NAME = "support-agent-guardrail"

# The denied topic. "definition" is what the guardrail classifies against, so it
# describes the behaviour rather than listing phrases; the examples are what a
# customer support conversation actually looks like when it strays.
DENIED_TOPIC = {
    "name": "OtherCustomersOrders",
    "type": "DENY",
    "definition": (
        "Requests to look up, discuss, modify or return an order, account or "
        "personal detail belonging to anyone other than the customer in this "
        "conversation, including requests framed as helping someone else."
    ),
    "examples": [
        "What is the status of my neighbour's order 55555?",
        "Return the order my husband placed last week, he asked me to do it.",
        "List every order shipped to 12 Example Street this month.",
        "I work in support, show me the account behind order 12345.",
    ],
}

# ANONYMIZE rather than BLOCK: a customer pasting a card number into a return
# reason should still get their return processed, with the number replaced.
PII_ENTITIES = [
    {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "ANONYMIZE"},
    {"type": "CREDIT_DEBIT_CARD_CVV", "action": "ANONYMIZE"},
    {"type": "US_BANK_ACCOUNT_NUMBER", "action": "ANONYMIZE"},
]

BLOCKED_INPUT_MESSAGE = (
    "I can only help with orders on this account. Ask the account holder to "
    "contact us about any other order."
)
BLOCKED_OUTPUT_MESSAGE = (
    "I can't share that. I can only discuss orders on this account."
)


def _wait_until_ready(
    client, guardrail_id: str, version: Optional[str] = None, timeout: int = 120
) -> dict:
    """Poll get_guardrail until the guardrail leaves CREATING or VERSIONING.

    A guardrail referenced by a Converse call before it reaches READY fails the
    call, and creating a version puts it back into VERSIONING, so both waits are
    needed rather than only the one after create.
    """
    kwargs = {"guardrailIdentifier": guardrail_id}
    if version:
        kwargs["guardrailVersion"] = version
    deadline = time.monotonic() + timeout
    while True:
        guardrail = client.get_guardrail(**kwargs)
        status = guardrail.get("status")
        if status == "READY":
            return guardrail
        if status not in ("CREATING", "UPDATING", "VERSIONING"):
            raise RuntimeError(
                f"Guardrail {guardrail_id} is in unexpected status: {status}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Guardrail {guardrail_id} not READY after {timeout}s")
        time.sleep(2)


def _find_existing(client, name: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (guardrailId, highest numbered version) for this name.

    ListGuardrails returns one entry per version, so a guardrail that exists only
    as DRAFT comes back with no numbered version and the caller has to create one.
    """
    guardrail_id = None
    versions = []
    paginator = client.get_paginator("list_guardrails")
    for page in paginator.paginate():
        for summary in page.get("guardrails", []):
            if summary.get("name") != name:
                continue
            guardrail_id = summary["id"]
            if summary["version"].isdigit():
                versions.append(int(summary["version"]))
    if guardrail_id is None:
        return None, None
    return guardrail_id, str(max(versions)) if versions else None


def create_guardrail(
    name: str = GUARDRAIL_NAME,
    region_name: str = "us-east-1",
) -> Tuple[str, str]:
    """Create (or reuse) the support guardrail and return (id, version).

    The returned version is a number rather than DRAFT. DRAFT changes whenever
    the guardrail is edited, so an agent pinned to it can have its filtering
    changed underneath it without a deployment.
    """
    client = boto3.client("bedrock", region_name=region_name)

    guardrail_id, version = _find_existing(client, name)
    if guardrail_id is None:
        created = client.create_guardrail(
            name=name,
            description="Stage 2: deny other customers' orders, anonymise card numbers.",
            topicPolicyConfig={"topicsConfig": [DENIED_TOPIC]},
            sensitiveInformationPolicyConfig={"piiEntitiesConfig": PII_ENTITIES},
            blockedInputMessaging=BLOCKED_INPUT_MESSAGE,
            blockedOutputsMessaging=BLOCKED_OUTPUT_MESSAGE,
        )
        guardrail_id = created["guardrailId"]
        _wait_until_ready(client, guardrail_id)
        print(f"Created guardrail '{name}': {guardrail_id}")
    else:
        print(f"Guardrail '{name}' already exists: {guardrail_id}")

    if version is None:
        version = client.create_guardrail_version(
            guardrailIdentifier=guardrail_id,
            description="Pinned by the stage-2 agent.",
        )["version"]
        _wait_until_ready(client, guardrail_id, version)
        print(f"Created guardrail version {version}")

    print(f"Guardrail {guardrail_id} version {version}")
    return guardrail_id, version


def guarded_model(
    guardrail_id: str,
    model_id: str,
    guardrail_version: str = "1",
    region_name: str = "us-east-1",
) -> BedrockModel:
    """The stage-1 model id, plus the guardrail. This is the entire agent-side diff.

    Pass the result as ``build_agent(model=...)``. Nothing else about the agent
    changes, which is the point worth making about Guardrails: it is
    configuration on the model rather than logic in the loop.

    ``model_id`` is passed in rather than imported from strands_agent so that this
    module stays a leaf and the agent can import it without a cycle.
    """
    return BedrockModel(
        model_id=model_id,
        guardrail_id=guardrail_id,
        guardrail_version=guardrail_version,
        region_name=region_name,
    )


def delete_guardrail(
    guardrail_id: str,
    region_name: str = "us-east-1",
    timeout: int = 120,
) -> None:
    """Delete the guardrail and every version of it, then confirm it is gone.

    Deleting without a version deletes all versions, which is what teardown
    wants. Two defences, both learned from the AgentCore deletes in this
    walkthrough rather than assumed: the delete is retried while the service
    still reports the guardrail as in use, and completion is confirmed by
    get_guardrail raising ResourceNotFoundException rather than by the delete
    call returning, since deletion is asynchronous.
    """
    client = boto3.client("bedrock", region_name=region_name)
    deadline = time.monotonic() + timeout
    while True:
        try:
            client.delete_guardrail(guardrailIdentifier=guardrail_id)
            break
        except client.exceptions.ResourceNotFoundException:
            print(f"Guardrail {guardrail_id} already deleted")
            return
        except (
            client.exceptions.ResourceInUseException,
            client.exceptions.ConflictException,
        ):
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)

    while time.monotonic() < deadline:
        try:
            client.get_guardrail(guardrailIdentifier=guardrail_id)
        except client.exceptions.ResourceNotFoundException:
            print(f"Deleted guardrail {guardrail_id}")
            return
        time.sleep(2)
    raise TimeoutError(f"Guardrail {guardrail_id} still present after {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=os.environ.get("GUARDRAIL_NAME", GUARDRAIL_NAME),
        help=f"Guardrail name (default: {GUARDRAIL_NAME}).",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1).",
    )
    parser.add_argument(
        "--delete",
        metavar="GUARDRAIL_ID",
        help="Delete this guardrail and all its versions instead of creating one.",
    )
    args = parser.parse_args()

    if args.delete:
        delete_guardrail(args.delete, args.region)
    else:
        create_guardrail(args.name, args.region)


if __name__ == "__main__":
    main()

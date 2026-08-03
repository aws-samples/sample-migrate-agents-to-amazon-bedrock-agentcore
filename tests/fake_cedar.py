# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""A Cedar evaluator for the subset support_tools.cedar uses, and nothing more.

Why this exists: Policy in AgentCore evaluates Cedar inside the Gateway, and
there is no synchronous authorization API to ask for a decision. Offline, the
only way to assert that the shipped rules deny process_return for the read-only
caller is to evaluate the shipped rules. So this reimplements the four Cedar
behaviours the file depends on:

1. default-deny — no matching permit is a deny,
2. forbid-wins — a matching forbid beats any permit,
3. scope matching on ``principal is T``, ``principal == T::"id"``,
   ``action == ...``, ``action in [...]`` and ``resource == ...``,
4. nothing else.

What it is not: Cedar. It has no schema validation, no entity hierarchy, no
sets besides an action list, no ``unless``, no ``when``, and no operators besides
``==``. That matters for how it fails — anything outside the subset raises rather
than being ignored, so a rule edited into a form this cannot evaluate breaks the
test instead of silently passing it. Real enforcement is only observable against a
live gateway.
"""

import re
from typing import List, NamedTuple, Optional, Tuple

# One statement: an effect, a parenthesised scope, an optional when block, a ";".
# The when block is captured only so that a conditioned rule is refused loudly
# rather than read as unconditional.
_STATEMENT = re.compile(
    r"(?P<effect>permit|forbid)\s*\((?P<scope>[^)]*)\)\s*"
    r"(?:when\s*\{(?P<when>[^}]*)\}\s*)?;",
    re.DOTALL,
)
_COMMENT = re.compile(r"//[^\n]*")
# AgentCore::IamEntity::"arn:aws:..." -> type "AgentCore::IamEntity", id the ARN.
_EQUALS = re.compile(r'^(principal|action|resource)\s*==\s*([A-Za-z:]+?)::"([^"]*)"$')
_IS = re.compile(r"^(principal|resource)\s+is\s+([A-Za-z][A-Za-z:]*)$")
_ACTION_IN = re.compile(r"^action\s+in\s+\[(?P<items>.*)\]$", re.DOTALL)
_ACTION_ITEM = re.compile(r'^AgentCore::Action::"([^"]*)"$')


class Entity(NamedTuple):
    """A Cedar entity: a type name and an id. IamEntity carries no tags."""

    type: str
    id: str


class Statement(NamedTuple):
    effect: str
    principal_type: Optional[str]
    principal_id: Optional[str]
    actions: Tuple[str, ...]
    resource_id: Optional[str]


def parse(text: str) -> List[Statement]:
    """Parse Cedar text into statements, raising on anything outside the subset."""
    body = _COMMENT.sub("", text)
    statements = []
    for match in _STATEMENT.finditer(body):
        if (match.group("when") or "").strip():
            raise ValueError(f"Unsupported Cedar condition: {match.group('when')!r}")
        statements.append(
            Statement(match.group("effect"), *_parse_scope(match.group("scope")))
        )
    if not statements:
        raise ValueError("No permit or forbid statement found")
    leftover = _STATEMENT.sub("", body).strip()
    if leftover:
        raise ValueError(f"Unparsed Cedar text: {leftover!r}")
    return statements


def _split_clauses(scope: str) -> List[str]:
    """Split the scope on its top-level commas.

    ``action in [A, B]`` contains a comma of its own, so splitting the whole scope
    on "," would cut the action list in half and lose one of the two tools.
    """
    clauses, depth, current = [], 0, ""
    for character in scope:
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        if character == "," and depth == 0:
            clauses.append(current)
            current = ""
            continue
        current += character
    clauses.append(current)
    return [c.strip() for c in clauses if c.strip()]


def _parse_scope(scope: str):
    """Read the three scope clauses. An unrecognised clause is an error."""
    principal_type = principal_id = resource_id = None
    actions: Tuple[str, ...] = ()
    for clause in _split_clauses(scope):
        equals = _EQUALS.match(clause)
        if equals:
            kind, entity_type, entity_id = equals.groups()
            if kind == "principal":
                principal_type, principal_id = entity_type, entity_id
            elif kind == "action":
                # AgentCore::Action::"supportTools___lookup_order" -> the tool.
                actions = (entity_id,)
            else:
                resource_id = entity_id
            continue
        action_in = _ACTION_IN.match(clause)
        if action_in:
            actions = _parse_action_list(action_in.group("items"))
            continue
        is_clause = _IS.match(clause)
        if is_clause and is_clause.group(1) == "principal":
            principal_type = is_clause.group(2)
            continue
        raise ValueError(f"Unsupported Cedar scope clause: {clause!r}")
    if not actions:
        # Every rule in this file names its actions; a rule without any would
        # apply far more widely than the file claims.
        raise ValueError(f"Scope names no action: {scope!r}")
    return principal_type, principal_id, actions, resource_id


def _parse_action_list(items: str) -> Tuple[str, ...]:
    """Read the entity ids out of an ``action in [ ... ]`` list."""
    actions = []
    for item in items.split(","):
        match = _ACTION_ITEM.match(item.strip())
        if not match:
            raise ValueError(f"Unsupported Cedar action list entry: {item.strip()!r}")
        actions.append(match.group(1))
    if not actions:
        raise ValueError("Empty Cedar action list")
    return tuple(actions)


def _matches(
    statement: Statement, principal: Entity, action: str, resource: Entity
) -> bool:
    if action not in statement.actions:
        return False
    if statement.resource_id is not None and statement.resource_id != resource.id:
        return False
    if statement.principal_type is not None and statement.principal_type != principal.type:
        return False
    if statement.principal_id is not None and statement.principal_id != principal.id:
        return False
    return True


def is_authorized(
    statements: List[Statement], principal: Entity, action: str, resource: Entity
) -> bool:
    """Cedar's decision: allow only on a matching permit with no matching forbid."""
    matched = [s for s in statements if _matches(s, principal, action, resource)]
    if any(s.effect == "forbid" for s in matched):
        return False
    return any(s.effect == "permit" for s in matched)

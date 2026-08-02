# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""A Cedar evaluator for the subset support_tools.cedar uses, and nothing more.

Why this exists: Policy in AgentCore evaluates Cedar inside the Gateway, and
there is no synchronous authorization API to ask for a decision. Offline, the
only way to assert that the shipped rules deny process_return for a non-owning
caller is to evaluate the shipped rules. So this reimplements the four Cedar
behaviours the file depends on:

1. default-deny — no matching permit is a deny,
2. forbid-wins — a matching forbid beats any permit,
3. scope matching on ``principal is T``, ``principal == T::"id"``,
   ``action == ...`` and ``resource == ...``,
4. a ``when`` condition over ``context.input.<name>``.

What it is not: Cedar. It has no schema validation, no entity hierarchy, no
``in``, no sets, no ``unless``, and no operators besides ``==`` and ``&&``. That
matters for how it fails — anything outside the subset raises rather than being
ignored, so a rule edited into a form this cannot evaluate breaks the test
instead of silently passing it. Real enforcement is only observable against a
live gateway.
"""

import re
from typing import Dict, List, NamedTuple, Optional, Tuple

# One statement: an effect, a parenthesised scope, an optional when block, a ";".
_STATEMENT = re.compile(
    r"(?P<effect>permit|forbid)\s*\((?P<scope>[^)]*)\)\s*"
    r"(?:when\s*\{(?P<when>[^}]*)\}\s*)?;",
    re.DOTALL,
)
_COMMENT = re.compile(r"//[^\n]*")
# AgentCore::IamEntity::"arn:aws:..." -> type "AgentCore::IamEntity", id the ARN.
_EQUALS = re.compile(r'^(principal|action|resource)\s*==\s*([A-Za-z:]+?)::"([^"]*)"$')
_IS = re.compile(r"^(principal|resource)\s+is\s+([A-Za-z][A-Za-z:]*)$")
_CONTEXT_EQUALS = re.compile(
    r'^context\.input\.([A-Za-z_][A-Za-z0-9_]*)\s*==\s*"([^"]*)"$'
)


class Entity(NamedTuple):
    """A Cedar entity: a type name and an id. IamEntity carries no tags."""

    type: str
    id: str


class Statement(NamedTuple):
    effect: str
    principal_type: Optional[str]
    principal_id: Optional[str]
    action: Optional[str]
    resource_id: Optional[str]
    conditions: Tuple[Tuple[str, str], ...]


def parse(text: str) -> List[Statement]:
    """Parse Cedar text into statements, raising on anything outside the subset."""
    body = _COMMENT.sub("", text)
    statements = []
    for match in _STATEMENT.finditer(body):
        statements.append(
            Statement(
                match.group("effect"),
                *_parse_scope(match.group("scope")),
                _parse_when(match.group("when")),
            )
        )
    if not statements:
        raise ValueError("No permit or forbid statement found")
    leftover = _STATEMENT.sub("", body).strip()
    if leftover:
        raise ValueError(f"Unparsed Cedar text: {leftover!r}")
    return statements


def _parse_scope(scope: str):
    """Read the three scope clauses. An unrecognised clause is an error."""
    principal_type = principal_id = action = resource_id = None
    for clause in (c.strip() for c in scope.split(",")):
        if not clause:
            continue
        equals = _EQUALS.match(clause)
        if equals:
            kind, entity_type, entity_id = equals.groups()
            if kind == "principal":
                principal_type, principal_id = entity_type, entity_id
            elif kind == "action":
                # AgentCore::Action::"supportTools___lookup_order" -> the tool.
                action = entity_id
            else:
                resource_id = entity_id
            continue
        is_clause = _IS.match(clause)
        if is_clause and is_clause.group(1) == "principal":
            principal_type = is_clause.group(2)
            continue
        raise ValueError(f"Unsupported Cedar scope clause: {clause!r}")
    if action is None:
        # Every rule in this file names one action; a rule without one would
        # apply far more widely than the file claims.
        raise ValueError(f"Scope names no action: {scope!r}")
    return principal_type, principal_id, action, resource_id


def _parse_when(when: Optional[str]) -> Tuple[Tuple[str, str], ...]:
    """Read a when block of context.input comparisons joined by &&."""
    if not when or not when.strip():
        return ()
    conditions = []
    for part in when.split("&&"):
        match = _CONTEXT_EQUALS.match(part.strip())
        if not match:
            raise ValueError(f"Unsupported Cedar condition: {part.strip()!r}")
        conditions.append((match.group(1), match.group(2)))
    return tuple(conditions)


def _matches(
    statement: Statement,
    principal: Entity,
    action: str,
    resource: Entity,
    tool_input: Dict[str, str],
) -> bool:
    if statement.action != action:
        return False
    if statement.resource_id is not None and statement.resource_id != resource.id:
        return False
    if statement.principal_type is not None and statement.principal_type != principal.type:
        return False
    if statement.principal_id is not None and statement.principal_id != principal.id:
        return False
    for name, expected in statement.conditions:
        # A missing attribute is an evaluation error in Cedar, and an errored
        # policy does not contribute a permit. Not matching is the same outcome.
        if tool_input.get(name) != expected:
            return False
    return True


def is_authorized(
    statements: List[Statement],
    principal: Entity,
    action: str,
    resource: Entity,
    tool_input: Optional[Dict[str, str]] = None,
) -> bool:
    """Cedar's decision: allow only on a matching permit with no matching forbid."""
    tool_input = tool_input or {}
    matched = [
        s for s in statements if _matches(s, principal, action, resource, tool_input)
    ]
    if any(s.effect == "forbid" for s in matched):
        return False
    return any(s.effect == "permit" for s in matched)

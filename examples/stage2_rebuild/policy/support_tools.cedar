// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
//
// Cedar rules for Policy in Amazon Bedrock AgentCore, evaluated on every tool
// call through the Gateway that stage 1 created. What they express is least
// privilege on tool calls: a read-only identity may look an order up, and only a
// privileged identity may start a return. That is authorization on a tool call,
// decided in the AgentCore data plane on a call the application cannot bypass,
// and no amount of prompt engineering substitutes for it.
//
// Cedar is default-deny and forbid-wins, so there is no forbid rule here. The
// read-only caller's deny on process_return is the absence of a matching permit.
//
// The principal type is AgentCore::IamEntity because the gateway in
// examples/gateway/create_gateway.py uses the AWS_IAM authorizer, whose principal
// carries an ARN and no tags. That choice bounds how finely authorization can be
// expressed here: an identity is all there is to match on, so these rules scope
// tools to callers and nothing narrower. A per-order ownership rule — this caller
// may return their own order and no one else's — has to compare a claim against
// the tool's argument, so it needs a CUSTOM_JWT gateway and the AgentCore
// Identity work this walkthrough deliberately leaves out.
//
// The three placeholders are substituted by attach_policy.render_policies at
// registration time. Cedar rejects a wildcard resource, so a policy naming
// specific actions has to name a specific Gateway ARN, and CreatePolicy rejects a
// gateway that does not exist with ResourceNotFoundException before it validates
// anything — measured, both times. So the gateway has to exist first.
//
// The parser in attach_policy.py splits this file on the "// === policy: NAME"
// markers and creates one AgentCore policy per block, so each rule can be
// created, inspected and deleted on its own.

// === policy: LookupOrderForReadOnlyCaller ===
// The read-only identity may read an order's status, and that is all it may do.
// There is no when clause: naming one principal ARN is the whole control, and the
// service's validator accepts it. What the validator rejects is a permit whose
// principal is a *type* — principal is AgentCore::IamEntity matches every
// authenticated caller, and it comes back as CREATE_FAILED with "Overly
// Permissive". The rejection is asynchronous: CreatePolicy returns a policyId
// first and the policy reaches its terminal status afterwards.
permit(
  principal == AgentCore::IamEntity::"<READ_ONLY_PRINCIPAL_ARN>",
  action == AgentCore::Action::"supportTools___lookup_order",
  resource == AgentCore::Gateway::"<GATEWAY_ARN>"
);

// === policy: LookupAndReturnForSupportAgent ===
// The privileged identity may also start a return. Nothing here permits
// process_return to anyone else, so the read-only caller's return request is
// denied by default while its lookup_order calls keep working — which is the
// whole point: the deny is specific to the tool, not to the caller.
permit(
  principal == AgentCore::IamEntity::"<SUPPORT_AGENT_PRINCIPAL_ARN>",
  action in [AgentCore::Action::"supportTools___lookup_order",
             AgentCore::Action::"supportTools___process_return"],
  resource == AgentCore::Gateway::"<GATEWAY_ARN>"
);

// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
//
// Cedar rules for Policy in Amazon Bedrock AgentCore, evaluated on every tool
// call through the Gateway that stage 1 created. The rule these express is one a
// LangChain or Strands agent cannot express at all: allow lookup_order for any
// authenticated caller, allow process_return only for the caller that owns the
// order. That is authorization on a tool call, decided outside the model, and no
// amount of prompt engineering substitutes for it.
//
// Cedar is default-deny and forbid-wins, so there is no forbid rule here. The
// deny for a non-owning caller is the absence of a matching permit, which is why
// the second rule's conditions are the whole security control.
//
// The three placeholders are substituted by attach_policy.render_policies at
// registration time. Cedar rejects wildcard resources, so a policy that names
// specific actions has to name a specific Gateway ARN, and the ARN only exists
// once the gateway does.
//
// The principal type is AgentCore::IamEntity because the gateway in
// examples/gateway/create_gateway.py uses the AWS_IAM authorizer. IAM principals
// carry an id (the caller's ARN) and no tags. On a CUSTOM_JWT gateway the
// principal would be AgentCore::OAuthUser instead and ownership would be a claim
// comparison rather than an ARN match:
//
//     principal.hasTag("order_id") &&
//     principal.getTag("order_id") == context.input.order_id
//
// The parser in attach_policy.py splits this file on the "// === policy: NAME"
// markers and creates one AgentCore policy per block, so each rule can be
// created, inspected and deleted on its own.

// === policy: LookupOrderForAnyCaller ===
// Reading an order's status is permitted for any caller the gateway
// authenticated. The when clause is not the security control here, it is what the
// service's policy validator demands: an unconditioned permit is rejected as
// overly permissive, and so is context.input has order_id, which the validator
// still reads as matching every request. A non-empty order_id is the weakest
// condition that passes. The rejection is asynchronous — CreatePolicy returns a
// policyId and the policy then reaches CREATE_FAILED.
permit(
  principal is AgentCore::IamEntity,
  action == AgentCore::Action::"supportTools___lookup_order",
  resource == AgentCore::Gateway::"<GATEWAY_ARN>"
)
when {
  context.input.order_id != ""
};

// === policy: ProcessReturnForOwnerOnly ===
// Starting a return is permitted only for the caller that owns the order, which
// is two conditions and not one. The principal clause pins who is asking; the
// when clause pins which order they are asking about, reading order_id straight
// out of the tool call's arguments through context.input. A caller who owns
// order A and asks to return order B matches neither the owner of B nor B's
// permit, so the request is denied by default while their lookup_order calls
// keep working.
permit(
  principal == AgentCore::IamEntity::"<OWNER_PRINCIPAL_ARN>",
  action == AgentCore::Action::"supportTools___process_return",
  resource == AgentCore::Gateway::"<GATEWAY_ARN>"
)
when {
  context.input.order_id == "<OWNED_ORDER_ID>"
};

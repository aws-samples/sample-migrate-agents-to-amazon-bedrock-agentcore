<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Security architecture: self-hosted vs. Amazon Bedrock AgentCore

This document compares the security responsibilities when running agentic workloads on self-hosted infrastructure versus Amazon Bedrock AgentCore.

The table below compares architectures, not this repository. It lists what each side of the migration makes available; most of it is not exercised by the samples. [What this sample implements](#what-this-sample-implements) says exactly which rows it does.

## Comparison

| Component | Self-hosted | Amazon Bedrock AgentCore |
|---|---|---|
| API authentication | API keys in environment variables or AWS Secrets Manager (you manage rotation, scoping) | AgentCore Identity with user delegation and pre-authorized consent (managed) |
| Network isolation | Self-managed VPC, ALB, WAF, security groups | AgentCore Runtime with optional VPC configuration, session isolation in dedicated microVMs |
| Agent-to-model auth | API keys per provider (OpenAI, Anthropic) or IAM roles for Amazon Bedrock | IAM execution role on AgentCore Runtime with least-privilege Bedrock model access |
| Agent-to-tool auth | Self-managed per tool (OAuth tokens, API keys, IAM) | AgentCore Gateway handles OAuth flows, token refresh, and secure credential storage |
| Content filtering | Self-implemented or third-party guardrails | Amazon Bedrock Guardrails (content filtering, PII redaction, topic denial) |
| Tool-call authorization | A conditional in the tool function, or a sentence in the system prompt | Policy in AgentCore: Cedar rules evaluated at the Gateway on the caller, the tool, and the tool's arguments |
| Secrets management | AWS Secrets Manager or HashiCorp Vault (you manage) | AgentCore credential management via Secrets Manager (integrated) |
| Audit logging | Custom CloudTrail and CloudWatch setup | AgentCore Observability with built-in tracing of agent reasoning, tool calls, and model interactions |
| Scaling security | Self-configured auto-scaling groups with security group rules | Serverless auto-scaling with session-isolated microVMs (no shared compute between sessions) |
| Patching | You manage OS, runtime, and dependency updates | Managed runtime environment, patched by AWS |

## Three ways to stop an agent doing something

Two rows above — content filtering and tool-call authorization — are routinely conflated, because both get called "guardrails" in casual use. There is also a third thing people reach for first, which is a sentence in the system prompt. All three are real options for the same-sounding requirement, "don't let the agent process a return it shouldn't". They are not substitutes.

| | A sentence in the system prompt | Amazon Bedrock Guardrails | Policy in Amazon Bedrock AgentCore |
|---|---|---|---|
| Protects | Nothing. It asks | The model surface: what a caller may put into the model, and what the model may say back | The tool surface: which tool calls are allowed to execute |
| Decides on | The model's disposition, one sample at a time | Text — a denied topic, a content filter, a PII entity | Identity, action and arguments — the caller, the tool, and the tool's input |
| Where it runs | Inside the model's own reasoning | In the Amazon Bedrock model invocation | In the AgentCore Gateway, before the target is invoked |
| Written as | Prose in the prompt | Guardrail configuration (`topicPolicyConfig`, `sensitiveInformationPolicyConfig`) | Cedar rules, default-deny and forbid-wins |
| Diff on the agent | The prompt string | Two model parameters, `guardrail_id` and `guardrail_version` | None — the decision lives at the gateway |
| Auditable | No | Yes, and it reports which policy intervened | Yes, and the Gateway logs the decision |
| Needs the migration | No | **No** — it works on any Amazon Bedrock call today | **Yes** — it needs a Gateway in front of the tools |

**A prompt instruction is not an access control: it is advice to a probabilistic system.** It is not auditable, it is not enforced anywhere, and the same prompt can produce a refusal one turn and a tool call the next. That is the whole reason the other two columns exist.

Between the other two, the distinction is what surface each one covers. One protects what the model may say; the other protects what a caller may do. **A guardrail cannot stop a `process_return` call that the model was correctly persuaded to make**, because by the time the tool runs the text has already passed. A Cedar rule cannot stop the model repeating a card number back to the customer, because it never sees the model's tokens.

The line is easy to cross by accident, and this sample crossed it once. A DENY topic was written for "an order belonging to anyone other than the customer in this conversation". Measured live, it blocked every prompt naming an order number, including the customer's own: a topic classifier reads the sentence, and "where is my order 12345" is indistinguishable from "where is order 12345" without knowing who is asking. **Ownership is not a property of the text.** It is a property of the caller, so it belongs in the column that can see one.

## Why Policy is the control this sample implements

The last row of that table is the reason. Bedrock Guardrails is an Amazon Bedrock feature: you can attach one to the model calls you are making right now, with no Runtime, no Gateway, no Memory and no Policy. It is a good control and it is not an argument for migrating, because you do not have to migrate to get it.

Policy in AgentCore is different. It evaluates Cedar rules at the Gateway, on the caller's identity, before the tool target is invoked — so it requires the Gateway that stage 1 stands up. It is the one control in this repository a reader cannot obtain without doing the migration, which is why it is the one stage 2 spends its complexity budget on. `examples/stage2_rebuild/policy/support_tools.cedar` holds two permits: the read-only caller may `lookup_order`, and only the support-agent role may `process_return`. The agent-side diff for that is empty.

What those rules use is the caller and the tool, not the tool's arguments, because the gateway's `AWS_IAM` authorizer presents an ARN and no claims. `context.input` conditions are available and validate; the sample does not need one, and a rule comparing a tool argument against *who the caller is* needs a `CUSTOM_JWT` authorizer to have a claim to compare against.

## What this sample implements

Of the rows in the first table, the samples exercise four:

- **Agent-to-model auth** — Amazon Bedrock model calls under the Runtime execution role, in every stage.
- **Agent-to-tool auth** — AgentCore Gateway with the `AWS_IAM` authorizer, SigV4-signed from the agent (`examples/tools/gateway_mcp_tools.py`), from stage 1 on.
- **Tool-call authorization** — Cedar rules in Policy in AgentCore, attached to the gateway in `ENFORCE` mode, in stage 2.
- **Patching** and the microVM session isolation that comes with Runtime, by virtue of running there at all.

It does **not** implement content filtering (no Bedrock Guardrail — see above), AgentCore Identity or its credential providers, VPC configuration, Observability wiring, or any secrets management beyond the execution role. Those are architectural options, listed for the comparison, not code in this repository.

## Key takeaway

With self-hosted infrastructure, you own every layer of the security stack. AgentCore shifts most of the undifferentiated security work (network isolation, credential management, session isolation, patching) to AWS, letting you focus on application-level security decisions. The one this sample shows is the Cedar rules that authorize tool calls, because it is the one the migration itself buys you.

For more information, refer to the [Amazon Bedrock AgentCore security documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/security.html).

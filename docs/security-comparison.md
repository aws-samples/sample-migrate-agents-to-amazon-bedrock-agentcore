<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# Security architecture: self-hosted vs. Amazon Bedrock AgentCore

This document compares the security responsibilities when running agentic workloads on self-hosted infrastructure versus Amazon Bedrock AgentCore.

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

## Guardrails and Policy are not the same control

The two rows above are the two security features stage 2 adds, and they are easy to conflate because both are described as "guardrails" in casual use. They protect different surfaces, and neither substitutes for the other.

| | Amazon Bedrock Guardrails | Policy in Amazon Bedrock AgentCore |
|---|---|---|
| Protects | The model surface: what a caller may put into the model, and what the model may say back | The tool surface: which tool calls are allowed to execute |
| Decides on | Text — a denied topic, a content filter, a PII entity | Identity, action and arguments — the caller, the tool, and the tool's input |
| Where it runs | In the Amazon Bedrock model invocation | In the AgentCore Gateway, before the target is invoked |
| Written as | Guardrail configuration (`topicPolicyConfig`, `sensitiveInformationPolicyConfig`) | Cedar rules, default-deny and forbid-wins |
| Diff on the agent | Two model parameters, `guardrail_id` and `guardrail_version` | None — the decision lives at the gateway |
| Failure it prevents | The agent discussing something it should not discuss | The agent doing something the caller is not entitled to do |

One protects what the model may say; the other protects what a caller may do. A guardrail cannot stop a `process_return` call that the model was correctly persuaded to make, because by the time the tool runs the text has already passed. A Cedar rule cannot stop the model describing another customer's order, because it never sees the model's tokens. In `examples/stage2_rebuild/` these are separate files for that reason: `guardrail.py` configures the model, and `policy/support_tools.cedar` authorizes the tools.

Both are also worth contrasting with the third option, which is a sentence in the system prompt. A prompt instruction is not an access control: it is advice to a probabilistic system, it is not auditable, and it is not enforced anywhere. Moving a rule from the prompt into a guardrail or a Cedar policy is the point of stage 2.

## Key takeaway

With self-hosted infrastructure, you own every layer of the security stack. AgentCore shifts most of the undifferentiated security work (network isolation, credential management, session isolation, patching) to AWS, letting you focus on application-level security decisions like IAM policy scoping, guardrail configuration, and the Cedar rules that authorize tool calls.

For more information, refer to the [Amazon Bedrock AgentCore security documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/security.html).

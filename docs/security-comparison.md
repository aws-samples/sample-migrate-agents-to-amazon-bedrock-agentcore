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
| Secrets management | AWS Secrets Manager or HashiCorp Vault (you manage) | AgentCore credential management via Secrets Manager (integrated) |
| Audit logging | Custom CloudTrail and CloudWatch setup | AgentCore Observability with built-in tracing of agent reasoning, tool calls, and model interactions |
| Scaling security | Self-configured auto-scaling groups with security group rules | Serverless auto-scaling with session-isolated microVMs (no shared compute between sessions) |
| Patching | You manage OS, runtime, and dependency updates | Managed runtime environment, patched by AWS |

## Key takeaway

With self-hosted infrastructure, you own every layer of the security stack. AgentCore shifts most of the undifferentiated security work (network isolation, credential management, session isolation, patching) to AWS, letting you focus on application-level security decisions like IAM policy scoping and guardrail configuration.

For more information, refer to the [Amazon Bedrock AgentCore security documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/security.html).

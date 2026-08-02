# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Stage 2: the same agent rebuilt on the Strands Agents SDK.

The reason to rebuild is the execution loop, not the security features. Stage 0
and stage 1 route with ``route_intent``, a hand-written branch that decides what
happens next, and every new behaviour is another edge in that graph. Strands runs
a model-driven loop instead: the model decides which tool to call and when the
task is done, and the code that used to encode that decision stops existing.
Stage 1 shows the branch surviving a migration untouched; stage 2 is choosing to
give it up, which is a different thing from being forced to.

Guardrails and Policy are layered on once you are already here. Neither one
required the rebuild:

- Bedrock Guardrails (guardrail.py) filters the model surface. It is an Amazon
  Bedrock feature rather than an AgentCore one, it attaches to the model rather
  than to the agent, and the same two parameters would have worked on stage 1.
- Policy in Amazon Bedrock AgentCore (policy/) filters the tool surface. It
  evaluates Cedar rules on every tool call through the Gateway stage 1 created,
  so the agent diff for it is empty: the rule lives outside the agent entirely.

One protects what the model may say, the other protects what a caller may do.
"""

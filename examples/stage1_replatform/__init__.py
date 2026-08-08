# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Stage 1: the same agent, replatformed onto Amazon Bedrock AgentCore.

This package copies nothing from stage 0. It imports it. The list of things the
migration did not change is therefore an import statement rather than a promise:
the graph topology, the router, the prompts and the tool bodies all come from
examples.stage0_langgraph and are not restated here.

Three services take over three burdens. Runtime hosts the loop
(agent_runtime.py), Gateway serves two of the three tools
(langchain_mcp_tools.py), and Memory holds the conversation state, through
langgraph-checkpoint-aws rather than through anything in this package. Model
inference does not move: it was already going to Amazon Bedrock in stage 0, and it
was never the problem.
"""

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Stage 0: the self-hosted LangGraph customer-support agent.

This is the agent before any migration. It makes no Amazon Bedrock AgentCore
calls. Everything it needs — the tool loop, the conversation state, the HTTP
backend for its tools — is something you run yourself.
"""

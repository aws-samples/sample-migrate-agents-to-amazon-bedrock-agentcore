# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Policy in Amazon Bedrock AgentCore: Cedar rules on the tools, not the agent.

support_tools.cedar holds the rules. attach_policy.py creates the policy engine,
registers each rule on it, attaches the engine to the Gateway in ENFORCE mode,
and deletes all of it again.

The agent diff for this feature is empty, which is the reason it is worth showing
in a post about migration: the authorization decision moves out of the agent's
code and into the gateway boundary, so it holds no matter which framework, or
which prompt, is on the other side.
"""

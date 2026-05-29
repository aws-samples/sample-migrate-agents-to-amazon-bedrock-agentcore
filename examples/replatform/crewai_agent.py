# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Replatform a CrewAI agent on Amazon Bedrock AgentCore Runtime.

Usage: Replace 'from my_crew import crew' with your existing CrewAI
crew module, then deploy to AgentCore Runtime.
"""

from bedrock_agentcore import BedrockAgentCoreApp
from my_crew import crew  # your existing CrewAI crew

app = BedrockAgentCoreApp()


@app.entrypoint
def agent_invocation(payload, context):
    result = crew.kickoff(inputs={"topic": payload.get("prompt", "")})
    return {"result": str(result)}


if __name__ == "__main__":
    app.run()

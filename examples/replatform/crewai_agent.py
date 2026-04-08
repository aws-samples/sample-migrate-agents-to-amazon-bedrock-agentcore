# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Replatform a CrewAI agent on Amazon Bedrock AgentCore Runtime."""

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from my_crew import crew  # your existing CrewAI crew

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload):
    result = crew.kickoff(inputs={"topic": payload.get("prompt")})
    return {"result": str(result)}


if __name__ == "__main__":
    app.run()

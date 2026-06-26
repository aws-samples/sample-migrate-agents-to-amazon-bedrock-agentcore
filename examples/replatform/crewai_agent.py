# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Replatform a CrewAI agent on Amazon Bedrock AgentCore Runtime.

This example includes a minimal CrewAI crew inline for testing.
Replace the crew definition with your own CrewAI crew.
"""

from bedrock_agentcore import BedrockAgentCoreApp
from crewai import Agent, Task, Crew


# --- Replace this section with your existing CrewAI crew ---
researcher = Agent(
    role="Research Assistant",
    goal="Provide helpful answers to questions",
    backstory="You are a knowledgeable assistant.",
    verbose=False,
)


def build_crew(topic: str) -> Crew:
    task = Task(
        description=f"Answer the following question: {topic}",
        expected_output="A concise, helpful answer",
        agent=researcher,
    )
    return Crew(agents=[researcher], tasks=[task], verbose=False)
# --- End of replaceable section ---


app = BedrockAgentCoreApp()


@app.entrypoint
def agent_invocation(payload, context):
    crew = build_crew(payload.get("prompt", ""))
    result = crew.kickoff()
    return {"result": str(result)}


if __name__ == "__main__":
    app.run()

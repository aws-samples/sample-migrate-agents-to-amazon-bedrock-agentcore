# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

FROM python:3.12-slim

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY examples/ examples/

# Which agent this image serves. Every stage's entrypoint is already in the image,
# so the stage is a build argument rather than a line you edit:
#
#   examples.stage1_replatform.agent_runtime   the LangGraph graph on Runtime
#   examples.rebuild.strands_agent            the Strands rewrite
#
# Build another one with:
#   finch build --build-arg AGENT_MODULE=examples.rebuild.strands_agent .
#
# Run as a module rather than copied to main.py, because each entrypoint imports
# its own stage's package from examples/ and -m keeps those imports resolvable
# from any working directory.
ARG AGENT_MODULE=examples.stage1_replatform.agent_runtime
ENV AGENT_MODULE=${AGENT_MODULE}

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# /ping is what BedrockAgentCoreApp serves; its only other route is
# POST /invocations. There is no route for /, so do not probe it.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/ping')" || exit 1

# Shell form on purpose: AGENT_MODULE has to be expanded, and exec keeps the
# server as PID 1 so Runtime's signals reach it.
CMD exec python -m "$AGENT_MODULE"

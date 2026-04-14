# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY examples/ examples/

# Replace with your agent entrypoint
COPY examples/rebuild/strands_agent.py main.py

EXPOSE 8080

CMD ["python", "main.py"]

#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -e

echo "=== Amazon Bedrock AgentCore Migration Sample Setup ==="
echo ""

# Check Python version
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$("$cmd" -c "import sys; print(sys.version_info.major)")
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)")
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.10+ is required. Found none."
    exit 1
fi
echo "[OK] Python: $("$PYTHON_CMD" --version)"

# Check the AWS CLI. Only ./examples/gateway/lambda_target/deploy.sh uses it; the
# walkthrough and every example call AWS through boto3. So a missing CLI is a
# warning, not a failure: installing and running the tests never touch it.
if ! command -v aws &>/dev/null; then
    echo "[WARN] AWS CLI not found. ./examples/gateway/lambda_target/deploy.sh needs it to"
    echo "       create the Lambda that backs the gateway target. Nothing else does."
    echo "       Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
else
    echo "[OK] AWS CLI: $(aws --version 2>&1 | head -1)"
fi

# Check AWS credentials. A warning, not an exit: the install below and the test
# suite both work without them, and that is where a first-time reader starts.
if command -v aws &>/dev/null && aws sts get-caller-identity &>/dev/null; then
    echo "[OK] AWS credentials configured"
else
    echo "[WARN] No usable AWS credentials. Run 'aws configure'. This does not block setup:"
    echo "       python -m unittest discover -s tests             works without credentials"
    echo "       python -m examples.stage0_langgraph.run_local    fails without Bedrock model access"
    echo "       examples/run_walkthrough.py                      creates real AWS resources"
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv .venv
fi

# Activate and install
source .venv/bin/activate
echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment: source .venv/bin/activate"
echo "Then the offline suite:      python -m unittest discover -s tests -v"
echo "Then the stage-0 agent:      python -m examples.stage0_langgraph.run_local"
echo ""
echo "run_local.py is the agent before any migration. It needs Bedrock model access and"
echo "nothing else — it starts its own orders-API stub on a loopback port. The later"
echo "stages need AWS resources that do not exist yet; see Run order in README.md."

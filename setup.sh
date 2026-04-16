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

# Check AWS CLI
if ! command -v aws &>/dev/null; then
    echo "ERROR: AWS CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi
echo "[OK] AWS CLI: $(aws --version 2>&1 | head -1)"

# Check AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS credentials not configured. Run 'aws configure' first."
    exit 1
fi
echo "[OK] AWS credentials configured"

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
echo "Then run any example:        python examples/rebuild/strands_agent.py"

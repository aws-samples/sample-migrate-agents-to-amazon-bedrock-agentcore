#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Create the IAM execution role and Lambda function that back the AgentCore
# Gateway target used by the walkthrough, then print the function ARN to pass as
# --lambda-arn.
#
# Nothing is hardcoded. Override any of these before running:
#   FUNCTION_NAME  Lambda name           (default: agentcore-support-tools)
#   ROLE_NAME      execution role name   (default: agentcore-support-tools-role)
#   REGION         AWS region            (default: $AWS_REGION, else us-east-1)
# The account id and role ARN are read back from the AWS CLI at runtime.
#
# The handler uses only the Python standard library, so there is no requirements
# file and nothing to install into the deployment package.
#
# Usage:
#   ./deploy.sh
set -euo pipefail

FUNCTION_NAME="${FUNCTION_NAME:-agentcore-support-tools}"
ROLE_NAME="${ROLE_NAME:-agentcore-support-tools-role}"
REGION="${REGION:-${AWS_REGION:-us-east-1}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Execution role the Lambda assumes, trusting the Lambda service.
aws iam create-role \
  --role-name "${ROLE_NAME}" \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# 2. Minimum permissions: write logs to CloudWatch Logs.
aws iam attach-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

ROLE_ARN="$(aws iam get-role --role-name "${ROLE_NAME}" \
  --query 'Role.Arn' --output text)"

# IAM role creation is eventually consistent; give it a moment to propagate
# before Lambda validates it.
sleep 10

# 3. Package the handler (single stdlib-only module) and create the function.
ZIP_PATH="$(mktemp -d)/function.zip"
(cd "${SCRIPT_DIR}" && zip -j "${ZIP_PATH}" lambda_function.py)

aws lambda create-function \
  --function-name "${FUNCTION_NAME}" \
  --runtime python3.12 \
  --handler lambda_function.handler \
  --role "${ROLE_ARN}" \
  --zip-file "fileb://${ZIP_PATH}" \
  --region "${REGION}" \
  --timeout 30 \
  --memory-size 128

LAMBDA_ARN="$(aws lambda get-function \
  --function-name "${FUNCTION_NAME}" \
  --region "${REGION}" \
  --query 'Configuration.FunctionArn' --output text)"

echo
echo "Created Lambda: ${LAMBDA_ARN}"
echo "Pass it to the walkthrough as --lambda-arn ${LAMBDA_ARN}"

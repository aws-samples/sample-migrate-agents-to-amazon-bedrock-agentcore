#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Create everything the walkthrough needs on the target side: the Lambda
# execution role, the Lambda function itself, and the gateway execution role
# that invokes that Lambda. Prints the function ARN for --lambda-arn and the
# gateway role ARN for --role-arn.
#
# Nothing is hardcoded. Override any of these before running:
#   FUNCTION_NAME      Lambda name              (default: agentcore-support-tools)
#   ROLE_NAME          Lambda role name         (default: agentcore-support-tools-role)
#   GATEWAY_ROLE_NAME  gateway role name        (default: agentcore-gateway-role)
#   REGION             AWS region               (default: $AWS_REGION, else us-east-1)
# The account id and both role ARNs are read back from the AWS CLI at runtime.
#
# The handler uses only the Python standard library, so there is no requirements
# file and nothing to install into the deployment package.
#
# Usage:
#   ./deploy.sh
set -euo pipefail

FUNCTION_NAME="${FUNCTION_NAME:-agentcore-support-tools}"
ROLE_NAME="${ROLE_NAME:-agentcore-support-tools-role}"
GATEWAY_ROLE_NAME="${GATEWAY_ROLE_NAME:-agentcore-gateway-role}"
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

# 4. Gateway execution role. The gateway assumes this role (trusting the
# bedrock-agentcore service) and invokes the Lambda target as this role. No Lambda
# resource policy (lambda add-permission) is required: with credentialProviderType
# GATEWAY_IAM_ROLE the gateway calls Invoke with this role's identity, and a
# same-account identity policy authorizes it on its own.
aws iam create-role \
  --role-name "${GATEWAY_ROLE_NAME}" \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# The second statement is what stage 2 needs and is easy to miss, because the
# service asks for it one permission at a time and two of the three actions are
# not in the SDK model at all. UpdateGateway does not simply record the policy
# engine: it runs a preflight *as this role* (visible as the session name
# GenesisPolicyEngineCheck) and refuses the attachment if the role cannot read the
# engine and authorize against it. Without these three actions, stage 2's
# attach_to_gateway fails on a gateway this script built — measured, in four
# rounds, one error per missing permission, and the error class changes from
# ValidationException to AccessDeniedException part way through.
#
# The resources are the account's gateways and policy engines in this region
# rather than two specific ARNs, because neither exists yet: this script runs
# before create_gateway and before the policy engine. AuthorizeAction is needed on
# both the engine and the gateway.
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

aws iam put-role-policy \
  --role-name "${GATEWAY_ROLE_NAME}" \
  --policy-name GatewayExecutionAccess \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Effect\": \"Allow\",
        \"Action\": \"lambda:InvokeFunction\",
        \"Resource\": \"${LAMBDA_ARN}\"
      },
      {
        \"Effect\": \"Allow\",
        \"Action\": [
          \"bedrock-agentcore:GetPolicyEngine\",
          \"bedrock-agentcore:AuthorizeAction\",
          \"bedrock-agentcore:PartiallyAuthorizeActions\"
        ],
        \"Resource\": [
          \"arn:aws:bedrock-agentcore:${REGION}:${ACCOUNT_ID}:policy-engine/*\",
          \"arn:aws:bedrock-agentcore:${REGION}:${ACCOUNT_ID}:gateway/*\"
        ]
      }
    ]
  }"

GATEWAY_ROLE_ARN="$(aws iam get-role --role-name "${GATEWAY_ROLE_NAME}" \
  --query 'Role.Arn' --output text)"

echo
echo "Created Lambda:       ${LAMBDA_ARN}"
echo "Created gateway role: ${GATEWAY_ROLE_ARN}"
echo
echo "Pass them to the walkthrough as:"
echo "  --lambda-arn ${LAMBDA_ARN}"
echo "  --role-arn   ${GATEWAY_ROLE_ARN}"

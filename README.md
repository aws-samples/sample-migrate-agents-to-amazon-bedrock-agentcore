# Migrating Agentic Workloads to Amazon Bedrock AgentCore

This repository contains sample code for the AWS blog post [Migrating agentic workloads to Amazon Bedrock AgentCore from other platforms](https://aws.amazon.com/blogs/machine-learning/). It demonstrates how to migrate existing AI agents from frameworks like LangGraph and CrewAI to [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/).

> **Important:** This sample code is for demonstration and educational purposes only. Review and adapt security configurations, error handling, and resource sizing for your production environment.

## Architecture

![Before and after a migration to Amazon Bedrock AgentCore. The left column shows self-hosted infrastructure: a load balancer, agent logic on Amazon ECS or AWS Lambda, a third-party model API, self-managed memory, third-party observability, and custom REST tools. The right column shows AgentCore Runtime hosting the same agent logic, with Amazon Bedrock models, AgentCore Memory, AgentCore Gateway fronting existing APIs, and a cross-cutting row holding AgentCore Identity, AgentCore Policy, and Amazon CloudWatch with AWS ADOT.](images/agentcore-migration-architecture.png)

## Overview

The samples show two migration paths:

- **Replatform**: Wrap your existing agent (LangGraph, CrewAI) with `BedrockAgentCoreApp` to run on AgentCore Runtime without rewriting agent logic.
- **Rebuild**: Rewrite your agent using the [Strands Agents SDK](https://github.com/strands-agents/sdk-python) for tighter AgentCore integration and model-driven orchestration.

The blog post describes a third path, handing the orchestration loop to the AgentCore harness, which has no sample code in this repository.

## Repository structure

```
Dockerfile                          # Sample ARM64 container for AgentCore Runtime
examples/
├── run_walkthrough.py          # Run the full sequence end to end (create -> invoke -> teardown)
├── gateway/
│   ├── create_gateway.py       # Create an AgentCore Gateway (MCP, AWS_IAM authorizer)
│   ├── register_target.py      # Register a Lambda target and its tool schema
│   └── lambda_target/
│       ├── lambda_function.py  # Lambda handler backing the gateway target
│       └── deploy.sh           # Create the execution role and function from scratch
├── replatform/
│   ├── langgraph_agent.py      # Wrap a LangGraph agent for AgentCore Runtime
│   └── crewai_agent.py         # Wrap a CrewAI agent for AgentCore Runtime
├── rebuild/
│   └── strands_agent.py        # Rebuild with Strands Agents SDK (memory-backed)
├── tools/
│   ├── gateway_mcp_tools.py    # Connect to a gateway as MCP tools (SigV4-signed)
│   └── inventory_tools.py      # Inventory existing tools before migration
├── memory/
│   └── configure_memory.py     # Set up AgentCore Memory (summary, preference, semantic)
└── validation/
    └── validate_migration.py   # Validate agent behavior post-migration

docs/
└── security-comparison.md      # Security architecture: self-hosted vs. AgentCore

.github/workflows/
└── deploy-agent.yml            # CI/CD pipeline for ARM64 builds and AgentCore deployment (manual trigger)
```

## Prerequisites

- An [AWS account](https://aws.amazon.com/free/) with [Amazon Bedrock](https://aws.amazon.com/bedrock/) model access enabled
- Python 3.10 or later
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed and configured

## Getting started

1. Clone this repository:

```bash
git clone https://github.com/aws-samples/sample-migrate-agents-to-amazon-bedrock-agentcore.git
cd sample-migrate-agents-to-amazon-bedrock-agentcore
```

2. Run the setup script (creates a virtual environment, installs dependencies, verifies AWS credentials):

```bash
./setup.sh
```

3. Activate the environment and run an example:

```bash
source .venv/bin/activate
python examples/rebuild/strands_agent.py
```

Alternatively, install manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run order

Before the walkthrough can register a gateway target, the Lambda that backs the
tools must exist. Create it once with the provided script, which builds the
Lambda execution role, the function, and the gateway execution role from
scratch, then prints the function ARN to use as `--lambda-arn` and the gateway
role ARN to use as `--role-arn`:

```bash
./examples/gateway/lambda_target/deploy.sh
```

The examples then form one continuous sequence in which each step consumes the
previous step's output. `examples/run_walkthrough.py` runs the whole path in
order and passes each stage's result to the next:

1. **Create a gateway** (`examples/gateway/create_gateway.py`) — creates an MCP
   gateway with the `AWS_IAM` authorizer and returns its `gatewayId` and
   `gatewayUrl`.
2. **Register a target** (`examples/gateway/register_target.py`) — registers a
   Lambda target and its `lookup_order` / `process_return` tool schema on that
   gateway id.
3. **Create memory** (`examples/memory/configure_memory.py`) — creates the
   AgentCore Memory resource and returns its `memoryId`.
4. **Build and invoke the agent** (`examples/rebuild/strands_agent.py`) —
   `build_agent()` wires the memory id into a session manager and appends the
   gateway-discovered MCP tools (signed with SigV4 by
   `examples/tools/gateway_mcp_tools.py`).

Run every stage in order, with a teardown at the end:

```bash
python -m examples.run_walkthrough \
  --role-arn arn:aws:iam::<account>:role/<gateway-execution-role> \
  --lambda-arn arn:aws:lambda:us-east-1:<account>:function:<tools-function> \
  --teardown
```

Both ARNs are printed by `examples/gateway/lambda_target/deploy.sh`:
`--lambda-arn` is the function ARN, and `--role-arn` is the gateway execution
role ARN that the same script creates.

Each stage can also be run on its own; pass the previous stage's output as an
argument or environment variable (`GATEWAY_ID`, `GATEWAY_URL`,
`AGENTCORE_MEMORY_ID`).

## Additional resources

- [Security architecture comparison](docs/security-comparison.md): Side-by-side comparison of security responsibilities when running agents on self-hosted infrastructure vs. Amazon Bedrock AgentCore.
- [CI/CD deployment workflow](.github/workflows/deploy-agent.yml): Sample GitHub Actions pipeline for building ARM64 containers and deploying to AgentCore Runtime. Uses `workflow_dispatch` (manual trigger only).

## Clean up

If you created AWS resources while following the examples, delete them to avoid ongoing charges.

Passing `--teardown` to `examples/run_walkthrough.py` deletes what the walkthrough created, in the required order: the gateway target first, then the gateway once target deletion has finished, then the memory resource. Deleting a gateway that still has a target attached returns `ValidationException`.

To remove resources by hand:

1. Delete the gateway target, then the gateway, in that order.
2. Delete the AgentCore Memory resource.
3. Delete the Lambda function and the two IAM roles created by `examples/gateway/lambda_target/deploy.sh`, which are the Lambda execution role and the gateway execution role.
4. Remove any AgentCore Runtime deployment from the [Amazon Bedrock console](https://console.aws.amazon.com/bedrock/).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.

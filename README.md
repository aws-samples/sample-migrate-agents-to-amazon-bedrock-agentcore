# Migrating Agentic Workloads to Amazon Bedrock AgentCore

This repository contains sample code for the AWS blog post "Migrating agentic workloads to Amazon Bedrock AgentCore from other platforms" — **not yet published; there is no link to give yet, and `TODO: post URL` is what goes here until there is.** It demonstrates how to migrate an existing LangGraph agent to [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/).

> **Important:** This sample code is for demonstration and educational purposes only. Review and adapt security configurations, error handling, and resource sizing for your production environment.

## Architecture

![Two-column before-and-after diagram of a migration to Amazon Bedrock AgentCore. Left column, "Before: self-hosted infrastructure, you manage everything": a client application calls a self-managed ALB or API Gateway, which reaches your agent logic (LangGraph, CrewAI, or custom) running on Amazon ECS or AWS Lambda that you operate. The agent logic fans out to four dependencies you own: an OpenAI or Anthropic API with API keys in environment variables, Redis or DynamoDB as self-managed memory, LangSmith or Datadog for third-party observability, and custom REST tools as self-managed APIs. A footer band lists what you carry — VPC, WAF, IAM policies, secrets rotation, OS patching, dependency updates, and auto-scaling rules. Right column, "After: Amazon Bedrock AgentCore, AWS manages infrastructure": the same client application calls serverless AgentCore Runtime directly, with no load balancer, and inside Runtime a BedrockAgentCoreApp wrapper hosts your agent logic unchanged. Runtime reaches three managed components — Amazon Bedrock models under IAM auth, AgentCore Memory with short-term and long-term memory, and AgentCore Gateway serving MCP tools with an inbound authorizerType and an outbound GATEWAY_IAM_ROLE — and the Gateway fronts your existing APIs, also unchanged. A cross-cutting row spanning Runtime and Gateway holds three more: AgentCore Identity, marked "not in this sample" because this repository configures no inbound authorizer or outbound credential providers; AgentCore Policy, doing Cedar-based authorization on tool calls; and Amazon CloudWatch with AWS ADOT for built-in tracing. A second footer band lists what AWS carries instead — auto-scaling, session-isolated microVMs, IAM roles, and managed patching.](images/agentcore-migration-architecture.png)

## Overview

One LangGraph customer-support agent is carried through three stages, so each migration path is a
diff against the same starting point rather than a separate demo:

- **Stage 0 — the agent you already have** (`examples/stage0_langgraph/`): a compiled `StateGraph`,
  a hand-written escalation router, three tools over HTTP, an in-process checkpointer. No AgentCore
  calls at all.
- **Stage 1 — replatform** (`examples/stage1_replatform/`): keep the agent, replace what surrounds
  it. Graph, router, prompts and tool bodies are imported unchanged; Runtime hosts the process,
  Gateway fronts two of the three tools, AgentCore Memory holds the conversation state.
  `deploy_runtime.py` is what puts it there — a zip in Amazon S3, with pip as the only build tool.
  See [Deploying to Runtime](#deploying-to-runtime).
- **Stage 2 — rebuild** (`examples/stage2_rebuild/`): the same agent on the
  [Strands Agents SDK](https://github.com/strands-agents/harness-sdk), trading the router for a
  model-driven loop, with Cedar rules in Policy in AgentCore over the gateway's tool calls.

The post describes a third path, handing the orchestration loop to the AgentCore harness, which has
no sample code here. `examples/validation/verify_diff_claim.py` measures the stage 0 → stage 1 cost
from the files on disk rather than asserting it, so the post's numbers can be re-checked.

## Repository structure

```
Dockerfile                          # Optional ARM64 image; the zip deploy is the path used here
requirements.txt                    # Six entries; langchain-aws is pinned, the rest are floors
setup.sh                            # Create .venv, install requirements, warn on missing credentials
examples/
├── run_walkthrough.py              # Run the stages in order (create -> invoke -> teardown)
├── stage0_langgraph/
│   ├── README.md                   # What stage 0 is, and what it costs you to operate
│   ├── agent.py                    # build_graph(): the StateGraph, the router, the tool loop
│   ├── prompts.py                  # System and classifier prompts
│   ├── tools.py                    # lookup_order, process_return, search_faq over HTTP
│   ├── local_api.py                # Localhost stub for the orders API
│   └── run_local.py                # Stage-0 entry point (needs Bedrock model access)
├── stage1_replatform/
│   ├── agent_runtime.py            # Stage-1 entry point for AgentCore Runtime
│   ├── agentcore_memory_saver.py   # LangGraph checkpointer over AgentCore Memory
│   ├── deploy_runtime.py           # Package the agent as a zip and deploy it to Runtime
│   └── langchain_mcp_tools.py      # Gateway MCP tools as LangChain tools
├── stage2_rebuild/
│   ├── strands_agent.py            # Stage-2 entry point, its tools, and build_agent()
│   └── policy/
│       ├── support_tools.cedar     # Cedar rules authorizing each gateway tool call
│       ├── attach_policy.py        # Register the rules and attach them to the gateway
│       └── demo_principals.py      # Two IAM-identical roles, so a denial is Cedar's
├── gateway/
│   ├── create_gateway.py           # Create an AgentCore Gateway (MCP, AWS_IAM authorizer)
│   ├── register_target.py          # Register a Lambda target and its tool schema
│   └── lambda_target/
│       ├── lambda_function.py      # Lambda handler backing the gateway target
│       └── deploy.sh               # Create the execution roles and function from scratch
├── memory/
│   └── configure_memory.py         # Set up AgentCore Memory (summary, preference, semantic)
├── tools/
│   └── gateway_mcp_tools.py        # Connect to a gateway as MCP tools (SigV4-signed)
└── validation/
    ├── verify_diff_claim.py        # Measure the post's two-number claim from the tree
    ├── measure_walkthrough.py      # The numbers the walkthrough prints, taken as it runs
    └── verify_policy_visibility.py # What each principal can see, not just call

docs/
└── security-comparison.md          # Self-hosted vs. AgentCore, and the three ways to stop an agent

tests/                              # Offline suite; fakes for Bedrock, MCP, Memory and Cedar

.github/workflows/
└── deploy-agent.yml                # CI/CD pipeline for the zip deploy to Runtime (manual trigger)
```

## Prerequisites

Not all of these are needed at once, and the order below is the order they start to matter:

- Python 3.10 or later, and a clone. That is the whole of what steps 1 to 3 need — the test suite runs
  offline, with no AWS account.
- An [AWS account](https://aws.amazon.com/free/) with [Amazon Bedrock](https://aws.amazon.com/bedrock/)
  model access enabled, and credentials. Needed from step 4 onwards, which is the first step that calls
  AWS.
- The [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), needed
  by one script only: `examples/gateway/lambda_target/deploy.sh`. Everything else uses boto3 from the
  virtual environment.

## Getting started

1. Clone this repository:

```bash
git clone https://github.com/aws-samples/sample-migrate-agents-to-amazon-bedrock-agentcore.git
cd sample-migrate-agents-to-amazon-bedrock-agentcore
```

2. Run the setup script. It creates a virtual environment, installs dependencies, and warns — rather
   than fails — if the AWS CLI or credentials are missing, because step 3 needs neither and step 4
   needs credentials but not the CLI:

```bash
./setup.sh
```

Or install manually, which is the same thing without the checks:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the test suite. **No credentials, and it creates nothing** — the chat model, the gateway, the
   memory service and the Cedar evaluator are faked, everything else is real:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

The suite is `unittest`, and that command needs nothing beyond `requirements.txt`. `pytest tests/ -q`
runs it too and prints less, but pytest is not a declared dependency, so `./setup.sh` does not install
it.

4. Run the stage-0 agent — the agent before any migration, and the last step that creates nothing in
   your account. **Needs AWS credentials and Amazon Bedrock model access for the model named in
   `examples/stage0_langgraph/run_local.py`, and nothing else:** no Gateway, no Memory, no Runtime, no
   Policy. That model id is a `us.` cross-region inference profile and the region defaults to
   `us-east-1`, so set `AWS_REGION` to a US region — or change `MODEL_ID` to a profile your region
   carries, since a mismatch surfaces as a model-access error rather than as a region one. It starts
   its own orders-API stub on a loopback port, so this one command is the whole of it:

```bash
python -m examples.stage0_langgraph.run_local
```

It holds two turns on one `thread_id` and answers the second without being told the order number
again, which is the in-process checkpointer being visible; then a third turn escalates without
reaching the model's tool loop.

5. Then work through [Run order](#run-order) for the migration itself. **Everything from here creates
   real AWS resources and costs money**: `examples/gateway/lambda_target/deploy.sh` first — the one
   script that needs the AWS CLI — then `examples/run_walkthrough.py`, which takes the two ARNs that
   script prints. `examples/stage2_rebuild/strands_agent.py` comes last of all, because it expects a
   gateway, a gateway target, an AgentCore Memory resource and attached Cedar policies to exist
   already.

## Run order

Before the walkthrough can register a gateway target, the Lambda that backs the
tools must exist. Create it once with the provided script, which builds the
Lambda execution role, the function, and the gateway execution role from
scratch, then prints the function ARN to use as `--lambda-arn` and the gateway
role ARN to use as `--role-arn`:

```bash
./examples/gateway/lambda_target/deploy.sh
```

The gateway execution role it creates carries two statements, and the second one
is only needed by stage 2: `bedrock-agentcore:GetPolicyEngine`,
`AuthorizeAction` and `PartiallyAuthorizeActions` on the account's gateways and
policy engines. `UpdateGateway` runs a preflight as the gateway role before it
accepts a policy engine, so without them stage 2's attach fails on a gateway this
script built.

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
4. **Build and invoke the agent** (`examples/stage0_langgraph/agent.py`) —
   `build_graph()` compiles the stage-0 graph with the gateway-discovered MCP
   tools (signed with SigV4 by `examples/tools/gateway_mcp_tools.py`, converted
   to LangChain tools by `examples/stage1_replatform/langchain_mcp_tools.py`) and
   the AgentCore Memory checkpointer keyed on the memory id.
5. **Deploy that same agent to Runtime**
   (`examples/stage1_replatform/deploy_runtime.py`) — packages
   `agent_runtime.py` as a zip, uploads it to Amazon S3, creates the runtime, and
   invokes it twice on one session id, cold then warm. The gateway URL from step 1
   and the memory id from step 3 reach it as Runtime environment variables, which
   is why the agent itself is deployed unmodified. This is where stage 1 ends;
   [Deploying to Runtime](#deploying-to-runtime) is what the artifact is and what
   the packaging costs to get wrong.
6. **Rebuild and authorize it** (`examples/stage2_rebuild/`) — builds the Strands
   agent on the gateway, target and memory from steps 1 to 3, then registers the
   Cedar rules and attaches them to the gateway in `ENFORCE` mode.

Run every stage in order, with a teardown at the end:

```bash
python -m examples.run_walkthrough \
  --role-arn arn:aws:iam::<account>:role/<gateway-execution-role> \
  --lambda-arn arn:aws:lambda:us-east-1:<account>:function:<tools-function> \
  --teardown
```

`--stage` selects one stage instead of all of them. `--stage 0` is the
self-hosted starting point and creates nothing, so it needs neither ARN;
`--stage 1` creates the gateway, target and memory above **and four more resources,
because the Runtime deploy is part of it**: an AgentCore Runtime
(`MigratedAgentRuntime`), the S3 bucket holding its zip
(`migrated-agent-runtime-<account-id>`), its IAM execution role
(`MigratedAgentRuntimeRole`), and a CloudWatch log group that nothing asked for —
the service creates it on the runtime's first log line. `--teardown` deletes all
four. `--stage 2` rebuilds
the agent on Strands and hardens it, reusing the gateway, target and memory from
stage 1 — not its runtime — and so needing the same two ARNs. Stage 2 creates a Cedar policy engine of its own, and
`--teardown` deletes it. The rebuilt agent also has its own Runtime entry point at
`examples/stage2_rebuild/strands_agent.py`.

Both ARNs are printed by `examples/gateway/lambda_target/deploy.sh`:
`--lambda-arn` is the function ARN, and `--role-arn` is the gateway execution
role ARN that the same script creates.

The individual scripts also run on their own, taking the previous step's output as an argument or
an environment variable — `register_target.py` and `stage2_rebuild/policy/attach_policy.py` read
`GATEWAY_ID`, `gateway_mcp_tools.py`, `agent_runtime.py` and `deploy_runtime.py` read `GATEWAY_URL`,
and both Runtime entry points and `deploy_runtime.py` read `AGENTCORE_MEMORY_ID`. `create_gateway.py`
reads neither, because it is the step that creates the gateway: it takes `GATEWAY_NAME` and
`GATEWAY_ROLE_ARN`. Every one of them takes `AWS_REGION` and defaults it to `us-east-1`.

## Deploying to Runtime

`CreateAgentRuntime` takes the agent two ways. `containerConfiguration` wants a `linux/arm64` image
in Amazon ECR, which is what the `Dockerfile` is for. `codeConfiguration` wants a zip in Amazon S3,
and that is what this repository uses, because the only build tool it needs is pip:

```bash
python -m examples.stage1_replatform.deploy_runtime \
  --gateway-url https://<gateway-id>.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp \
  --memory-id MigratedAgentMemory-xxxx
```

`agent_runtime.py` is deployed unmodified; the gateway URL and memory id reach it as Runtime
environment variables. `--stage 1` of the walkthrough runs this for you.

Two things about the zip are not obvious, and each costs a failed deploy to learn:

- **Cross-compile the dependencies.** Runtime is ARM64, so a plain `pip install --target` on macOS
  or an x86 machine produces the wrong binaries and the deploy fails with a message about
  incompatible binary files that names none of them. Install with
  `--platform manylinux2014_aarch64 --only-binary=:all:` — a pip flag, not a container.
- **A `requirements.txt` inside the zip is inert.** Nothing installs it; the dependencies have to be
  vendored into the archive. **The error message for getting this wrong points away from the
  cause:** a missing module surfaces as `Runtime initialization time exceeded ... initialization
  completes in 30s`, a timeout message for an `ImportError`, with the real reason visible only in
  CloudWatch. Vendor the dependency; do not tune the timeout.

`deploy_runtime.py` does both of these, and its error paths name the cause and the fix rather than
repeating the service's message.

## Additional resources

- [Security architecture comparison](docs/security-comparison.md): what self-hosting owns versus what AgentCore owns, and the difference between a sentence in the system prompt, a Bedrock Guardrail, and Cedar rules in Policy in AgentCore.
- [CI/CD deployment workflow](.github/workflows/deploy-agent.yml): Sample GitHub Actions pipeline that runs the zip deploy described above — no container build and no Node.js on the runner. Uses `workflow_dispatch` (manual trigger only).

## Clean up

If you created AWS resources while following the examples, delete them to avoid ongoing charges.

Passing `--teardown` to `examples/run_walkthrough.py` deletes what the walkthrough created, in the order the dependencies require: the agent runtime first, then the gateway target, then the gateway once target deletion has finished, then the memory resource, then the Cedar policies and the policy engine holding them, then the two demo IAM roles stage 2 creates, and last the three things the runtime leaves behind — its artifact bucket, its execution role and its log group. Deleting a gateway that still has a target attached returns `ValidationException`, and it does so for a while after `ListGatewayTargets` already returns nothing, so the delete is retried and completion is confirmed with `GetGateway`. The runtime goes first and the bucket and role go last because the runtime holds a reference to the zip and assumes the role; the log group is last of all, because deleting it while the runtime is still alive only means its next log line recreates it. Every step is attempted even when an earlier one fails, and the failures are reported together at the end, because a teardown that stops at its first error orphans everything later in the list.

`--teardown` on its own, with no `--stage`, is the recovery path for a run that already died: it finds all of the above in the account by the names this walkthrough gives them, so it works from a different process than the one that created them, and `--dry-run` lists what it would delete without deleting it.

To remove resources by hand — the names are the ones the scripts use, so they are what to search for:

1. Delete the gateway target, then the gateway (`MigratedAgentGateway`), in that order.
2. Delete the AgentCore Memory resource (`MigratedAgentMemory-` plus a service-assigned suffix).
3. Delete the Cedar policies, then the policy engine that holds them (`SupportToolsPolicyEngine`).
4. Delete the Lambda function and the two IAM roles created by `examples/gateway/lambda_target/deploy.sh`, which are the Lambda execution role and the gateway execution role.
5. If you ran stage 2, delete its two demo IAM roles, `MigratedAgentReadOnlyCaller` and `MigratedAgentSupportAgent`. They are Cedar principals, not callers you use, and a role that can reach a deleted gateway is a permission nobody is auditing.
6. Delete the four resources the Runtime deploy leaves, none of which the AgentCore console will clear for you:
   - the agent runtime `MigratedAgentRuntime`, in the [Amazon Bedrock AgentCore console](https://console.aws.amazon.com/bedrock/) or with `DeleteAgentRuntime`;
   - the S3 bucket `migrated-agent-runtime-<account-id>`, which holds `agent.zip` and has to be emptied before it will delete;
   - the IAM role `MigratedAgentRuntimeRole`, whose inline `RuntimeExecution` policy has to go first;
   - the CloudWatch log group `/aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT`, which `DeleteAgentRuntime` does not remove. Copy the runtime id before deleting the runtime: afterwards the id is no longer discoverable and the group is the one resource with nothing left pointing at it.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.

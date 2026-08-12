# Migrating agentic workloads to Amazon Bedrock AgentCore

This repository contains sample code for the AWS blog post "Migrating agentic workloads to Amazon Bedrock AgentCore from other platforms". It demonstrates how to migrate an existing LangGraph agent to [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/).

> **Important:** This sample code is for demonstration and educational purposes only. Review and adapt security configurations, error handling, and resource sizing for your production environment.

## Architecture

One migration, read as a ladder of descending ownership. It's a recommended order rather than a
menu, and nothing forces you down the whole chain. Two things move independently along it. One is
how much of the layer underneath the agent you operate. The other is whose code plans the next
step. At stage 0, self-hosted LangGraph, all ten operational burdens in the matrix are yours and
your `route_intent` plans. Stage 1 replatforms, and five of the ten stay yours while your code
still plans. Stage 2 rebuilds. The same five stay yours, because ownership does not change, but
the model now plans instead of your router. Stage 3 hands the loop to the AgentCore harness, where
four stay yours and the model plans. That last stage isn't demonstrated in this sample:

![Diagram titled "Sample migration flow, stage by stage." A subtitle states that stages 1 and 2 are built and run in the sample. Three stacked cards read from top to bottom, joined by down arrows, and every card pairs a problem in grey with the result beneath it. The first card, stage 0, is headed the agent you already have. Its problem reads that a support agent is already answering customers, and that our team spent its week on the servers underneath it rather than on the answers it gave. Its result, in brick red, reads that we counted what we operate before touching anything: ten things, all ours, and every stage below has to beat that number. An arrow leads to the second card, stage 1, headed the same agent, replatformed, carrying an Amazon Bedrock AgentCore icon and a chip marked built and run. Its stage-level problem reads that three separate things were ours to run and none of them was the agent's actual job, and its result, in teal, reads that we moved them one at a time and checked after each, so we could tell what each move actually bought. Four moments nest inside that card, each pairing a problem with a result the same way. Runtime: the operating system under the agent still needed patching, and that was our weekend rather than the model's; AgentCore Runtime took over the patching and not a line of the agent changed. Gateway: only this agent could call its own tools, because the permissions lived in our code; Gateway holds those permissions now, which means the next agent gets the same tools for free. Memory: restarting the process lost the conversation and customers noticed, because they had to start over; AgentCore Memory keeps the thread now, and a different copy of the agent answered about an order it had never seen. Verify: three things had changed and nobody should take our word that it still behaved; we re-ran the same three conversations and compared, the behavior held, and five of the ten burdens are AWS's now with five still ours. A second arrow leads to the third card, stage 2, headed rebuilding the loop because you chose to, carrying a Strands SDK icon and a chip marked built and run. Its problem reads that each new kind of question meant another branch in our router and we wrote every one of them. Its result, in teal, reads that Strands lets the model plan instead of our router, that nothing operational moved, and that we lost a branch we could audit, closing on the line that we would not call that a free upgrade. A footer reads that stage 3 hands the loop to an AgentCore harness, documented and not built.](images/agentcore-migration-roadmap.png)

![Comparison matrix titled "Migrating the sample LangGraph agent to AgentCore, stage by stage." A subtitle states that each stage is measured for what we stopped operating and who plans the next step, and a note beneath it adds that one variable moved at a time: stage 1 moved where the agent ran, stage 2 moved how it planned. A band across the top, marked unchanged at every stage, holds the agent itself: classify, route, escalate, three tools. Four numbered stage columns are grouped under three headings. Where we started holds stage 0, self-hosted LangGraph on compute we ran and patched, marked baseline, reporting ten still ours. What this post builds, labeled with an Amazon Bedrock AgentCore icon, holds two highlighted columns: stage 1, replatform, adopting Runtime, Gateway and Memory, marked walked in this post, reporting five moved and five still ours; and stage 2, rebuild, where the loop becomes model-driven planning, also marked walked in this post, reporting the same five moved and five still ours. Beyond this post holds stage 3, hand the loop over, where AgentCore runs it, marked documented only, reporting six moved and four still ours. A row labeled who planned the next step reads our code at stage 0, our code at stage 1, the model at stage 2, and the model at stage 3. A row labeled what moves at this stage reads: nothing moved yet at stage 0, all ten ours, our router planned; five moved to AWS at stage 1, the same five still ours, our router still planned; nothing operational moved at stage 2, the same five still ours, the model took over planning, and stage 2 runs standalone; one more would move at stage 3, dependency updates, not built here. Ten operational burdens then run down the left as rows: VPC, WAF, IAM policies, secrets rotation, OS patching, dependency updates, auto-scaling rules, session isolation, checkpoint storage, and tool auth. Stage 0 shows a filled square on all ten. Stages 1 and 2 are identical to each other: filled circles on OS patching, auto-scaling rules, session isolation, checkpoint storage and tool auth, and filled squares on VPC, WAF, IAM policies, secrets rotation and dependency updates. Stage 3 shows filled circles on six, adding dependency updates, with VPC, WAF, IAM policies and secrets rotation the four squares left. A legend defines the two marks: a filled circle for moved to AgentCore, and a filled square for still ours to operate. A footer reads that the counts are measured from the committed sample, not asserted.](images/agentcore-migration-architecture.png)

This repository ships stages 1 and 2. The order is recommended because stage 1 validates the move
while your logic is unchanged. The same graph, router and prompts answer on AgentCore Runtime,
which separates the infrastructure variable from the behavior variable. Entering at stage 2 is
how a team already committed to a rewrite uses this. `--stage 2` stands up the same gateway, target
and memory itself, because steps 1 to 3 of the [run order](#run-order) run for either stage and
none of them needs the stage-1 agent. What that team skips is the entry-point diff and the
before-and-after parity validation. Stage 1 replatforms the runtime and leaves the graph alone:

![Architecture diagram titled "Stage 1: replatform the runtime." Subtitle: direct invocation, unchanged orchestration, partial tool migration. A band across the top, marked deploy time and not request path, traces source plus vendored dependencies as a zip, into Amazon S3 as agent.zip, into a CreateAgentRuntime call whose codeConfiguration points back at S3, ending at Runtime deployed; there is no container and no Amazon ECR anywhere along it. Below that, a band marked request path starts at a client application card that calls InvokeAgentRuntime with a runtimeSessionId of at least 33 characters and is marked direct call. One arrow runs from it straight into the Amazon Bedrock AgentCore Runtime card, described as serverless with one microVM per session, with no load balancer and no API Gateway in between. Inside the Runtime card, three boxes stack down the page. First, agent_runtime.py, holding BedrockAgentCoreApp with an @app.entrypoint function and mapping RequestContext.session_id to thread_id. Second, the LangGraph graph, marked imported, not rewritten, built by build_graph from stage 0, and noted as the same graph calling all three tools. Third, the tools, split in two: a box outlined in the accent color and marked local Python holds search_faq and states it stays in Runtime, while a box marked gateway-served holds supportTools___lookup_order and supportTools___process_return, so two of the three tools are served by Gateway and one is not. Four service cards sit down the right side. Amazon Bedrock, using ChatBedrockConverse, is marked did not move and already Bedrock. AgentCore Memory, reached from Runtime, is annotated AgentCoreMemorySaver, a first-party saver from langgraph-checkpoint-aws. AgentCore Gateway speaks MCP over SigV4-signed calls with an AWS_IAM authorizer, and is reached from the gateway-served tool box. AWS Lambda is the Gateway target, receiving arguments as the raw event and the tool name in the client context. Along the bottom, an observability card carries the Amazon CloudWatch and AWS Distro for OpenTelemetry icons and reports Runtime logs, metrics, and traces. Beside it, an AgentCore Policy card marked attaches to this gateway is noted as demonstrated at stage 2 and describes Cedar on each tool call in the data plane, joined to Gateway by a dashed line. A line above the footer records that stage 1 was validated by re-running the stage 0 baseline over the same three turns and comparing the tools that ran and the final state, and states that VPC configuration, WAF, IAM policies, secrets rotation and dependency updates are still ours. A footer line reads: stage 1 replatformed runtime, no load balancer, no API Gateway, no container, no Amazon ECR.](images/agentcore-replatform-architecture.png)

Stage 2 rebuilds the loop, for when the hand-written router is the thing holding you back:

![Architecture diagram titled "Stage 2: rebuild the loop." Subtitle: model-driven planning, SDK-shipped memory wiring, Cedar on every tool call. The upper half, headed what the rebuild changes, holds two cards. The first, marked the loop, rebuilt, shows a Strands Agent running strands_agent.py on AgentCore Runtime, with model-driven planning rather than a router, no route_intent and no add_conditional_edges, and a note that what is lost is a deterministic, auditable branch. The second, marked memory wiring, SDK-shipped, shows AgentCore Memory reusing stage 1's store, with the SDK session manager named as AgentCoreMemorySessionManager. The lower half, headed the proof, Cedar in the data plane, holds AgentCore Policy described as two callers times two tools, evaluated at the Gateway in the data plane on every tool call, with a note that two by two is the smallest experiment that attributes a refusal because one refused call looks like missing IAM. A two by two table follows: a read-only role is allowed supportTools___lookup_order but denied supportTools___process_return for no matching permit, and a support-agent role is allowed both. Two lines state that no forbid rule exists anywhere because Cedar is default-deny, so the refusal is the absence of a permit, and that both roles' IAM policies are identical, so the one refusal is provably Cedar's. A dashed arrow leads down to a final band whose label lists what is reused, the Gateway, the target and Memory but not the stage 1 runtime, and notes that walkthrough steps 1 to 3 build them for either stage; the band shows AgentCore Gateway with its AWS_IAM authorizer, AWS Lambda as an unchanged target, and Amazon Bedrock as the same model in every stage, adding that the Memory store is also stage 1's, the two Gateway tools supersede the same-named local stubs, search_faq stays local as in every stage, and deploy uses the same zip path as stage 1. One footer line states that ownership is unchanged, VPC configuration, WAF, IAM policies, secrets rotation and dependency updates, and that what changed is who plans. A second footer line notes that AWS_IAM principals carry an ARN and no tags, so the rules scope tools to callers and nothing finer.](images/agentcore-rebuild-architecture.png)

## Overview

One LangGraph customer-support agent goes through all three stages, so each stage is a diff
against the same starting point rather than a separate demo:

- **Stage 0, the agent you already have** (`examples/stage0_langgraph/`): a compiled `StateGraph`,
  a hand-written escalation router, three tools over HTTP, an in-process checkpointer. No AgentCore
  calls at all.
- **Stage 1, replatform** (`examples/stage1_replatform/`): keep the agent, replace what surrounds
  it. Graph, router, prompts and tool bodies are imported unchanged. Runtime hosts the process,
  Gateway fronts two of the three tools, and AgentCore Memory holds the conversation state.
  `deploy_runtime.py` is what puts it there, as a zip in Amazon S3, with pip as the only build tool.
  See [Deploying to Runtime](#deploying-to-runtime).
- **Stage 2, rebuild** (`examples/stage2_rebuild/`): the same agent on the
  [Strands Agents SDK](https://github.com/strands-agents/harness-sdk). The router gives way to a
  model-driven loop, and Cedar rules in Policy in AgentCore sit over the gateway's tool calls.

The post describes the final stage, handing the orchestration loop to the AgentCore harness. There is
no sample code for it here. `examples/validation/verify_diff_claim.py` measures the stage 0 → stage 1
cost from the files on disk rather than asserting it, so you can re-check the post's numbers.

## Repository structure

```
Dockerfile                          # Optional ARM64 image; the zip deploy is the path used here
requirements.txt                    # Seven entries; two are pinned, the rest are floors
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
│   └── configure_memory.py         # Set up AgentCore Memory (raw events only, no strategies)
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

- Python 3.12, and a clone. That is the whole of what steps 1 to 3 need, because the test
  suite runs offline, with no AWS account. `setup.sh` accepts any Python from 3.10 and the offline
  steps run on it. 3.12 is the version everything else here names, in `.python-version`, in the
  `Dockerfile`, and in the `PYTHON_3_12` runtime the deploy creates. The pair that has to match is
  the vendored wheels and that named runtime, not your local interpreter, and `deploy_runtime.py`
  pins both sides of it. So running 3.12 locally just means one version everywhere.
- An [AWS account](https://aws.amazon.com/free/) with [Amazon Bedrock](https://aws.amazon.com/bedrock/)
  model access enabled, and credentials. Needed from step 4 onwards, which is the first step that calls
  AWS.
- The [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), needed
  by one script only: `examples/gateway/lambda_target/deploy.sh`. Everything else uses boto3 from the
  virtual environment.
- AgentCore Runtime instruments the agents it hosts for you. No OpenTelemetry package, no `OTEL_*`
  environment variables. Viewing their spans and traces does have a one-time per-account
  prerequisite, so enable
  [CloudWatch Transaction Search](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-get-started.html),
  or the traces the stage-1 diagram shows cannot be seen at all.

## Getting started

1. Clone this repository:

```bash
git clone https://github.com/aws-samples/sample-migrate-agents-to-amazon-bedrock-agentcore.git
cd sample-migrate-agents-to-amazon-bedrock-agentcore
```

2. Run the setup script. It creates a virtual environment, installs dependencies, and warns rather
   than fails if the AWS CLI or credentials are missing, because step 3 needs neither and step 4
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

3. Run the test suite. **No credentials, and it creates nothing.** The chat model, the gateway, the
   memory service and the Cedar evaluator are faked. Everything else is real:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

The suite is `unittest`, and that command needs nothing beyond `requirements.txt`. `pytest tests/ -q`
runs it too and prints less, but pytest isn't a declared dependency, so `./setup.sh` does not install
it.

4. Run the stage-0 agent. This is the agent before any migration, and the last step that creates
   nothing in your account. **Needs AWS credentials and Amazon Bedrock model access for the model
   named in `examples/stage0_langgraph/run_local.py`, and nothing else:** no Gateway, no Memory, no
   Runtime, no Policy. That model id is a `us.` cross-region inference profile and the region
   defaults to `us-east-1`, so set `AWS_REGION` to a US region, or change `MODEL_ID` to a profile
   your region carries. A mismatch surfaces as a model-access error rather than as a region one. It
   starts its own orders-API stub on a loopback port, so this one command is the whole of it:

```bash
python -m examples.stage0_langgraph.run_local
```

It holds two turns on one `thread_id` and answers the second without being told the order number
again. You're watching the in-process checkpointer. A third turn then escalates without reaching
the model's tool loop.

5. Then work through [Run order](#run-order) for the migration itself. **Everything from here creates
   real AWS resources and costs money**: `examples/gateway/lambda_target/deploy.sh` first (the one
   script that needs the AWS CLI), then `examples/run_walkthrough.py`, which takes the two ARNs that
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

1. **Create a gateway** (`examples/gateway/create_gateway.py`). This creates an
   MCP gateway with the `AWS_IAM` authorizer and returns its `gatewayId` and
   `gatewayUrl`.
2. **Register a target** (`examples/gateway/register_target.py`): a Lambda target
   and its `lookup_order` / `process_return` tool schema, on that gateway id.
3. **Create memory** (`examples/memory/configure_memory.py`), which creates the
   AgentCore Memory resource and returns its `memoryId`.
4. **Build and invoke the agent** (`examples/stage0_langgraph/agent.py`).
   `build_graph()` compiles the stage-0 graph with the gateway-discovered MCP
   tools (signed with SigV4 by `examples/tools/gateway_mcp_tools.py`, converted
   to LangChain tools by `examples/stage1_replatform/langchain_mcp_tools.py`) and
   `AgentCoreMemorySaver` from `langgraph-checkpoint-aws`, constructed on the
   memory id. Each invocation supplies the `thread_id` and `actor_id` that
   together name the event stream the checkpoint is written into.
5. **Deploy that same agent to Runtime**
   (`examples/stage1_replatform/deploy_runtime.py`). This packages
   `agent_runtime.py` as a zip, uploads it to Amazon S3, creates the runtime, and
   invokes it twice on one session id, cold then warm. The gateway URL from step 1
   and the memory id from step 3 reach it as Runtime environment variables, which
   is why the agent itself is deployed unmodified. Stage 1 ends here.
   [Deploying to Runtime](#deploying-to-runtime) covers what the artifact is and
   what the packaging costs to get wrong.
6. **Rebuild and authorize it** (`examples/stage2_rebuild/`) builds the Strands
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
(`MigratedAgentRuntimeRole`), and a CloudWatch log group that nothing asked for,
which the service creates on the runtime's first log line. `--teardown` deletes all
four.

`--stage 2` rebuilds the agent on Strands and hardens it. It needs the same two
ARNs because it stands on the same foundation. The gateway, target and memory
are created by steps 1 to 3 whichever stage runs, so on a clean account
`--stage 2` builds them itself, and after a stage-1 run it reuses stage 1's
(never its runtime). Stage 2 creates a Cedar policy engine of its own, and `--teardown` deletes
it. The rebuilt agent also has its own Runtime entry point at
`examples/stage2_rebuild/strands_agent.py`.

Both ARNs are printed by `examples/gateway/lambda_target/deploy.sh`:
`--lambda-arn` is the function ARN, and `--role-arn` is the gateway execution
role ARN that the same script creates.

The individual scripts also run on their own, taking the previous step's output as an argument or
an environment variable. `register_target.py` and `stage2_rebuild/policy/attach_policy.py` read
`GATEWAY_ID`; `gateway_mcp_tools.py`, `agent_runtime.py` and `deploy_runtime.py` read `GATEWAY_URL`;
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

`agent_runtime.py` is deployed unmodified. The gateway URL and memory id reach it as Runtime
environment variables. `--stage 1` of the walkthrough runs this for you.

Two things about the zip aren't obvious, and each costs a failed deploy to learn:

- **Cross-compile the dependencies.** Runtime is ARM64, so a plain `pip install --target` on macOS
  or an x86 machine produces the wrong binaries and the deploy fails with a message about
  incompatible binary files that names none of them. Install with
  `--platform manylinux2014_aarch64 --python-version 3.12 --only-binary=:all:`. Those are pip flags, not
  a container. The version has to match the runtime the deploy names, and omitting it vendors wheels for
  whichever interpreter you happen to be running.
- **A `requirements.txt` inside the zip is inert.** Nothing installs it, and the dependencies have
  to be vendored into the archive. **The error message for getting this wrong points away from the
  cause:** a missing module surfaces as `Runtime initialization time exceeded ... initialization
  completes in 30s`, a timeout message for an `ImportError`, with the real reason visible only in
  CloudWatch. Vendor the dependency. Do not tune the timeout.

`deploy_runtime.py` does both of these, and its error paths name the cause and the fix rather than
repeating the service's message.

## Additional resources

- [Security architecture comparison](docs/security-comparison.md): what self-hosting owns versus what AgentCore owns, and the difference between a sentence in the system prompt, a Bedrock Guardrail, and Cedar rules in Policy in AgentCore.
- [CI/CD deployment workflow](.github/workflows/deploy-agent.yml): Sample GitHub Actions pipeline that runs the zip deploy described above, with no container build and no Node.js on the runner. Uses `workflow_dispatch` (manual trigger only).

## Clean up

If you created AWS resources while following the examples, delete them to avoid ongoing charges.

Passing `--teardown` to `examples/run_walkthrough.py` deletes what the walkthrough created, in the order the dependencies require. The agent runtime goes first, then the gateway target, then the gateway once target deletion has finished. Then the memory resource, then the Cedar policies and the policy engine holding them, then the two demo IAM roles stage 2 creates. Last come the three things the runtime leaves behind: its artifact bucket, its execution role and its log group.

Deleting a gateway that still has a target attached returns `ValidationException`, and it does so for a while after `ListGatewayTargets` already returns nothing, so the teardown retries the delete and confirms completion with `GetGateway`. The runtime goes first and the bucket and role go last because the runtime holds a reference to the zip and assumes the role. The log group is last of all: deleting it while the runtime is still alive only means its next log line recreates it.

The teardown attempts every step even when an earlier one fails, and reports the failures together at the end. A teardown that stops at its first error orphans everything later in the list.

`--teardown` on its own, with no `--stage`, is the recovery path for a run that already died. It finds all of the above in the account by the names this walkthrough gives them, so it works from a different process than the one that created them, and `--dry-run` lists what it would delete without deleting it.

To remove resources by hand, search on the names the scripts use:

1. Delete the gateway target, then the gateway (`MigratedAgentGateway`), in that order.
2. Delete the AgentCore Memory resource (`MigratedAgentMemory-` plus a service-assigned suffix).
3. Delete the Cedar policies, then the policy engine that holds them (`SupportToolsPolicyEngine`).
4. Delete the Lambda function and the two IAM roles created by `examples/gateway/lambda_target/deploy.sh`, which are the Lambda execution role and the gateway execution role.
5. If you ran stage 2, delete its two demo IAM roles, `MigratedAgentReadOnlyCaller` and `MigratedAgentSupportAgent`. They are Cedar principals, not callers you use, and a role that can reach a deleted gateway is a permission nobody is auditing.
6. Delete the four resources the Runtime deploy leaves, none of which the AgentCore console will clear for you:
   - the agent runtime `MigratedAgentRuntime`, in the [Amazon Bedrock AgentCore console](https://console.aws.amazon.com/bedrock/) or with `DeleteAgentRuntime`;
   - the S3 bucket `migrated-agent-runtime-<account-id>`, which holds `agent.zip` and has to be emptied before it will delete;
   - the IAM role `MigratedAgentRuntimeRole`, whose inline `RuntimeExecution` policy has to go first;
   - the CloudWatch log group `/aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT`, which `DeleteAgentRuntime` does not remove. Copy the runtime id before deleting the runtime. Afterwards the id is no longer discoverable, and the group is the one resource with nothing left pointing at it.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.

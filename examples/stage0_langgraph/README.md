# Stage 0 — the agent you already have

A self-hosted LangGraph customer-support agent for ExampleCorp. This is the
starting point of the migration walkthrough, and it makes **zero Amazon Bedrock
AgentCore calls**. Bedrock `Converse` for the model is its only AWS dependency.

Nothing here is a toy. It is a compiled `StateGraph` with a model-driven tool
loop, three tools over an HTTP backend, a checkpointer keyed on `thread_id`, and
a hand-written escalation branch — which is what "my agent" means to a LangGraph
user, and what the later stages have to preserve.

## The graph

```
START
  └─> classify_intent                  model call, returns state["intent"]
        ├─(intent == "escalate")─> escalate ─> END
        └─(intent == "assist")──> assist     model call with tools bound
                                    ├─(tool calls present)─> tools
                                    └─(else)──────────────> END
              tools ─────────────────────────> assist       the cycle
```

Four nodes, two conditional edges, one cycle.

`route_intent` in `agent.py` is the one to watch. It is business routing you
tuned by hand, no service replaces it, and it appears in none of the migration
diffs — it is hosted unchanged from here to the end.

## The three tools

| Tool | Backend | What happens to it later |
|---|---|---|
| `lookup_order` | `GET {ORDERS_API_BASE}/orders/{order_id}` | Moves to an AgentCore Gateway target in stage 1, as `supportTools___lookup_order` |
| `process_return` | `POST {ORDERS_API_BASE}/returns` | Moves to Gateway in stage 1, as `supportTools___process_return` |
| `search_faq` | `GET {ORDERS_API_BASE}/faq/search` | Stays a local function, unchanged, through every stage |

`search_faq` staying put is the point, not an oversight. Migration is not
all-or-nothing, and a tool that never moves is what proves the gateway-bound
tools supersede their local twins without taking the rest of your tool list with
them.

`ORDERS_API_BASE` defaults to `https://api.example.com`, which does not resolve.
`local_api.py` serves the real thing on localhost, and it imports its order and
return payloads straight from the Gateway Lambda in
`examples/gateway/lambda_target/lambda_function.py` — so stage 0 and stage 1
return identical payloads by construction, which is what makes the stage-1
before-and-after real rather than two runs that merely look alike.

## Run it

Offline, no AWS, no credentials — the chat model is faked and everything else is
real:

```bash
python -m unittest discover -s tests -v
```

Live against Bedrock, needs credentials and model access for
`us.anthropic.claude-sonnet-5`. That `us.` prefix is a cross-region inference
profile and `run_local.py` defaults to `us-east-1`, so set `AWS_REGION` to a US
region — or change `MODEL_ID` to a profile your region carries, since a mismatch
comes back as a model-access error rather than a region one:

```bash
python -m examples.stage0_langgraph.run_local
```

That runs two turns on one `thread_id` — turn 2 never repeats the order number,
so an answer that knows it came from the checkpointer — and then one angry prompt
that takes the escalation branch without reaching the tool loop.

To poke at the tools by hand, serve the stub on a fixed port instead:

```bash
python -m examples.stage0_langgraph.local_api          # port 8080
export ORDERS_API_BASE=http://127.0.0.1:8080
```

## What it costs you to operate

Everything above runs on your machine. In production, all of it is still yours:

- **The process.** A container, a web server in front of it, a VPC, a WAF, IAM
  policies, secrets rotation, OS patching, dependency updates, and auto-scaling
  rules — none of which are agent logic, all of which are on-call for you.
- **Conversation state.** `MemorySaver` is in-process. It dies with the container
  and two replicas cannot share it, so a conversation is pinned to whichever
  instance started it. `SqliteSaver` is the durable local variant, in a separate
  package this sample does not install; either way the storage is yours to run,
  back up and expire.
- **Tool credentials.** The agent holds whatever the orders API needs. Every tool
  call is authorized by your code, in-process, and a tool the model should not
  have called is a bug you find in your own logs.
- **Session isolation.** One `thread_id` collision is one customer reading
  another customer's order.

Stage 1 hands the first, second and third of those to AgentCore Runtime, Memory
and Gateway. It does not touch the graph, the router, the prompts, or the tool
bodies — which is the claim the rest of the walkthrough exists to demonstrate.

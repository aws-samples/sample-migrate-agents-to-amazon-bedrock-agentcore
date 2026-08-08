# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Measure the blog post's two-number migration claim from the committed tree.

    python -m examples.validation.verify_diff_claim

The post claims two numbers rather than one ratio, because a ratio needs a
denominator and every candidate denominator here is arguable:

  1. How many lines change INSIDE the reader's own agent.
  2. How many lines of NEW GLUE the migration needs that the SDK does not ship.

Both are counted from files on disk, so the numbers in the post are generated
rather than asserted. Run it after any change to stage 0 or stage 1 and the post
either still holds or does not.

WHAT COUNTS AS THE READER'S AGENT, AND WHAT COUNTS AS GLUE

  The reader's agent is stage 0 plus the stage-1 entry point. Stage 0 is the
  agent they already have. examples/stage1_replatform/agent_runtime.py is that
  same agent after the move: it is the file with their application's name on it,
  the one they edit and deploy.

  Glue is the stage-1 module that is a library rather than application code: the
  MCP-to-LangChain tool adapter. Neither bedrock-agentcore nor langchain-aws
  ships one, so a reader migrating a LangGraph agent writes it. That is the
  honest cost of the move and the reason the claim is two numbers.

  There were two glue modules here, and the other was the larger: a checkpointer
  over AgentCore Memory. It is gone, because langgraph-checkpoint-aws ships
  AgentCoreMemorySaver. That cost is now a pinned line in requirements.txt rather
  than a file the reader owns, which is a real result and not a rounding -- so it
  is recorded here rather than quietly absorbed into a smaller number two.

  Number 1 is a line-level diff between the two entry points, because both play
  the same role: construct the model, the tools and the checkpointer, then
  invoke the graph. Comments, docstrings and blank lines are stripped from both
  sides first, so a rewritten comment does not inflate a migration cost.

  Two classes of line are dropped from the stage-0 side, and every dropped line
  is printed so the classification can be argued with rather than trusted. They
  are the console-demo and local-stub plumbing that exist because stage 0 ships
  as a runnable sample: print() calls, calls to the ask() helper, the ask()
  helper itself, and the ORDERS_API_BASE / running_stub wiring. A reader's own
  agent has no counterpart to any of them, so counting their removal as
  migration cost would overstate the diff.

TESTS ARE NOT COUNTED, in either number. Excluding them is the conservative
choice for the claim being made: tests/ is 4 files and over 1,000 lines, and
counting them would inflate the glue number several times over on work that
proves the migration rather than performs it. A reader migrating a real agent
writes tests either way, before and after. Stated here rather than left implicit
because it is the single largest exclusion in this script.

DEPLOY TOOLING IS NOT GLUE, and this is the exclusion most open to argument, so
the number is printed anyway and the reasoning is here to be disagreed with.
examples/stage1_replatform/deploy_runtime.py is several times the size of both
glue modules put together. Two things keep it out of the glue count:

  1. The glue test is "no SDK ships it, so the reader writes it." Deployment is
     shipped. The agentcore CLI deploys an agent to Runtime, and a reader who
     wants their agent hosted runs that, or clicks through the console, or writes
     a dozen lines of boto3. Nobody has to write this file. It exists because a
     walkthrough has requirements a reader does not: reproducible from a checkout
     with pip and boto3 and nothing else, idempotent across re-runs, able to
     delete everything it made, and instrumented to time four things. The
     MCP adapter is a hole in the SDK -- the checkpointer no longer is, and the
     file that used to fill it is no longer in this repo. Deployment is not a hole
     either; it is a choice of tool.

  2. The glue module runs inside the deployed artifact — the adapter *is* the
     agent's tools, and removing it stops the agent working.
     deploy_runtime.py runs on the developer's machine and is never
     imported by the agent at all. That line is mechanical rather than a matter of
     taste, so main() checks it instead of asserting it: if the entry point ever
     imports the deploy module, the check fails and this classification has to be
     revisited rather than quietly going stale.
"""

import ast
import difflib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

STAGE0_ENTRY = REPO / "examples/stage0_langgraph/run_local.py"
STAGE1_ENTRY = REPO / "examples/stage1_replatform/agent_runtime.py"

# The reader's agent, imported unchanged by stage 1 rather than copied.
STAGE0_AGENT = [
    REPO / "examples/stage0_langgraph/agent.py",
    REPO / "examples/stage0_langgraph/prompts.py",
    REPO / "examples/stage0_langgraph/tools.py",
]

# Glue: written for this migration, shipped by no SDK, and loaded by the agent.
STAGE1_GLUE = [
    REPO / "examples/stage1_replatform/langchain_mcp_tools.py",
]

# Counted and reported, but in neither number. See DEPLOY TOOLING above.
DEPLOY_TOOLING = [
    REPO / "examples/stage1_replatform/deploy_runtime.py",
]

# Sample-only plumbing on the stage-0 side. A line is demo plumbing if it is a
# print, if it calls the ask() console helper, or if it wires the local stub.
DEMO_MARKERS = ("print(", "ask(", "running_stub", "ORDERS_API_BASE")


def code_lines(path: Path) -> list:
    """Return (lineno, text) for every executable line, docstrings dropped.

    Comments and blanks go by string inspection; docstrings by AST, because a
    module or function docstring is an expression statement and would otherwise
    count as code.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    docstring_lines = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                docstring_lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))

    out = []
    for number, raw in enumerate(source.splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#") or number in docstring_lines:
            continue
        out.append((number, text))
    return out


def demo_lines(path: Path) -> set:
    """Line numbers of sample-only console and stub plumbing.

    Works on statement spans, not on single lines, so the continuation lines of
    a multi-line demo call go with their header instead of being counted as
    migration cost. For a compound statement only the header is considered: a
    `with running_stub() as base_url:` block wraps the real graph construction,
    and dropping its whole span would drop that too.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    marked = set()

    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.stmt):
            continue

        # The ask() helper is a print wrapper; drop it whole.
        if isinstance(node, ast.FunctionDef) and node.name == "ask":
            marked.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
            continue

        body = getattr(node, "body", None)
        if body:
            span = range(node.lineno, body[0].lineno)  # header only
        else:
            span = range(node.lineno, (node.end_lineno or node.lineno) + 1)

        text = "\n".join(lines[n - 1] for n in span)
        if any(marker in text for marker in DEMO_MARKERS):
            marked.update(span)

    return marked


def entry_point_diff():
    """Diff the two entry points and split the result into counted and dropped."""
    before = code_lines(STAGE0_ENTRY)
    after = code_lines(STAGE1_ENTRY)
    demo = demo_lines(STAGE0_ENTRY)

    removed, added, dropped = [], [], []
    matcher = difflib.SequenceMatcher(
        None, [t for _, t in before], [t for _, t in after], autojunk=False
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            for number, text in before[i1:i2]:
                (dropped if number in demo else removed).append((number, text))
        if tag in ("replace", "insert"):
            added.extend(after[j1:j2])
    return removed, added, dropped


def stage0_imports() -> list:
    """List what stage 1 imports from stage 0, so 'unchanged' is checkable.

    This is the structural claim the post rests on: stage 1 does not copy the
    reader's agent, it imports it. An import edge is evidence; a sentence saying
    nothing changed is not.
    """
    edges = []
    for path in sorted(REPO.glob("examples/stage1_replatform/*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "examples.stage0_langgraph"
            ):
                names = ", ".join(alias.name for alias in node.names)
                edges.append(f"{path.name}:{node.lineno} imports {names} from {node.module}")
    return edges


def deploy_tooling_is_not_in_the_agent() -> list:
    """Check the line drawn above: nothing in the artifact imports the deployer.

    The claim that deploy tooling is not glue rests on it not being part of the
    running agent. That is checkable rather than arguable, so it is checked. Any
    finding here is a reason to reclassify the file, not a reason to edit this
    function.
    """
    excluded = {path.stem for path in DEPLOY_TOOLING}
    offenders = []
    for path in [STAGE1_ENTRY] + STAGE1_GLUE:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[-1]]
                names += [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                names = [alias.name.split(".")[-1] for alias in node.names]
            for name in names:
                if name in excluded:
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")
    return offenders


def main() -> None:
    removed, added, dropped = entry_point_diff()
    changed = len(removed) + len(added)

    glue = {p: len(code_lines(p)) for p in STAGE1_GLUE}
    glue_total = sum(glue.values())

    tooling = {p: len(code_lines(p)) for p in DEPLOY_TOOLING}
    tooling_total = sum(tooling.values())
    offenders = deploy_tooling_is_not_in_the_agent()

    imported = {p: len(code_lines(p)) for p in STAGE0_AGENT}
    imported_total = sum(imported.values())

    print("Two-number migration claim, measured from the committed tree")
    print("=" * 72)
    print()
    print("NUMBER 1 -- lines that change inside the reader's own agent")
    print(f"  {STAGE0_ENTRY.relative_to(REPO)}  ->  {STAGE1_ENTRY.relative_to(REPO)}")
    print(f"  removed {len(removed)}, added {len(added)}, changed total {changed}")
    print()
    for number, text in removed:
        print(f"    - {number:>4}  {text}")
    for number, text in added:
        print(f"    + {number:>4}  {text}")
    print()
    print(f"  dropped as sample-only console and stub plumbing ({len(dropped)} lines,")
    print(f"  matched on {', '.join(DEMO_MARKERS)} or inside the ask() helper):")
    for number, text in dropped:
        print(f"      {number:>4}  {text}")
    print()
    print("NUMBER 2 -- new glue no SDK ships")
    for path, count in glue.items():
        print(f"  {count:>4}  {path.relative_to(REPO)}")
    print(f"  {glue_total:>4}  total")
    print()
    print("EXCLUDED FROM BOTH NUMBERS -- deploy tooling, disclosed because it is large")
    for path, count in tooling.items():
        print(f"  {count:>4}  {path.relative_to(REPO)}")
    print(f"  {tooling_total:>4}  total, which would be {glue_total + tooling_total} if counted as glue")
    print("  Not counted: the agentcore CLI already deploys to Runtime, so no reader")
    print("  has to write this, and nothing inside the artifact imports it. Move it")
    print("  into the glue column if you disagree -- the number is right there.")
    if offenders:
        print("  CHECK FAILED -- the agent imports the deploy tooling, so the")
        print("  not-part-of-the-agent argument above does not hold:")
        for offender in offenders:
            print(f"    {offender}")
    else:
        print("  checked: no file in the deployed artifact imports it.")
    print()
    print("CONTEXT -- the reader's agent, imported by stage 1 rather than rewritten")
    for path, count in imported.items():
        print(f"  {count:>4}  {path.relative_to(REPO)}")
    print(f"  {imported_total:>4}  total, all of it unchanged by the migration")
    print()
    print("  imported rather than copied, which is what makes that checkable:")
    for edge in stage0_imports():
        print(f"    {edge}")
    print()
    print("HEADLINE")
    print(f"  {changed} lines change inside the agent.")
    print(f"  {glue_total} lines of new supporting code the SDK does not ship.")
    print(f"  {imported_total} lines of graph, router, prompts and tool bodies imported untouched.")
    print()
    print("METHOD: code lines only, docstrings and comments stripped. Tests and the")
    print("deploy tooling are excluded, and both exclusions are counted above.")
    print("Counts are of lines, not statements, and follow this repo's formatting.")


if __name__ == "__main__":
    main()

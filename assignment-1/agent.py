"""Assignment 1 — a tool-using research agent built on LangGraph.

The graph is a real reason/act loop: the model picks the next tool on every hop and
decides for itself when it has enough to answer. Nothing in here sequences the tools.
The only hardcoded control flow is the tool-call budget, which is enforced by the
graph so the model cannot talk its way past it.

Usage:
    python agent.py                        # clean run
    python agent.py --fail-mode malformed  # first web_search returns garbage
"""

import argparse
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

import tools as tools_module
from llm import Usage, build_llm
from tools import ALL_TOOLS, TOOLS_BY_NAME

# Hard cap on tool calls per run. Overridable with --max-tool-calls, which is how the
# budget-exhausted path is demonstrated without waiting for a genuinely hard question.
MAX_TOOL_CALLS = 6

QUESTION = (
    "For our product read API — which currently serves about 10k reads/sec straight "
    "from PostgreSQL with no cache — what caching strategy should we adopt? Recommend "
    "one concrete design, justify it against our actual traffic numbers and constraints, "
    "and say what it costs us in memory and in staleness."
)

SYSTEM_PROMPT_TEMPLATE = """You are a research agent answering an open-ended engineering question.

Your tools:
- web_search(query)      — public/industry knowledge.
- service_metrics(metric) — our real production numbers. Valid metrics: read_qps, write_qps,
  p99_latency_ms, avg_response_bytes, distinct_keys, hot_key_share, current_cache_hit_rate,
  backing_store, tolerable_staleness_seconds — or "all" to get every one in a single call.
- read_notes(filename)   — our internal notes. Available: architecture.md, constraints.md.
- calculator(expression) — arithmetic.

Those inventories are exact, so you never need a call just to discover what exists.

How to work:
- Decide your own path. There is no prescribed order — pick whichever tool actually moves
  you toward the answer next, based on what you have learned so far.
- Every tool takes a `reason` argument. State plainly why you are making that specific
  call before you make it.
- A tool result starting with TOOL_ERROR means that call failed. You must acknowledge it
  and adapt: retry once with different arguments if that seems likely to help, otherwise
  get the information from a different tool or proceed explicitly without it and say so in
  your final answer. Never pretend a failed call succeeded, and never silently ignore one.
- You have a hard budget of {MAX_TOOL_CALLS} tool calls for the whole run. Spend them
  deliberately. Stop calling tools and write your answer as soon as you have enough —
  there is no requirement to use the whole budget.
- You MUST use at least two distinct tools during your investigation before providing your
  final answer.

When you are ready, reply with the final answer as prose and no further tool calls. Ground
it in the specific numbers you found, note any recommendation you are less sure about, and
mention any tool failure that limited you.
"""


def _append(left: list, right: list) -> list:
    return left + right


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls_used: int
    # Every failed tool call, recorded so the run can report failures regardless of
    # whether the model remembers to mention them.
    failures: Annotated[list, _append]


class Trace:
    """Writes the reasoning trace to stdout and, optionally, to a transcript file."""

    def __init__(self, path: Path | None) -> None:
        self.handle = path.open("w") if path else None

    def __call__(self, text: str = "") -> None:
        print(text)
        if self.handle:
            self.handle.write(text + "\n")
            self.handle.flush()

    def close(self) -> None:
        if self.handle:
            self.handle.close()


def _truncate(text: str, limit: int = 400) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + f" ... [{len(text)} chars total]"


def build_graph(llm, trace: Trace, usage: Usage):
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState) -> dict:
        called_tools = set()
        for m in state["messages"]:
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    called_tools.add(tc["name"])

        remaining = MAX_TOOL_CALLS - state["tool_calls_used"]
        note = f"Budget check: {remaining} of {MAX_TOOL_CALLS} tool calls remaining.\nDistinct tools used so far: {len(called_tools)} (Minimum required: 2)."
        if state["failures"]:
            note += (
                "\nTool calls that failed so far: "
                + "; ".join(state["failures"])
                + "\nIf you are writing your final answer now, state explicitly that these "
                "calls failed, how you worked around them, and anything you could not "
                "verify as a result."
            )
        budget_note = SystemMessage(content=note)
        response = llm_with_tools.invoke(state["messages"] + [budget_note])
        usage.record(response)

        if not response.tool_calls:
            trace()
            trace("--- agent decided it has enough information ---")
        return {"messages": [response]}

    def tools_node(state: AgentState) -> dict:
        last: AIMessage = state["messages"][-1]
        outputs = []
        failures = []
        used = state["tool_calls_used"]

        for call in last.tool_calls:
            used += 1
            args = dict(call["args"])
            reason = args.pop("reason", "(no reason given)")

            trace()
            trace(f"--- step {used}/{MAX_TOOL_CALLS} ---")
            trace(f"decided : call {call['name']}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
            trace(f"why     : {reason}")

            tool = TOOLS_BY_NAME.get(call["name"])
            if tool is None:
                result = f"TOOL_ERROR: no such tool {call['name']!r}."
            else:
                # Tools return errors as strings; this guards against an unexpected raise.
                try:
                    result = tool.invoke(call["args"])
                except Exception as exc:
                    result = f"TOOL_ERROR: {call['name']} raised {type(exc).__name__}: {exc}"

            failed = str(result).startswith("TOOL_ERROR")
            trace(f"result  : [{'FAILED' if failed else 'ok'}] {_truncate(result)}")

            content = str(result)
            if failed:
                failures.append(f"step {used} {call['name']} — {_truncate(result, 120)}")
                content += (
                    "\n\n(This call failed. Adapt — retry differently or use another tool — "
                    "and say in your final answer that this call failed and how you worked "
                    "around it.)"
                )
            outputs.append(ToolMessage(content=content, tool_call_id=call["id"]))

        return {"messages": outputs, "tool_calls_used": used, "failures": failures}

    def give_up_node(state: AgentState) -> dict:
        trace()
        trace(f"--- budget exhausted: {MAX_TOOL_CALLS} tool calls used, no answer yet ---")
        trace("The agent is stopping itself rather than continuing past its limit.")
        # Summarise from a fresh, tool-free conversation: flattening the findings into
        # plain text means no tools need to be bound, so the model cannot respond with
        # another tool call instead of the write-up we need here.
        findings = "\n".join(
            f"- {m.content}" for m in state["messages"] if isinstance(m, ToolMessage)
        ) or "- (no tool call returned usable information)"

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are wrapping up a research run that hit its tool-call budget "
                        "before reaching an answer. You have no tools. Reply in prose with: "
                        "(1) that you stopped because the budget ran out, (2) what the "
                        "findings below do establish, (3) what you would have checked next. "
                        "Use only the findings below — do not fill gaps with "
                        "plausible-looking numbers."
                    )
                ),
                HumanMessage(
                    content=f"Question:\n{QUESTION}\n\nFindings gathered so far:\n{findings}"
                ),
            ]
        )
        usage.record(response)
        text = (response.content or "").strip() or (
            "Stopped after exhausting the tool-call budget without reaching an answer."
        )
        return {"messages": [AIMessage(content=text)]}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return "done"
        if state["tool_calls_used"] + len(last.tool_calls) > MAX_TOOL_CALLS:
            return "give_up"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("give_up", give_up_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", route, {"tools": "tools", "give_up": "give_up", "done": END}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("give_up", END)
    return graph.compile()


def main() -> None:
    global MAX_TOOL_CALLS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-mode",
        choices=["none", "timeout", "empty", "malformed"],
        default="none",
        help="Inject a mocked failure into one tool call.",
    )
    parser.add_argument(
        "--fail-tool",
        default="first",
        help=(
            "Which tool the injected failure hits: 'first' (whichever tool the agent "
            "reaches for first) or a tool name."
        ),
    )
    parser.add_argument(
        "--max-tool-calls", type=int, default=MAX_TOOL_CALLS, help="Tool-call budget."
    )
    parser.add_argument("--transcript", help="Also write the trace to this file.")
    parser.add_argument("--question", default=QUESTION, help="Override the question.")
    args = parser.parse_args()
    MAX_TOOL_CALLS = args.max_tool_calls

    tools_module.FAIL_MODE = args.fail_mode
    tools_module.FAIL_TOOL = args.fail_tool
    tools_module.reset_failure_injection()

    trace = Trace(Path(args.transcript) if args.transcript else None)
    usage = Usage()

    trace("=" * 78)
    trace("ASSIGNMENT 1 — TOOL-USING RESEARCH AGENT")
    trace("=" * 78)
    trace(f"question   : {args.question}")
    trace(f"tools      : {', '.join(TOOLS_BY_NAME)}")
    trace(f"budget     : {MAX_TOOL_CALLS} tool calls")
    target = "the first tool called" if args.fail_tool == "first" else args.fail_tool
    trace(
        f"fail mode  : {args.fail_mode}"
        + (f" (injected into {target})" if args.fail_mode != "none" else "")
    )

    graph = build_graph(build_llm(), trace, usage)
    final = graph.invoke(
        {
            "messages": [
                SystemMessage(
                    content=SYSTEM_PROMPT_TEMPLATE.replace(
                        "{MAX_TOOL_CALLS}", str(MAX_TOOL_CALLS)
                    )
                ),
                HumanMessage(content=args.question),
            ],
            "tool_calls_used": 0,
            "failures": [],
        },
        # Generous: the budget above is the real limit, this is just a runaway guard.
        {"recursion_limit": 50},
    )

    trace()
    trace("=" * 78)
    trace("FINAL ANSWER")
    trace("=" * 78)
    trace(final["messages"][-1].content.strip())
    trace()
    trace("-" * 78)
    if final["failures"]:
        trace(f"tool failures this run: {len(final['failures'])}")
        for failure in final["failures"]:
            trace(f"  - {failure}")
    else:
        trace("tool failures this run: none")
    trace(f"tool calls used: {final['tool_calls_used']}/{MAX_TOOL_CALLS}")
    trace(usage.report())
    trace.close()


if __name__ == "__main__":
    main()

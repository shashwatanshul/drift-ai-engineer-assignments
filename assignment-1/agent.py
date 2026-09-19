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

from llm import Usage, build_llm
from tools import TOOL_NAMES, make_tools

# The tool-call budget is a hard range, not a suggestion: the assignment caps a run at
# six tool calls, and a run with none is not a tool-using agent. MAX_TOOL_CALLS is the
# default; --max-tool-calls moves it within the range, which is how the budget-exhausted
# path is demonstrated without waiting for a genuinely hard question.
MIN_TOOL_CALLS = 1
MAX_TOOL_CALLS = 6
TOOL_CALL_BUDGET_RANGE = range(MIN_TOOL_CALLS, MAX_TOOL_CALLS + 1)

# The agent may not finish until it has had this many *distinct* tools return a usable
# result. Which tools, and in what order, remain entirely the agent's choice.
MIN_DISTINCT_TOOLS = 2

# A model that keeps trying to answer without meeting the bar is sent back this many
# times before the run is stopped, so a refusal cannot spin forever.
MAX_NUDGES = 3

QUESTION = (
    "For our product read API — which currently serves about 10k reads/sec straight "
    "from PostgreSQL with no cache — what caching strategy should we adopt? Recommend "
    "one concrete design, justify it against our actual traffic numbers and constraints, "
    "and say what it costs us in memory and in staleness."
)

SYSTEM_PROMPT_TEMPLATE = """You are a research agent answering an open-ended engineering question.

Your tools:
- web_search(query)      — public/industry knowledge.
- service_metrics(metric) — our real production numbers. Pass one name, several separated
  by commas, or "all" to get every one in a single call. Valid names: read_qps, write_qps,
  p99_latency_ms, avg_response_bytes, distinct_keys, hot_key_share, current_cache_hit_rate,
  backing_store, tolerable_staleness_seconds.
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
- Before you can answer, at least {MIN_DISTINCT_TOOLS} different tools must have returned
  a usable result. A call that fails does not count towards that. Which tools those are,
  and in what order you use them, is entirely your choice — pick the ones that genuinely
  help with this question. If you try to answer before then you will be sent back.

When you are ready, reply with the final answer as prose and no further tool calls. Ground
it in the specific numbers you found, note any recommendation you are less sure about, and
mention any tool failure that limited you.
"""


def _append(left: list, right: list) -> list:
    return left + right


def _union(left: list, right: list) -> list:
    """Merge preserving first-seen order, so the trace reads chronologically."""
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
    return merged


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls_used: int
    # Every failed tool call, recorded so the run can report failures regardless of
    # whether the model remembers to mention them.
    failures: Annotated[list, _append]
    # Names of tools that have returned a usable result. A failed call does not count,
    # so the two-tool bar cannot be met by two calls that both errored.
    tools_succeeded: Annotated[list, _union]
    # How many times the agent has been sent back for trying to answer too early.
    # Capped so a model that simply refuses to call tools cannot loop forever.
    nudges: int


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


def build_graph(llm, trace, usage: Usage, tools, tools_by_name, max_tool_calls: int, question: str):
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        remaining = max_tool_calls - state["tool_calls_used"]
        note = f"Budget check: {remaining} of {max_tool_calls} tool calls remaining."
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
        succeeded = []
        used = state["tool_calls_used"]

        for call in last.tool_calls:
            used += 1
            args = dict(call["args"])
            reason = args.pop("reason", "(no reason given)")

            trace()
            trace(f"--- step {used}/{max_tool_calls} ---")
            trace(f"decided : call {call['name']}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
            trace(f"why     : {reason}")

            tool = tools_by_name.get(call["name"])
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
            if not failed:
                succeeded.append(call["name"])
            if failed:
                failures.append(f"step {used} {call['name']} — {_truncate(result, 120)}")
                content += (
                    "\n\n(This call failed. Adapt — retry differently or use another tool — "
                    "and say in your final answer that this call failed and how you worked "
                    "around it.)"
                )
            outputs.append(ToolMessage(content=content, tool_call_id=call["id"]))

        return {
            "messages": outputs,
            "tool_calls_used": used,
            "failures": failures,
            "tools_succeeded": succeeded,
        }

    def give_up_node(state: AgentState) -> dict:
        used = state["tool_calls_used"]
        wanted = len(getattr(state["messages"][-1], "tool_calls", []) or [])
        distinct = len(state["tools_succeeded"])
        trace()
        if wanted:
            trace(
                f"--- budget stop: {used}/{max_tool_calls} tool calls used, and the next "
                f"{wanted} would exceed the budget ---"
            )
            trace("The agent is stopping itself rather than continuing past its limit.")
            reason = "its tool-call budget ran out"
        else:
            # Reached by trying to answer with too few distinct tools working, and with
            # no budget or no patience left to fix it.
            trace(
                f"--- stopping short: only {distinct} of {MIN_DISTINCT_TOOLS} distinct "
                f"tools returned a usable result, and there is no budget left to fix it ---"
            )
            trace("The agent is stopping rather than answering on too little evidence.")
            reason = (
                f"only {distinct} of the {MIN_DISTINCT_TOOLS} distinct working tool "
                "results it needed were available"
            )
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
                        "You are wrapping up a research run that stopped early because "
                        f"{reason}. You have no tools. Reply in prose with: (1) that you "
                        f"stopped because {reason}, (2) what the findings below do "
                        "establish, (3) what you would have checked next. Use only the "
                        "findings below — do not fill gaps with plausible-looking numbers."
                    )
                ),
                HumanMessage(
                    content=f"Question:\n{question}\n\nFindings gathered so far:\n{findings}"
                ),
            ]
        )
        usage.record(response)
        text = (response.content or "").strip() or (
            "Stopped after exhausting the tool-call budget without reaching an answer."
        )
        return {"messages": [AIMessage(content=text)]}

    def needs_more_tools_node(state: AgentState) -> dict:
        """The agent tried to answer before enough distinct tools had worked.

        Sends it back with a short instruction. Deliberately says nothing about which
        tool to reach for next — choosing that is the agent's job, and naming one here
        would turn the run into a script.
        """
        used_names = state["tools_succeeded"]
        remaining = max_tool_calls - state["tool_calls_used"]
        trace()
        trace(
            f"--- too early: {len(used_names)} of {MIN_DISTINCT_TOOLS} distinct tools have "
            f"returned a usable result ({', '.join(used_names) or 'none'}) ---"
        )
        trace("Sending the agent back to gather more before it answers.")
        return {
            "messages": [
                SystemMessage(
                    content=(
                        f"Not yet. So far {len(used_names)} distinct tool(s) have returned "
                        f"a usable result: {', '.join(used_names) or 'none'}. You need at "
                        f"least {MIN_DISTINCT_TOOLS} different ones before you can answer. "
                        f"You have {remaining} tool call(s) left. Choose whichever tool "
                        "genuinely adds something you do not already have, and call it now "
                        "instead of answering."
                    )
                )
            ],
            "nudges": state["nudges"] + 1,
        }

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        used = state["tool_calls_used"]
        distinct = len(state["tools_succeeded"])

        if not getattr(last, "tool_calls", None):
            # The agent wants to answer. It may only do so once enough distinct tools
            # have actually worked.
            if distinct >= MIN_DISTINCT_TOOLS:
                return "done"
            # Sending it back is pointless if it has no budget left to comply with, or
            # if it has already ignored the instruction repeatedly. Either way the run
            # ends through give_up, which reports honestly rather than pretending.
            if used >= max_tool_calls or state["nudges"] >= MAX_NUDGES:
                return "give_up"
            return "needs_more_tools"

        if used + len(last.tool_calls) > max_tool_calls:
            return "give_up"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("needs_more_tools", needs_more_tools_node)
    graph.add_node("give_up", give_up_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        route,
        {
            "tools": "tools",
            "needs_more_tools": "needs_more_tools",
            "give_up": "give_up",
            "done": END,
        },
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("needs_more_tools", "agent")
    graph.add_edge("give_up", END)
    return graph.compile()


def run(
    trace,
    *,
    question: str = QUESTION,
    fail_mode: str = "none",
    fail_tool: str = "first",
    max_tool_calls: int = MAX_TOOL_CALLS,
    api_key: str | None = None,
) -> dict:
    """Run the agent once and return its result.

    Everything the run depends on is an argument, so two runs in the same process
    cannot interfere. `trace` is any callable taking one string — the CLI passes a
    Trace, the Streamlit page passes a live log widget.

    Returns:
        {answer, tool_calls_used, max_tool_calls, failures, usage}
    """
    if not isinstance(max_tool_calls, int) or isinstance(max_tool_calls, bool):
        raise ValueError(f"max_tool_calls must be an integer, got {max_tool_calls!r}")
    if max_tool_calls not in TOOL_CALL_BUDGET_RANGE:
        raise ValueError(
            f"max_tool_calls must be between {MIN_TOOL_CALLS} and {MAX_TOOL_CALLS}, "
            f"got {max_tool_calls}"
        )
    if not question.strip():
        raise ValueError("question must not be empty")

    tools, tools_by_name = make_tools(fail_mode, fail_tool)
    usage = Usage()

    trace("=" * 78)
    trace("ASSIGNMENT 1 — TOOL-USING RESEARCH AGENT")
    trace("=" * 78)
    trace(f"question   : {question}")
    trace(f"tools      : {', '.join(tools_by_name)}")
    trace(f"budget     : {max_tool_calls} tool calls")
    target = "the first tool called" if fail_tool == "first" else fail_tool
    trace(
        f"fail mode  : {fail_mode}"
        + (f" (injected into {target})" if fail_mode != "none" else "")
    )

    graph = build_graph(
        build_llm(api_key=api_key), trace, usage, tools, tools_by_name, max_tool_calls, question
    )
    final = graph.invoke(
        {
            "messages": [
                SystemMessage(
                    content=SYSTEM_PROMPT_TEMPLATE.replace(
                        "{MAX_TOOL_CALLS}", str(max_tool_calls)
                    ).replace("{MIN_DISTINCT_TOOLS}", str(MIN_DISTINCT_TOOLS))
                ),
                HumanMessage(content=question),
            ],
            "tool_calls_used": 0,
            "failures": [],
            "tools_succeeded": [],
            "nudges": 0,
        },
        # Generous: the budget above is the real limit, this is just a runaway guard.
        {"recursion_limit": 50},
    )

    answer = final["messages"][-1].content.strip()

    trace()
    trace("=" * 78)
    trace("FINAL ANSWER")
    trace("=" * 78)
    trace(answer)
    trace()
    trace("-" * 78)
    if final["failures"]:
        trace(f"tool failures this run: {len(final['failures'])}")
        for failure in final["failures"]:
            trace(f"  - {failure}")
    else:
        trace("tool failures this run: none")
    succeeded = final["tools_succeeded"]
    trace(
        f"distinct tools that worked: {len(succeeded)}/{MIN_DISTINCT_TOOLS} required"
        + (f" ({', '.join(succeeded)})" if succeeded else "")
    )
    trace(f"tool calls used: {final['tool_calls_used']}/{max_tool_calls}")
    if final["nudges"]:
        trace(f"times sent back for answering too early: {final['nudges']}")
    trace(usage.report())

    return {
        "answer": answer,
        "tool_calls_used": final["tool_calls_used"],
        "max_tool_calls": max_tool_calls,
        "failures": final["failures"],
        "tools_succeeded": succeeded,
        "min_distinct_tools": MIN_DISTINCT_TOOLS,
        "nudges": final["nudges"],
        "usage": usage,
    }


def _budget_arg(value: str) -> int:
    """argparse type for --max-tool-calls: an integer inside the allowed range.

    Rejecting at parse time means an out-of-range budget fails immediately with a
    usage message, rather than after the run has started.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if number not in TOOL_CALL_BUDGET_RANGE:
        raise argparse.ArgumentTypeError(
            f"must be between {MIN_TOOL_CALLS} and {MAX_TOOL_CALLS}, got {number}"
        )
    return number


def main() -> None:
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
        choices=["first", *TOOL_NAMES],
        help=(
            "Which tool the injected failure hits: 'first' (whichever tool the agent "
            "reaches for first) or a tool name."
        ),
    )
    parser.add_argument(
        "--max-tool-calls",
        type=_budget_arg,
        default=MAX_TOOL_CALLS,
        metavar=f"{{{MIN_TOOL_CALLS}..{MAX_TOOL_CALLS}}}",
        help=f"Tool-call budget, {MIN_TOOL_CALLS} to {MAX_TOOL_CALLS}. "
        f"Default {MAX_TOOL_CALLS}.",
    )
    parser.add_argument("--transcript", help="Also write the trace to this file.")
    parser.add_argument("--question", default=QUESTION, help="Override the question.")
    args = parser.parse_args()

    trace = Trace(Path(args.transcript) if args.transcript else None)
    try:
        run(
            trace,
            question=args.question,
            fail_mode=args.fail_mode,
            fail_tool=args.fail_tool,
            max_tool_calls=args.max_tool_calls,
        )
    finally:
        trace.close()


if __name__ == "__main__":
    main()

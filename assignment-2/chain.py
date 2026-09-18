"""Assignment 2 — a worker agent and a reviewer agent in a single LangGraph chain.

Agent A writes a Python function. Agent B reviews that one attempt against six concrete
criteria and returns approved, or rejected with specific reasons. There is no revision
loop: B reviews once and the chain ends.

Usage:
    python chain.py           # normal worker — expected to be approved
    python chain.py --weak    # deliberately sloppy worker — expected to be rejected
"""

import argparse
from pathlib import Path
from typing import Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from llm import Usage, build_llm

TASK = """Write a Python function `merge_intervals(intervals)`.

It takes a list of [start, end] pairs and returns a new list of non-overlapping intervals
covering the same points, sorted by start. Intervals that overlap or merely touch
(e.g. [1, 3] and [3, 5]) must be merged into one.
"""

# Agent B's approval bar. These are stated once here and used verbatim in its prompt, so
# the README, the prompt and the printed report can never drift apart.
CRITERIA = [
    (
        "correct_on_overlaps",
        "Correctly merges overlapping intervals and touching intervals such as "
        "[1,3] and [3,5]. Trace the code on [[1,3],[2,6],[8,10],[3,5]] to check.",
    ),
    (
        "handles_unsorted_input",
        "Produces correct output when the input is not already sorted by start.",
    ),
    (
        "handles_empty_input",
        "Returns [] for an empty input list without raising IndexError.",
    ),
    (
        "efficient",
        "Runs in O(n log n) — a sort plus a single pass. Reject an O(n^2) approach such "
        "as repeatedly rescanning the list for something to merge.",
    ),
    (
        "no_caller_mutation",
        "Does not mutate the caller's list or the inner lists it was given. Sorting the "
        "argument in place with intervals.sort() fails this; sorted(intervals) passes.",
    ),
    (
        "documented",
        "Has a docstring stating what it takes and what it returns.",
    ),
]

WORKER_PROMPT = """You are a Python engineer. Write the function you are asked for.

Return the function in a single ```python code block, with no commentary around it.
Use only the standard library.
"""

# The --weak worker: a realistically bad first attempt, not a broken one. It runs and
# looks plausible, and fails specific criteria that Agent B has to actually find.
WEAK_WORKER_PROMPT = """You are a hurried junior engineer producing a first draft under
time pressure. Write the function you are asked for, but write it the way someone would
when rushing:

- Sort the input list in place with intervals.sort() rather than copying it first.
- Use a repeated-rescan approach: loop over the list looking for a pair to merge, merge
  it, and start over, until no more merges are possible. Do not write the single-pass
  version.
- Do not write a docstring.
- Do not handle the empty-list case specially.

The code must still run and must be plausible. Return it in a single ```python code block
with no commentary.
"""

REVIEWER_PROMPT = """You are a code reviewer. You are given one attempt at a task and a
fixed list of criteria. Judge that attempt against each criterion — nothing else.

For each criterion, decide pass or fail and give evidence: quote the specific line or
construct in the code that made you decide, or name the concrete input that breaks it.
"Looks fine" and "seems correct" are not evidence and are not acceptable.

The verdict is "approved" only if every criterion passes. If any criterion fails, the
verdict is "rejected" and your reasons must be specific enough that the author knows
exactly what to change.

Do not rewrite the code. Do not suggest a revision round. Review this one attempt.
"""


class CriterionResult(BaseModel):
    name: str = Field(description="The criterion name, exactly as given.")
    passed: bool = Field(description="Whether the attempt satisfies this criterion.")
    evidence: str = Field(
        description="The line, construct, or concrete failing input that decided it."
    )


class Review(BaseModel):
    criteria: list[CriterionResult] = Field(description="One entry per criterion, in order.")
    verdict: Literal["approved", "rejected"] = Field(
        description="'approved' only if every criterion passed, otherwise 'rejected'."
    )
    reasons: list[str] = Field(
        description="Specific reasons for a rejection. Empty when approved."
    )


class ChainState(TypedDict):
    task: str
    weak: bool
    attempt: Optional[str]
    review: Optional[Review]


def build_graph(llm, usage: Usage, log):
    def worker_node(state: ChainState) -> dict:
        prompt = WEAK_WORKER_PROMPT if state["weak"] else WORKER_PROMPT
        log(f"[Agent A] working{' (--weak: deliberately sloppy draft)' if state['weak'] else ''}")
        response = llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=state["task"])]
        )
        usage.record(response)
        log("[Agent A] produced one attempt, handing off to Agent B")
        return {"attempt": response.content.strip()}

    def reviewer_node(state: ChainState) -> dict:
        log("[Agent B] reviewing that attempt against the criteria")
        criteria_text = "\n".join(f"- {name}: {desc}" for name, desc in CRITERIA)
        # Structured output is what stops the review degenerating into "looks good":
        # the model cannot return a verdict without also filling in per-criterion evidence.
        reviewer = llm.with_structured_output(Review, include_raw=True)
        raw = reviewer.invoke(
            [
                SystemMessage(content=REVIEWER_PROMPT),
                HumanMessage(
                    content=(
                        f"TASK GIVEN TO THE AUTHOR:\n{state['task']}\n\n"
                        f"CRITERIA:\n{criteria_text}\n\n"
                        f"THE ATTEMPT:\n{state['attempt']}"
                    )
                ),
            ]
        )
        usage.record(raw["raw"])
        return {"review": raw["parsed"]}

    graph = StateGraph(ChainState)
    graph.add_node("worker", worker_node)
    graph.add_node("reviewer", reviewer_node)
    graph.set_entry_point("worker")
    graph.add_edge("worker", "reviewer")  # single pass — no edge back to worker
    graph.add_edge("reviewer", END)
    return graph.compile()


class Log:
    """Writes to stdout and, optionally, to a transcript file."""

    def __init__(self, path: Optional[Path]) -> None:
        self.handle = path.open("w") if path else None

    def __call__(self, text: str = "") -> None:
        print(text)
        if self.handle:
            self.handle.write(text + "\n")
            self.handle.flush()

    def close(self) -> None:
        if self.handle:
            self.handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weak", action="store_true", help="Make Agent A produce a deliberately sloppy draft."
    )
    parser.add_argument("--transcript", help="Also write the run to this file.")
    args = parser.parse_args()

    log = Log(Path(args.transcript) if args.transcript else None)
    usage = Usage()

    log("=" * 78)
    log("ASSIGNMENT 2 — WORKER + REVIEWER, SINGLE PASS")
    log("=" * 78)
    log("task        : merge_intervals(intervals)")
    log(f"worker mode : {'weak (deliberately sloppy)' if args.weak else 'normal'}")
    log()
    log("Agent B's approval criteria:")
    for name, desc in CRITERIA:
        log(f"  - {name}: {desc}")
    log()

    graph = build_graph(build_llm(), usage, log)
    final = graph.invoke({"task": TASK, "weak": args.weak, "attempt": None, "review": None})

    log()
    log("-" * 78)
    log("AGENT A — THE ATTEMPT")
    log("-" * 78)
    log(final["attempt"])

    review: Review = final["review"]
    log()
    log("-" * 78)
    log("AGENT B — THE REVIEW")
    log("-" * 78)
    for item in review.criteria:
        log(f"[{'PASS' if item.passed else 'FAIL'}] {item.name}")
        log(f"       {item.evidence}")

    log()
    log("=" * 78)
    log(f"VERDICT: {review.verdict.upper()}")
    log("=" * 78)
    if review.verdict == "approved":
        log("Final output is Agent A's attempt above, as approved by Agent B.")
    else:
        log("Rejected. Agent B's specific reasons:")
        for i, reason in enumerate(review.reasons, 1):
            log(f"  {i}. {reason}")

    log()
    log("-" * 78)
    log(usage.report())
    log.close()


if __name__ == "__main__":
    main()

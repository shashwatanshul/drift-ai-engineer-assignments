"""Assignment 3 — a resumable agent with a self-check, built on LangGraph.

Summarises four files one at a time, checkpointing after each one with LangGraph's own
SqliteSaver. If the run is interrupted, re-running it picks up where it stopped and skips
what was already done. When all four are finished it re-reads the results and asks the
model, once per item, whether each summary actually matches its file.

Usage:
    python agent.py --reset --stop-after 2   # do items 1-2, then stop
    python agent.py                          # resume: skip 1-2, do 3-4, then self-check
    python agent.py --reset --sabotage 3     # blank item 3 so the check catches it
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from llm import build_llm

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
CHECKPOINT_DB = HERE / "checkpoints.sqlite"
THREAD_ID = "summarise-docs"

ITEMS = ["caching.md", "queues.md", "indexing.md", "observability.md"]


class StopRequested(Exception):
    """Raised by --stop-after to interrupt the run the way a Ctrl+C would."""


class AgentState(TypedDict):
    """Deliberately plain. This is what gets checkpointed, and it should be readable.

    items       — the filenames to process, in order
    index       — how many are done; also the position of the next one to process
    results     — filename -> summary, for the items finished so far
    llm_calls   — running total, carried in state so it survives a resume
    check       — the self-check report, filled in at the end
    """

    items: list
    index: int
    results: dict
    llm_calls: int
    check: Optional[list]


def build_graph(llm, log, stop_after: Optional[int], sabotage: Optional[int], sabotage_mode: str):
    def process_one(state: AgentState) -> dict:
        i = state["index"]
        name = state["items"][i]
        text = (DOCS_DIR / name).read_text()

        log(f"  PROCESS  item {i + 1}/{len(state['items'])}: {name}")
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Summarise the document in two or three sentences. Cover what it "
                        "actually says — the specifics, not a generic description of the "
                        "topic. Reply with the summary only."
                    )
                ),
                HumanMessage(content=f"Document: {name}\n\n{text}"),
            ]
        )
        summary = response.content.strip()

        if sabotage == i + 1:
            # Corrupt one result on purpose so the self-check has something to catch.
            if sabotage_mode == "blank":
                summary = ""
                log(f"           (SABOTAGED: storing a blank summary for {name})")
            else:
                # A plausible-looking summary that is simply about something else. The
                # emptiness test cannot catch this — only the model comparison can.
                summary = (
                    "This document describes the company's expense reimbursement policy, "
                    "including the approval thresholds for travel spend and the deadline "
                    "for submitting receipts at the end of each quarter."
                )
                log(f"           (SABOTAGED: storing a summary of the wrong topic for {name})")

        log(f"           stored {len(summary)} chars, checkpointing")

        # Returning is what writes the checkpoint, so --stop-after must not raise here —
        # it raises on the next hop, in should_continue, once this item is safely saved.
        return {
            "index": i + 1,
            "results": {**state["results"], name: summary},
            "llm_calls": state["llm_calls"] + 1,
        }

    def self_check(state: AgentState) -> dict:
        log()
        log("All items processed. Running the self-check.")
        report = []
        calls = state["llm_calls"]

        for name in state["items"]:
            summary = state["results"].get(name, "")

            # Cheap local check first — no point asking the model about an empty string.
            if not summary.strip():
                report.append((name, "FAIL", "summary is empty"))
                log(f"  FAIL  {name} (empty — caught locally, no LLM call needed)")
                continue

            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are checking one summary against its source document. "
                            "Does the summary accurately describe this document? Answer "
                            "with 'yes' or 'no' on the first line, then one line of "
                            "reasoning."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Document ({name}):\n{(DOCS_DIR / name).read_text()}\n\n"
                            f"Summary:\n{summary}"
                        )
                    ),
                ]
            )
            calls += 1
            answer = response.content.strip()
            verdict = "PASS" if answer.lower().lstrip("*# ").startswith("yes") else "FAIL"
            reason = " ".join(answer.split("\n")[1:]).strip() or answer
            report.append((name, verdict, reason))
            log(f"  {verdict}  {name} (asked the model: does this summary match the file?)")

        return {"check": report, "llm_calls": calls}

    def should_continue(state: AgentState) -> str:
        if state["index"] < len(state["items"]):
            if stop_after == state["index"]:
                raise StopRequested(
                    f"stopping after item {state['index']} as requested by --stop-after"
                )
            return "next"
        return "check"

    def entry(state: AgentState) -> str:
        """Where to start. On a resume, everything may already be done."""
        return "next" if state["index"] < len(state["items"]) else "check"

    graph = StateGraph(AgentState)
    graph.add_node("process_one", process_one)
    graph.add_node("self_check", self_check)
    graph.set_conditional_entry_point(entry, {"next": "process_one", "check": "self_check"})
    graph.add_conditional_edges(
        "process_one", should_continue, {"next": "process_one", "check": "self_check"}
    )
    graph.add_edge("self_check", END)
    return graph


class Log:
    """Writes to stdout and, optionally, appends to a transcript file."""

    def __init__(self, path: Optional[Path]) -> None:
        self.handle = path.open("a") if path else None

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
    parser.add_argument("--reset", action="store_true", help="Delete the checkpoint and start over.")
    parser.add_argument(
        "--stop-after", type=int, metavar="N", help="Stop after item N, to demonstrate resume."
    )
    parser.add_argument(
        "--sabotage", type=int, metavar="N", help="Deliberately store a bad result for item N."
    )
    parser.add_argument(
        "--sabotage-mode",
        choices=["wrong", "blank"],
        default="wrong",
        help="'wrong' stores an off-topic summary (only the LLM check catches it); "
        "'blank' stores an empty one (the local check catches it).",
    )
    parser.add_argument("--transcript", help="Also append the run to this file.")
    args = parser.parse_args()

    if args.reset and CHECKPOINT_DB.exists():
        CHECKPOINT_DB.unlink()

    log = Log(Path(args.transcript) if args.transcript else None)
    log("=" * 78)
    log("ASSIGNMENT 3 — RESUMABLE AGENT WITH SELF-CHECK")
    log("=" * 78)
    log(f"items      : {', '.join(ITEMS)}")
    log(f"checkpoint : {CHECKPOINT_DB.name}" + (" (reset)" if args.reset else ""))

    config = {"configurable": {"thread_id": THREAD_ID}}

    with sqlite3.connect(CHECKPOINT_DB, check_same_thread=False) as conn:
        checkpointer = SqliteSaver(conn)
        graph = build_graph(
            build_llm(), log, args.stop_after, args.sabotage, args.sabotage_mode
        ).compile(checkpointer=checkpointer)

        # A checkpoint for this thread means a previous run got part way through.
        saved = graph.get_state(config)
        resuming = bool(saved.values)

        if resuming:
            done = saved.values["index"]
            log(f"status     : resuming — {done} of {len(ITEMS)} items already done")
            log()
            for name in ITEMS[:done]:
                log(f"  SKIP     {name} (already completed in an earlier run)")
            # Feed the checkpointed state straight back in. The graph re-enters at the
            # top, but process_one works from `index`, so completed items are skipped
            # rather than redone — no work and no LLM call is repeated.
            start = dict(saved.values)
        else:
            log("status     : fresh start, no checkpoint found")
            log()
            start = {
                "items": ITEMS,
                "index": 0,
                "results": {},
                "llm_calls": 0,
                "check": None,
            }

        try:
            final = graph.invoke(start, config)
        except StopRequested as exc:
            state = graph.get_state(config).values
            log()
            log(f"INTERRUPTED: {exc}")
            log(f"Progress is checkpointed: {state['index']}/{len(ITEMS)} items done.")
            log("Re-run without --stop-after to pick up from here.")
            log(f"LLM calls so far (all runs): {state['llm_calls']}")
            log.close()
            return
        except KeyboardInterrupt:
            state = graph.get_state(config).values
            log()
            log("INTERRUPTED: Ctrl+C")
            log(f"Progress is checkpointed: {state['index']}/{len(ITEMS)} items done.")
            log("Re-run to pick up from here.")
            log.close()
            return

    log()
    log("-" * 78)
    log("SELF-CHECK REPORT")
    log("-" * 78)
    failures = [r for r in final["check"] if r[1] == "FAIL"]
    for name, verdict, reason in final["check"]:
        log(f"  [{verdict}] {name} — {reason}")
    log()
    if failures:
        log(f"RESULT: {len(failures)} of {len(ITEMS)} results failed the check.")
        for name, _, reason in failures:
            log(f"  - {name}: {reason}")
    else:
        log(f"RESULT: all {len(ITEMS)} results passed the check.")

    log()
    log("-" * 78)
    # Counted in the checkpointed state, so this is the total across every run that
    # contributed to these results — not just this process.
    log(f"Total LLM calls across all runs: {final['llm_calls']}")
    log.close()


if __name__ == "__main__":
    main()

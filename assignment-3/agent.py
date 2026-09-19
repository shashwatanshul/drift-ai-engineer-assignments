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


def read_checkpoint(checkpoint_path: Path, thread_id: str = THREAD_ID) -> Optional[dict]:
    """Return the checkpointed state for a thread, or None if there is no checkpoint.

    Used by the Streamlit page to show what a resume would skip before running anything.
    """
    if not Path(checkpoint_path).exists():
        return None
    config = {"configurable": {"thread_id": thread_id}}
    with sqlite3.connect(checkpoint_path, check_same_thread=False) as conn:
        graph = build_graph(None, lambda *_: None, None, None, "wrong").compile(
            checkpointer=SqliteSaver(conn)
        )
        values = graph.get_state(config).values
    return dict(values) if values else None


def run(
    log,
    *,
    checkpoint_path: Path = CHECKPOINT_DB,
    thread_id: str = THREAD_ID,
    stop_after: Optional[int] = None,
    sabotage: Optional[int] = None,
    sabotage_mode: str = "wrong",
    reset: bool = False,
    api_key: Optional[str] = None,
) -> dict:
    """Run (or resume) the agent once.

    `checkpoint_path` is an argument so each caller can have its own checkpoint — the
    Streamlit app gives every browser session a separate file. `log` is any callable
    taking one string.

    Returns:
        {status: "interrupted"|"complete", index, results, check, llm_calls}
    """
    if stop_after is not None and not 1 <= stop_after <= len(ITEMS):
        raise ValueError(f"stop_after must be between 1 and {len(ITEMS)}, got {stop_after}")
    if sabotage is not None and not 1 <= sabotage <= len(ITEMS):
        raise ValueError(f"sabotage must be between 1 and {len(ITEMS)}, got {sabotage}")
    if sabotage_mode not in ("wrong", "blank"):
        raise ValueError(f"sabotage_mode must be 'wrong' or 'blank', got {sabotage_mode!r}")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if reset and checkpoint_path.exists():
        checkpoint_path.unlink()

    log("=" * 78)
    log("ASSIGNMENT 3 — RESUMABLE AGENT WITH SELF-CHECK")
    log("=" * 78)
    log(f"items      : {', '.join(ITEMS)}")
    log(f"checkpoint : {checkpoint_path.name}" + (" (reset)" if reset else ""))

    config = {"configurable": {"thread_id": thread_id}}

    with sqlite3.connect(checkpoint_path, check_same_thread=False) as conn:
        checkpointer = SqliteSaver(conn)
        graph = build_graph(
            build_llm(api_key=api_key), log, stop_after, sabotage, sabotage_mode
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
        except (StopRequested, KeyboardInterrupt) as exc:
            state = graph.get_state(config).values
            stopped_by_flag = isinstance(exc, StopRequested)
            log()
            log(f"INTERRUPTED: {exc if stopped_by_flag else 'Ctrl+C'}")
            log(f"Progress is checkpointed: {state['index']}/{len(ITEMS)} items done.")
            log(
                "Re-run without --stop-after to pick up from here."
                if stopped_by_flag
                else "Re-run to pick up from here."
            )
            log(f"LLM calls so far (all runs): {state['llm_calls']}")
            return {
                "status": "interrupted",
                "index": state["index"],
                "results": dict(state["results"]),
                "check": None,
                "llm_calls": state["llm_calls"],
            }

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

    return {
        "status": "complete",
        "index": final["index"],
        "results": dict(final["results"]),
        "check": final["check"],
        "llm_calls": final["llm_calls"],
    }


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

    log = Log(Path(args.transcript) if args.transcript else None)
    try:
        run(
            log,
            stop_after=args.stop_after,
            sabotage=args.sabotage,
            sabotage_mode=args.sabotage_mode,
            reset=args.reset,
        )
    finally:
        log.close()


if __name__ == "__main__":
    main()

"""/assignment-3 — the resumable agent with a self-check."""

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app import loader, ui  # noqa: E402

agent = loader.get("assignment_3")

ui.page_header(
    "Resumable Agent with Self-Check",
    "Four documents summarised one at a time, checkpointed after each with LangGraph's "
    "SqliteSaver. Stop it partway and resume, and it skips what is already done. Then "
    "it re-reads its own results and checks them.",
    "assignment-3",
)


def checkpoint_path() -> Path:
    """This browser session's own checkpoint file.

    Keyed by session so two people using the deployed app at the same time cannot
    resume into each other's progress.
    """
    directory = Path(tempfile.gettempdir()) / "assignment-3-checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{ui.session_id()}.sqlite"


def saved_state():
    """The checkpointed state for this session, or None. Never raises."""
    try:
        return agent.read_checkpoint(checkpoint_path())
    except Exception:
        # A truncated or unreadable checkpoint should present as "nothing saved"
        # rather than breaking the page.
        return None


with st.expander("What gets saved between runs"):
    st.markdown(
        """
Persistence is LangGraph's own `SqliteSaver`, not a hand-rolled state file. The state it
checkpoints is deliberately plain:

```python
{
  "items":     ["caching.md", "queues.md", "indexing.md", "observability.md"],
  "index":     2,          # how many are done = the next one to process
  "results":   {...},      # filename -> summary, for the items finished so far
  "llm_calls": 2,          # carried in state, so it survives a resume
  "check":     None,       # the self-check report, filled in at the end
}
```

`index` is the whole resume mechanism. Returning from the node is what writes the
checkpoint, so an item is only marked done once its summary is safely stored.
"""
    )

if not ui.api_key():
    ui.missing_key_panel()
    st.stop()

state = saved_state()
items = agent.ITEMS
done = state["index"] if state else 0
has_checkpoint = state is not None

st.subheader("Checkpoint")

if not has_checkpoint:
    st.info(
        "No checkpoint for this session. Start a run below.", icon=":material/info:"
    )
else:
    if done >= len(items):
        st.success(
            f"All {len(items)} items are complete. Reset to run again.",
            icon=":material/check_circle:",
        )
    else:
        st.warning(
            f"{done} of {len(items)} items done. Resume picks up at item {done + 1}.",
            icon=":material/pause_circle:",
        )
    st.progress(done / len(items))

st.dataframe(
    [
        {
            "#": i + 1,
            "Item": name,
            "Status": "complete" if i < done else "pending",
            "Summary stored": (
                "empty"
                if i < done and not (state["results"].get(name) or "").strip()
                else f"{len(state['results'].get(name, ''))} chars"
                if i < done
                else "-"
            ),
        }
        for i, name in enumerate(items)
    ],
    hide_index=True,
    width="stretch",
)

st.subheader("Run settings")

col_left, col_right = st.columns(2)

with col_left:
    stop_enabled = st.checkbox(
        "Stop partway",
        value=False,
        help="Interrupts the run after an item is checkpointed, to demonstrate resume.",
        disabled=ui.is_running(),
    )
    stop_after = st.slider(
        "Stop after item",
        min_value=1,
        max_value=len(items) - 1,
        value=2,
        disabled=ui.is_running() or not stop_enabled,
        help="Capped below the last item, since stopping after the last one is just "
        "finishing.",
    )

with col_right:
    sabotage_enabled = st.checkbox(
        "Sabotage one result",
        value=False,
        help="Deliberately stores a bad summary so the self-check has something to catch.",
        disabled=ui.is_running(),
    )
    sabotage = st.slider(
        "Sabotage item",
        min_value=1,
        max_value=len(items),
        value=3,
        disabled=ui.is_running() or not sabotage_enabled,
    )
    sabotage_mode = st.radio(
        "How to sabotage it",
        options=["wrong", "blank"],
        horizontal=True,
        captions=[
            "An off-topic summary. Only the model comparison catches this.",
            "An empty summary. The local check catches it without an LLM call.",
        ],
        disabled=ui.is_running() or not sabotage_enabled,
    )

if sabotage_enabled and has_checkpoint and sabotage <= done:
    st.caption(
        f"Item {sabotage} is already complete in this checkpoint, so resuming will not "
        "re-process it and the sabotage will not apply. Reset first if you want it to."
    )

st.subheader("Run")

buttons = st.columns(3)

start_clicked = buttons[0].button(
    "Start fresh run",
    type="primary",
    disabled=ui.is_running(),
    icon=":material/play_arrow:",
    help="Discards any checkpoint for this session and processes from item 1.",
    width="stretch",
)
resume_clicked = buttons[1].button(
    "Resume",
    disabled=ui.is_running() or not has_checkpoint or done >= len(items),
    icon=":material/resume:",
    help=(
        "Picks up from the checkpoint, skipping completed items."
        if has_checkpoint and done < len(items)
        else "Available once a run has been interrupted partway."
    ),
    width="stretch",
)
reset_clicked = buttons[2].button(
    "Reset checkpoint",
    disabled=ui.is_running() or not has_checkpoint,
    icon=":material/delete:",
    help="Deletes this session's checkpoint without running anything.",
    width="stretch",
)

if reset_clicked:
    try:
        checkpoint_path().unlink(missing_ok=True)
        st.toast("Checkpoint deleted.")
    except OSError as exc:
        st.error(f"Could not delete the checkpoint: {exc}", icon=":material/error:")
    st.rerun()

if start_clicked or resume_clicked:
    st.subheader("Run log")
    log = ui.LiveLog(st.empty())
    result = None

    with ui.run_guard():
        spinner_text = (
            "Resuming. Completed items are skipped."
            if resume_clicked
            else "Processing items one at a time."
        )
        with st.spinner(spinner_text):
            result = agent.run(
                log,
                checkpoint_path=checkpoint_path(),
                stop_after=stop_after if stop_enabled else None,
                sabotage=sabotage if sabotage_enabled else None,
                sabotage_mode=sabotage_mode if sabotage_enabled else "wrong",
                reset=bool(start_clicked),
                api_key=ui.api_key(),
            )

    if result:
        st.subheader("Result")

        if result["status"] == "interrupted":
            st.warning(
                f"Interrupted after item {result['index']} of {len(items)}. Progress is "
                "checkpointed. Use Resume to pick up from here.",
                icon=":material/pause_circle:",
            )
            st.metric("LLM calls so far, across all runs", result["llm_calls"])
            ui.transcript_download(log, "assignment-3-interrupted-run.txt")
            # Any button click reruns the script, which re-reads the checkpoint and
            # enables Resume. No callback needed.
            st.button("Refresh to enable Resume", icon=":material/refresh:")
        else:
            report = result["check"] or []
            failures = [r for r in report if r[1] == "FAIL"]

            if failures:
                st.error(
                    f"The self-check flagged {len(failures)} of {len(items)} results.",
                    icon=":material/error:",
                )
            else:
                st.success(
                    f"All {len(items)} results passed the self-check.",
                    icon=":material/check_circle:",
                )

            metrics = st.columns(2)
            metrics[0].metric("Results flagged", f"{len(failures)} of {len(items)}")
            metrics[1].metric("LLM calls, across all runs", result["llm_calls"])

            st.markdown("**Self-check report**")
            st.dataframe(
                [
                    {"Item": name, "Result": verdict, "Reason": reason}
                    for name, verdict, reason in report
                ],
                hide_index=True,
                width="stretch",
            )

            with st.expander("The summaries it produced"):
                for name in items:
                    summary = result["results"].get(name, "")
                    st.markdown(f"**{name}**")
                    st.markdown(summary if summary.strip() else "_(empty)_")

            ui.transcript_download(log, "assignment-3-run.txt")

"""/assignment-2 — the worker and reviewer chain."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app import loader, ui  # noqa: E402

chain = loader.get("assignment_2")

ui.page_header(
    "Multi-Agent Task with Review",
    "Agent A writes a Python function. Agent B reviews that one attempt against six "
    "concrete criteria and returns a verdict. No revision loop: B reviews once and the "
    "chain ends.",
    "assignment-2",
)

with st.expander("The task and Agent B's approval criteria"):
    st.markdown("**Task given to Agent A**")
    st.markdown(chain.TASK)
    st.markdown(
        "**Agent B's criteria.** The verdict is approved only if all six pass. These are "
        "read from `CRITERIA` in `chain.py` and injected verbatim into B's prompt, so "
        "this list cannot drift from what the reviewer is actually told."
    )
    st.dataframe(
        [{"Criterion": name, "Bar": desc} for name, desc in chain.CRITERIA],
        hide_index=True,
        width="stretch",
    )

if not ui.api_key():
    ui.missing_key_panel()
    st.stop()

st.subheader("Run settings")

mode = st.radio(
    "Agent A's mode",
    options=["Normal", "Weak"],
    horizontal=True,
    captions=[
        "A straightforward attempt. Usually approved.",
        "A deliberately hurried draft: sorts in place, rescans repeatedly, no docstring.",
    ],
    disabled=ui.is_running(),
)
weak = mode == "Weak"

st.caption(
    "Only Agent A's prompt changes between the two modes. Agent B and its criteria are "
    "identical, and B never sees which mode produced the code, so a rejection is earned "
    "rather than staged."
)

start = st.button(
    "Run chain",
    type="primary",
    disabled=ui.is_running(),
    icon=":material/play_arrow:",
)

if start:
    st.subheader("Run log")
    log = ui.LiveLog(st.empty(), height=300)
    result = None

    with ui.run_guard():
        with st.spinner("Agent A is writing, then Agent B will review it."):
            result = chain.run(log, weak=weak, api_key=ui.api_key())

    if result:
        review = result["review"]
        approved = review.verdict == "approved"

        st.subheader("Verdict")
        if approved:
            st.success(
                "Approved. All six criteria passed.", icon=":material/check_circle:"
            )
        else:
            failed = [c.name for c in review.criteria if not c.passed]
            st.error(
                f"Rejected. {len(failed)} of {len(review.criteria)} criteria failed: "
                + ", ".join(failed),
                icon=":material/cancel:",
            )

        metrics = st.columns(3)
        metrics[0].metric(
            "Criteria passed",
            f"{sum(1 for c in review.criteria if c.passed)} of {len(review.criteria)}",
        )
        metrics[1].metric("LLM calls", result["usage"].calls)
        metrics[2].metric(
            "Tokens", result["usage"].input_tokens + result["usage"].output_tokens
        )

        attempt_col, review_col = st.columns([1, 1])

        with attempt_col:
            st.markdown("**Agent A — the attempt**")
            code = result["attempt"]
            # The worker is asked for a fenced block; show just the code inside it.
            if "```" in code:
                parts = code.split("```")
                if len(parts) >= 2:
                    body = parts[1]
                    code = body[len("python") :] if body.startswith("python") else body
            st.code(code.strip(), language="python")

        with review_col:
            st.markdown("**Agent B — criterion by criterion**")
            st.dataframe(
                [
                    {
                        "Criterion": c.name,
                        "Result": "PASS" if c.passed else "FAIL",
                        "Evidence": c.evidence,
                    }
                    for c in review.criteria
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "The review is a structured output, so the model cannot return a verdict "
                "without also filling in a per-criterion result and the evidence for it."
            )

        if not approved and review.reasons:
            st.markdown("**Agent B's specific reasons for rejection**")
            for i, reason in enumerate(review.reasons, 1):
                st.markdown(f"{i}. {reason}")

        ui.transcript_download(
            log, f"assignment-2-{'rejected' if not approved else 'approved'}-run.txt"
        )

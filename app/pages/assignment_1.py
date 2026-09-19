"""/assignment-1 — the tool-using research agent."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app import loader, ui  # noqa: E402

agent = loader.get("assignment_1")

ui.page_header(
    "Tool-Using Research Agent",
    "The agent picks its own path through four tools and decides for itself when it "
    "has enough to answer. The tool-call budget is enforced by the graph, not the "
    "prompt, so it cannot be talked past.",
    "assignment-1",
)

TOOL_DESCRIPTIONS = {
    "web_search": "Real DuckDuckGo search. No API key.",
    "service_metrics": "Stand-in for an internal metrics API. Fixed values, so runs compare.",
    "read_notes": "Reads the two internal notes in assignment-1/notes/.",
    "calculator": "Arithmetic over a restricted AST. Not eval.",
}

with st.expander("What the agent has to work with"):
    st.markdown(
        "Every tool takes a `reason` argument, so the model must state why it is making "
        "a call before making it. That is what turns the log below into a reasoning "
        "trace rather than a dump of inputs and outputs."
    )
    st.markdown(
        f"The graph will not let the agent answer until **{agent.MIN_DISTINCT_TOOLS} "
        "different tools have returned a usable result** — a failed call does not count. "
        "Which tools, and in what order, is left entirely to the agent."
    )
    st.dataframe(
        [{"Tool": name, "What it does": desc} for name, desc in TOOL_DESCRIPTIONS.items()],
        hide_index=True,
        width="stretch",
    )

if not ui.api_key():
    ui.missing_key_panel()
    st.stop()

st.subheader("Run settings")

question = st.text_area(
    "Question",
    value=agent.QUESTION,
    height=120,
    help="Open-ended by design. A single lookup cannot answer it.",
    disabled=ui.is_running(),
)

col_left, col_right = st.columns(2)

with col_left:
    fail_mode = st.selectbox(
        "Injected tool failure",
        options=["none", "timeout", "empty", "malformed"],
        help=(
            "Swaps one real tool call for a mocked bad response. The agent has to "
            "notice and adapt rather than crash or carry on regardless."
        ),
        disabled=ui.is_running(),
    )
    fail_tool = st.selectbox(
        "Which call the failure hits",
        options=["first", *agent.TOOL_NAMES],
        help=(
            "'first' hits whichever tool the agent reaches for first, so the failure "
            "always fires whatever path it picks. Naming a tool only fires if the "
            "agent chooses to call it."
        ),
        disabled=ui.is_running() or fail_mode == "none",
    )

with col_right:
    max_tool_calls = st.slider(
        "Tool-call budget",
        min_value=agent.MIN_TOOL_CALLS,
        max_value=agent.MAX_TOOL_CALLS,
        value=agent.MAX_TOOL_CALLS,
        help=(
            f"The hard cap for the run, {agent.MIN_TOOL_CALLS} to {agent.MAX_TOOL_CALLS}. "
            "Set it low to push the agent into the give-up path, where it stops itself "
            "and reports what it did establish."
        ),
        disabled=ui.is_running(),
    )
    if max_tool_calls < agent.MIN_DISTINCT_TOOLS:
        st.warning(
            f"A budget of {max_tool_calls} cannot satisfy the requirement that "
            f"{agent.MIN_DISTINCT_TOOLS} distinct tools return a usable result, so this "
            "run will stop short of an answer by design.",
            icon=":material/warning:",
        )
    elif max_tool_calls == agent.MIN_DISTINCT_TOOLS:
        st.info(
            "With this budget every call has to land: one failure leaves the agent "
            "unable to reach two distinct working tools.",
            icon=":material/info:",
        )

start = st.button(
    "Run agent",
    type="primary",
    disabled=ui.is_running() or not question.strip(),
    icon=":material/play_arrow:",
)

if start:
    st.subheader("Reasoning trace")
    log = ui.LiveLog(st.empty())
    result = None

    with ui.run_guard():
        with st.spinner("Running. The trace updates as the agent works."):
            result = agent.run(
                log,
                question=question,
                fail_mode=fail_mode,
                fail_tool=fail_tool if fail_mode != "none" else "first",
                max_tool_calls=max_tool_calls,
                api_key=ui.api_key(),
            )

    if result:
        st.subheader("Result")

        stopped_early = result["tool_calls_used"] >= result["max_tool_calls"] and (
            "budget ran out" in result["answer"].lower()
            or "budget was exhausted" in result["answer"].lower()
        )
        if stopped_early:
            st.warning(
                "The agent stopped itself at the budget without reaching an answer.",
                icon=":material/timer_off:",
            )

        metrics = st.columns(4)
        metrics[0].metric(
            "Tool calls", f"{result['tool_calls_used']} of {result['max_tool_calls']}"
        )
        metrics[1].metric(
            "Distinct tools that worked",
            f"{len(result['tools_succeeded'])} of {result['min_distinct_tools']} required",
            help="Which tools these are is the agent's choice; only the count is enforced.",
        )
        metrics[2].metric("Failed calls", len(result["failures"]))
        metrics[3].metric("LLM calls", result["usage"].calls)

        if result["tools_succeeded"]:
            st.caption("Tools that returned a usable result: " + ", ".join(result["tools_succeeded"]))
        if result["nudges"]:
            st.info(
                f"The agent tried to answer before {result['min_distinct_tools']} distinct "
                f"tools had worked and was routed back {result['nudges']} "
                f"time{'s' if result['nudges'] > 1 else ''}. It chose what to call next "
                "itself.",
                icon=":material/u_turn_left:",
            )

        if result["failures"]:
            st.markdown("**Tool calls that failed during this run**")
            st.dataframe(
                [{"Failure": f} for f in result["failures"]],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Failures are tracked in graph state, not just in the prompt, so they "
                "are reported here whether or not the model mentions them in its answer."
            )
        elif fail_mode != "none":
            st.caption(
                "No failure was recorded. If you targeted a specific tool, the agent "
                "may simply not have called it on this run."
            )

        st.markdown("**Final answer**")
        st.markdown(result["answer"])

        ui.transcript_download(log, f"assignment-1-{fail_mode}-run.txt")

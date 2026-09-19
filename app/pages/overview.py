"""Landing page: what the three assignments are and how to exercise them here."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app import ui  # noqa: E402

st.title("Agent Assignments")
st.caption(
    "Three LangGraph agents, each demonstrating a different thing that matters when an "
    "agent runs without supervision. Every page here runs the same code the command "
    "line runs."
)

with st.sidebar:
    st.subheader("Configuration")
    st.text(f"Model: {ui.model_name()}")
    if ui.api_key():
        st.success("API key configured", icon=":material/check_circle:")
    else:
        st.error("No API key configured", icon=":material/error:")

if not ui.api_key():
    ui.missing_key_panel()
    st.divider()

CARDS = [
    {
        "title": "1. Tool-Using Research Agent",
        "body": (
            "An agent that plans its own path through four tools, decides for itself "
            "when it has enough to answer, stops at a hard tool-call budget, and adapts "
            "when a tool call fails."
        ),
        "try": "inject a malformed tool response and watch it recover.",
        "page": "app/pages/assignment_1.py",
    },
    {
        "title": "2. Multi-Agent Task with Review",
        "body": (
            "A worker agent writes a Python function; a reviewer agent judges that one "
            "attempt against six concrete criteria and returns a structured verdict with "
            "evidence. One pass, no revision loop."
        ),
        "try": "switch the worker to weak and watch the review reject it.",
        "page": "app/pages/assignment_2.py",
    },
    {
        "title": "3. Resumable Agent with Self-Check",
        "body": (
            "Four documents summarised one at a time, checkpointed after each with "
            "LangGraph's SqliteSaver. Stop it partway, resume, and it skips what is "
            "already done. Then it checks its own results."
        ),
        "try": "stop after item 2, resume, then sabotage an item.",
        "page": "app/pages/assignment_3.py",
    },
]

# The three cards carry different amounts of text. Stretching each container to its
# column, and pushing the link down with a stretched spacer above it, keeps the borders
# the same height and the links on one line regardless.
for column, card in zip(st.columns(len(CARDS)), CARDS):
    with column:
        with st.container(border=True, height="stretch"):
            st.subheader(card["title"])
            st.markdown(card["body"])
            st.markdown(f"**Try:** {card['try']}")
            st.container(height="stretch", border=False)
            st.page_link(card["page"], label="Open", icon=":material/arrow_forward:")

st.divider()

st.subheader("How this app relates to the repository")
st.markdown(
    """
The assignments are the deliverable; this app is a way to try them. Each page imports
the same `run()` function the command line calls, so what you see here is the real
agent, not a demonstration of one. Nothing is mocked except the failures the
assignments are supposed to inject deliberately.

Each assignment folder holds its own README, the agent code, and transcripts committed
from command-line runs. Those transcripts are the graded artefacts; the pages here let
you produce fresh ones.

Runs take roughly fifteen to sixty seconds depending on the assignment and how many
tools the agent decides to use. The trace streams in as it happens, so you can follow
the agent's reasoning rather than waiting at a blank page.
"""
)

st.info(
    "This deployment uses a single shared OpenAI API key, so every run here draws on "
    "the same account. A run costs a fraction of a cent; if several people run agents "
    "at once you may briefly see a rate-limit message.",
    icon=":material/info:",
)

"""Streamlit entry point.

Exposes the three assignments at /assignment-1, /assignment-2 and /assignment-3 so
their behaviour can be exercised from a browser. The agents themselves are unchanged
and still run from the command line exactly as their READMEs describe.
"""

import streamlit as st

st.set_page_config(
    page_title="Agent Assignments",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    # The default page is served at "/", so giving it a url_path as well would advertise
    # a second address that does not resolve.
    st.Page("app/pages/overview.py", title="Overview", default=True),
    st.Page(
        "app/pages/assignment_1.py",
        title="1. Tool-Using Research Agent",
        url_path="assignment-1",
    ),
    st.Page(
        "app/pages/assignment_2.py",
        title="2. Multi-Agent Task with Review",
        url_path="assignment-2",
    ),
    st.Page(
        "app/pages/assignment_3.py",
        title="3. Resumable Agent with Self-Check",
        url_path="assignment-3",
    ),
]

st.navigation(PAGES).run()

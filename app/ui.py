"""Shared chrome for the Streamlit pages.

Holds the three things every page needs: the API key, a live log that streams an
agent's trace into the page as it is produced, and one place that turns an exception
into a readable message instead of a Streamlit traceback.
"""

import contextlib
import os
import re
import traceback
import uuid
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gpt-4.1-mini"
CONSOLE_KEYS_URL = "https://platform.openai.com/api-keys"


def model_name() -> str:
    """The model the agents will use, resolved the same way llm.py resolves it."""
    return _secret("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


def _secret(name: str):
    """Read a Streamlit secret without raising when no secrets file exists at all."""
    try:
        return st.secrets.get(name)
    except Exception:
        # No secrets.toml locally and none configured on the host — not an error here.
        return None


def api_key() -> str:
    """The OpenAI key from Streamlit secrets, falling back to the environment locally."""
    return (_secret("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()


def session_id() -> str:
    """A stable id for this browser session, used to keep per-session state apart."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]
    return st.session_state.session_id


def is_running() -> bool:
    return bool(st.session_state.get("running"))


def page_header(title: str, subtitle: str, folder: str) -> None:
    """Page title, one-line framing, and the sidebar status panel."""
    st.title(title)
    st.caption(subtitle)

    with st.sidebar:
        st.subheader("Configuration")
        st.text(f"Model: {model_name()}")
        if api_key():
            st.success("API key configured", icon=":material/check_circle:")
        else:
            st.error("No API key configured", icon=":material/error:")
        st.divider()
        st.subheader("Source")
        st.markdown(
            f"The code and full write-up for this assignment are in `{folder}/` "
            f"in the repository, along with the transcripts committed from CLI runs."
        )


def missing_key_panel() -> None:
    """Shown in place of the run controls when there is no key."""
    st.error(
        "This app needs an OpenAI API key before it can run anything.",
        icon=":material/key_off:",
    )
    st.markdown(
        f"""
**Running locally.** Create `.streamlit/secrets.toml` next to `streamlit_app.py`:

```toml
OPENAI_API_KEY = "sk-your_key_here"
```

**Deployed on Streamlit Community Cloud.** Open the app's menu, choose Settings, then
Secrets, and paste the same line.

Create or manage keys at {CONSOLE_KEYS_URL}.
"""
    )


class LiveLog:
    """Streams an agent's trace into the page as it is produced.

    The agents take a `log` callable and call it once per line. Rewriting a single
    placeholder on each call means the reasoning trace appears while the run is in
    flight rather than after a thirty-second blank wait.
    """

    def __init__(self, placeholder, height: int = 420) -> None:
        self._placeholder = placeholder
        self._height = height
        self._lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        self._lines.append(str(text))
        self._placeholder.code(self.text(), language="text", height=self._height)

    def text(self) -> str:
        return "\n".join(self._lines)

    def is_empty(self) -> bool:
        return not self._lines


def describe_exception(exc: BaseException) -> tuple[str, str]:
    """Turn an exception into (headline, detail) aimed at someone using the app.

    Matching is on class name rather than on imported OpenAI exception types, so this
    keeps working if the client library reorganises them.
    """
    name = type(exc).__name__
    text = str(exc)

    if name == "AuthenticationError" or "invalid_api_key" in text or "401" in text[:64]:
        return (
            "The configured OpenAI API key was rejected.",
            f"Check the key in the app's secrets, or issue a new one at {CONSOLE_KEYS_URL}. "
            "A key that was rotated or revoked will fail this way.",
        )
    if name == "RateLimitError" or "rate_limit" in text or "429" in text[:64]:
        # A 429 from OpenAI means either "too fast" or "out of credit", and the advice
        # is completely different, so they are worth separating.
        if "insufficient_quota" in text or "exceeded your current quota" in text:
            return (
                "This account has no OpenAI credit left.",
                "The key is valid, but its quota or billing balance is exhausted, so the "
                "API will not accept further requests. Add credit at "
                "https://platform.openai.com/settings/organization/billing.",
            )
        wait = ""
        match = re.search(r"try again in ([0-9hms.]+)", text)
        if match:
            wait = f" OpenAI suggests retrying in about {match.group(1)}."
        return (
            "OpenAI's rate limit was hit.",
            "Requests are arriving faster than this account's limit allows, and the "
            f"client already retried with backoff before giving up.{wait} Assignment 3 is "
            "the most likely to trigger this, since it makes eight calls in quick "
            "succession.",
        )
    if name == "NotFoundError" or "does not exist or you do not have access" in text:
        return (
            f"The model {model_name()} is not available to this API key.",
            "Not every account can reach every model. Set OPENAI_MODEL in the app's "
            "secrets to one your key supports, such as gpt-4.1-mini.",
        )
    if name in ("APIConnectionError", "APITimeoutError", "ConnectionError"):
        return (
            "Could not reach the OpenAI API.",
            "This looks like a network problem rather than anything in the agent. "
            "Check connectivity and try again.",
        )
    if name == "BadRequestError" or "400" in text[:64]:
        return (
            "The API rejected the request as malformed.",
            "This usually means the configured model does not support tool calling or "
            "structured output, both of which these agents rely on. "
            f"Detail from the API: {text}",
        )
    if isinstance(exc, ValueError):
        return ("That combination of settings is not valid.", text)
    if isinstance(exc, SystemExit):
        return ("The agent stopped before it could start.", text or "No detail was given.")

    return (f"The run failed with {name}.", text or "No further detail was available.")


@contextlib.contextmanager
def run_guard():
    """Wrap a run: mark the session busy, and turn any failure into a readable message.

    The busy flag is what stops a second click starting an overlapping run while the
    first is still going.
    """
    st.session_state.running = True
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 — the point is that nothing escapes
        headline, detail = describe_exception(exc)
        st.error(headline, icon=":material/error:")
        st.markdown(detail)
        with st.expander("Technical detail"):
            st.code("".join(traceback.format_exception(exc)), language="text")
    finally:
        st.session_state.running = False


def transcript_download(log: LiveLog, filename: str) -> None:
    """Offer the trace as a file, the same text the CLI would have written."""
    if log.is_empty():
        return
    st.download_button(
        "Download this transcript",
        data=log.text(),
        file_name=filename,
        mime="text/plain",
        icon=":material/download:",
    )

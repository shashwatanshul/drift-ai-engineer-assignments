"""OpenAI chat model plus a counter for LLM calls and tokens.

Each assignment folder carries its own copy so it can be run standalone.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# .env lives at the repo root, one level up from this folder.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Chosen for cost rather than capability: these agents need reliable tool calling and
# structured output, both of which this model does well, and a full run costs a fraction
# of a cent. Override with OPENAI_MODEL if you want something stronger.
DEFAULT_MODEL = "gpt-4.1-mini"


class Usage:
    """Running total of LLM calls and tokens across a run."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, response) -> None:
        self.calls += 1
        meta = getattr(response, "usage_metadata", None) or {}
        self.input_tokens += meta.get("input_tokens", 0)
        self.output_tokens += meta.get("output_tokens", 0)

    def report(self) -> str:
        total = self.input_tokens + self.output_tokens
        return (
            f"LLM calls: {self.calls} | "
            f"tokens: {self.input_tokens} in + {self.output_tokens} out = {total} total"
        )


def current_model() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def build_llm(temperature: float = 0.0, api_key: str | None = None) -> ChatOpenAI:
    """Build the chat model.

    `api_key` lets a caller pass the key explicitly — the Streamlit app reads it from
    st.secrets rather than the environment. Falling back to OPENAI_API_KEY keeps the CLI
    working from .env unchanged.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy .env.example to .env at the repo root "
            "and put your key in it (https://platform.openai.com/api-keys)."
        )
    return ChatOpenAI(
        model=current_model(),
        temperature=temperature,
        api_key=key,
        # Rate limits are per-account and a multi-call run can bump into them.
        # The client honours the Retry-After header, so this just waits it out.
        max_retries=5,
        timeout=120,
    )

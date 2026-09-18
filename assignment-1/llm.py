"""Groq chat model plus a counter for LLM calls and tokens.

Each assignment folder carries its own copy so it can be run standalone.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# .env lives at the repo root, one level up from this folder.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_MODEL = "openai/gpt-oss-120b"


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


def build_llm(temperature: float = 0.0) -> ChatGroq:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Copy .env.example to .env at the repo root "
            "and put your key in it (free key: https://console.groq.com/keys)."
        )
    return ChatGroq(
        model=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        temperature=temperature,
        # Groq's free tier caps tokens per minute, and a multi-call run bumps into it.
        # The client honours the Retry-After header, so this just waits it out.
        max_retries=8,
    )

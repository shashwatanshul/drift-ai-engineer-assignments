"""OpenAI chat model plus a counter for LLM calls and tokens.

Each assignment folder carries its own copy so it can be run standalone.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# .env lives at the repo root, one level up from this folder.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_MODEL = "gpt-4o"


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


def build_llm(temperature: float = 0.0) -> ChatOpenAI:
    api_key = os.environ.get("OPEN_AI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPEN_AI_API_KEY is not set. Copy .env.example to .env at the repo root "
            "and put your key in it (https://platform.openai.com/api-keys)."
        )
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        temperature=temperature,
        # Rate limits on a multi-call run are transient; the client honours Retry-After.
        max_retries=8,
    )

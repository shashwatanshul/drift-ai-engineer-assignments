# Junior AI Engineer — Take-Home Assignments

All three assignments, built on **LangGraph**. Each folder is self-contained and has its
own README covering setup, how to run it, and what its transcripts show.

| Folder | Assignment | What it demonstrates |
|---|---|---|
| [`assignment-1/`](assignment-1/) | Tool-Using Research Agent | An agent that plans its own path through four tools, decides for itself when it has enough, stops at a hard tool-call budget, and adapts when a tool call fails. |
| [`assignment-2/`](assignment-2/) | Multi-Agent Task with Review | A worker agent and a reviewer agent in a single pass. The reviewer returns a structured, evidence-backed verdict against six concrete criteria. |
| [`assignment-3/`](assignment-3/) | Resumable Agent with Self-Check | Four items processed one at a time, checkpointed with LangGraph's `SqliteSaver`, resumable after an interruption, with a self-check that catches a deliberately bad result. |

## Setup

Once, from this directory:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then put your API keys in `.env`. Assignment 1 runs on OpenAI; assignments 2 and 3 run on
Groq.

```
OPEN_AI_API_KEY=sk-...    # assignment 1 — https://platform.openai.com/api-keys
GROQ_API_KEY=gsk_...      # assignments 2 and 3 — https://console.groq.com/keys
```

That is all three assignments set up. Each folder also carries its own
`requirements.txt` if you would rather install them separately.

**Models:** assignment 1 uses `gpt-4o` on OpenAI, overridable with `OPENAI_MODEL` in
`.env`. Assignments 2 and 3 use `openai/gpt-oss-120b` on Groq's free tier, which supports
tool calling and structured output, overridable with `GROQ_MODEL`. The Groq free tier caps
tokens per minute, so that client is configured to retry and wait out a 429 rather than
fail the run.

## Quick tour

```bash
cd assignment-1 && python agent.py                      # research agent, clean run
cd assignment-1 && python agent.py --fail-mode malformed   # ... with a tool failure

cd assignment-2 && python chain.py                      # worker + reviewer → approved
cd assignment-2 && python chain.py --weak               # ... → rejected, with reasons

cd assignment-3 && python agent.py --reset --stop-after 2  # process 2 of 4, stop
cd assignment-3 && python agent.py                      # resume, finish, self-check
cd assignment-3 && python agent.py --reset --sabotage 3    # self-check catches a bad result
```

## Transcripts

Every transcript in the repo is the literal stdout of the run that produced it, not a
reconstruction. The runs are not seeded and the model is not deterministic, so re-running
will produce different wording — the behaviour each transcript demonstrates has been
stable across runs.

```
assignment-1/transcripts/  clean-run.txt, failure-run.txt, budget-exhausted-run.txt
assignment-2/transcripts/  approved-run.txt, rejected-run.txt
assignment-3/transcripts/  resume-run.txt, self-check-catches-bad-result.txt,
                           self-check-blank-result.txt
```

## Notes on shared code

`llm.py` is duplicated across the three folders rather than factored into a shared package,
so each assignment stands alone as the brief asks. It is about forty lines: it builds the
Groq client and tallies LLM calls and tokens from each response's `usage_metadata`.

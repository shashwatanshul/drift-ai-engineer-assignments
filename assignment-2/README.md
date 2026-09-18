# Assignment 2 — Multi-Agent Task with Review

Two LangGraph agents in a single chain. Agent A writes a Python function; Agent B reviews
that one attempt against a fixed list of criteria and returns a verdict. No revision loop —
B reviews once and the chain ends.

## The task

> Write a Python function `merge_intervals(intervals)`. It takes a list of `[start, end]`
> pairs and returns a new list of non-overlapping intervals covering the same points,
> sorted by start. Intervals that overlap or merely touch (e.g. `[1, 3]` and `[3, 5]`)
> must be merged into one.

Small, with a clear "good enough" bar, and enough edge cases that a review has something
real to say.

## Agent B's approval criteria

The verdict is **approved** only if all six pass. Any failure means **rejected**, with
reasons. These live in `CRITERIA` in `chain.py` and are injected verbatim into B's prompt,
so this list, the prompt, and the printed report cannot drift apart.

| Criterion | Bar |
|---|---|
| `correct_on_overlaps` | Merges overlapping *and* touching intervals. B is told to trace `[[1,3],[2,6],[8,10],[3,5]]`. |
| `handles_unsorted_input` | Correct when input is not already sorted by start. |
| `handles_empty_input` | Returns `[]` for `[]` rather than raising `IndexError`. |
| `efficient` | O(n log n) — a sort plus one pass. An O(n²) repeated-rescan fails. |
| `no_caller_mutation` | Does not mutate the caller's list. `intervals.sort()` fails; `sorted(intervals)` passes. |
| `documented` | Has a docstring saying what it takes and returns. |

Each criterion has a concrete test attached — an input to trace, or a specific construct to
look for — which is what makes a verdict explainable rather than a vibe.

## Setup

From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then put your free Groq key in it
```

Needs `langgraph`, `langchain-groq`, `python-dotenv`. Default model is
`openai/gpt-oss-120b` on Groq's free tier; override with `GROQ_MODEL` in `.env`.

## Running it

```bash
cd assignment-2
python chain.py                                     # normal worker → approved
python chain.py --weak                              # sloppy worker → rejected
python chain.py --weak --transcript transcripts/my-run.txt
```

## The chain

```
worker (Agent A) ──▶ reviewer (Agent B) ──▶ END
```

Two nodes, one edge between them, and no edge back to the worker. The single pass is a
property of the graph, not something the prompt asks for.

**Agent A** gets the task and returns one attempt. Nothing else.

**Agent B** gets the task, the criteria, and A's attempt — but not A's prompt, so it cannot
tell a normal draft from a `--weak` one and has to judge the code in front of it. It
returns a Pydantic-structured `Review` via `with_structured_output`:

```python
class Review(BaseModel):
    criteria: list[CriterionResult]   # name, passed, evidence — one per criterion
    verdict: Literal["approved", "rejected"]
    reasons: list[str]                # specific; empty when approved
```

Structured output is what keeps the review honest: the model cannot emit a verdict without
also filling in a per-criterion pass/fail and the evidence for it. Its prompt states that
"looks fine" and "seems correct" are not evidence — it must quote the line or name the
input that decided the call.

## How the rejection is forced

`--weak` swaps Agent A's system prompt for one that asks for a realistically hurried draft:
sort in place, use a repeated-rescan merge, no docstring. The code still runs and looks
plausible — it is a bad draft, not a broken one, so B has to find the problems rather than
trip over them. The reviewer and its criteria are identical in both runs, so the rejection
is earned, not staged.

In `transcripts/rejected-run.txt` B passes the three correctness criteria and fails
`efficient`, `no_caller_mutation` and `documented`, quoting `intervals.sort(...)` and the
`while True` rescan loop as evidence.

## Reporting

Both runs end with total LLM calls and token counts, summed from each response's
`usage_metadata` (see `Usage` in `llm.py`). Two calls per run — one per agent.

## Transcripts

| File | What it shows |
|---|---|
| `transcripts/approved-run.txt` | Normal worker, all six criteria pass, approved. |
| `transcripts/rejected-run.txt` | `--weak` worker, three failures, rejected with specific reasons. |

Each is the literal stdout of the run that produced it.

## Assumptions

- "Good enough" is defined entirely by the six criteria above. B is told to judge against
  those and nothing else, so it cannot reject over style preferences.
- B reviews by reading the code, not by executing it. That is why the criteria name
  specific inputs to trace — it keeps the judgement concrete without a sandbox.
- Token counts come from Groq's `usage_metadata`. `Usage` treats a missing field as zero
  rather than failing, so the report still prints on a provider that omits them.
- The run is not seeded and the model is not deterministic, so re-running will not
  reproduce these transcripts word for word. The verdicts have been stable across runs.

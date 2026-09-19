# Assignment 2 — Multi-Agent Task with Review

**Live deployed app:**  
https://drift-ai-engineer-assignments-krtt7sujdpij3m7jepsm9j.streamlit.app/assignment-2

This assignment implements a two-agent single-pass workflow using **LangGraph**.

Agent A produces one attempt. Agent B reviews that single attempt against concrete,
predefined criteria and returns either an approved verdict or a rejected verdict with
specific reasons.

There is no revision or negotiation loop.

## Task

Agent A is asked to implement:

```python
merge_intervals(intervals)
```

The function receives a list of `[start, end]` intervals and must return a new list of
non-overlapping intervals covering the same points, sorted by start.

Intervals that overlap or merely touch must be merged.

Example:

```text
[1, 3] and [3, 5] → [1, 5]
```

## Agent B approval criteria

Agent B evaluates the attempt against six concrete criteria.

| Criterion | Requirement |
|---|---|
| `correct_on_overlaps` | Correctly merges overlapping and touching intervals. |
| `handles_unsorted_input` | Correctly handles input that is not already sorted. |
| `handles_empty_input` | Returns `[]` for an empty list without raising an error. |
| `efficient` | Uses O(n log n) behavior: sorting followed by a single merge pass. |
| `no_caller_mutation` | Does not mutate the caller's outer list or the supplied inner interval lists. |
| `documented` | Includes a docstring explaining the function's input and returned value. |

The verdict is **approved only when all six criteria pass**.

If one or more criteria fail, Agent B returns **rejected** with specific reasons and
per-criterion evidence.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key:

```env
OPENAI_API_KEY=your_key_here
```

The default model is `gpt-4.1-mini`.

It can be overridden with:

```env
OPENAI_MODEL=your_model_name
```

## Running the chain

Enter the assignment folder:

```bash
cd assignment-2
```

Normal run:

```bash
python chain.py
```

Deliberately weaker Agent A output:

```bash
python chain.py --weak
```

Save another transcript:

```bash
python chain.py --weak --transcript transcripts/my-run.txt
```

## LangGraph workflow

The workflow is deliberately single-pass:

```text
Agent A / Worker
       |
       v
Agent B / Reviewer
       |
       v
      END
```

There is no edge from Agent B back to Agent A.

Agent A therefore produces exactly one attempt, Agent B reviews it once, and the chain
ends.

## Agent A — Worker

Agent A receives the programming task and returns one implementation attempt.

In normal mode, it is instructed to produce a complete solution.

With:

```bash
python chain.py --weak
```

only Agent A's prompt changes. It is asked to produce a realistically weaker first draft.

The reviewer, criteria, and graph structure remain unchanged.

## Agent B — Reviewer

Agent B receives:

- the original programming task,
- the six approval criteria,
- Agent A's single attempt.

The reviewer uses structured output containing:

```python
criteria
verdict
reasons
```

For every criterion it returns a pass/fail decision together with evidence.

The overall verdict is:

```text
approved
```

only when every criterion passes.

Otherwise it is:

```text
rejected
```

with specific rejection reasons.

## Approved transcript

The required approved example is:

```text
transcripts/approved-run.txt
```

In this run, Agent B passes all six criteria and reports:

```text
VERDICT: APPROVED
```

The approved Agent A implementation is reported as the final output.

## Rejected transcript

The required rejected example is:

```text
transcripts/rejected-run.txt
```

In the current transcript Agent B passes:

```text
correct_on_overlaps
handles_unsorted_input
handles_empty_input
no_caller_mutation
```

and fails:

```text
efficient
documented
```

The weak implementation first copies the supplied interval lists, so the caller's data is
not mutated.

However, its repeated scan-and-pop approach may require O(n²) work, and the implementation
does not contain a docstring.

Agent B therefore returns:

```text
VERDICT: REJECTED
```

with specific reasons for both failed criteria.

## LLM usage

Each run uses two agent calls:

```text
1. Agent A — Worker
2. Agent B — Reviewer
```

The program reports total LLM calls and available token usage at the end of the run.

Example:

```text
LLM calls: 2 | tokens: ...
```

## Transcripts

| File | What it demonstrates |
|---|---|
| `transcripts/approved-run.txt` | Required run where Agent B approves Agent A's output. |
| `transcripts/rejected-run.txt` | Required run where Agent B rejects Agent A's output and provides specific reasons. |

## Assumptions

- "Good enough" is defined entirely by the six explicit criteria above.
- Agent B reviews against those criteria rather than introducing unrelated style
  preferences.
- Agent B reviews the code rather than executing it, so the criteria contain concrete
  examples and constructs that can be inspected directly.
- The model is not seeded, so exact code and explanation wording may vary between runs.

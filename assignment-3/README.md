# Assignment 3 — Resumable Agent with Basic Self-Check

**Live deployed app:**  
https://drift-ai-engineer-assignments-krtt7sujdpij3m7jepsm9j.streamlit.app/assignment-3

This assignment implements a resumable agent using **LangGraph** and LangGraph's built-in
`SqliteSaver` checkpointing.

The agent processes four files one at a time, persists progress after each completed item,
can resume after an interruption without repeating completed work, and performs a final
self-check over the stored results.

## Task

The agent summarises these four files from `docs/`, one at a time and in order:

```text
caching.md
queues.md
indexing.md
observability.md
```

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

It can be overridden using:

```env
OPENAI_MODEL=your_model_name
```

## Resume demonstration

Enter the assignment folder:

```bash
cd assignment-3
```

Start with a fresh checkpoint and deliberately stop after item 2:

```bash
python agent.py --reset --stop-after 2
```

The first run processes:

```text
caching.md
queues.md
```

and then stops after their progress has been checkpointed.

Run the program again:

```bash
python agent.py
```

The second invocation loads the saved LangGraph state, skips the first two completed
files, processes items 3 and 4, and runs the final self-check.

## Deliberately bad result

To deliberately store a wrong result for item 3:

```bash
python agent.py --reset --sabotage 3
```

This replaces the generated summary for `indexing.md` with an unrelated but non-empty
summary.

The final self-check should detect that the result does not match the source file.

A blank result can also be tested:

```bash
python agent.py --reset --sabotage 2 --sabotage-mode blank
```

## Options

| Flag | Meaning |
|---|---|
| `--reset` | Deletes the current checkpoint and starts again from item 1. |
| `--stop-after N` | Stops after item N has completed and been checkpointed. |
| `--sabotage N` | Deliberately corrupts the result stored for item N. |
| `--sabotage-mode wrong` | Stores an unrelated non-empty summary. |
| `--sabotage-mode blank` | Stores an empty summary. |
| `--transcript PATH` | Writes the run output to the specified transcript file. |

Ctrl+C can also interrupt execution. Items that completed and were checkpointed before
the interruption remain saved for the next run.

## Saved state

Persistence uses LangGraph's built-in:

```python
SqliteSaver
```

For the command-line version, the checkpoint database is:

```text
checkpoints.sqlite
```

The state is intentionally simple and readable:

```python
{
    "items": [
        "caching.md",
        "queues.md",
        "indexing.md",
        "observability.md"
    ],
    "index": 2,
    "results": {
        "caching.md": "...",
        "queues.md": "..."
    },
    "llm_calls": 2,
    "check": None
}
```

### `items`

The ordered list of files to process.

### `index`

The number of completed items and therefore the position of the next item.

### `results`

The summaries already completed.

### `llm_calls`

The running LLM-call count. It is stored in checkpoint state so it survives a resume.

### `check`

The final self-check report, populated once all items are complete.

## Processing and checkpointing

The graph processes one item at a time:

```text
items[index]
```

After processing an item, the node returns:

```text
index + 1
updated results
updated LLM-call total
```

Returning from the LangGraph node causes the updated state to be checkpointed.

The deliberate `--stop-after` interruption occurs only after the completed item's state
has been returned, so completed work is safely persisted before the run stops.

## Resume behavior

When the program starts, it checks LangGraph for saved state.

If no checkpoint exists, processing begins with the first item.

If a checkpoint exists, previously completed items are logged as:

```text
SKIP
```

and the graph continues from the saved `index`.

For example, after stopping at item 2, the resumed execution shows:

```text
SKIP caching.md
SKIP queues.md
PROCESS indexing.md
PROCESS observability.md
```

Items 1 and 2 therefore do not run again and do not consume additional summarisation
calls.

## Self-check

After all four items have been processed, the graph runs a final validation step.

### Empty-result check

If a stored result is blank or only whitespace, it fails locally:

```text
FAIL — summary is empty
```

No LLM call is needed for an empty result.

### Content-match check

For every non-empty result, the original document and stored summary are given to the
model.

The model is asked whether the summary accurately describes the source document.

The result is recorded as either:

```text
PASS
```

or:

```text
FAIL
```

with a short explanation.

## Demonstrating that the checker works

The command:

```bash
python agent.py --reset --sabotage 3
```

deliberately replaces the `indexing.md` summary with unrelated information about an
expense reimbursement policy.

The final checker compares this result with the real `indexing.md` document and flags it
as incorrect.

This demonstrates that the self-check does not automatically approve every stored result.

## LLM-call reporting

The LLM-call total is stored in checkpoint state.

This means the final count includes calls made before an interruption as well as calls
made after resuming.

In the normal resume demonstration:

```text
4 summary calls
+
4 self-check calls
=
8 total LLM calls
```

Completed items are not summarised again after the resume.

## Transcripts

| File | What it demonstrates |
|---|---|
| `transcripts/resume-run.txt` | Required demonstration: first run stops after item 2, second run skips completed items and finishes. |
| `transcripts/self-check-catches-bad-result.txt` | Required demonstration of the checker catching an intentionally incorrect result. |
| `transcripts/self-check-blank-result.txt` | Additional demonstration of the local blank-result validation. |

`resume-run.txt` contains both invocations so the stop and subsequent resume can be seen
in a single log.

## Assumptions

- Progress is considered persisted once the LangGraph node handling an item returns and
  its state is checkpointed.
- `checkpoints.sqlite` is a runtime artifact and is intentionally excluded from Git.
- `--reset` is used when a completely fresh demonstration is required.
- LLM-call count is persisted across resumed executions.
- A non-empty summary must roughly match its source to pass the model-based check.
- The model is not seeded, so generated summaries and explanations may vary between runs.

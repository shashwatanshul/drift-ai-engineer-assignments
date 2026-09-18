# Assignment 3 — Resumable Agent with Basic Self-Check

A LangGraph agent that summarises four files one at a time, checkpointing after each. If
it is interrupted, re-running it picks up where it stopped and skips what is already done.
Once all four are finished it re-reads the results and checks each one.

## The task

Summarise the four files in `docs/` — `caching.md`, `queues.md`, `indexing.md`,
`observability.md` — one at a time, in order.

## Setup

From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then put your free Groq key in it
```

Needs `langgraph`, `langgraph-checkpoint-sqlite`, `langchain-groq`, `python-dotenv`.
Default model is `openai/gpt-oss-120b` on Groq's free tier; override with `GROQ_MODEL`.

## Running it

```bash
cd assignment-3

# The resume demo:
python agent.py --reset --stop-after 2    # does items 1-2, stops
python agent.py                           # skips 1-2, does 3-4, then self-checks

# The self-check catching a bad result:
python agent.py --reset --sabotage 3                      # off-topic summary for item 3
python agent.py --reset --sabotage 2 --sabotage-mode blank  # blank summary for item 2
```

| Flag | Meaning |
|---|---|
| `--reset` | Delete the checkpoint and start from item 1. |
| `--stop-after N` | Interrupt the run after item N is checkpointed. |
| `--sabotage N` | Deliberately store a bad result for item N. |
| `--sabotage-mode {wrong,blank}` | `wrong` (default) stores an off-topic summary; `blank` stores an empty one. |
| `--transcript PATH` | Also append the run to a file. |

Ctrl+C works the same way as `--stop-after`: the checkpoint is written after each item, so
whatever finished is kept and the next run resumes from there.

## The saved state

Persistence is LangGraph's own `SqliteSaver`, writing to `checkpoints.sqlite` under a
fixed thread ID (`summarise-docs`). Nothing is hand-rolled. The state it checkpoints is
deliberately plain:

```python
{
  "items":     ["caching.md", "queues.md", "indexing.md", "observability.md"],
  "index":     2,                       # how many are done = the next one to process
  "results":   {"caching.md": "...", "queues.md": "..."},
  "llm_calls": 2,                       # carried in state, so it survives a resume
  "check":     None,                    # the self-check report, filled in at the end
}
```

`index` is the whole resume mechanism: it is the count of finished items and the position
of the next one, so "where did I get to" is a single integer.

## How resume works

```
        ┌──────────────────────────┐
        ▼                          │
  ┌─────────────┐  more items?  ───┘
  │ process_one │──────────────▶
  └─────────────┘  all done? ──▶ ┌────────────┐
                                 │ self_check │──▶ END
                                 └────────────┘
```

`process_one` handles `items[index]`, then returns `index + 1` along with the new result.
Returning from the node is what writes the checkpoint, so an item is only ever marked done
after its summary is safely stored. That ordering is why `--stop-after` raises on the
*next* hop rather than inside the node — raising inside would throw away the checkpoint
for the item that just finished.

On startup the agent reads the checkpoint for its thread. If one exists, it logs a `SKIP`
line for each completed item and feeds the saved state straight back into the graph. The
graph re-enters at the top, but `process_one` works from `index`, so completed items are
never re-read and never cost another LLM call. The entry point is conditional, so a run
that was interrupted after the last item goes straight to the self-check.

In `transcripts/resume-run.txt` the first invocation processes items 1–2 and stops; the
second skips those two, processes 3–4, and runs the check. Total LLM calls across both
invocations is 8 — four summaries plus four checks — which is what it would have been
without the interruption. Nothing was redone.

## The self-check

After all four items are done, `self_check` re-reads `results` and checks each one:

1. **Empty test, locally.** A blank or whitespace-only summary fails immediately — no
   point asking the model about an empty string.
2. **Match test, one LLM call per item.** The model gets the original file and the stored
   summary and answers "does this summary accurately describe this document? yes or no",
   plus one line of reasoning, which is printed as the reason for the verdict.

`--sabotage` corrupts one result on purpose so the check has something to catch. The two
modes exercise the two halves:

- `--sabotage 3` (default, `wrong`) stores a plausible-looking summary about an expense
  reimbursement policy for `indexing.md`. Nothing structural is wrong with it — only the
  model comparison can catch it. See `transcripts/self-check-catches-bad-result.txt`:

  ```
  [FAIL] indexing.md — The summary talks about expense reimbursement policies, which
  bears no relation to the document that discusses OpenSearch indexing, CDC streams,
  reindexing, shard counts ...
  ```

- `--sabotage 2 --sabotage-mode blank` stores an empty summary, caught by the local test
  without an LLM call. See `transcripts/self-check-blank-result.txt`.

## Transcripts

| File | What it shows |
|---|---|
| `transcripts/resume-run.txt` | Run → stopped after item 2 → re-run → completes. Shows which items were skipped and which were newly processed. |
| `transcripts/self-check-catches-bad-result.txt` | An off-topic summary for item 3, caught by the model check. |
| `transcripts/self-check-blank-result.txt` | A blank summary for item 2, caught by the local check. |

Each is the literal stdout of the runs that produced it. `--transcript` appends, which is
how the two invocations of the resume demo end up in one file.

## Assumptions

- "Persists progress after each item" is taken literally: one checkpoint per item, written
  when the node returns.
- `checkpoints.sqlite` is gitignored — it is a runtime artifact, and the resume behaviour
  is evidenced by the transcripts. Any run starting fresh should pass `--reset`.
- The LLM-call total is the count across every run that contributed to the results, since
  it lives in the checkpointed state. That is the more useful number for a resumable agent
  than a per-process count.
- The self-check reads the model's first word for yes/no. A response that starts any other
  way is treated as a failure, which errs toward flagging rather than silently passing.
- The run is not seeded and the model is not deterministic, so re-running will not
  reproduce these transcripts word for word.

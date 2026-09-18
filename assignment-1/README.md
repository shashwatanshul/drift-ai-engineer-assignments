# Assignment 1 — Tool-Using Research Agent

A LangGraph agent that answers an open-ended engineering question by deciding its own
path through four tools, stopping when it judges it has enough, and adapting when a tool
call fails.

## The question it answers

> For our product read API — which currently serves about 10k reads/sec straight from
> PostgreSQL with no cache — what caching strategy should we adopt? Recommend one concrete
> design, justify it against our actual traffic numbers and constraints, and say what it
> costs us in memory and in staleness.

No single lookup answers this. A useful answer needs our production numbers, our internal
constraints (staleness budget, no new managed services, stampede risk), some arithmetic to
size the cache, and general caching knowledge — and the agent has to combine them.

## Setup

From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then put your free Groq key in it
```

Needs `langgraph`, `langchain-groq`, `python-dotenv`, `ddgs`. Default model is
`openai/gpt-oss-120b` on Groq's free tier; override with `GROQ_MODEL` in `.env`.

## Running it

```bash
cd assignment-1
python agent.py                                    # clean run
python agent.py --fail-mode malformed              # one tool call returns garbage
python agent.py --fail-mode timeout --fail-tool web_search
python agent.py --max-tool-calls 1                 # force the budget-exhausted path
python agent.py --transcript transcripts/my-run.txt
```

| Flag | Meaning |
|---|---|
| `--fail-mode {none,timeout,empty,malformed}` | Inject a mocked bad response into one tool call. |
| `--fail-tool {first,<tool name>}` | Which call the failure hits. `first` (default) hits whichever tool the agent reaches for first, so it always fires. |
| `--max-tool-calls N` | Tool-call budget. Default 6. |
| `--transcript PATH` | Also write the reasoning trace to a file. |
| `--question TEXT` | Ask something else. |

## Tools

| Tool | What it does |
|---|---|
| `web_search` | Real DuckDuckGo search via `ddgs`. No API key. |
| `service_metrics` | Stand-in for our internal metrics API — read QPS, payload size, hot-key share, staleness budget. Fixed values, so runs are reproducible. |
| `read_notes` | Reads `notes/architecture.md` and `notes/constraints.md`. |
| `calculator` | Arithmetic over a restricted AST. Not `eval`. |

Every tool takes a `reason` argument, so the model must state why it is making a call
before making it. That is what turns the log into a reasoning trace instead of a dump of
inputs and outputs.

## How it plans

`agent.py` builds a three-node graph:

```
          ┌─────────────────────────────┐
          ▼                             │
    ┌──────────┐   wants a tool    ┌──────────┐
──▶ │  agent   │──────────────────▶│  tools   │
    └──────────┘                   └──────────┘
        │     │
        │     └── would exceed budget ──▶ give_up ──▶ END
        └── no tool call ─────────────────────────▶ END
```

The model picks the next tool on every hop. Nothing sequences the tools, and nothing tells
it how many iterations to run — the loop ends when it stops emitting tool calls. Across
runs it takes different paths (three calls in one transcript, four in another) for the same
question, which is the difference between a planner and a pipeline in a costume.

The one piece of hardcoded control flow is the budget, and that is deliberate: it is
enforced in the routing function, not in the prompt, so the model cannot talk its way past
it. If the model's next batch of tool calls would push the run over `--max-tool-calls`,
routing diverts to `give_up`, which reports that it stopped without an answer, what it did
establish, and what it would have checked next. See
`transcripts/budget-exhausted-run.txt`.

## How failure is handled

`--fail-mode` swaps one real tool call for a mocked bad response — a timeout, an empty
result set, or a truncated JSON body. Two rules keep this from crashing or being ignored:

1. **Tools never raise.** Failures come back as a `TOOL_ERROR: ...` string, so a bad
   response is something the agent has to read and react to. Real exceptions (network
   down, rate limit) are caught and converted to the same shape.
2. **The failure is carried in graph state**, not just in the prompt. Every failed call is
   appended to `state["failures"]`, which is re-injected into the model's context on each
   subsequent turn with an instruction to acknowledge it, and printed as a
   `tool failures this run:` block at the end of the run. So the transcript records the
   failure whether or not the model chooses to mention it.

In `transcripts/failure-run.txt` the first `read_notes` call returns a truncated JSON body.
The agent marks it `[FAILED]`, re-lists the available notes, reads the other one, pulls the
numbers it still needs from `service_metrics`, and answers — never getting the content of
the file that failed, and never pretending it did.

## Transcripts

| File | What it shows |
|---|---|
| `transcripts/clean-run.txt` | Clean run, no failures, final answer. |
| `transcripts/failure-run.txt` | Same question with a mocked malformed response, and the recovery. |
| `transcripts/budget-exhausted-run.txt` | The agent stopping itself at the budget without an answer. |

Each is the literal stdout of the run that produced it (`--transcript` tees the trace).
The trace format is one block per tool call:

```
--- step 1/6 ---
decided : call read_notes(filename='architecture.md')
why     : Understand overall system architecture before choosing a caching strategy
result  : [FAILED] TOOL_ERROR: read_notes returned malformed data ...
```

## Assumptions

- The internal notes and `service_metrics` values are invented for the exercise. They are
  fixed rather than random so runs are comparable.
- "At least 2 distinct tools" is satisfied by the agent's own choices, not forced. Most
  runs use two or three of the four; which ones vary.
- The budget-exhausted transcript was produced with `--max-tool-calls 1`. Whether that
  path triggers on any given run depends on whether the model decides to answer from the
  first tool result — re-run if it answers instead. The behaviour under the default budget
  of 6 is the normal case, shown in the other two transcripts.
- The run is not seeded and the model is not deterministic, so re-running will not
  reproduce these transcripts word for word.

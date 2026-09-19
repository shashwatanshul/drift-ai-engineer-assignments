# Assignment 1 — Tool-Using Research Agent

A LangGraph agent that answers an open-ended engineering question by deciding its own
path through four tools, stopping when it judges it has enough, and adapting when a tool
call fails.

Two limits are enforced by the graph rather than the prompt: a tool-call budget of 1 to 6,
and a floor of two distinct tools that must return a usable result before the agent is
allowed to answer. Which tools those are, and in what order, is left entirely to the agent.

## The question it answers

> For our product read API — which currently serves about 10k reads/sec straight from
> PostgreSQL with no cache — what caching strategy should we adopt? Recommend one concrete
> design, justify it against our actual traffic numbers and constraints, and say what it
> costs us in memory and in staleness.

No single lookup answers this. A useful answer needs our production numbers, our internal
constraints (staleness budget, no new managed services, stampede risk), some arithmetic to
size the cache, and general caching knowledge — and the agent has to combine them.

> **Try it in a browser.** `streamlit run streamlit_app.py` from the repo root, then open `/assignment-1`. The page exposes the same options as the flags below and streams the trace live. See [DEPLOY.md](../DEPLOY.md).

## Setup

From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then put your OpenAI key in it
```

Needs `langgraph`, `langchain-openai`, `python-dotenv`, `ddgs`. Default model is
`gpt-4.1-mini`; override with `OPENAI_MODEL` in `.env`.

## Running it

```bash
cd assignment-1
python agent.py                                    # clean run
python agent.py --fail-mode malformed              # one tool call returns garbage
python agent.py --fail-mode timeout --fail-tool web_search
python agent.py --max-tool-calls 1                 # force the give-up path (1 to 6 only)
python agent.py --transcript transcripts/my-run.txt
```

| Flag | Meaning |
|---|---|
| `--fail-mode {none,timeout,empty,malformed}` | Inject a mocked bad response into one tool call. |
| `--fail-tool {first,<tool name>}` | Which call the failure hits. `first` (default) hits whichever tool the agent reaches for first, so it always fires. |
| `--max-tool-calls N` | Tool-call budget. An integer from 1 to 6; anything outside that range is rejected by argparse before the run starts. Default 6. |
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

`agent.py` builds a four-node graph:

```
          ┌───────────────────────────────────────────────┐
          ▼                                               │
    ┌──────────┐   wants a tool, and affords it     ┌──────────┐
──▶ │  agent   │───────────────────────────────────▶│  tools   │
    └──────────┘                                    └──────────┘
        │  │  │
        │  │  └── would exceed budget ──────────────▶ give_up ──▶ END
        │  │
        │  └── wants to answer, under 2 distinct working tools:
        │         budget or patience left ──▶ needs_more_tools ──┘
        │         neither ──────────────────▶ give_up ──▶ END
        │
        └── wants to answer, 2+ distinct working tools ────────▶ END
```

The model picks the next tool on every hop. Nothing sequences the tools, and nothing tells
it how many iterations to run. Across runs it takes different paths for the same question,
which is the difference between a planner and a pipeline in a costume.

Two rules are hardcoded, and both live in the routing function rather than the prompt, so
the model cannot talk its way past either.

### The tool-call budget is 1 to 6

Six is the assignment's cap and one is the floor; a run with no tool calls is not a
tool-using agent. The range is enforced in three places, so there is no path to a value
outside it:

- `_budget_arg` is the argparse type for `--max-tool-calls`, so an out-of-range value
  fails with a usage message before the run starts.
- `run()` re-validates, since it is also called directly by the Streamlit page and by
  tests. Non-integers are rejected too.
- The slider on `/assignment-1` is bounded by the same constants, so the UI cannot offer
  a value the agent would refuse.

If the model's next batch of tool calls would push the run over the budget, routing
diverts to `give_up`, which reports that it stopped without an answer, what it did
establish, and what it would have checked next. See `transcripts/budget-exhausted-run.txt`.

### At least two distinct tools must work before it may answer

A single lookup cannot answer this question, so the graph does not accept an answer built
on one. `state["tools_succeeded"]` accumulates the names of tools that returned a usable
result, deduplicated. When the model stops emitting tool calls, routing checks that list:

- Two or more distinct names, and the run ends normally.
- Fewer, and it goes to `needs_more_tools`, which appends one short system message and
  routes straight back to `agent`.

**A failed call does not count.** Two calls that both returned `TOOL_ERROR` leave the
agent exactly where it started, which is what stops the requirement being satisfied by
noise. Neither does calling one tool twice — the list is deduplicated.

**Nothing names a tool.** The check counts distinct successes; it has no opinion about
which tools or in what order, and the message sent back deliberately does not suggest one.
Picking the next tool stays the agent's job. In `transcripts/sent-back-run.txt` the agent
is sent back after one successful call and chooses `read_notes` on its own.

Two escape hatches keep this from looping forever, because a rule that can deadlock is
worse than no rule:

- If the budget is spent, being sent back is pointless — routing goes to `give_up`.
- If the model has already been sent back `MAX_NUDGES` (3) times and still will not call a
  tool, routing goes to `give_up` rather than spinning.

Either way `give_up` says which of the two reasons stopped the run, and the final report
prints `distinct tools that worked: N/2 required`, so a run that fell short is visible
rather than quietly passed off as an answer.

A budget of 1 therefore cannot produce an answer by construction: one call can never
satisfy two distinct tools. That is not a bug, and the Streamlit page warns about it
before you run.

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
| `transcripts/budget-exhausted-run.txt` | The agent stopping itself at the budget without an answer, produced with `--max-tool-calls 1`. |
| `transcripts/sent-back-run.txt` | The agent trying to answer after one successful tool, being routed back, and choosing its own next tool. |

Each is the literal stdout of the run that produced it (`--transcript` tees the trace).
The trace format is one block per tool call:

```
--- step 1/6 ---
decided : call read_notes(filename='architecture.md')
why     : Understand overall system architecture before choosing a caching strategy
result  : [FAILED] TOOL_ERROR: read_notes returned malformed data ...
```

and every run ends with the two enforced rules accounted for:

```
distinct tools that worked: 2/2 required (service_metrics, read_notes)
tool calls used: 2/6
times sent back for answering too early: 1
```

## Tests

The control-flow rules are covered by `tests/test_assignment_1_budget.py` at the repo
root, which drives the routing function directly with stub messages rather than through a
live model, so the assertions are about the graph rather than whatever the model felt like
doing on the day:

```bash
pytest tests/test_assignment_1_budget.py      # no API key needed
```

## Assumptions

- The internal notes and `service_metrics` values are invented for the exercise. They are
  fixed rather than random so runs are comparable.
- The two-distinct-tools rule is a floor, not a script. Most runs clear it without ever
  being sent back, using two to four of the tools; which ones vary between runs.
- `sent-back-run.txt` was produced with a deliberately narrow question
  (`--question "What is our read_qps? Reply with just the number."`), since the default
  question needs several tools anyway and rarely trips the rule.
- The run is not seeded and the model is not deterministic, so re-running will not
  reproduce these transcripts word for word.

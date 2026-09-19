# Assignment 1 — Tool-Using Research Agent

**Live deployed app:**  
https://drift-ai-engineer-assignments-krtt7sujdpij3m7jepsm9j.streamlit.app/assignment-1

This assignment implements a single autonomous research agent using **LangGraph**.

The agent decides which tools to use and in what order, gathers enough information to
answer an open-ended engineering question, records a readable reasoning trace, respects
a hard tool-call limit, and adapts when a tool fails.

## Question

> For our product read API — which currently serves about 10k reads/sec straight from
> PostgreSQL with no cache — what caching strategy should we adopt? Recommend one concrete
> design, justify it against our actual traffic numbers and constraints, and say what it
> costs us in memory and in staleness.

The question cannot be answered from a single lookup. The agent must reason across
production-style metrics, internal architecture and constraint notes, general caching
information, and arithmetic when useful.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key to `.env`:

```env
OPENAI_API_KEY=your_key_here
```

The default model is `gpt-4.1-mini`.

To use another supported OpenAI model:

```env
OPENAI_MODEL=your_model_name
```

## Running the agent

Enter the assignment folder:

```bash
cd assignment-1
```

Clean run:

```bash
python agent.py
```

Run with a mocked malformed tool response:

```bash
python agent.py --fail-mode malformed
```

Target a particular tool with a mocked timeout:

```bash
python agent.py --fail-mode timeout --fail-tool web_search
```

Change the tool-call budget:

```bash
python agent.py --max-tool-calls 3
```

The accepted range is **1 to 6**.

Save another transcript:

```bash
python agent.py --transcript transcripts/my-run.txt
```

Ask a different question:

```bash
python agent.py --question "Your question here"
```

## Tools

The agent has four available tools.

| Tool | Purpose |
|---|---|
| `web_search` | Performs a real web search using DuckDuckGo through `ddgs`. |
| `service_metrics` | Provides fixed production-style metrics such as read QPS, write QPS, latency, payload size, number of distinct keys, hot-key traffic share, backing store, and tolerated staleness. |
| `read_notes` | Reads the supplied `notes/architecture.md` and `notes/constraints.md` files. |
| `calculator` | Performs arithmetic using a restricted expression parser. |

The agent chooses which tools to use. Their order is not hardcoded.

## Autonomous planning

The LangGraph workflow contains four nodes:

```text
agent
  ├── tools ────────────────> agent
  ├── needs_more_tools ─────> agent
  ├── give_up ──────────────> END
  └── valid final answer ───> END
```

The model selects its next action based on what it has learned so far.

If it requests one or more tools, execution moves to `tools`, the requested calls are
performed, and the results are returned to the model.

If it attempts to answer before enough distinct tools have succeeded,
`needs_more_tools` sends it back to continue investigating without prescribing which
tool it must choose.

The research path is therefore selected dynamically by the agent rather than implemented
as a fixed step-by-step pipeline.

## Tool-call limit

The assignment requires a maximum of six tool calls per run.

This implementation allows a configurable budget from **1 to 6**, with a default of 6.

The range is validated before execution. The routing logic also checks requested calls
before executing them.

If a requested batch would exceed the remaining budget, the graph routes to `give_up`
rather than executing calls beyond the limit.

The additional transcript:

```text
transcripts/budget-exhausted-run.txt
```

demonstrates this behavior.

## At least two distinct tools

A normal final answer is accepted only after at least **two distinct tools** have returned
usable results.

Repeated calls to the same tool count only once.

A failed call does not count as a successful tool.

If the model attempts to answer too early, the graph routes it through
`needs_more_tools` and then back to the agent.

The graph checks the number of distinct successful tools but does not prescribe which
tools must be used or their order.

The additional transcript:

```text
transcripts/sent-back-run.txt
```

shows the model attempting to answer after one successful tool, being sent back, and then
choosing another tool itself.

## Reasoning trace

Every executed tool call logs:

```text
decided : the action/tool selected
why     : why the agent chose that action
result  : the returned result
```

For example:

```text
--- step 1/6 ---
decided : call service_metrics(...)
why     : gather current production metrics
result  : [ok] ...
```

This provides a readable decision trace rather than only raw tool inputs and outputs.

## Failure handling

Failures can be deliberately injected with:

```bash
--fail-mode timeout
--fail-mode empty
--fail-mode malformed
```

A mocked failure returns a `TOOL_ERROR` result instead of crashing the program.

Failures are also stored in graph state so that they remain visible throughout the run
and are included in the final report.

The agent is instructed to notice a failed result and adapt by doing one of the following:

- retrying with different arguments,
- obtaining the information from another tool,
- or continuing without that information while explicitly acknowledging the limitation.

## Failure transcript

`transcripts/failure-run.txt` demonstrates the required failure case.

The run proceeds as follows:

1. The first `service_metrics(...)` call is deliberately replaced with malformed data.
2. The tool result is logged as `[FAILED]`.
3. The agent switches to `read_notes("constraints.md")`.
4. The agent later retries `service_metrics` with a narrower `read_qps` request.
5. That retry succeeds.
6. The final answer explicitly acknowledges that the original metrics request failed and
   that some information could not be verified.

The failure is therefore visible and the agent adapts instead of crashing or silently
acting as though the call succeeded.

## Transcripts

| File | What it demonstrates |
|---|---|
| `transcripts/clean-run.txt` | Required clean run with no mocked failure and a final answer. |
| `transcripts/failure-run.txt` | Required similar run with the mocked failure visible and the agent's recovery. |
| `transcripts/budget-exhausted-run.txt` | Additional demonstration that the agent does not exceed its tool-call budget. |
| `transcripts/sent-back-run.txt` | Additional demonstration of the two-distinct-tools rule. |

The extra transcripts supplement the two transcripts explicitly required by the brief.

## Final run report

A run reports information such as:

```text
tool failures this run: ...
distinct tools that worked: .../2 required
tool calls used: .../6
LLM calls: ...
tokens: ...
```

This makes tool usage, failures, and resource use visible.

## Assumptions

- The internal notes and `service_metrics` data are synthetic inputs created for the
  exercise.
- Their values are fixed so runs can reason over stable information.
- The two-distinct-tools rule is treated as a minimum requirement rather than a fixed
  sequence.
- Failed calls do not count toward the successful distinct-tool requirement.
- The model is not seeded, so different executions may choose different valid research
  paths or produce different wording.

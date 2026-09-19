# Junior AI Engineer — Take-Home Assignments

All three assignments, built on **LangGraph**. Each folder is self-contained and has its
own README covering setup, how to run it, and what its transcripts show.

| Folder | Assignment | What it demonstrates |
|---|---|---|
| [`assignment-1/`](assignment-1/) | Tool-Using Research Agent | An agent that plans its own path through four tools, decides for itself when it has enough, stops at a hard tool-call budget, and adapts when a tool call fails. |
| [`assignment-2/`](assignment-2/) | Multi-Agent Task with Review | A worker agent and a reviewer agent in a single pass. The reviewer returns a structured, evidence-backed verdict against six concrete criteria. |
| [`assignment-3/`](assignment-3/) | Resumable Agent with Self-Check | Four items processed one at a time, checkpointed with LangGraph's `SqliteSaver`, resumable after an interruption, with a self-check that catches a deliberately bad result. |

## Try them in a browser

There is a Streamlit app that runs all three, with each at its own path:

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your OpenAI key
streamlit run streamlit_app.py
```

| Path | Assignment |
|---|---|
| `/assignment-1` | Tool-using research agent |
| `/assignment-2` | Multi-agent task with review |
| `/assignment-3` | Resumable agent with self-check |

Each page exposes that assignment's real controls — inject a tool failure, force a
rejection, stop a run partway and resume it — and streams the agent's reasoning trace
as it happens. The pages call the same `run()` function the command line calls, so
nothing is mocked except the failures the assignments inject on purpose.

[DEPLOY.md](DEPLOY.md) covers deploying to Streamlit Community Cloud.

## Setup

Once, from this directory:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then put an OpenAI API key in `.env` — https://platform.openai.com/api-keys.

```
OPENAI_API_KEY=sk-...
```

That is all three assignments set up. Each folder also carries its own
`requirements.txt` if you would rather install them separately.

**Model:** `gpt-4.1-mini`, chosen for cost rather than capability — these agents need
reliable tool calling and structured output, both of which it does well, and a full run
costs a fraction of a cent. Override it by setting `OPENAI_MODEL` in `.env`. The client
retries and waits out a 429 rather than failing the run.

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

## Repository layout

```
assignment-1/  assignment-2/  assignment-3/   the assignments: agent, README, transcripts
streamlit_app.py                              the app entry point; routes live here
app/                                          pages and shared UI; imports the agents
tests/                                        page rendering, validation, error handling
DEPLOY.md                                     running locally and deploying
```

Each agent exposes a `run(log, ...)` function taking its configuration as arguments and a
`log` callable for output. The CLI passes a file-writing logger; the Streamlit pages pass
a widget that streams the trace into the browser. Nothing about a run is stored at module
level, so two people using the deployed app at once cannot affect each other's runs.

`llm.py` is duplicated across the three folders rather than factored into a shared package,
so each assignment stands alone as the brief asks. It is about fifty lines: it builds the
OpenAI client and tallies LLM calls and tokens from each response's `usage_metadata`.

## Tests

```bash
pytest -m "not live"   # every page renders, validation and error handling; no key needed
pytest                 # also runs the agents against the real API
```

# Junior AI Engineer — Take-Home Assignments

**Live deployed app:**  
https://drift-ai-engineer-assignments-krtt7sujdpij3m7jepsm9j.streamlit.app/

This repository contains all three Junior AI Engineer take-home assignments, implemented
using **LangGraph**. Each assignment is in its own folder with its own README, source code,
and transcripts demonstrating the required behavior.

| Folder | Assignment | What it demonstrates |
|---|---|---|
| [`assignment-1/`](assignment-1/) | Tool-Using Research Agent | Autonomous planning, tool use, reasoning traces, tool-call limits, and graceful failure recovery. |
| [`assignment-2/`](assignment-2/) | Multi-Agent Task with Review | A worker agent followed by a reviewer agent that produces an approved/rejected verdict against concrete criteria. |
| [`assignment-3/`](assignment-3/) | Resumable Agent with Basic Self-Check | Sequential processing, LangGraph checkpointing, interruption/resume, skipping completed work, and final result validation. |

## Framework and LLM

All three assignments use **LangGraph**.

Assignment 3 specifically uses LangGraph's built-in `SqliteSaver` checkpointing rather
than a custom persistence implementation.

All three assignments use OpenAI through `langchain-openai`.

The default model is:

```text
gpt-4.1-mini
```

It can be overridden with the `OPENAI_MODEL` environment variable.

## Setup

From the repository root:

```bash
python -m venv .venv
```

Activate the environment.

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

Add your OpenAI API key:

```env
OPENAI_API_KEY=your_key_here
```

Optionally override the model:

```env
OPENAI_MODEL=gpt-4.1-mini
```

Each assignment also contains its own `requirements.txt` so it can be installed
independently.

## Run the assignments

### Assignment 1 — Tool-Using Research Agent

```bash
cd assignment-1

python agent.py
python agent.py --fail-mode malformed
```

### Assignment 2 — Multi-Agent Task with Review

```bash
cd assignment-2

python chain.py
python chain.py --weak
```

### Assignment 3 — Resumable Agent with Basic Self-Check

```bash
cd assignment-3

python agent.py --reset --stop-after 2
python agent.py
python agent.py --reset --sabotage 3
```

See the README inside each assignment folder for the complete task description,
controls, implementation details, and transcript descriptions.

## Streamlit app

The same assignments can also be run through the deployed Streamlit interface:

https://drift-ai-engineer-assignments-krtt7sujdpij3m7jepsm9j.streamlit.app/

The available pages are:

| Path | Assignment |
|---|---|
| `/assignment-1` | Tool-Using Research Agent |
| `/assignment-2` | Multi-Agent Task with Review |
| `/assignment-3` | Resumable Agent with Basic Self-Check |

To run the Streamlit app locally:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Add your key to `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your_key_here"
```

Then run:

```bash
streamlit run streamlit_app.py
```

The Streamlit pages call the same assignment logic used by the command-line programs.

## Included transcripts

The repository contains all transcripts required by the assignment brief, plus a few
additional demonstrations.

```text
assignment-1/transcripts/
├── clean-run.txt
├── failure-run.txt
├── budget-exhausted-run.txt
└── sent-back-run.txt

assignment-2/transcripts/
├── approved-run.txt
└── rejected-run.txt

assignment-3/transcripts/
├── resume-run.txt
├── self-check-catches-bad-result.txt
└── self-check-blank-result.txt
```

The additional transcripts demonstrate control-flow behavior beyond the minimum
deliverables.

## Repository structure

```text
.
├── assignment-1/
│   ├── README.md
│   ├── agent.py
│   ├── tools.py
│   ├── llm.py
│   ├── notes/
│   ├── requirements.txt
│   └── transcripts/
│
├── assignment-2/
│   ├── README.md
│   ├── chain.py
│   ├── llm.py
│   ├── requirements.txt
│   └── transcripts/
│
├── assignment-3/
│   ├── README.md
│   ├── agent.py
│   ├── llm.py
│   ├── docs/
│   ├── requirements.txt
│   └── transcripts/
│
├── app/
├── .streamlit/
├── streamlit_app.py
├── .env.example
├── .gitignore
└── requirements.txt
```

## Assignment summary

### Assignment 1

A single autonomous research agent that:

- decides its own research path,
- uses multiple tools,
- stops when it has enough information,
- never exceeds six tool calls,
- logs each decision, reason, and result,
- and adapts when a tool call fails.

### Assignment 2

A two-agent single-pass workflow:

```text
Agent A / Worker → Agent B / Reviewer → END
```

Agent B reviews Agent A's single attempt against concrete approval criteria and returns
either an approved result or a rejected result with specific reasons.

### Assignment 3

A resumable agent that:

- processes four items one at a time,
- persists progress using LangGraph checkpointing,
- resumes after interruption,
- skips completed work,
- and performs a final self-check that catches deliberately incorrect results.

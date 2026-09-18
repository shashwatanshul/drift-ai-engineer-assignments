# Assignment 1 — Tool-Using Research Agent

## Task

Build a single autonomous research agent using LangGraph or LangChain that answers an open-ended question requiring reasoning across multiple pieces of information.

The agent must:

* Plan its own steps toward the answer
* Avoid a hardcoded step-by-step pipeline
* Use at least 2 distinct tools
* Decide on its own when it has enough information to answer
* Use a maximum of 6 tool calls per run
* Track its own tool-call count
* Log what it decided to do, why it decided to do it, and the result
* Handle a mocked tool failure without crashing

## Question

[Add the open-ended research question here]

## Tools Used

1. [Tool 1]
2. [Tool 2]

## Framework

[LangGraph or LangChain]

## Setup

```bash
[Add setup commands here]
```

## How to Run

```bash
[Add run command here]
```

## Failure Handling

One tool call can be configured to return a mocked failure such as:

* Timeout
* Empty result
* Malformed response

The agent detects the failure and adapts rather than crashing or treating the failed response as valid information.

## Transcripts

### Transcript 1 — Clean Run

See:

```text
[Add transcript file path]
```

This transcript contains a complete run without mocked failures and includes the final answer.

### Transcript 2 — Failure Run

See:

```text
[Add transcript file path]
```

This transcript shows the mocked tool failure and how the agent handles it.

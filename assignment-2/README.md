# Assignment 2 — Multi-Agent Task with Review

## Task

Build a two-agent workflow using LangGraph or LangChain.

The workflow contains:

### Agent A — Worker

Agent A produces one attempt at the selected task.

### Agent B — Reviewer

Agent B reviews Agent A's output once against clearly defined approval criteria.

Agent B must return one of the following:

* `approved` with the final output
* `rejected` with specific reasons

There is no revision loop. After Agent B reviews the output, the workflow ends.

## Selected Task

[Add the selected task here]

## Framework

[LangGraph or LangChain]

## Reviewer Approval Criteria

Agent B evaluates Agent A's output using the following concrete criteria:

1. [Criterion 1]
2. [Criterion 2]
3. [Criterion 3]
4. [Add more criteria if required]

## Setup

```bash
[Add setup commands here]
```

## How to Run

```bash
[Add run command here]
```

## LLM Usage

At the end of each run, the application reports:

* Total LLM calls
* Total tokens used, if available

## Transcripts

### Transcript 1 — Approved

See:

```text
[Add approved transcript file path]
```

This transcript shows a run where Agent B approves Agent A's output.

### Transcript 2 — Rejected

See:

```text
[Add rejected transcript file path]
```

This transcript shows a run where Agent B rejects Agent A's output and provides specific reasons.

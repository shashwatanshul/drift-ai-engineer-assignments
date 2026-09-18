# Assignment 3 — Resumable Agent with Basic Self-Check

## Task

Build a resumable agent using LangGraph or LangChain that processes 4–5 items one at a time and persists its progress after each completed item.

The agent must:

* Process items one at a time
* Process items in order
* Persist progress after each item
* Support being stopped partway through execution
* Resume from the saved progress when run again
* Skip items that were already completed
* Perform a final self-check after all items are processed
* Detect one deliberately incorrect or blank result

## Items to Process

1. [Item 1]
2. [Item 2]
3. [Item 3]
4. [Item 4]
5. [Optional Item 5]

## Framework

[LangGraph or LangChain]

## Saved State

The persisted state contains the progress required for the agent to resume processing.

Saved state format:

```text
[Add saved-state structure here]
```

The saved state is kept simple and readable.

## Setup

```bash
[Add setup commands here]
```

## How to Run

```bash
[Add run command here]
```

## How to Trigger Stop / Resume

To demonstrate interruption:

```text
[Add instructions for triggering the intentional stop here]
```

After the program stops, run it again using:

```bash
[Add run command here]
```

The resumed run skips already completed items and continues from the next unfinished item.

## Final Self-Check

After all items are processed, the agent re-reads the final results and checks that each result:

* Is non-empty
* Roughly matches what it is expected to contain

One result is deliberately made incorrect or blank so that the self-check can demonstrate that it detects the bad result.

## LLM Usage

At the end of the run, the application reports:

```text
Total LLM calls: [value]
```

## Logs

### Stop / Resume Log

See:

```text
[Add stop-resume log file path]
```

This log shows:

* Initial run
* Interruption partway through
* Re-run
* Previously completed items being skipped
* Remaining items being processed

### Self-Check Log

See:

```text
[Add self-check log file path]
```

This log shows the final check detecting the deliberately incorrect or blank result.

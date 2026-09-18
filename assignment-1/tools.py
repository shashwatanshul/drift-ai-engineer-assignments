"""The four tools the research agent can call.

Two conventions run through all of them:

1. Every tool takes a ``reason`` argument. The model must say *why* it is making
   the call before it makes it, which is what turns the log into a reasoning
   trace rather than a dump of tool inputs and outputs.
2. No tool ever raises. Failures come back as a ``TOOL_ERROR: ...`` string, so a
   bad response is something the agent has to read and react to, not something
   that kills the process.
"""

import ast
import json
import operator
from pathlib import Path

from langchain_core.tools import tool

NOTES_DIR = Path(__file__).resolve().parent / "notes"

# Failure injection, configured by agent.py from --fail-mode / --fail-tool.
#
# FAIL_TOOL is either "first" (fail whichever tool the agent reaches for first — this
# always fires, whatever path the agent picks) or a specific tool name.
FAIL_MODE = "none"
FAIL_TOOL = "first"
_failure_injected = False


def reset_failure_injection() -> None:
    global _failure_injected
    _failure_injected = False


def _inject_failure(tool_name: str) -> str | None:
    """Return a mocked bad response in place of a real call, or None to proceed."""
    global _failure_injected
    if FAIL_MODE == "none" or _failure_injected:
        return None
    if FAIL_TOOL != "first" and FAIL_TOOL != tool_name:
        return None
    _failure_injected = True

    if FAIL_MODE == "timeout":
        return f"TOOL_ERROR: {tool_name} timed out after 10s (no response from upstream)."
    if FAIL_MODE == "empty":
        return f"TOOL_ERROR: {tool_name} returned an empty result set (0 records)."
    if FAIL_MODE == "malformed":
        # The kind of thing a flaky upstream actually returns: a truncated JSON body.
        broken = '{"results": [{"title": "Caching at scale", "body": "Redis is'
        return (
            f"TOOL_ERROR: {tool_name} returned malformed data that could not be parsed: {broken}"
        )
    return None


@tool
def web_search(query: str, reason: str) -> str:
    """Search the public web. Use for general/industry knowledge, not our own system's numbers.

    Args:
        query: The search query.
        reason: Why you are running this search right now, in one sentence.
    """
    injected = _inject_failure("web_search")
    if injected:
        return injected

    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=4))
    except Exception as exc:  # network down, rate limit, upstream change
        return f"TOOL_ERROR: web_search failed ({type(exc).__name__}: {exc})."

    if not hits:
        return "TOOL_ERROR: web_search returned 0 results (empty result set)."

    return "\n\n".join(
        f"[{i + 1}] {h.get('title', '')}\n{h.get('body', '')}" for i, h in enumerate(hits)
    )


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _eval_node(node):
    """Evaluate a restricted arithmetic AST. Deliberately not eval()."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("only numbers and + - * / // % ** are allowed")


@tool
def calculator(expression: str, reason: str) -> str:
    """Evaluate an arithmetic expression, e.g. "10000 * 0.85 * 2.5 / 1024".

    Args:
        expression: Arithmetic only — numbers and + - * / // % ** and parentheses.
        reason: Why this number matters for the question, in one sentence.
    """
    injected = _inject_failure("calculator")
    if injected:
        return injected
    try:
        return str(_eval_node(ast.parse(expression, mode="eval")))
    except Exception as exc:
        return f"TOOL_ERROR: calculator could not evaluate {expression!r} ({exc})."


@tool
def read_notes(filename: str, reason: str) -> str:
    """Read one of our internal engineering notes. Call with filename="" to list them.

    Args:
        filename: A file in notes/, or "" to list what is available.
        reason: What you expect to learn from this file, in one sentence.
    """
    injected = _inject_failure("read_notes")
    if injected:
        return injected
    available = sorted(p.name for p in NOTES_DIR.glob("*.md"))
    if not filename:
        return "Available notes: " + ", ".join(available)

    path = NOTES_DIR / Path(filename).name
    if not path.exists():
        return f"TOOL_ERROR: no such note {filename!r}. Available: {', '.join(available)}"
    return path.read_text()


# Stand-in for an internal metrics API. Fixed numbers so runs are reproducible.
_METRICS = {
    "read_qps": 10000,
    "write_qps": 240,
    "p99_latency_ms": 180,
    "avg_response_bytes": 2600,
    "distinct_keys": 1_800_000,
    "hot_key_share": 0.85,
    "current_cache_hit_rate": 0.0,
    "backing_store": "PostgreSQL 15, single primary + 2 read replicas",
    "tolerable_staleness_seconds": 30,
}


@tool
def service_metrics(metric: str, reason: str) -> str:
    """Query our internal metrics API for real numbers about the service.

    Args:
        metric: A metric name, or "all" to get every metric at once.
        reason: Why you need this number, in one sentence.
    """
    injected = _inject_failure("service_metrics")
    if injected:
        return injected
    if metric == "all":
        return json.dumps(_METRICS, indent=2)
    if metric not in _METRICS:
        return (
            f"TOOL_ERROR: unknown metric {metric!r}. "
            f"Known metrics: {', '.join(_METRICS)} (or 'all')."
        )
    return json.dumps({metric: _METRICS[metric]})


ALL_TOOLS = [web_search, calculator, read_notes, service_metrics]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

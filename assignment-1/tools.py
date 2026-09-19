"""The four tools the research agent can call.

Two conventions run through all of them:

1. Every tool takes a ``reason`` argument. The model must say *why* it is making
   the call before it makes it, which is what turns the log into a reasoning
   trace rather than a dump of tool inputs and outputs.
2. No tool ever raises. Failures come back as a ``TOOL_ERROR: ...`` string, so a
   bad response is something the agent has to read and react to, not something
   that kills the process.

The tools are built per run by ``make_tools``, which closes over that run's failure
injection settings. Nothing about a run is stored at module level, so two runs in the
same process — two people using the Streamlit app at once, say — cannot affect each
other.
"""

import ast
import json
import operator
from pathlib import Path

from langchain_core.tools import tool

NOTES_DIR = Path(__file__).resolve().parent / "notes"

FAIL_MODES = ("none", "timeout", "empty", "malformed")

# Stand-in for an internal metrics API. Fixed numbers so runs are reproducible.
METRICS = {
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

TOOL_NAMES = ("web_search", "calculator", "read_notes", "service_metrics")


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


def make_tools(fail_mode: str = "none", fail_tool: str = "first"):
    """Build this run's tools.

    Args:
        fail_mode: "none", or the kind of bad response to inject into one call.
        fail_tool: "first" to fail whichever tool the agent reaches for first (which
            always fires, whatever path the agent picks), or a specific tool name.

    Returns:
        (tools, tools_by_name)
    """
    if fail_mode not in FAIL_MODES:
        raise ValueError(f"fail_mode must be one of {FAIL_MODES}, got {fail_mode!r}")
    if fail_tool != "first" and fail_tool not in TOOL_NAMES:
        raise ValueError(f"fail_tool must be 'first' or one of {TOOL_NAMES}, got {fail_tool!r}")

    # Per-run, not module-level: the injection fires at most once per run.
    state = {"injected": False}

    def inject(tool_name: str):
        """Return a mocked bad response in place of a real call, or None to proceed."""
        if fail_mode == "none" or state["injected"]:
            return None
        if fail_tool != "first" and fail_tool != tool_name:
            return None
        state["injected"] = True

        if fail_mode == "timeout":
            return f"TOOL_ERROR: {tool_name} timed out after 10s (no response from upstream)."
        if fail_mode == "empty":
            return f"TOOL_ERROR: {tool_name} returned an empty result set (0 records)."
        # malformed — the kind of thing a flaky upstream actually returns: a truncated body.
        broken = '{"results": [{"title": "Caching at scale", "body": "Redis is'
        return f"TOOL_ERROR: {tool_name} returned malformed data that could not be parsed: {broken}"

    @tool
    def web_search(query: str, reason: str) -> str:
        """Search the public web. Use for general/industry knowledge, not our own system's numbers.

        Args:
            query: The search query.
            reason: Why you are running this search right now, in one sentence.
        """
        injected = inject("web_search")
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

    @tool
    def calculator(expression: str, reason: str) -> str:
        """Evaluate an arithmetic expression, e.g. "10000 * 0.85 * 2.5 / 1024".

        Args:
            expression: Arithmetic only — numbers and + - * / // % ** and parentheses.
            reason: Why this number matters for the question, in one sentence.
        """
        injected = inject("calculator")
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
        injected = inject("read_notes")
        if injected:
            return injected

        available = sorted(p.name for p in NOTES_DIR.glob("*.md"))
        if not filename:
            return "Available notes: " + ", ".join(available)

        path = NOTES_DIR / Path(filename).name
        if not path.exists():
            return f"TOOL_ERROR: no such note {filename!r}. Available: {', '.join(available)}"
        return path.read_text()

    @tool
    def service_metrics(metric: str, reason: str) -> str:
        """Query our internal metrics API for real numbers about the service.

        Args:
            metric: A metric name, several separated by commas, or "all" for everything.
            reason: Why you need this number, in one sentence.
        """
        injected = inject("service_metrics")
        if injected:
            return injected
        if metric.strip() == "all":
            return json.dumps(METRICS, indent=2)

        # Asking for several at once is a reasonable thing to want, and a real metrics
        # API would support it, so accept a comma-separated list rather than failing.
        requested = [name.strip() for name in metric.split(",") if name.strip()]
        unknown = [name for name in requested if name not in METRICS]
        if not requested or unknown:
            return (
                f"TOOL_ERROR: unknown metric {', '.join(unknown) or metric!r}. "
                f"Known metrics: {', '.join(METRICS)} (or 'all')."
            )
        return json.dumps({name: METRICS[name] for name in requested}, indent=2)

    tools = [web_search, calculator, read_notes, service_metrics]
    return tools, {t.name: t for t in tools}

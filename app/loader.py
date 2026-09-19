"""Import the agent modules out of the hyphenated assignment folders.

`assignment-1` is not a valid Python identifier, so the folders are not importable
packages and the agents cannot be reached with a plain import. They are loaded by file
path instead.

Two details make this fiddly enough to deserve its own module:

1. Each agent does sibling imports (`from llm import build_llm`, `from tools import ...`),
   which only resolve when its own folder is on `sys.path`. The loader prepends it for
   the duration of the load.
2. All three folders contain a module named `llm`, so whichever loads first would
   otherwise satisfy the import for the other two. The loader clears those names from
   `sys.modules` around each load, and registers each agent under a unique key.

Everything is cached, so this happens once per process rather than on every rerun.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent

# (cache key, folder, file) for each agent entry point.
_AGENTS = {
    "assignment_1": ("assignment-1", "agent.py"),
    "assignment_2": ("assignment-2", "chain.py"),
    "assignment_3": ("assignment-3", "agent.py"),
}

# Module names the assignment folders define locally and would otherwise share.
_SHARED_NAMES = ("llm", "tools")

_cache: dict = {}


def _load(key: str) -> ModuleType:
    folder, filename = _AGENTS[key]
    directory = REPO_ROOT / folder
    path = directory / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. The Streamlit app expects to run from the repo root, "
            f"alongside the assignment folders."
        )

    saved = {name: sys.modules.pop(name, None) for name in _SHARED_NAMES}
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        # Registered before exec so the module is importable from within itself.
        sys.modules[key] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(directory))
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def get(key: str) -> ModuleType:
    """Return an agent module by key: assignment_1, assignment_2 or assignment_3."""
    if key not in _AGENTS:
        raise KeyError(f"unknown agent {key!r}; expected one of {', '.join(_AGENTS)}")
    if key not in _cache:
        _cache[key] = _load(key)
    return _cache[key]

---
chapter: ch-04
course: boson-agent
phase: read
excerpt_of: boson-agent/packages/basement/basement/tools/registry.py
created_at: "2026-04-19"
---

# Excerpt: registry.py — Full Walkthrough

**File:** `boson-agent/packages/basement/basement/tools/registry.py`
**Role:** Auto-discovers `@tool`-decorated functions from a directory; manages the name → ToolSpec map.
**Calling spec:** Called by `__main__` at startup and by `agent_loop` for `get_all_specs()`; calls `tools/decorator` indirectly (reads `__tool_spec__`).

---

## Full Source with Line-by-Line Commentary

```python
# boson-agent/packages/basement/basement/tools/registry.py, lines 23-91

def _import_module_from_path(path: Path):
    """Import a Python module from file path using importlib.

    Shared utility used by both ToolRegistry and HookRegistry.
    """
    module_name = f"_basement_dynamic_{path.stem}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")

    # Add parent dir to sys.path for relative imports
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ToolRegistry:
    """Discover and manage agent tools."""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def discover_tools(self, tools_dir: Path) -> int:
        """Import all .py files in tools_dir, find @tool functions.

        Returns count of discovered tools.
        """
        if not tools_dir.exists():
            return 0

        count = 0
        for py_file in sorted(tools_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                module = _import_module_from_path(py_file)
                for obj in vars(module).values():
                    if hasattr(obj, "__tool_spec__"):
                        self.register(obj.__tool_spec__)
                        count += 1
            except Exception as e:
                logger.error("Failed to load tool from %s: %s", py_file, e)
                continue

        return count

    def register(self, spec: ToolSpec) -> None:
        """Register a tool. Raises ValueError on duplicate name."""
        if spec.name in self._tools:
            raise ValueError(f"Duplicate tool name: '{spec.name}'")
        self._tools[spec.name] = spec
        logger.debug("Registered tool: %s", spec.name)

    def get(self, name: str) -> ToolSpec:
        """Get tool by name. Raises ToolNotFoundError if not found."""
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Tool '{name}' not found. Available: {list(self._tools)}"
            )
        return self._tools[name]

    def get_all_specs(self) -> list[ToolSpec]:
        """Return all tool specs (for LLM API tools parameter)."""
        return list(self._tools.values())
```

---

## Line-by-Line Analysis

### `_import_module_from_path()` (lines 23–41)

This is the discovery engine. Four steps:

1. **Unique module name:** `f"_basement_dynamic_{path.stem}_{id(path)}"` avoids collisions
   in `sys.modules` when two different agent folders contain files with the same stem (e.g.,
   two agents both have `tools/search.py`). The `id(path)` component makes the name unique
   per `Path` object.

2. **`importlib.util.spec_from_file_location`:** constructs an import spec from a
   filesystem path instead of a dotted module name. This is the standard way to load
   arbitrary files without them being on `sys.path` at process start.

3. **`sys.path` injection (line 34–36):** inserts the file's parent directory at the
   front of `sys.path` before executing the module. This is what lets tool files use
   relative imports like `from _session import get_active_customer` (seen in test-lina
   tools). Without this, those imports would fail with `ModuleNotFoundError`.

4. **`sys.modules[module_name] = module` then `exec_module`:** the module must be
   registered in `sys.modules` *before* executing it. If the module contains
   `from __future__ import annotations` or any self-referential import, it needs to find
   itself in `sys.modules` to avoid infinite recursion.

This utility is shared with `HookRegistry` — both subsystems use identical dynamic
import mechanics. That sharing is deliberate: the pattern is complicated enough that
duplicating it would be error-prone.

---

### `discover_tools()` (lines 50–71)

The scan loop has four notable properties:

1. **`sorted()`:** deterministic discovery order. Tool names must be unique, but
   registration order affects log output and `get_all_specs()` list order, which in turn
   affects the order tools appear in the LLM API request. Alphabetical ordering makes
   this reproducible.

2. **`startswith("_")` skip:** conventional Python private-file convention. `_session.py`
   in the test-lina agent is a shared utility module, not a tool file. The underscore
   prefix opts it out of scanning automatically — no configuration required.

3. **`hasattr(obj, "__tool_spec__")`:** the entire discovery protocol. The decorator
   stamped `__tool_spec__` on the function object; the registry reads it back. No import
   side effects, no registration calls, no metaclass magic. The coupling is purely through
   the attribute name.

4. **`except Exception: continue`:** fail-open per file. A broken tool file does not
   kill the agent startup. The error is logged (available for debugging) but the registry
   continues loading other files. This is a deliberate operational choice: a partially
   working agent is better than a crashed one.

---

### `register()` (lines 73–79)

Duplicate name check at registration time is strict — raises `ValueError`. This surfaces
collisions at startup rather than silently clobbering an existing tool. The error message
includes the conflicting name so the developer knows exactly which tool to rename.

---

### `get()` and `get_all_specs()` (lines 81–91)

`get()` raises `ToolNotFoundError` (a domain-specific exception, not `KeyError`) and
includes the list of available tools in the message. This is the error the LLM would
trigger if it hallucinated a tool name — the message is readable enough to appear in
logs and be useful without a stack trace.

`get_all_specs()` returns `list(self._tools.values())` — a copy, not a view. Callers
(specifically `agent_loop.py` line 110: `runtime.tool_registry.get_all_specs() or None`)
can mutate the list without affecting the registry's internal state.

---

## Discovery Flow Diagram

```
agent startup
    │
    ▼
ToolRegistry()
    │
    ▼
discover_tools(tools_dir)
    │
    ├── sorted(tools_dir.glob("*.py"))
    │       filters out _private.py files
    │
    └── for each py_file:
            │
            ▼
        _import_module_from_path(py_file)
            ├── unique module name via id()
            ├── sys.path.insert(0, parent_dir)
            └── exec_module → module object
            │
            ▼
        vars(module).values()
            │
            for each obj:
                hasattr(obj, "__tool_spec__")?
                    yes → register(obj.__tool_spec__)
                    no  → skip
```

## What This Means for Agent Authors

Zero registration code is required. The full tool lifecycle is:

1. Create `agents/my_agent/tools/my_tool.py`
2. Write a function, apply `@tool`, add a docstring and type hints
3. Start the agent — `discover_tools()` finds and loads it automatically

No `__init__.py`, no import in any config file, no registration call anywhere.

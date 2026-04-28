---
chapter: ch-04
course: boson-agent
phase: read
excerpt_of: boson-agent/packages/basement/basement/tools/decorator.py
created_at: "2026-04-19"
---

# Excerpt: decorator.py — Full Walkthrough

**File:** `boson-agent/packages/basement/basement/tools/decorator.py`
**Role:** Defines the `@tool` decorator and `_generate_schema` / `_type_to_schema` helpers.
**Calling spec:** Called by user tool files; calls `basement.schemas.tool_schema.ToolSpec`.

---

## Full Source with Line-by-Line Commentary

```python
# boson-agent/packages/basement/basement/tools/decorator.py, lines 1-113

# === CALLING SPEC ===
# PURPOSE: @tool decorator — extracts metadata and generates JSON Schema
# CALLED BY: User tool files in agent's tools/ folder
# CALLS: schemas/tool_schema
# PURE: yes
# DETERMINISTIC: yes (sealed)

"""@tool decorator — auto-extracts name, description, and JSON Schema from type hints."""

from __future__ import annotations

import inspect
from typing import Callable, get_args, get_origin, Union

from basement.schemas.tool_schema import ToolSpec


TYPE_MAP: dict[type, dict] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}


def tool(fn: Callable | None = None, *, name: str | None = None) -> Callable:
    """Mark a function as an agent tool.

    Usage:
        @tool
        def search(query: str, limit: int = 10) -> str:
            '''Search the knowledge base.'''
            ...

        @tool(name='custom_search')
        def my_search(query: str) -> str:
            '''Search with custom name.'''
            ...
    """

    def decorator(func: Callable) -> Callable:
        if not func.__doc__:
            raise ValueError(
                f"Tool '{func.__name__}' must have a docstring (used as description)"
            )

        spec = ToolSpec(
            name=name or func.__name__,
            description=func.__doc__.strip(),
            input_schema=_generate_schema(func),
            handler=func,
        )
        func.__tool_spec__ = spec
        return func

    if fn is not None:
        return decorator(fn)
    return decorator


def _generate_schema(func: Callable) -> dict:
    """Generate JSON Schema from function type hints.

    Supports: str, int, float, bool, list[str], Optional[X]
    """
    sig = inspect.signature(func)
    properties: dict[str, dict] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            properties[param_name] = {"type": "string"}
        else:
            properties[param_name] = _type_to_schema(annotation)

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _type_to_schema(annotation: type) -> dict:
    """Map Python type hint to JSON Schema type."""
    # Direct type match
    if annotation in TYPE_MAP:
        return dict(TYPE_MAP[annotation])

    origin = get_origin(annotation)
    args = get_args(annotation)

    # list[X] -> {"type": "array", "items": ...}
    if origin is list:
        if args:
            return {"type": "array", "items": _type_to_schema(args[0])}
        return {"type": "array"}

    # Optional[X] = Union[X, None]
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_schema(non_none[0])

    # Fallback
    return {"type": "string"}
```

---

## Line-by-Line Analysis

### `TYPE_MAP` (lines 18–23)

A module-level dict mapping the four JSON Schema primitives. The keys are the actual
Python `type` objects (not strings), so `annotation in TYPE_MAP` is an identity check —
O(1) and impossible to confuse with a string-keyed dispatch. Every other type falls
through to `_type_to_schema` for structural decomposition.

Notice that `list` and `Optional` are **absent** from `TYPE_MAP`. They cannot be there
because they are generic aliases (`list[str]` is not the same object as `list`), which
means `get_origin` / `get_args` must handle them separately.

---

### `tool()` dual-mode entry point (lines 26–58)

The function signature `def tool(fn: Callable | None = None, *, name: str | None = None)`
enables two call patterns without any separate `@overload`:

```
@tool                    # fn=<function>, name=None → takes the if-branch at line 56
@tool(name="foo")       # fn=None, name="foo"    → returns decorator, Python calls it next
```

The `if fn is not None` branch at line 56 is the key: when used bare (`@tool`), Python
passes the decorated function as `fn`; when used with parens (`@tool(name="foo")`),
Python calls `tool(name="foo")` first, gets back `decorator`, then applies that to the
function. This is the standard Python "optional-arguments decorator" pattern.

The inner `decorator` function (lines 41–54) does three things:

1. **Guards** on `func.__doc__` — raises `ValueError` if missing. The docstring is the
   tool's description; without it the LLM receives no description, which is a hard error.
2. **Builds** `ToolSpec` from the function's metadata. `name or func.__name__` lets
   callers override the tool name for aliasing without changing the Python identifier.
3. **Stamps** `func.__tool_spec__ = spec` on the function object itself. This is how
   `ToolRegistry.discover_tools()` finds decorated functions without any import-time
   registration — it just checks `hasattr(obj, "__tool_spec__")`.

The function is returned unchanged (`return func`). The decorator is transparent to
callers of the function directly (useful in tests).

---

### `_generate_schema()` (lines 61–87)

Uses `inspect.signature(func)` to iterate parameters. Two decisions per parameter:

1. **Type → JSON Schema:** missing annotation falls back to `{"type": "string"}` (line 76).
   This is a reasonable default; an untyped parameter at least gets a schema slot.
2. **Required vs optional:** a parameter is required if and only if it has no default
   (`param.default is inspect.Parameter.empty`). Python's sentinel `inspect.Parameter.empty`
   is the canonical way to detect "no default provided".

`self` and `cls` are explicitly skipped (line 72). This matters for any tool defined
as a method (though the framework's discovery pattern uses module-level functions, the
defensive skip future-proofs it).

The returned dict is the `input_schema` field of `ToolSpec` and maps directly to the
`input_schema` key Anthropic's tool-use API expects.

---

### `_type_to_schema()` (lines 90–113)

A recursive type-mapper. Resolution order:

1. **Primitive:** `annotation in TYPE_MAP` → return the dict from the map.
2. **Generic alias:** `get_origin(annotation)` returns the base generic (`list`, `Union`).
   - `list[X]` → `{"type": "array", "items": _type_to_schema(X)}` (recursive for nested lists).
   - `Union[X, None]` (i.e., `Optional[X]`) → strip `None`, recurse on `X`.
3. **Fallback:** anything not matched → `{"type": "string"}`. Silently degrades rather
   than raising, which matches the framework's fail-open philosophy for tool dispatch.

The recursion on `list[X]` means `list[list[str]]` would produce
`{"type": "array", "items": {"type": "array", "items": {"type": "string"}}}` — though
no current example uses nested lists. The framework supports it for free.

`Optional[X]` being handled as `Union[X, None]` is correct because `Optional[X]` is
literally `typing.Union[X, type(None)]` in CPython's type system since 3.9.

---

## Key Design Choices

| Choice | Alternative | Why this way |
|--------|-------------|--------------|
| Stamp `__tool_spec__` on the function | Central import-time registry | Discovery is deferred to `ToolRegistry.discover_tools()`; the decorator itself has zero side effects |
| Raise on missing docstring | Silently use empty description | Empty description is worse than an error; LLM gets no context for tool selection |
| Return `func` unchanged | Return a wrapper | Tests can call the function directly without stripping the decorator |
| Fallback to `"string"` for unknown types | Raise `TypeError` | Fail-open keeps partially-typed tools working; schema is advisory, not enforced at call time |

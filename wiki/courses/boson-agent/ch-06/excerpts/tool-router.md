---
calling_spec:
  purpose: Deep walkthrough of ToolRouter — unified dispatch table
  chapter: ch-06
  course: boson-agent
  phase: read
  excerpt_of: read.md
---

# ToolRouter — Deep Walkthrough

`ToolRouter` lives in `packages/basement/basement/metatool/router.py`. It is the
single class responsible for unifying native Python tools and MCP tools behind one
dispatch interface, and for enforcing both global permission policy and per-stage
access control.

---

## Class Structure and Constants

```python
# packages/basement/basement/metatool/router.py, lines 26-51

class ToolRouter:
    """Universal dispatch table that unifies native and MCP tools.

    Usage::

        router = ToolRouter(permission_checker=checker)
        router.register_native(registry)
        router.register_mcp(mcp_manager)
        result = await router.dispatch("greet", {"name": "World"})
    """

    # Meta-tools that are always allowed regardless of stage filtering
    _META_TOOLS = {"use_tool", "use_skill"}

    def __init__(self, permission_checker: PermissionChecker | None = None) -> None:
        self._permission_checker = permission_checker
        self._dispatch_table: dict[str, ToolSpec] = {}
        self._allowed_tools: set[str] | None = None  # None = all allowed

    def set_allowed_tools(self, tool_names: list[str] | None) -> None:
        """Restrict which tools can be dispatched per stage.

        Args:
            tool_names: List of allowed tool names, or None to allow all.
        """
        self._allowed_tools = set(tool_names) if tool_names is not None else None
```

**Notice:** `_META_TOOLS = {"use_tool", "use_skill"}` is a class-level constant, not
per-instance. This means the meta-tools bypass stage filtering unconditionally for
every `ToolRouter` instance. The LLM can always call `use_tool` or `use_skill`
regardless of which stage is active — stages narrow the *domain* tools, never the
*dispatch mechanism* itself.

`set_allowed_tools(None)` sets `_allowed_tools = None`, which opens the stage gate
entirely (all domain tools pass). `set_allowed_tools(["search", "calculate"])` narrows
to exactly those two names plus the always-allowed meta-tools. This is called by the
Gateway's stage machine on every stage transition (see [[ch-12]]).

---

## Registration

```python
# packages/basement/basement/metatool/router.py, lines 57-79

def register_native(self, tool_registry: ToolRegistry) -> int:
    """Add all native tools from registry to the dispatch table.

    Returns count of tools added.
    """
    count = 0
    for spec in tool_registry.get_all_specs():
        self._dispatch_table[spec.name] = spec
        logger.debug("ToolRouter registered native tool: %s", spec.name)
        count += 1
    return count

def register_mcp(self, mcp_manager: MCPManager) -> int:
    """Add all MCP tools from manager to the dispatch table.

    Returns count of tools added.
    """
    count = 0
    for spec in mcp_manager.get_all_tools():
        self._dispatch_table[spec.name] = spec
        logger.debug("ToolRouter registered MCP tool: %s", spec.name)
        count += 1
    return count
```

Both methods write into the same `_dispatch_table` dict. From dispatch's perspective,
there is no distinction between a native `ToolSpec` and an MCP-bridged `ToolSpec` —
both are just a name mapped to a handler. The naming convention
`mcp:{server_name}:{tool_name}` (applied by `mcp/bridge.py`) is the only marker
distinguishing MCP tools in the table.

---

## `dispatch()` — The Core Algorithm

```python
# packages/basement/basement/metatool/router.py, lines 85-142

async def dispatch(self, tool_name: str, arguments: dict) -> ToolResultBlock:
    """Execute a tool by name with given arguments.

    1. Checks permissions (raises PermissionDeniedError if denied).
    2. Looks up tool in dispatch table (raises ToolNotFoundError if absent).
    3. Executes handler — supports both sync and async handlers.
    4. Returns ToolResultBlock with result or error content.
    """
    if self._permission_checker is not None:
        self._permission_checker.check_tool(tool_name)  # raises on denial

    # Stage-based access control (meta-tools always allowed)
    if (self._allowed_tools is not None
            and tool_name not in self._allowed_tools
            and tool_name not in self._META_TOOLS):
        logger.info("Tool '%s' blocked — not in current stage's allowed tools", tool_name)
        return ToolResultBlock(
            tool_use_id=f"toolu_{uuid4().hex[:12]}",
            content=f"Tool '{tool_name}' is not available in the current stage.",
            is_error=True,
        )

    if tool_name not in self._dispatch_table:
        raise ToolNotFoundError(
            f"Tool '{tool_name}' not found. Available: {list(self._dispatch_table)}"
        )

    spec = self._dispatch_table[tool_name]

    try:
        if inspect.iscoroutinefunction(spec.handler):
            result = await spec.handler(**arguments)
        else:
            result = spec.handler(**arguments)

        return ToolResultBlock(
            tool_use_id=f"toolu_{uuid4().hex[:12]}",
            content=str(result),
            is_error=False,
        )
    except Exception as e:
        logger.error("Tool '%s' raised: %s", tool_name, e, exc_info=True)
        return ToolResultBlock(
            tool_use_id=f"toolu_{uuid4().hex[:12]}",
            content=f"Tool error: {type(e).__name__}: {e}",
            is_error=True,
        )
```

**Mechanical explanation — the five-step dispatch sequence:**

**Step 1 — Global permission check.** `_permission_checker.check_tool(tool_name)` runs
first. If it raises `PermissionDeniedError`, dispatch exits immediately without touching
the dispatch table. This exception propagates to the agent loop's `ON_ERROR` path.

**Step 2 — Stage gate check.** If `_allowed_tools` is not `None` (a stage is active
with explicit tool restrictions), the tool name must appear in `_allowed_tools` OR in
`_META_TOOLS`. If neither, dispatch does **not** raise — it returns a `ToolResultBlock`
with `is_error=True`. This is a deliberate design choice: the LLM receives an error
result ("Tool X is not available in the current stage") rather than an uncaught
exception. The agent loop continues; the LLM can try a different tool.

**Step 3 — Dispatch table lookup.** `ToolNotFoundError` is raised (not returned) if the
name is absent. This is a programming error (misconfigured agent), not a runtime
tool-call failure, hence the exception path.

**Step 4 — Sync/async handler invocation.** `inspect.iscoroutinefunction(spec.handler)`
is checked at call time to support both `def` and `async def` handlers uniformly.
The result is `str(result)` — every return value is coerced to string before wrapping.

**Step 5 — Exception wrapping.** Any exception from the handler is caught, logged with
`exc_info=True` (full traceback in logs), and returned as an error `ToolResultBlock`.
This prevents a buggy tool from crashing the entire agent turn. Compare this to step 1
(permission error, which propagates) and step 3 (not-found error, which also propagates)
— the only failures that are **swallowed into a result** are errors from within the
handler itself.

---

## Wiring in `__main__.py`

```python
# packages/basement/basement/__main__.py, lines 76-87

if config.enable_tool_router:
    tool_router = ToolRouter(permission_checker=permissions)
    tool_router.register_native(tool_reg)
    if mcp_mgr:
        tool_router.register_mcp(mcp_mgr)
    # Register use_tool meta-tool
    use_tool_spec = create_use_tool(tool_router)
    tool_reg.register(use_tool_spec)
```

`create_use_tool(tool_router)` creates a `ToolSpec` whose handler closes over the
router and calls `router.dispatch(tool_name, arguments)`. That spec is then added to
`tool_reg` — so the LLM sees `use_tool` as a normal tool in its schema, and when it
calls it, the handler delegates to the router, which dispatches to native or MCP.

When `enable_tool_router: false` (default), `tool_router` is `None` and native tools are
exposed directly to the LLM via `tool_reg`. The router only activates when you opt in.

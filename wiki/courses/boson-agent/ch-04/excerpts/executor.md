---
chapter: ch-04
course: boson-agent
phase: read
excerpt_of: boson-agent/packages/basement/basement/tools/executor.py
created_at: "2026-04-19"
---

# Excerpt: executor.py — Full Walkthrough

**File:** `boson-agent/packages/basement/basement/tools/executor.py`
**Role:** Bridges `ToolRegistry` and actual handler invocation; normalises sync/async handlers
and wraps all results in a `ToolResultBlock` for the LLM conversation.
**Calling spec:** Called exclusively by `loop/agent_loop._execute_tool_uses()`; calls `tools/registry` and `schemas/message_schema`.

---

## Full Source with Line-by-Line Commentary

```python
# boson-agent/packages/basement/basement/tools/executor.py, lines 23-61

async def execute_tool(
    registry: ToolRegistry,
    name: str,
    tool_input: dict,
) -> ToolResultBlock:
    """Execute a tool by name with given input.

    Handles both sync and async handlers.
    Catches exceptions and returns is_error=True result.

    Args:
        registry: ToolRegistry to look up the tool.
        name: Tool name to execute.
        tool_input: Dict of arguments to pass to the handler.

    Returns:
        ToolResultBlock with result content or error message.
    """
    spec = registry.get(name)  # raises ToolNotFoundError

    try:
        if inspect.iscoroutinefunction(spec.handler):
            result = await spec.handler(**tool_input)
        else:
            result = spec.handler(**tool_input)

        return ToolResultBlock(
            tool_use_id=f"toolu_{uuid4().hex[:12]}",
            content=str(result),
            is_error=False,
        )
    except Exception as e:
        logger.error("Tool '%s' raised: %s", name, e, exc_info=True)
        return ToolResultBlock(
            tool_use_id="",
            content=f"Tool error: {type(e).__name__}: {e}",
            is_error=True,
        )
```

---

## Line-by-Line Analysis

### Signature: always `async` (line 23)

`execute_tool` is always `async` even though many tool handlers are synchronous. This is
the standard Python pattern for an async wrapper that needs to conditionally `await`:
you cannot `await` inside a sync function, so the outer function must be a coroutine.
The caller (`_execute_tool_uses` in `agent_loop.py`) is also async, so the `await
execute_tool(...)` call fits naturally into the async generator context of the agent loop.

This asymmetry — async wrapper around possibly-sync handlers — means tool authors never
need to think about the event loop. They write whatever feels natural for their use case.

### `registry.get(name)` (line 41)

Raises `ToolNotFoundError` if the name is unknown. This exception propagates up to
`_execute_tool_uses()` in `agent_loop.py`, where it is caught by the outer `except
Exception` block and converted to an error `ToolResultBlock` with `is_error=True`. The
LLM sees the error as a tool result, not as a crash — it can react to the error message
and try a different tool or ask for clarification.

### `inspect.iscoroutinefunction()` (line 44)

The sync/async branch. `iscoroutinefunction` returns `True` for functions defined with
`async def` (including methods on classes). It is evaluated per-call but is effectively
free — it just inspects `CO_COROUTINE` in the function's code flags.

The two branches produce identical `ToolResultBlock` structure; only the invocation
mechanism differs:
- Async: `await spec.handler(**tool_input)` — yields control to the event loop during I/O.
- Sync: `spec.handler(**tool_input)` — blocks the event loop until completion.

Notice that blocking the event loop is a known limitation for sync handlers that do
network I/O. For the framework's current use cases (thin wrappers, YAML reads, simple
computation) this is acceptable. A future version could use `asyncio.run_in_executor()`
to offload CPU/IO-bound sync handlers to a thread pool.

### `str(result)` (line 50)

Every return value is coerced to a string before being stored in `content`. This is the
boundary where Python's type system ends and the LLM's string world begins. The tool
author can return anything with a reasonable `__str__` — a `dict`, a `list`, a Pydantic
model, a multiline formatted string. The LLM receives it as text.

This is why the real tools (`check_available_products`, `escalate_to_human`) build and
return formatted strings — they are already pre-rendering for readability, but the
`str()` coercion means returning a structured object would also work.

### `tool_use_id=f"toolu_{uuid4().hex[:12]}"` (line 49)

In the success path, a fresh UUID fragment is generated for the `tool_use_id`. In the
error path (line 58), `tool_use_id=""` is used instead. This matters because Anthropic's
API requires `tool_use_id` in tool results to match the corresponding `tool_use` block
from the assistant message.

The agent loop corrects this at line 239 of `agent_loop.py`:
```python
result.tool_use_id = tu["id"]   # agent_loop.py, line 239
```
The `tool_use_id` generated inside `execute_tool` is immediately overwritten by the
actual ID from the streaming response. The UUID generated here is therefore a placeholder
that is never used. This is slightly redundant but harmless — `ToolResultBlock` requires
the field, and the executor cannot know the streaming ID at call time.

### Exception handling (lines 54–60)

The `except Exception` is intentionally broad. Tool handlers are user code and can raise
anything. The framework's contract is: **a tool call never crashes the agent loop**.
Instead it produces an `is_error=True` result that the LLM can see and respond to.

`exc_info=True` in the logger call ensures the full stack trace appears in logs even
though the exception is swallowed from the loop's perspective.

---

## How `execute_tool` Fits Into the Agent Loop

From `agent_loop.py`, the call site (lines 217–219 in the direct execution path):

```python
# agent_loop.py, lines 215-219
if runtime.permissions:
    runtime.permissions.check_tool(tu["name"])
result = await execute_tool(
    runtime.tool_registry, tu["name"], tool_input
)
```

And from the hook context built immediately after (lines 242–246):

```python
# agent_loop.py, lines 242-246
hook_ctx = _make_context(
    api, HookEvent.POST_TOOL_CALL,
    tool_name=tu["name"], tool_input=tool_input, tool_result=result,
)
await fire_event(hooks, HookEvent.POST_TOOL_CALL, hook_ctx)
```

The `ToolResultBlock` returned by `execute_tool` is passed directly into the
`POST_TOOL_CALL` hook context. Hooks that run at `POST_TOOL_CALL` therefore have
access to the full result, including `is_error` and `content`, before it is appended
to the conversation. This is the integration point covered in [[../ch-05-hooks]] (hooks
chapter).

---

## Sync vs Async Decision Tree for Tool Authors

```
Is your tool doing I/O (network, disk, subprocess)?
    ├── Yes, and the I/O has an async API → use async def
    │       Example: aiohttp fetch, asyncpg query
    ├── Yes, but the library is sync-only → sync def is fine
    │       Risk: blocks event loop during I/O
    │       Mitigation: keep calls fast (<100ms) or use run_in_executor
    └── No (pure computation, in-memory data) → sync def
            Example: calculate(), get_weather() stub

Both work identically from the LLM's perspective.
The difference is only in concurrency behaviour.
```

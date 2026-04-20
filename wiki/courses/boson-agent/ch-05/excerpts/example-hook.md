---
chapter: ch-05
course: boson-agent
phase: excerpt
title: "Example hook — agents/demo/hooks/logger.py"
source_file: boson-agent/agents/demo/hooks/logger.py
created_at: 2026-04-19
---

# Example Hook — agents/demo/hooks/logger.py

**Source:** `boson-agent/agents/demo/hooks/logger.py`
**Role in system:** The minimal reference implementation of a multi-event hook file. Shows the drop-file convention: no registration calls, no imports beyond the decorator and enum.

---

## Full source

```python
# boson-agent/agents/demo/hooks/logger.py, lines 1-27

from basement.hooks.events import hook, HookEvent


@hook(HookEvent.ON_TURN_START)
async def log_turn_start(ctx):
    """Log when a turn starts."""
    print("[HOOK] Turn started")


@hook(HookEvent.PRE_TOOL_CALL)
async def log_pre_tool(ctx):
    """Log before tool execution."""
    print(f"[HOOK] PRE_TOOL_CALL: {ctx.tool_name}({ctx.tool_input})")


@hook(HookEvent.POST_TOOL_CALL)
async def log_post_tool(ctx):
    """Log after tool execution."""
    result_preview = str(ctx.tool_result.content)[:80] if ctx.tool_result else "N/A"
    print(f"[HOOK] POST_TOOL_CALL: {ctx.tool_name} -> {result_preview}")


@hook(HookEvent.ON_TURN_END)
async def log_turn_end(ctx):
    """Log when a turn ends."""
    print(f"[HOOK] Turn ended (messages: {ctx.conversation.message_count})")
```

---

## Line-by-line walkthrough

**Import (line 1):** Only two names imported: `hook` (the decorator) and `HookEvent` (the enum). `HookContext` is not imported because none of these handlers use the type annotation — Python does not require it at runtime. A production hook would typically annotate `ctx: HookContext` for IDE support.

**`log_turn_start` (lines 4-6):** Subscribes to `ON_TURN_START` with default priority 100. No `ctx` fields are accessed except implicitly via the `HookContext` object being passed. This fires once per user turn, before any LLM call. Useful for initializing per-turn state (e.g., a timer or request counter stored in `ctx.metadata`).

**`log_pre_tool` (lines 9-11):** Accesses `ctx.tool_name` and `ctx.tool_input` — both guaranteed to be populated for `PRE_TOOL_CALL` (see the field population table in [[excerpts/events]]). This is the right place to validate tool arguments before execution, log the call for audit trails, or inject a warning via `ctx.conversation.inject_system_reminder`.

**`log_post_tool` (lines 14-19):** Accesses `ctx.tool_result.content`. The guard `if ctx.tool_result else "N/A"` handles the theoretical case where `tool_result` is None — in practice, `POST_TOOL_CALL` always has `tool_result` populated (it fires after the result is constructed, including error results). The `.content` attribute of `ToolResultBlock` is a string (or list of content blocks for multi-part results). The `[:80]` slice prevents log spam from large tool outputs.

**`log_turn_end` (lines 22-24):** Accesses `ctx.conversation.message_count`. This is the only handler in the file that uses the `ConversationAPI`. At `ON_TURN_END`, all tool calls for the turn have completed and all messages have been appended. The count here reflects the full turn's message additions. Note that `flush_pending` runs after `ON_TURN_END` fires (see `agent_loop.py` line 186), so this count does not yet reflect any deferred removals or compaction.

---

## What this file demonstrates about the drop-file convention

The entire file has four functions, no class, no `__init__`, no registration call. When Basement discovers `hooks/logger.py`:

1. `_import_module_from_path(py_file)` imports the module.
2. `vars(module).values()` iterates all names defined in the module.
3. `hasattr(obj, "__hook_event__")` matches each of the four functions.
4. Each is registered with its event and priority.

The framework's only requirement: files must be importable (no syntax errors, no broken imports at module load time). The `@hook` decorator does the rest.

**Notice:** Four handlers in one file, covering four different events. The registry stores them by event — they end up in four separate lists. There is no requirement that all handlers in a file share the same event. Grouping by concern (all logging in `logger.py`, all recovery in `recovery.py`) is idiomatic. The `sorted(hooks_dir.glob("*.py"))` load order only matters for handlers sharing the same event and same priority.

---

## Extending this pattern — a production-grade version

The demo logger shows the skeleton. A production variant would:

```python
# hooks/audit_logger.py — production extension of the same pattern
import logging
from basement.hooks.events import hook, HookEvent, HookContext

logger = logging.getLogger("agent.audit")

@hook(HookEvent.PRE_TOOL_CALL, priority=10)  # early — catches all tool calls
async def audit_pre_tool(ctx: HookContext) -> None:
    logger.info("TOOL_CALL tool=%s input=%r", ctx.tool_name, ctx.tool_input)

@hook(HookEvent.ON_ERROR, priority=75)  # after supervisors (50), before default (100)
async def audit_error(ctx: HookContext) -> None:
    logger.error("TOOL_ERROR tool=%s error=%r", ctx.tool_name, ctx.error, exc_info=True)
```

Key differences from the demo:
- Uses `logging` instead of `print` (structured, level-aware, goes to log aggregators).
- Explicit `ctx: HookContext` annotation for IDE support and documentation.
- Explicit priority values rather than default 100, to control ordering relative to other hooks.
- `ON_ERROR` handler at priority 75 — after supervisor recovery hooks at 50, before default observational hooks at 100.

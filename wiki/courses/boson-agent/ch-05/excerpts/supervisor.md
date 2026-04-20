---
chapter: ch-05
course: boson-agent
phase: excerpt
title: "supervisor_hook — syntactic sugar for ON_ERROR recovery"
source_file: boson-agent/packages/basement/basement/hooks/supervisor.py
created_at: 2026-04-19
---

# supervisor_hook — supervisor.py

**Source:** `boson-agent/packages/basement/basement/hooks/supervisor.py`
**Role in system:** Pure decorator definition. Wraps `@hook(HookEvent.ON_ERROR)` with a lower default priority (50 vs 100) and a simpler call signature.

---

## Full source

```python
# boson-agent/packages/basement/basement/hooks/supervisor.py, lines 1-37

# === CALLING SPEC ===
# PURPOSE: @supervisor_hook decorator — syntactic sugar for ON_ERROR hooks
# CALLED BY: User hook files in agent's hooks/ folder
# CALLS: hooks/events
# PURE: yes
# DETERMINISTIC: yes (sealed)

"""Supervisor hook — error-retry pattern.

@supervisor_hook is syntactic sugar for @hook(HookEvent.ON_ERROR).
A supervisor hook receives the error in ctx.error and can:
- Inspect the failure
- Inject corrective messages via ctx.conversation
- Let the agent loop retry
"""

from typing import Callable

from basement.hooks.events import HookEvent, hook


def supervisor_hook(fn: Callable | None = None, *, priority: int = 50) -> Callable:
    """Mark a function as a supervisor hook (fires on ON_ERROR).

    Usage:
        @supervisor_hook
        async def retry_on_error(ctx):
            if ctx.error:
                await ctx.conversation.inject_system_reminder(
                    f"Previous action failed: {ctx.error}. Try again."
                )

    Priority defaults to 50 (fires before regular hooks at 100).
    """
    if fn is not None:
        return hook(HookEvent.ON_ERROR, priority=priority)(fn)
    return hook(HookEvent.ON_ERROR, priority=priority)
```

---

## Line-by-line walkthrough

**Signature `fn: Callable | None = None` (line 22):** This is the Python decorator dual-call pattern. It lets users write either:

```python
@supervisor_hook                    # bare — fn is passed directly
async def my_handler(ctx): ...

@supervisor_hook(priority=10)       # called with args — fn is None, returns decorator
async def my_handler(ctx): ...
```

The `if fn is not None` branch handles the bare form; the `return hook(...)` branch handles the called form. Without this pattern, `@supervisor_hook` (bare) would pass the function as `priority`, causing a type error.

**`priority: int = 50` (line 22):** The default 50 is half of the regular hook default (100). Lower fires first. This means supervisor hooks — which are meant to inject corrective guidance before anything else reacts to the error — run ahead of any observational ON_ERROR hook registered with the default priority. A user can still override: `@supervisor_hook(priority=10)` for a supervisor that must fire first among supervisors.

**`hook(HookEvent.ON_ERROR, priority=priority)(fn)` (line 35):** The implementation is a one-liner delegating entirely to `@hook`. `supervisor_hook` adds zero new runtime behavior — it only provides a more expressive name and a lower default priority. Any `@supervisor_hook` handler is indistinguishable from `@hook(HookEvent.ON_ERROR, priority=50)` in the registry; both set `fn.__hook_event__ = HookEvent.ON_ERROR` and `fn.__hook_priority__ = 50`.

---

## ON_ERROR in the agent loop — the full exception flow

The key question: when a tool raises, does the loop bail out or retry?

```python
# boson-agent/packages/basement/basement/loop/agent_loop.py, lines 226-237

        except Exception as e:
            # Fire ON_ERROR hook
            error_ctx = _make_context(
                api, HookEvent.ON_ERROR,
                tool_name=tu["name"], tool_input=tool_input, error=e,
            )
            await fire_event(hooks, HookEvent.ON_ERROR, error_ctx)
            result = ToolResultBlock(
                tool_use_id=tu["id"],
                content=f"Tool error: {type(e).__name__}: {e}",
                is_error=True,
            )
```

The loop **does not bail**. The exception is caught, `ON_ERROR` hooks fire (giving supervisor hooks a chance to inject guidance via `ctx.conversation.inject_system_reminder`), and then a `ToolResultBlock` with `is_error=True` is synthesized and added to the conversation. The inner loop continues with the next tool use (if any) and then loops back to the LLM call. The LLM sees the error result and the injected system reminder, and can decide to retry.

**The retry mechanism is LLM-driven, not framework-driven.** The supervisor hook cannot force a retry directly. What it can do:

1. Call `await ctx.conversation.inject_system_reminder("...")` — this queues a reminder that will appear in the next user message sent to the LLM.
2. Call `await ctx.conversation.inject_assistant_tool_use(...)` / `await ctx.conversation.inject_tool_result(...)` — synthesize a corrective exchange in the history.

Then the loop calls the LLM again with this enriched context, and the LLM may choose to retry the tool call with corrected arguments.

**Notice:** `PermissionDeniedError` takes a separate branch (lines 220-225 of agent_loop.py) and does NOT fire `ON_ERROR`. Only unhandled exceptions from the tool executor reach the `ON_ERROR` path. A supervisor hook cannot observe permission denials.

---

## Priority ordering for ON_ERROR — worked example

Suppose an agent has three ON_ERROR handlers:

| Handler | Registered via | Priority |
|---|---|---|
| `smart_retry` | `@supervisor_hook(priority=10)` | 10 |
| `log_error` | `@supervisor_hook` (bare) | 50 |
| `alert_ops` | `@hook(HookEvent.ON_ERROR)` | 100 |

Execution order: `smart_retry` → `log_error` → `alert_ops`. All three see the same `ctx.error`. If `smart_retry` calls `inject_system_reminder`, that reminder is already queued before `log_error` runs — but `log_error` cannot read what `smart_retry` injected (the pending reminders live in `ContextManager`, not in `HookContext.metadata`). Use `ctx.metadata` for inter-handler communication within the same event firing.

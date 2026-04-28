---
chapter: ch-05
course: boson-agent
phase: excerpt
title: "HookEvent enum and HookContext dataclass — events.py"
source_file: boson-agent/packages/basement/basement/hooks/events.py
created_at: 2026-04-19
---

# HookEvent and HookContext — events.py

**Source:** `boson-agent/packages/basement/basement/hooks/events.py`
**Role in system:** Pure definitions module — enum, dataclass, decorator. No side effects. Imported by everything in the hook subsystem.

---

## The HookEvent Enum

```python
# boson-agent/packages/basement/basement/hooks/events.py, lines 17-28

class HookEvent(str, Enum):
    """Events that hooks can subscribe to."""

    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"
    ON_ERROR = "on_error"
    ON_COMPACT = "on_compact"
    ON_TURN_START = "on_turn_start"
    ON_TURN_END = "on_turn_end"
    ON_SKILL_INVOKE = "on_skill_invoke"
```

`HookEvent` inherits from both `str` and `Enum`. This is a Python pattern that lets enum members compare equal to their string values and serialize to JSON without extra conversion. When the runner does `event.value` in a log message, it gets the human-readable string (`"pre_tool_call"`) directly.

The ordering of members in the source has no functional significance — the registry uses a dict keyed by event, so lookup is O(1) regardless. The nine events map cleanly onto the agent loop's four phases:

| Phase | Events |
|---|---|
| Turn boundary | `ON_TURN_START`, `ON_TURN_END` |
| LLM boundary | `PRE_LLM_CALL`, `POST_LLM_CALL` |
| Tool boundary | `PRE_TOOL_CALL`, `POST_TOOL_CALL`, `ON_ERROR` |
| System events | `ON_COMPACT`, `ON_SKILL_INVOKE` |

**Notice:** `POST_LLM_CALL` only fires on a **text-only** response path (line 167 of agent_loop.py). When the LLM returns tool calls, the loop skips `POST_LLM_CALL` and goes directly back to `PRE_LLM_CALL` for the next iteration. A hook registered for `POST_LLM_CALL` fires zero times in a multi-tool turn and exactly once at the end of a turn that terminates with text. This is non-obvious and worth testing explicitly.

---

## HookContext Dataclass

```python
# boson-agent/packages/basement/basement/hooks/events.py, lines 31-44

@dataclass
class HookContext:
    """Context passed to every hook handler.

    Provides access to ConversationAPI for message injection/mutation.
    """

    event: HookEvent
    conversation: Any  # ConversationAPI instance
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_result: Any | None = None  # ToolResultBlock
    error: Exception | None = None
    metadata: dict = field(default_factory=dict)
```

Every hook, regardless of event type, receives the same `HookContext` shape. The optional fields are populated selectively by `_make_context()` in `agent_loop.py`. The type annotation `Any` on `conversation` is intentional — it avoids a circular import between `hooks/events.py` and `context/conversation_api.py`. In practice, `conversation` is always a `ConversationAPI` instance.

**Field population by event:**

| Event | `tool_name` | `tool_input` | `tool_result` | `error` |
|---|---|---|---|---|
| `ON_TURN_START` | None | None | None | None |
| `PRE_LLM_CALL` | None | None | None | None |
| `POST_LLM_CALL` | None | None | None | None |
| `PRE_TOOL_CALL` | set | set | None | None |
| `POST_TOOL_CALL` | set | set | set | None |
| `ON_ERROR` | set | set | None | set |
| `ON_TURN_END` | None | None | None | None |
| `ON_COMPACT` | None | None | None | None |
| `ON_SKILL_INVOKE` | None | None | None | None |

The `metadata: dict` field is an escape hatch. The framework never populates it, but a hook can set `ctx.metadata["key"] = value` to pass data to lower-priority hooks running in the same event's handler list. Since `fire_event` passes the same `HookContext` object to every handler in sequence, mutations to `metadata` are visible to subsequent handlers.

---

## The @hook Decorator

```python
# boson-agent/packages/basement/basement/hooks/events.py, lines 47-66

def hook(event: HookEvent, *, priority: int = 100) -> Callable:
    """Decorator to mark a function as a hook handler.

    Usage in agent's hooks/ folder:
        from basement.hooks.events import hook, HookEvent

        @hook(HookEvent.PRE_TOOL_CALL)
        async def my_hook(ctx: HookContext):
            print(f"About to call: {ctx.tool_name}")

    Args:
        event: Which event to subscribe to.
        priority: Lower values fire first (default 100).
    """

    def decorator(fn: Callable) -> Callable:
        fn.__hook_event__ = event
        fn.__hook_priority__ = priority
        return fn

    return decorator
```

`@hook` is a pure metadata attachment — it sets two dunder attributes on the function object and returns it unchanged. There is no registry side-effect at decoration time. The registry discovery (`HookRegistry.discover_hooks`) reads these attributes later via `hasattr(obj, "__hook_event__")`. This is the same pattern as `@tool` in [[ch-04]], which sets `__tool_spec__` on the decorated function.

**Notice:** The decorator returns the original function unmodified (no wrapper). The function can be called directly in tests without any framework involvement:

```python
# Tests can call hook handlers directly — no mocking needed
ctx = HookContext(event=HookEvent.PRE_TOOL_CALL, conversation=fake_api, tool_name="search", tool_input={})
await my_hook(ctx)  # Works fine outside the registry
```

This is a deliberate LLM-Oriented Design choice: hook handlers are plain async functions; the framework only discovers and dispatches them.

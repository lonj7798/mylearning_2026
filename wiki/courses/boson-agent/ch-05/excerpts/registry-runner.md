---
chapter: ch-05
course: boson-agent
phase: excerpt
title: "HookRegistry and fire_event — registry.py + runner.py"
source_files:
  - boson-agent/packages/basement/basement/hooks/registry.py
  - boson-agent/packages/basement/basement/hooks/runner.py
created_at: 2026-04-19
---

# HookRegistry and fire_event — registry.py + runner.py

These two files form the dispatch spine of the hook system. `registry.py` owns the sorted handler lists; `runner.py` owns the async dispatch loop with error isolation.

---

## HookRegistry — sorted handler storage

```python
# boson-agent/packages/basement/basement/hooks/registry.py, lines 22-73

class HookRegistry:
    """Discover and manage hook handlers."""

    def __init__(self):
        self._hooks: dict[HookEvent, list[tuple[int, Callable]]] = {
            event: [] for event in HookEvent
        }

    def discover_hooks(self, hooks_dir: Path) -> int:
        """Import all .py files in hooks_dir, find @hook functions.

        Returns count of discovered hooks.
        """
        if not hooks_dir.exists():
            return 0

        count = 0
        for py_file in sorted(hooks_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                module = _import_module_from_path(py_file)
                for obj in vars(module).values():
                    if hasattr(obj, "__hook_event__"):
                        self.register(
                            obj.__hook_event__,
                            obj,
                            obj.__hook_priority__,
                        )
                        count += 1
            except Exception as e:
                logger.error("Failed to load hook from %s: %s", py_file, e)
                continue

        return count

    def register(
        self, event: HookEvent, handler: Callable, priority: int = 100
    ) -> None:
        """Register a hook handler for an event."""
        self._hooks[event].append((priority, handler))
        self._hooks[event].sort(key=lambda x: x[0])
        logger.debug(
            "Registered hook '%s' for event '%s' (priority=%d)",
            handler.__name__,
            event.value,
            priority,
        )

    def get_handlers(self, event: HookEvent) -> list[Callable]:
        """Get handlers for event, sorted by priority (lower = first)."""
        return [handler for _, handler in self._hooks[event]]
```

### Line-by-line walkthrough

**`__init__` (lines 25-28):** The internal store is `dict[HookEvent, list[tuple[int, Callable]]]` — one list per event, initialized empty for all nine events at construction time. Using a list of `(priority, handler)` tuples rather than a dict allows duplicate priorities (two handlers at priority 100 both run) and makes sort straightforward.

**`discover_hooks` (lines 30-56):** The discovery loop uses `sorted(hooks_dir.glob("*.py"))` — the `sorted()` call is load-order determinism. On macOS/Linux, `glob` returns filesystem order (effectively undefined). Sorting by filename means `a_logger.py` always loads before `z_auditor.py` within the same priority level. Files starting with `_` are skipped (standard Python convention for private/internal modules).

**`register` (lines 58-69):** Every call to `register` appends then sorts. This is O(n log n) per registration, but hook discovery is a one-time startup cost. The sort key is `x[0]` (the priority integer). After every registration the list is in ascending priority order, so `get_handlers` simply strips the priority value with a list comprehension.

**Notice:** The `except Exception` catch in `discover_hooks` uses `continue` — a malformed hook file does not abort discovery of subsequent files. This fail-open policy means a syntax error in `hooks/bad_hook.py` produces a logged error but all other hooks in the directory still load. The agent runs with partial hook coverage rather than crashing. This is the same fail-open philosophy as `fire_event` (see below).

**`get_handlers` (lines 71-73):** Returns a plain list of callables with the priority stripped. The caller (`fire_event`) does not need to know about priorities after this point — ordering is already encoded in list position.

---

## fire_event — async dispatch with error isolation

```python
# boson-agent/packages/basement/basement/hooks/runner.py, lines 20-46

async def fire_event(
    registry: HookRegistry,
    event: HookEvent,
    context: HookContext,
) -> None:
    """Fire all hooks for an event.

    - Iterates handlers in priority order
    - Each handler receives HookContext with ConversationAPI access
    - Errors are logged but do NOT stop other hooks or the agent loop
    - All handlers must be async
    """
    handlers = registry.get_handlers(event)
    if not handlers:
        return

    for handler in handlers:
        try:
            await handler(context)
        except Exception as e:
            logger.error(
                "Hook '%s' for event '%s' raised: %s",
                handler.__name__,
                event.value,
                e,
                exc_info=True,
            )
```

### Line-by-line walkthrough

**Early return (line 33):** `if not handlers: return` is a hot-path optimization. Most events fire zero or one handler in a minimal agent. Avoiding the loop entirely when the list is empty saves a trivial but real allocation.

**Sequential `await` (lines 35-45):** Handlers run one at a time in priority order. This is a deliberate choice against `asyncio.gather()`. Parallel execution would break the case where handler A at priority 10 mutates `ctx.metadata` and handler B at priority 50 reads it. Sequential also means handler A's `await ctx.conversation.inject_system_reminder(...)` completes and the message is in the context before B runs.

**Per-handler `try/except` (lines 36-45):** The exception boundary wraps each handler individually. If handler A raises, handlers B and C still run. The agent loop itself never sees hook exceptions. This is why the calling spec for `runner.py` says `# PURE: no (runs user code)` — it executes arbitrary user-provided callables.

**`exc_info=True` (line 44):** The `logger.error` call passes `exc_info=True`, which attaches the full traceback to the log record. When debugging a hook that silently fails, check structured logs — the traceback will be there even though the loop continues normally.

**Notice:** `fire_event` is a module-level `async def`, not a method. It takes the registry as a parameter rather than closing over it. This is the LOD "pure function over method" principle applied to async functions — it makes the function trivially testable by passing a mock registry without any class instantiation.

---

## How registry and runner compose in the agent loop

The agent loop (see [[ch-02]]) always calls `fire_event` through this pattern:

```python
# boson-agent/packages/basement/basement/loop/agent_loop.py, lines 80-84
await fire_event(
    hooks,                          # HookRegistry from runtime
    HookEvent.ON_TURN_START,
    _make_context(api, HookEvent.ON_TURN_START),
)
```

`hooks` is `runtime.hook_registry` — a single `HookRegistry` instance shared across the entire turn. The `_make_context` helper builds the `HookContext` with only the fields relevant to the event populated (see [[excerpts/events]]).

The async generator nature of `run_agent_loop` (it `yield`s `StreamEvent` objects) does not conflict with `await fire_event(...)` — Python's async model allows `await` expressions inside an `async def` that also uses `yield`. The hook dispatch is a fully `await`-ed coroutine; control returns to the generator body only after all handlers for that event have completed (or raised and been logged).

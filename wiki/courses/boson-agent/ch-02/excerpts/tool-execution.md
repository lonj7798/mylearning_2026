---
chapter: ch-02
course: boson-agent
phase: read
kind: excerpt
source: packages/basement/basement/tools/executor.py
created_at: 2026-04-17T00:00:00Z
---

# Excerpt: `tools/executor.py` — Tool Execution

← Back to [[../read]]

---

## Source description

`tools/executor.py` is 61 lines. It does one thing: given a `ToolRegistry`, a tool name, and a dict of arguments, call the handler and return a `ToolResultBlock`. It handles sync and async handlers transparently and converts all exceptions to error results rather than propagating them. It is called by `_execute_tool_uses()` in `agent_loop.py`, and also directly by `GatewayCore._run_stage_preloads()` for tool preloading at stage transitions.

---

## Excerpt 1 — `execute_tool()` (lines 23-60)

```python
# packages/basement/basement/tools/executor.py, lines 23-60
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

**What this shows mechanically.** `execute_tool()` is a pure execution shim. It has no knowledge of hooks, permissions, the agent loop, or the conversation. Its contract is narrow: look up a spec, call a handler, return a result.

`registry.get(name)` raises `ToolNotFoundError` if the name is not registered. This exception is *not* caught inside `execute_tool()` — it propagates to the caller (`_execute_tool_uses()`), where it is caught by the outer `except Exception as e:` block and converted to a `ToolResultBlock(is_error=True)`. This is intentional: a missing tool is a different class of error from a tool that exists but fails. The framework's error model distinguishes them.

`inspect.iscoroutinefunction(spec.handler)` is the sync/async detection mechanism. If the handler is an `async def`, it is awaited. If it is a plain `def`, it is called synchronously. This means tool authors can use either style without any framework-level registration flag — the framework inspects at call time. The trade-off: synchronous handlers block the event loop. For I/O-heavy tools, `async def` is the correct choice.

`str(result)` converts the handler's return value unconditionally. Tools that return `int`, `float`, `list`, `dict`, or any custom object all produce string tool results. The LLM receives a string; it cannot receive a structured object. This is a constraint of the Anthropic tool-result protocol, not a framework limitation.

`tool_use_id=f"toolu_{uuid4().hex[:12]}"` generates a placeholder ID. The calling code in `_execute_tool_uses()` immediately overwrites this with `result.tool_use_id = tu["id"]` (agent_loop.py line 239), setting it to the actual `id` from the `ToolUseStart` event. The UUID here is a defensive default — if `execute_tool()` is called directly (as in stage preloads), the ID is meaningful. When called via `_execute_tool_uses()`, it is overwritten.

**Notice.** The `except Exception` clause inside `execute_tool()` is a *second* layer of exception handling. The outer `try/except` in `_execute_tool_uses()` (agent_loop.py lines 205-237) also catches exceptions. In practice, `execute_tool()`'s inner `except` fires only if an exception occurs during handler execution itself — `spec.handler(**tool_input)`. The `ToolNotFoundError` from `registry.get(name)` escapes the inner `try` block (it is before the `try`) and is caught by the outer handler in `_execute_tool_uses()`. So the two layers handle distinct failure points:

| Failure point | Caught by | Effect |
|---|---|---|
| `registry.get(name)` → `ToolNotFoundError` | Outer `except Exception` in `_execute_tool_uses` | `ON_ERROR` fires; `is_error=True` result |
| `spec.handler(**tool_input)` raises | Inner `except Exception` in `execute_tool` | No `ON_ERROR`; `is_error=True` result; `logger.error` |
| `permissions.check_tool()` → `PermissionDeniedError` | `except PermissionDeniedError` in `_execute_tool_uses` | No `ON_ERROR`; clean "Permission denied" result |

This three-way split is the full error taxonomy for tool execution.

**Connection to universal pattern.** `execute_tool()` is step 5b of the universal pseudocode — the "ACT" phase reduced to its mechanical minimum. Everything else (hooks, permissions, error routing) lives in the orchestration layer `_execute_tool_uses()`.

---

## Excerpt 2 — How tool results reach context (agent_loop.py lines 239-257)

```python
# packages/basement/basement/loop/agent_loop.py, lines 239-257
        result.tool_use_id = tu["id"]

        # 5c. Fire POST_TOOL_CALL
        hook_ctx = _make_context(
            api, HookEvent.POST_TOOL_CALL,
            tool_name=tu["name"], tool_input=tool_input, tool_result=result,
        )
        await fire_event(hooks, HookEvent.POST_TOOL_CALL, hook_ctx)

        # 5d. Add tool result to context
        ctx.add_message("user", [result])

        # 5e. Flush pending reminders from skill activation (use_skill injects via pending)
        reminders = ctx.pop_pending_reminders()
        if reminders:
            reminder_text = "\n".join(
                f"<system-reminder>{r}</system-reminder>" for r in reminders
            )
            ctx.add_message("user", reminder_text)
```

**What this shows mechanically.** After `execute_tool()` returns, three things happen in order:

1. `result.tool_use_id = tu["id"]` — the ID is pinned to the actual `ToolUseBlock.id` from the LLM response. The Anthropic API requires that every `tool_result` block in a user message reference the `id` of the corresponding `tool_use` in the preceding assistant message. Mismatching IDs produces an API error.

2. `POST_TOOL_CALL` fires with `tool_result=result` in the context. A hook registered here can inspect the result and, for example, call `await ctx.conversation.inject_system_reminder("The result was very long — summarize key points")`. That reminder goes into the pending queue.

3. `ctx.add_message("user", [result])` appends the result as a *list* containing one `ToolResultBlock`. The Anthropic API requires tool results in the user role as a list of blocks, not as a plain string.

4. `ctx.pop_pending_reminders()` — this is step 5e, and it is the skill-injection mechanism. If tool N was `use_skill(skill_name="explain")`, the skill injector (called from within the `use_skill` handler) called `await api.inject_system_reminder(skill_prompt)`. That reminder is now in the pending queue. Popping it here appends a `<system-reminder>...</system-reminder>` user message to context. When the loop iterates back to the LLM call (step 3), the LLM sees the skill prompt as part of the conversation history.

**Notice.** The `ctx.add_message("user", [result])` call uses role `"user"` not `"assistant"`. This is the Anthropic API convention: the agent submits tool results as if the user (or the execution environment) is reporting back to the assistant. The conversation alternates `user → assistant → user → assistant`, with tool results always on the `user` side. If you add a `ToolResultBlock` as an assistant message, the API will reject the request.

**Connection to universal pattern.** Steps 5c and 5d of the pseudocode. The `pop_pending_reminders()` at step 5e is the mechanism that links tool execution to the skill system — it is what makes the pending-reminders protocol useful inside a multi-tool turn, not just between turns.

---

## End-to-end trace: one tool call

To make the layers concrete, here is the complete path for a single `calculate(expression="2+2")` call from LLM response to LLM next input:

```
1. LLM emits ToolUseStart(id="tu_abc", name="calculate")
   → agent_loop.py line 120: tool_uses.append({"id":"tu_abc","name":"calculate","input_json":""})

2. LLM emits InputJsonDelta(partial_json='{"expression": "2+2"}')
   → agent_loop.py line 127: tool_uses[-1]["input_json"] += '{"expression": "2+2"}'

3. LLM emits ToolUseEnd(id="tu_abc")
   → agent_loop.py line 129: pass

4. LLM emits MessageEnd(stop_reason="tool_use")
   → agent_loop.py line 131: yield MessageEnd(...)

5. agent_loop.py line 134: if tool_uses: → True
   → build assistant message:
      ctx.add_message("assistant", [ToolUseBlock(id="tu_abc", name="calculate", input={"expression":"2+2"})])

6. _execute_tool_uses() called:
   a. tool_input = json.loads('{"expression": "2+2"}') = {"expression": "2+2"}
   b. fire PRE_TOOL_CALL
   c. no tool_router → permissions.check_tool("calculate") → OK
   d. execute_tool(registry, "calculate", {"expression":"2+2"})
      → spec = registry.get("calculate")  # ToolSpec with handler=calculate_fn
      → inspect.iscoroutinefunction(calculate_fn) → False (sync)
      → result_val = calculate_fn(expression="2+2") → "4"
      → return ToolResultBlock(tool_use_id="toolu_<uuid>", content="4", is_error=False)
   e. result.tool_use_id = "tu_abc"  # overwrite UUID with real id
   f. fire POST_TOOL_CALL (hook sees result.content == "4")
   g. ctx.add_message("user", [ToolResultBlock(tool_use_id="tu_abc", content="4")])
   h. pop_pending_reminders() → [] (no skill activated)

7. continue → turn_count=2, back to PRE_LLM_CALL

8. LLM receives updated messages (now includes the assistant tool_use + user tool_result)
   → LLM responds: TextDelta("The answer is 4."), MessageEnd(stop_reason="end_turn")
   → text-only branch → break
```

Total `turn_count`: 2. Total `PRE_LLM_CALL` fires: 2. Total `POST_LLM_CALL` fires: 1 (only on the final text-only iteration).

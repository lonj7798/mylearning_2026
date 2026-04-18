---
chapter: ch-02
course: boson-agent
phase: read
kind: excerpt
source: packages/basement/basement/loop/agent_loop.py
created_at: 2026-04-17T00:00:00Z
---

# Excerpt: `agent_loop.py` — The Orchestrator

← Back to [[../read]]

---

## Source description

`loop/agent_loop.py` is the single file that implements the entire think → act → observe cycle. At 263 lines it is the longest "core" file in the framework, and deliberately so — this is LOD Pattern 5 (Orchestrator Recipe): one file that calls everything else, calls nothing twice, and owns no persistent state of its own.

The module-level docstring is itself an algorithm specification:

```python
# packages/basement/basement/loop/agent_loop.py, lines 9-28
"""Agent loop — think -> act -> observe cycle.

Algorithm:
    1. Add user message to context (with pending system-reminders)
    2. Fire ON_TURN_START hooks
    3. Fire PRE_LLM_CALL hooks
    4. Stream LLM response, yield TextDelta events
    5. If tool_use in response:
       a. Fire PRE_TOOL_CALL hooks (per tool)
       b. Execute tool
       c. Fire POST_TOOL_CALL hooks (per tool)
       d. Add tool result to context
       e. Go to 3 (inner loop for tool chaining)
    6. If text-only response:
       a. Add assistant message to context
       b. Fire POST_LLM_CALL hooks
    7. Fire ON_TURN_END hooks
    8. Flush pending mutations (AD1)
    9. Return
"""
```

This docstring is load-bearing: it is the specification the tests are written against. When you read the implementation below, map each code block to a numbered step here.

---

## Excerpt 1 — Function signature and initial setup (lines 55-84)

```python
# packages/basement/basement/loop/agent_loop.py, lines 55-84
async def run_agent_loop(
    runtime: AgentRuntime,
    user_input: str,
) -> AsyncIterator[StreamEvent]:
    """Main agent loop. Yields StreamEvents for the caller to render.

    Safety: max_turns enforced. If exceeded, yield warning and stop.
    """
    ctx = runtime.context_manager
    api = runtime.conversation_api
    hooks = runtime.hook_registry

    # 1. Add user message (with system-reminders if any)
    # Skip if gateway pipeline already appended the message (v0.6)
    if not getattr(runtime, "skip_user_append", False):
        reminders = ctx.pop_pending_reminders()
        content = user_input
        if reminders:
            reminder_text = "\n".join(
                f"<system-reminder>{r}</system-reminder>" for r in reminders
            )
            content = f"{user_input}\n{reminder_text}"
        ctx.add_message("user", content)

    # 2. Fire ON_TURN_START
    await fire_event(
        hooks,
        HookEvent.ON_TURN_START,
        _make_context(api, HookEvent.ON_TURN_START),
    )
```

**What this shows mechanically.** The function is typed as `async def … -> AsyncIterator[StreamEvent]` — it is an async generator because it contains `yield` statements further down. The three local aliases (`ctx`, `api`, `hooks`) are a readability idiom repeated in every LOD orchestrator: unpack the runtime bundle once at the top, then work with flat names.

The `skip_user_append` path (line 69) is a v0.6 seam. When the Gateway's `handle_message()` calls `run_agent_loop()`, it has already appended the user message to `session.messages` (which is the same list `SharedHistory` wraps as `ctx`). Without the flag the message would be appended twice — once by the Gateway, once by the loop. `getattr(runtime, "skip_user_append", False)` is a graceful default: standalone CLI use sets no such attribute, so the loop appends normally.

`pop_pending_reminders()` is the flush mechanism for `inject_system_reminder()` calls made by hooks during the *previous* turn. Any `<system-reminder>` queued by a `POST_TOOL_CALL` or `ON_TURN_END` hook last turn is now prepended to the new user message as XML tags. The LLM receives them embedded in the user turn — not as a system message — because the Anthropic API only supports one system prompt.

**Notice.** `ON_TURN_START` fires *after* the user message is appended to context, not before. This means an `ON_TURN_START` hook can read the new user message via `ctx.conversation.get_messages()`. If it fired before the append, hooks could not react to the user's input.

**Connection to universal pattern.** This is the setup phase before step 1 of the pseudocode. The `while` loop has not started yet.

---

## Excerpt 2 — The `while` loop and LLM streaming (lines 86-131)

```python
# packages/basement/basement/loop/agent_loop.py, lines 86-131
    turn_count = 0
    while turn_count < runtime.config.max_turns:
        turn_count += 1

        # 3. Fire PRE_LLM_CALL
        await fire_event(
            hooks,
            HookEvent.PRE_LLM_CALL,
            _make_context(api, HookEvent.PRE_LLM_CALL),
        )

        # 4. Stream LLM response
        tool_uses: list[dict] = []
        text_parts: list[str] = []

        # When ToolRouter enabled: only expose meta-tools (use_tool, use_skill)
        # Native tools are reachable via use_tool("name", args)
        if runtime.tool_router:
            exposed = getattr(runtime, "exposed_meta_tools", {"use_tool", "use_skill"})
            tools = [
                s for s in runtime.tool_registry.get_all_specs()
                if s.name in exposed
            ] or None
        else:
            tools = runtime.tool_registry.get_all_specs() or None

        async for event in runtime.provider.stream(
            messages=ctx.get_messages(),
            system=ctx.get_system_prompt(),
            tools=tools,
        ):
            if isinstance(event, TextDelta):
                text_parts.append(event.text)
                yield event
            elif isinstance(event, ToolUseStart):
                tool_uses.append(
                    {"id": event.id, "name": event.name, "input_json": ""}
                )
                yield event  # Expose tool call start to caller (gateway)
            elif isinstance(event, InputJsonDelta):
                if tool_uses:
                    tool_uses[-1]["input_json"] += event.partial_json
            elif isinstance(event, ToolUseEnd):
                pass  # tool_use complete, will execute below
            elif isinstance(event, MessageEnd):
                yield event
```

**What this shows mechanically.** `tool_uses` and `text_parts` are initialized as empty lists *inside* the `while` loop on every iteration, not outside it. This is deliberate: each LLM call starts fresh. A tool chain of three iterations accumulates three separate `tool_uses` lists, each flushed to context before the next call.

The `ToolRouter` branch (lines 103-110) filters the `tools=` list before it reaches the LLM. When the router is active, the LLM receives at most `{"use_tool", "use_skill"}` — it cannot name native tools in its response because it does not know they exist. The `or None` at the end of both branches converts an empty list to `None` (meaning "no tools available"), which avoids sending an empty `tools=[]` array to the provider.

The `async for` over `runtime.provider.stream()` is the core streaming loop. Each `StreamEvent` subtype is handled inline:

- `TextDelta` → accumulate + `yield` immediately (caller sees tokens as they arrive)
- `ToolUseStart` → initialize a new accumulator dict; `yield` so the Gateway can emit a filler message
- `InputJsonDelta` → concatenate into the last accumulator's `"input_json"` string
- `ToolUseEnd` → `pass` (accumulation is complete; execution happens after the `async for` exits)
- `MessageEnd` → `yield` so the caller knows the LLM is done

**Notice.** `InputJsonDelta` has a guard: `if tool_uses:`. This protects against a malformed provider emitting an `InputJsonDelta` before any `ToolUseStart`. Without the guard, `tool_uses[-1]` would raise `IndexError`. The guard silently discards stray JSON fragments — a fail-open choice consistent with the framework's general error philosophy.

**Connection to universal pattern.** This is steps 3 and 4 of the pseudocode — PRE_LLM_CALL and the streaming accumulation. After the `async for` exits, the branch decision (tool vs text) happens next.

---

## Excerpt 3 — Branch: tool execution path (lines 133-156)

```python
# packages/basement/basement/loop/agent_loop.py, lines 133-156
        # 5. If tool_use: execute tools
        if tool_uses:
            # Build assistant message with tool_use blocks
            assistant_blocks = []
            if text_parts:
                from basement.schemas.message_schema import TextBlock
                text = re.sub(r'<system-reminder>.*?</system-reminder>\s*', '', "".join(text_parts), flags=re.DOTALL).strip()
                if text:
                    assistant_blocks.append(TextBlock(text=text))
            for tu in tool_uses:
                tool_input = (
                    json.loads(tu["input_json"]) if tu["input_json"] else {}
                )
                assistant_blocks.append(
                    ToolUseBlock(
                        id=tu["id"], name=tu["name"], input=tool_input
                    )
                )
            ctx.add_message("assistant", assistant_blocks)

            # Execute each tool
            await _execute_tool_uses(runtime, tool_uses, hooks, api, ctx)

            continue  # 5e. Loop back to LLM call

        # 6. Text-only response
        if text_parts:
            text = "".join(text_parts)
            # Strip any <system-reminder> tags the LLM may have echoed
            text = re.sub(r'<system-reminder>.*?</system-reminder>\s*', '', text, flags=re.DOTALL).strip()
            if text:
                ctx.add_message("assistant", text)

        # Fire POST_LLM_CALL
        await fire_event(
            hooks,
            HookEvent.POST_LLM_CALL,
            _make_context(api, HookEvent.POST_LLM_CALL),
        )
        break  # Done — text response means end of turn
```

**What this shows mechanically.** The branch is on `if tool_uses:` — a simple truthiness check. If the list has any entries, the turn is a tool-use turn. The assistant message is built as a mixed-content list: optionally a `TextBlock` (if the LLM emitted pre-tool reasoning text), then one `ToolUseBlock` per requested tool. This mixed block format is required by the Anthropic API — you cannot send a `ToolUseBlock` without first posting an assistant message that contains it.

`json.loads(tu["input_json"])` happens here, not in the streaming loop. The `input_json` field was a raw string built by concatenating `InputJsonDelta.partial_json` fragments. It becomes a Python dict only at this point, when the LLM has closed the JSON stream. If `input_json` is empty (`""`), the fallback is `{}` — another fail-open choice.

After `_execute_tool_uses()` returns, the code hits `continue` — which sends execution back to the top of the `while` loop. `turn_count` increments. The LLM is called again with the full conversation (now including all tool results). This is the tool chaining mechanism.

The `re.sub` stripping `<system-reminder>` tags from `text_parts` (line 139) handles a specific failure mode: the LLM occasionally echoes injected system-reminder text verbatim in its response. The regex strips it before the assistant message is committed to context, so the history stays clean.

**Notice.** `POST_LLM_CALL` fires only on the text-only branch (line 168), never on the tool-use branch. A tool chain of N iterations fires `PRE_LLM_CALL` N times but `POST_LLM_CALL` exactly once — at the final iteration when the LLM stops calling tools. This means `POST_LLM_CALL` hooks semantically mean "the LLM has finished its reasoning for this turn," not "the LLM responded."

**Connection to universal pattern.** This is step 5 (tool path → `continue`) and step 6 (text path → `break`) from the pseudocode.

---

## Excerpt 4 — `_execute_tool_uses()` — hooks, permissions, error isolation (lines 189-258)

```python
# packages/basement/basement/loop/agent_loop.py, lines 189-258
async def _execute_tool_uses(runtime, tool_uses, hooks, api, ctx):
    """Execute tool_use blocks with hooks, permissions, and ON_ERROR support."""
    for tu in tool_uses:
        tool_input = json.loads(tu["input_json"]) if tu["input_json"] else {}

        # 5a. Fire PRE_TOOL_CALL
        hook_ctx = _make_context(
            api, HookEvent.PRE_TOOL_CALL,
            tool_name=tu["name"], tool_input=tool_input,
        )
        await fire_event(hooks, HookEvent.PRE_TOOL_CALL, hook_ctx)

        # 5b. Execute tool (ToolRouter or direct)
        try:
            if runtime.tool_router and tu["name"] not in ("use_tool", "use_skill"):
                result = await runtime.tool_router.dispatch(tu["name"], tool_input)
            elif runtime.tool_router and tu["name"] in ("use_tool", "use_skill"):
                result = await execute_tool(
                    runtime.tool_registry, tu["name"], tool_input
                )
            else:
                # v0.1 fallback path — permissions still enforced (AD7)
                if runtime.permissions:
                    runtime.permissions.check_tool(tu["name"])
                result = await execute_tool(
                    runtime.tool_registry, tu["name"], tool_input
                )
        except PermissionDeniedError as e:
            result = ToolResultBlock(
                tool_use_id=tu["id"],
                content=f"Permission denied: {e}",
                is_error=True,
            )
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

        result.tool_use_id = tu["id"]

        # 5c. Fire POST_TOOL_CALL
        hook_ctx = _make_context(
            api, HookEvent.POST_TOOL_CALL,
            tool_name=tu["name"], tool_input=tool_input, tool_result=result,
        )
        await fire_event(hooks, HookEvent.POST_TOOL_CALL, hook_ctx)

        # 5d. Add tool result to context
        ctx.add_message("user", [result])

        # 5e. Flush pending reminders from skill activation
        reminders = ctx.pop_pending_reminders()
        if reminders:
            reminder_text = "\n".join(
                f"<system-reminder>{r}</system-reminder>" for r in reminders
            )
            ctx.add_message("user", reminder_text)
```

**What this shows mechanically.** Tool execution goes through three dispatch paths inside a single `try` block:

1. **ToolRouter + non-meta-tool** → `router.dispatch(name, input)` — the router handles permission checking internally and can reach MCP tools.
2. **ToolRouter + meta-tool** (`use_tool` or `use_skill`) → `execute_tool(tool_registry, name, input)` directly — meta-tools live in the registry, not behind the router.
3. **No router (v0.1 path)** → `permissions.check_tool(name)` then `execute_tool(tool_registry, name, input)` — explicit permission check before execution.

The two `except` clauses handle distinct error classes:
- `PermissionDeniedError` — raised only by `permissions.check_tool()` in path 3. **Does not** fire `ON_ERROR`. The LLM receives a structured "Permission denied" result and can decide how to proceed.
- `Exception` (all others) — fires `ON_ERROR` hook before constructing the error result. This is where supervisor hooks for retry logic attach.

`result.tool_use_id = tu["id"]` on line 239 (after the `try/except`) overwrites whatever `tool_use_id` the executor assigned. This ensures the result is always correlated to the correct `ToolUseBlock` even if `execute_tool()` generated a placeholder ID.

After `POST_TOOL_CALL`, `pop_pending_reminders()` is called again (line 252). This is the skill-injection hook: when a tool call is `use_skill(skill_name="explain")`, the skill injector pushes the skill's prompt into the pending reminders queue. The flush here converts it to a user-role `<system-reminder>` message that the LLM will see at the top of its next call. This is the mechanism that makes skills work inside tool chains.

**Notice.** `_execute_tool_uses()` is a module-level coroutine, not a method. It receives all collaborators as parameters. This makes it independently testable and keeps `run_agent_loop()` readable — you can read the outer loop without understanding tool execution internals.

**Connection to universal pattern.** This is steps 5a-5d of the pseudocode, implemented with three dispatch paths, two error classes, and a skill-injection flush that enables the "pending reminders" protocol.

---

## Excerpt 5 — Turn end: max_turns guard, ON_TURN_END, flush (lines 174-186)

```python
# packages/basement/basement/loop/agent_loop.py, lines 174-186
    else:
        # max_turns exceeded
        logger.warning("Max turns (%d) exceeded", runtime.config.max_turns)
        yield TextDelta(text="\n[Max turns exceeded — stopping]")
        yield MessageEnd(stop_reason="max_turns")

    # 7-8. Fire ON_TURN_END + flush pending mutations (AD1)
    await fire_event(
        hooks,
        HookEvent.ON_TURN_END,
        _make_context(api, HookEvent.ON_TURN_END),
    )
    await api.flush_pending()
```

**What this shows mechanically.** The `else` clause on a `while` loop in Python fires when the loop condition becomes false (i.e., `turn_count >= max_turns`) and does NOT fire when the loop exits via `break`. This is a precise, idiomatic use of Python's `while/else`: the normal text-only exit path `break`s (skipping `else`), while the max-turns path falls through to `else`. Both paths then fall through to the `fire_event` + `flush_pending()` block below the loop, which is unconditional.

`ON_TURN_END` fires in all cases: normal completion, max-turns exceeded, and — because the code is structured sequentially, not in a try/finally — it also fires after the loop regardless of how the loop exited. If you want a hook that fires even on unhandled exceptions, you would need a `try/finally`; the current design does not guarantee that.

`api.flush_pending()` applies all deferred mutations queued by hooks: `remove_message`, `replace_message`, `trigger_compact`. The ordering is significant: `ON_TURN_END` hooks fire before `flush_pending()`, which means an `ON_TURN_END` hook can still queue additional mutations that will be included in the same flush.

**Notice.** The max-turns yield produces a `TextDelta` and a `MessageEnd` with `stop_reason="max_turns"`. This means the caller's streaming loop sees a synthetic response that can be rendered to the user ("Max turns exceeded — stopping"). The `stop_reason` in `MessageEnd` is how callers distinguish this case from a normal `"end_turn"` or `"tool_use"` response.

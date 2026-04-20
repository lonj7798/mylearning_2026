---
chapter: ch-07
course: boson-agent
phase: read
excerpt_of: gateway/core.py — GatewayCore.handle_message
created_at: "2026-04-19"
---

# Excerpt: GatewayCore — handle_message

**Source:** `boson-agent/packages/gateway/gateway/core.py`
**Class:** `GatewayCore`
**Method:** `handle_message` (lines 108–256)

---

## Full method walkthrough

### Signature and ownership declaration

```python
# boson-agent/packages/gateway/gateway/core.py, lines 108-109
async def handle_message(self, session_id: str, content: str) -> AsyncIterator[str]:
    """Per-turn flow: rules → executor → agent loop. Yields text chunks."""
```

`handle_message` is declared as an `async` function that returns `AsyncIterator[str]`. In Python, any `async def` that contains `yield` becomes an **async generator**. The caller — the WebSocket handler in `__main__.py` — does `async for chunk in core.handle_message(session_id, content)` and sends each chunk down the wire. This is the outermost level of the streaming pipeline; every `yield` here is a token delivery to the client.

---

### Step 1 — Session retrieval (lines 110-113)

```python
# boson-agent/packages/gateway/gateway/core.py, lines 110-113
session = self._get_or_create_session(session_id)

# v0.4: Reset cancellation flag at turn start
InterruptHandler.reset_cancellation(session)
```

The very first thing `handle_message` does is recover the session. `_get_or_create_session` delegates to `SessionStore.has()` / `SessionStore.get()` / `SessionStore.create()`. The cancellation flag is reset so a barge-in from a previous turn does not bleed into the new one.

**Notice:** the session is not fetched from a database or reconstructed from a token. It is a live Python object kept in `SessionStore._sessions: dict[str, SessionState]`. The entire conversation history for this user is in memory, mutable, for the duration of the server process. This is a deliberate design choice for low-latency; durability is delegated to `on_disconnect`.

---

### Step 2 — SharedHistory adapter (lines 115-116)

```python
# boson-agent/packages/gateway/gateway/core.py, lines 115-116
# 2. Create SharedHistory adapter
shared_history = SharedHistory(session)
```

`SharedHistory` is a turn-scoped adapter object — created fresh each turn, but it wraps the same `SessionState` that persists across turns. Its only job at construction time is to hold a reference to `session`; the expensive work (`create_context_manager`, `create_conversation_api`) is lazy and cached on `session.context_manager` / `session.conversation_api`.

**Notice:** `SharedHistory` is re-instantiated every turn but the underlying objects it exposes (`ctx`, `api`) are singletons on the session. This means pending reminders queued to `ctx` in turn N are still visible in turn N+1, which is why `ContextManager` and `ConversationAPI` must not be re-created.

---

### Step 3 — Apply pending compact (lines 118-120)

```python
# boson-agent/packages/gateway/gateway/core.py, lines 118-120
# 3. Apply any pending compact from previous turn
if self._compact_pipeline is not None:
    self._compact_pipeline.apply_pending(session, shared_history)
```

Background compaction runs in a `asyncio.create_task` during a previous turn. Its result sits in `session.pending_compact`. Before touching history for the current turn, `apply_pending` is called. It calls `shared_history.swap_compact(...)` which does a `.clear()` + `.extend()` on the live `session.messages` list — leaving the list object itself (and thus `ctx._messages`) unchanged while replacing its contents. See [[excerpts/shared-history]] for the swap mechanics.

---

### Step 4 — User message append (lines 129-146)

```python
# boson-agent/packages/gateway/gateway/core.py, lines 129-146
pipeline_appended = getattr(session, "_pipeline_appended", False)
if pipeline_appended:
    session._pipeline_appended = False  # reset for next turn
else:
    # v0.4: Set initial stage on first message (combined with user message)
    if self._stage_machine and session.active_stage is None:
        stage = self._set_initial_stage(session)
        if stage:
            injection = build_stage_injection(stage)
            session.messages.append(Message(
                role="user",
                content=f"<system-reminder>{injection}</system-reminder>\n{content}",
            ))
        else:
            session.messages.append(Message(role="user", content=content))
    else:
        session.messages.append(Message(role="user", content=content))
```

This block handles the two paths by which a user message gets into history:

1. **LayerPipeline path** (`pipeline_appended=True`): the outer `LayerPipeline` in `__main__.py` appended the message before calling `handle_message`. Gateway skips the append here.
2. **Direct path** (`pipeline_appended=False`): `handle_message` is the entry point (no layer pipeline). Gateway appends directly to `session.messages`.

On first turn with a stage machine configured, the stage injection is embedded directly into the user message content rather than appended as a separate message. This keeps the Claude API message structure valid (a `<system-reminder>` inside a user turn, not a standalone system turn).

**Notice:** `session.messages.append(...)` is a direct write to the same list that `ctx._messages` points at. The agent loop will see this message without any synchronisation step because they share the same list object.

---

### Step 5 — Rule engine (lines 148-154)

```python
# boson-agent/packages/gateway/gateway/core.py, lines 148-154
# 5. Run rule engine
if self._rule_engine is not None:
    actions = await self._rule_engine.evaluate(
        session.messages, content, session
    )
else:
    actions = [Continue()]
```

Rules receive the full `session.messages` list (including the just-appended user message) and the raw `content` string. Rules can read history to make decisions; they do not append to history. See [[ch-10]] for the full rule system.

---

### Step 6 — Action execution (lines 156-168)

```python
# boson-agent/packages/gateway/gateway/core.py, lines 156-168
result = await self._action_executor.execute(actions, session)

# v0.4: Execute pending stage transition from rules
if result.pending_transition:
    skill_fillers = await self._apply_stage_transition(session, result.pending_transition)
    for sf in skill_fillers:
        yield f"\n{sf}\n"

# 7. Fixed response — skip agent
if not result.should_continue:
    yield result.response or ""
    return
```

`ActionExecutor.execute` processes the list of actions from the rule engine. If any action requests an immediate response (e.g., `Respond`, `Filter`), `result.should_continue` is `False` and `handle_message` yields that fixed text and returns — the agent loop never runs for this turn.

**Notice:** the `yield` on filler text and the `yield` + `return` for fixed responses happen *before* the agent loop. This is the Gateway's conversation ownership in action: it can decide an entire turn without ever invoking the LLM.

---

### Steps 8-9 — AgentRuntime construction and loop invocation (lines 174-195)

```python
# boson-agent/packages/gateway/gateway/core.py, lines 174-195
runtime = self._build_agent_runtime(session, shared_history)
# v0.6: skip user append in agent loop if pipeline already did it
if pipeline_appended:
    runtime.skip_user_append = True

# v0.6-lina: filter tools and meta-tools by current stage
if runtime.tool_router and session.active_stage and self._stage_machine:
    stage_def = self._stage_machine.get_stage(session.active_stage)
    if stage_def and stage_def.tools is not None:
        runtime.tool_router.set_allowed_tools(stage_def.tools)
    else:
        runtime.tool_router.set_allowed_tools(None)
    exposed = set()
    if stage_def:
        if stage_def.tools:
            exposed.add("use_tool")
        if getattr(stage_def, "skills", None):
            exposed.add("use_skill")
    runtime.exposed_meta_tools = exposed
```

`_build_agent_runtime` wraps all session-scoped objects (`ctx`, `api`, provider, registries) into an `AgentRuntime` value object. This is the Aggregated Dependency (AD3) pattern: instead of passing a dozen arguments to `run_agent_loop`, a single struct carries everything.

The `skip_user_append` flag is set on the runtime before passing it down, so `run_agent_loop` knows not to double-append the user message.

Stage-aware tool filtering happens here: the runtime's `tool_router` has its allowed set narrowed to only the tools declared for `session.active_stage`. The LLM will only be offered `use_tool` or `use_skill` meta-tools if the current stage actually has tools or skills configured.

---

### The streaming bridge (lines 196-256) — critical section

```python
# boson-agent/packages/gateway/gateway/core.py, lines 196-256
from basement.llm.base import TextDelta, ToolUseStart
filler_count = 0
# Buffer initial text to catch system-reminder echoes (always at response start)
# Once cleared, switch to direct streaming for low latency
initial_buf: list[str] = []
streaming = False

async for event in run_agent_loop(runtime, content):
    if isinstance(event, TextDelta):
        if streaming:
            yield event.text
        else:
            initial_buf.append(event.text)
            combined = ''.join(initial_buf)
            # Check if system-reminder tag is opening
            if '<system-reminder>' in combined:
                if '</system-reminder>' in combined:
                    # Complete tag — strip and flush remainder
                    clean = _SR_RE.sub('', combined)
                    clean = _TOOL_CALL_RE.sub('', clean).strip()
                    if clean:
                        yield clean
                    initial_buf = []
                    streaming = True
                # else: tag still open, keep buffering
            elif len(combined) > 30 or '\n' in combined:
                # No tag after enough text — safe to stream
                clean = _TOOL_CALL_RE.sub('', combined).strip()
                if clean:
                    yield clean
                initial_buf = []
                streaming = True
    elif isinstance(event, ToolUseStart):
        # Flush any buffered text as filler (pre-tool text is transition, not real response)
        if initial_buf:
            raw = ''.join(initial_buf)
            clean = _SR_RE.sub('', raw)
            clean = _TOOL_CALL_RE.sub('', clean).strip()
            if clean:
                yield f"[FILLER]{clean}[/FILLER]"
            initial_buf = []
        streaming = False  # reset — post-tool text needs buffering again
        limit = self._max_fillers_per_turn  # 0=off, -1=unlimited, N=cap
        if limit != 0 and (limit < 0 or filler_count < limit):
            raw = self._tool_fillers.get(
                event.name, self._tool_fillers.get("_default", ""))
            if raw:
                filler = random.choice(raw) if isinstance(raw, list) else raw
                yield f"\n{filler}\n"
                filler_count += 1
        if self._show_tool_calls:
            yield f"[tool: {event.name}]"

# Flush any remaining buffered text
if initial_buf:
    raw = ''.join(initial_buf)
    clean = _SR_RE.sub('', raw)
    clean = _TOOL_CALL_RE.sub('', clean).strip()
    if clean:
        yield clean
```

This is the most mechanically complex section. Walk through it state-by-state:

**State machine variables:**
- `initial_buf: list[str]` — accumulates `TextDelta.text` fragments before the streaming gate opens
- `streaming: bool` — `False` at start, set `True` when the buffer is flushed clean; resets to `False` after each `ToolUseStart`
- `filler_count: int` — cap enforcement for tool fillers

**TextDelta handling (two-path logic):**

When `streaming=True`, every `TextDelta` is yielded immediately — zero buffering, zero copy. This is the low-latency hot path for all text after the initial guard check passes.

When `streaming=False`, fragments are accumulated in `initial_buf`. After each append, the concatenated string is checked for a `<system-reminder>` opening tag. There are three outcomes:
1. Tag is open but not closed yet: keep buffering (no yield).
2. Complete tag found: strip it with `_SR_RE`, strip any `use_tool(...)` artifacts with `_TOOL_CALL_RE`, yield whatever remains, clear buf, set `streaming=True`.
3. No tag after 30 chars or a newline: the LLM is clearly not echoing a reminder, flush cleaned text, set `streaming=True`.

**ToolUseStart handling:**

A `ToolUseStart` event means the LLM is beginning a tool call. Any accumulated `initial_buf` is flushed as a `[FILLER]...[/FILLER]` tagged string (pre-tool "thinking" text that UIs may display as a loading indicator). The streaming gate resets to `False` because post-tool LLM text must also go through the guard — the LLM might emit another system-reminder echo after tool results are injected.

Tool filler text (`self._tool_fillers`) is then looked up by tool name and yielded if configured and the per-turn cap allows it.

**After the loop:**

Any remaining `initial_buf` is flushed. This handles the edge case where the entire LLM response was short enough to stay in the buffer without triggering the >30-char or newline threshold.

**Notice:** `streaming` resets to `False` on every `ToolUseStart`. This means a turn with N tool calls applies the system-reminder guard N+1 times: once before the first LLM call, and once after each tool result. This is correct because the LLM re-reads history (including injected tool results) on each inner loop iteration and can produce a new reminder-echo at any point.

---

### _build_agent_runtime (lines 384-414)

```python
# boson-agent/packages/gateway/gateway/core.py, lines 384-414
def _build_agent_runtime(
    self, session: SessionState, shared_history: SharedHistory
) -> AgentRuntime:
    """Build an AgentRuntime for this session turn."""
    ctx = shared_history.create_context_manager()
    api = shared_history.create_conversation_api(ctx)

    provider = get_provider(self._agent_config.llm)
    permissions = load_permissions(self._agent_config)

    # Register use_skill per session (needs per-session ConversationAPI)
    # Re-register each turn to capture the new ConversationAPI instance
    if self._skill_registry and self._skill_registry.get_all():
        from basement.skills.injector import create_use_skill
        use_skill_spec = create_use_skill(
            self._skill_registry, api, self._hook_registry, permissions
        )
        self._tool_registry._tools[use_skill_spec.name] = use_skill_spec

    return AgentRuntime(
        config=self._agent_config,
        provider=provider,
        tool_registry=self._tool_registry,
        hook_registry=self._hook_registry,
        context_manager=ctx,
        conversation_api=api,
        system_prompt=self._system_prompt,
        skill_registry=self._skill_registry,
        permissions=permissions,
        tool_router=self._tool_router,
    )
```

`create_context_manager()` and `create_conversation_api(ctx)` both check `session.context_manager` / `session.conversation_api` before creating. If they already exist (all turns after turn 1), the cached objects are returned. This is what preserves `ctx._messages` as the same list object across turns.

`use_skill` is re-registered every turn because `ConversationAPI` — even though it is cached — might be needed in a closure captured by the skill spec. In practice for v0.6 this is a no-op on turns 2+ since the same `api` object is returned, but the pattern is defensive.

**Notice:** `ctx` returned here is the same object as `session.context_manager`. So `ctx._messages is session.messages` is `True` for every turn after turn 1. The agent loop never holds a distinct copy of the message list.

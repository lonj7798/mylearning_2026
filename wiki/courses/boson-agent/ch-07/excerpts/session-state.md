---
chapter: ch-07
course: boson-agent
phase: read
excerpt_of: gateway/schemas/session.py — SessionState
created_at: "2026-04-19"
---

# Excerpt: SessionState — the conversation-scoped record

**Source:** `boson-agent/packages/gateway/gateway/schemas/session.py`
**Class:** `SessionState` (lines 30–53)

---

## Full dataclass definition

```python
# boson-agent/packages/gateway/gateway/schemas/session.py, lines 20-53
@dataclass
class SessionState:
    """State for a single Gateway session.

    Persists across turns. ContextManager and ConversationAPI are created
    once at session start and reused (prevents pending_reminder loss).
    """

    session_id: str
    messages: list[Message] = field(default_factory=list)
    system_prompt: str = ""
    active_skill: str | None = None
    active_stage: str | None = None
    pending_compact: dict | None = None
    compact_in_progress: bool = False
    # Persisted across turns (Architect/Critic fix)
    context_manager: Any = None
    conversation_api: Any = None
    # v0.4: Interruption support
    cancellation_flag: CancellationFlag = field(default_factory=CancellationFlag)
    partial_buffer: PartialBuffer | None = None
    # v0.5: Agent status tracking for layer rules
    status_tracker: AgentStatusTracker = field(default_factory=AgentStatusTracker)
    # v0.6: History write lock for WebSocket concurrency
    history_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
```

---

## Field-by-field analysis

### `session_id: str`
The primary key. In the WebSocket server, this is derived from the client connection object (e.g., a UUID per connection). It is the key into `SessionStore._sessions`.

### `messages: list[Message]`
**The single source of truth for conversation history.** This list is:
- Written by `GatewayCore.handle_message` when appending user messages.
- Written by `run_agent_loop` (via `ctx.add_message`) when appending assistant messages and tool results.
- Read by `RuleEngine.evaluate` to give rules access to full history.
- Read by `ctx.get_messages()` (returns a deep copy) when calling the LLM provider.
- Mutated in-place by `SharedHistory.swap_compact` during compaction.

Because `ctx._messages` is assigned to point at this exact list object, every append to `session.messages` is immediately visible to the agent loop through `ctx`, and every `ctx.add_message(...)` call appends to the same physical list.

### `active_skill: str | None`
The name of the currently active skill, if any. Persists across turns. Used by `swap_compact` to re-inject the skill reminder after a compact operation (see [[excerpts/shared-history]]). See [[ch-12]] for full skill lifecycle.

### `active_stage: str | None`
The name of the current conversation stage. `None` until the first message is processed (stage machine initialises it in `_get_or_create_session` / first-turn handling). Checked by `_build_agent_runtime` to decide which tools to expose via the `ToolRouter`. See [[ch-12]] for stage machine details.

### `pending_compact: dict | None`
Set by `AsyncCompactPipeline._compact_task` when a background summarisation completes. The dict contains `{"summary": str, "keep_recent": int}`. Consumed and cleared by `apply_pending` at the top of the next `handle_message` call. The one-turn lag is intentional: the compact task runs asynchronously during a turn while the main path is streaming; applying it at the START of the next turn ensures the current turn's LLM call sees a consistent history.

### `compact_in_progress: bool`
Guard flag set by `trigger()` before spawning the asyncio task and cleared in the `finally` block of `_compact_task`. Prevents stacking multiple background compaction tasks for the same session.

### `context_manager: Any` and `conversation_api: Any`
Stored on the session after first creation by `SharedHistory.create_context_manager()` / `create_conversation_api()`. Typed as `Any` to avoid a circular import between `gateway.schemas` and `basement.context`. The docstring comment "Persisted across turns (Architect/Critic fix)" refers to a design iteration where recreating these objects each turn caused loss of pending reminders queued by skills during a previous turn.

### `cancellation_flag: CancellationFlag`
A simple flag object with an `is_set` attribute. Reset at the top of each `handle_message`. Set by `InterruptHandler` when a barge-in is detected. Checked before the LLM call to short-circuit the turn.

### `history_lock: asyncio.Lock`
Added in v0.6 to protect against concurrent writes from multiple WebSocket frames arriving within a single turn (e.g., very fast typing or reconnect races). The lock is per-session so sessions never block each other.

---

## Why a plain dataclass, not a Pydantic model?

`SessionState` uses `@dataclass` rather than `BaseModel` because:
1. It holds non-serialisable objects (`asyncio.Lock`, `CancellationFlag`, `ContextManager`). Pydantic validation would reject or mangle these.
2. It is never serialised to JSON at request time (only at disconnect, in `on_disconnect`, where it is manually `.model_dump()`-ed per message).
3. Construction speed matters: sessions are created per connection, not per request.

**Notice:** the `messages` field uses `field(default_factory=list)` — a fresh empty list per instance. This is the Python dataclass idiom to avoid the mutable-default-argument trap. If `messages=[]` were written as a class-level default, all sessions would share the same list.

---
chapter: ch-07
course: boson-agent
phase: read
excerpt_of: gateway/session/history.py — SharedHistory
created_at: "2026-04-19"
---

# Excerpt: SharedHistory — the shared-reference adapter

**Source:** `boson-agent/packages/gateway/gateway/session/history.py`
**Class:** `SharedHistory` (lines 17–101)

---

## Full class listing

```python
# boson-agent/packages/gateway/gateway/session/history.py, lines 17-101
class SharedHistory:
    """Adapts a SessionState for use with basement's ContextManager.

    The ContextManager's internal _messages list is pointed directly at
    session.messages so that both the gateway and the agent loop share
    the same list object. This is intentional: it avoids synchronisation
    overhead and ensures both sides see mutations immediately.
    """

    def __init__(self, session: SessionState) -> None:
        self._session = session

    def create_context_manager(self) -> ContextManager:
        """Return the session's ContextManager, creating it once.

        The manager's _messages attribute is set directly to
        session.messages (intentional — shared reference, not a copy).
        """
        if self._session.context_manager is not None:
            return self._session.context_manager

        ctx = ContextManager(system_prompt=self._session.system_prompt)
        # Intentional direct assignment: share the same list object so that
        # gateway writes to session.messages are visible to the agent loop
        # through ctx._messages and vice-versa.
        ctx._messages = self._session.messages
        self._session.context_manager = ctx
        return ctx

    def create_conversation_api(self, ctx: ContextManager) -> ConversationAPI:
        """Return the session's ConversationAPI, creating it once."""
        if self._session.conversation_api is not None:
            return self._session.conversation_api

        api = ConversationAPI(ctx)
        self._session.conversation_api = api
        return api

    def swap_compact(self, compact_summary: str, keep_recent: int = 10) -> None:
        """Replace old messages with a compact summary + recent messages.

        Builds a new message list:
          1. A user message wrapping the compact summary.
          2. The last `keep_recent` messages from the current history.
          3. Optional system-reminder messages for active_skill / active_stage.

        After the swap:
          - session.messages is replaced with the new list.
          - The ContextManager's _messages is re-pointed to the new list.
          - session.pending_compact is cleared.
        """
        session = self._session

        summary_msg = Message(
            role="user",
            content=f"[Compact Summary]\n{compact_summary}",
        )

        recent = list(session.messages[-keep_recent:]) if session.messages else []

        new_messages: list[Message] = [summary_msg] + recent

        if session.active_skill:
            new_messages.append(
                Message(
                    role="user",
                    content=f"<system-reminder>Active skill: {session.active_skill}</system-reminder>",
                )
            )

        if session.active_stage:
            new_messages.append(
                Message(
                    role="user",
                    content=f"<system-reminder>Active stage: {session.active_stage}</system-reminder>",
                )
            )

        # Replace session.messages contents in-place so any existing
        # shared reference (e.g. ctx._messages) stays valid.
        session.messages.clear()
        session.messages.extend(new_messages)

        session.pending_compact = None
```

---

## The shared-reference trick, explained precisely

### Construction (first turn)

When `create_context_manager()` runs for the first time:

```python
ctx = ContextManager(system_prompt=self._session.system_prompt)
# ContextManager.__init__ sets: self._messages = []  (a new empty list)
ctx._messages = self._session.messages  # OVERWRITE with session's list
self._session.context_manager = ctx
```

`ContextManager.__init__` creates a fresh `[]` for `self._messages`. Then the very next line overwrites that attribute with `session.messages` — the list already owned by `SessionState`. After this assignment:

```
session.messages  ──┐
                    ├──► [ same list object in memory ]
ctx._messages     ──┘
```

Both names are aliases for the same Python list object. There is no copying, no proxy, no observer pattern — just two attribute names pointing at one list.

### Why no lock is needed within a single turn

The Python asyncio event loop is single-threaded. Within a single turn of `handle_message`:

1. Gateway appends user message to `session.messages` (synchronous).
2. Gateway runs the rule engine (awaited, but no other coroutine can run while a sequential await is active unless the awaited code yields control).
3. `run_agent_loop` runs as an async generator. Each `async for event in run_agent_loop(runtime, content)` iteration resumes the generator until the next `yield` or `await`. Between yields, control is in `handle_message`.
4. `run_agent_loop` calls `ctx.add_message(...)` to append assistant messages and tool results — these writes land in the shared list while the generator is running (i.e., between yields from the generator back to `handle_message`).

No two coroutines are ever simultaneously executing against the shared list within a turn. The asyncio event loop's cooperative scheduling is the "lock." The v0.6 `history_lock` on `SessionState` protects against *inter-turn* races (two WebSocket frames arriving concurrently for the same session), not intra-turn races.

### `get_messages()` returns a deep copy — and why that is fine

`ContextManager.get_messages()` returns `deepcopy(self._messages)`. This copy is passed to the LLM provider's `stream()` method. The deep copy is made because:
1. The provider might serialise the list asynchronously while the loop continues appending tool results.
2. A snapshot is semantically correct: the LLM should see messages as of the moment it is called, not messages that arrive during streaming.

The deep copy is of the *snapshot sent to the API*, not of the working history. Writes from both sides still land in the shared list.

---

## `swap_compact` — in-place replacement

The compact swap is the trickiest operation on the shared list. It must replace the list *contents* without replacing the list *object* (because `ctx._messages` holds a reference to the object, not to `session.messages` the attribute):

```python
# WRONG — would break the shared reference:
session.messages = new_messages      # ctx._messages still points at old list

# CORRECT — modifies the list object in place:
session.messages.clear()
session.messages.extend(new_messages)
```

After `clear()` + `extend()`:
- `session.messages` still refers to the same list object.
- `ctx._messages` still refers to the same list object.
- Both now see the compacted contents.

The test `test_compact_ctx_reflects_new_messages` in `test_shared_history.py` explicitly verifies this:

```python
# boson-agent/packages/gateway/tests/test_shared_history.py, lines 150-163
def test_compact_ctx_reflects_new_messages():
    """After swap_compact, ContextManager still sees the new message list."""
    session = _make_session()
    sh = SharedHistory(session)
    ctx = sh.create_context_manager()

    for i in range(10):
        session.messages.append(Message(role="user", content=f"msg {i}"))

    sh.swap_compact("compacted", keep_recent=3)

    # ctx._messages is the same list object as session.messages
    assert ctx.message_count == len(session.messages)
```

**Notice:** `active_skill` and `active_stage` are re-injected as `<system-reminder>` messages at the *end* of the compacted list. This ensures that after a compact wipes history, the LLM still knows what stage and skill it is operating under — without having to re-run the full stage injection flow. These re-injections are conversation-scoped state (owned by `SessionState`) surviving the compaction, which is exactly why they live on `SessionState` rather than being derived dynamically each turn.

---

## How hooks see session history

Hooks registered via `@hook` in the agent folder run inside `run_agent_loop` with access to `ConversationAPI`. `ConversationAPI` wraps `ContextManager` (which wraps `session.messages`). So a `PRE_LLM_CALL` hook that calls `ctx.conversation.get_messages()` is reading the same list as `GatewayCore.handle_message` when it appended the user message two steps earlier. This is the mechanism described in [[ch-05]]: hooks see session history through `SharedHistory` without any additional wiring.

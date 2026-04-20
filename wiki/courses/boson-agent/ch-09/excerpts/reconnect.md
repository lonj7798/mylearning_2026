---
chapter: ch-09
course: boson-agent
phase: read
excerpt_for: reconnect
created_at: "2026-04-19"
sources_cited:
  - "boson-agent/packages/gateway/gateway/session/store.py"
  - "boson-agent/packages/gateway/gateway/core.py"
  - "boson-agent/packages/gateway/gateway/server/websocket.py"
  - "boson-agent/packages/gateway/gateway/session/history.py"
---

# Excerpt: Reconnect Behavior — Session Rebind and State Restoration

---

## The reconnect contract

Boson Gateway's reconnect behavior is implicit, not explicit. There is no
`reconnect` message type in the protocol. A client that drops and reconnects
simply sends a `user_message` with the same `session_id` it used before. The
server-side machinery does the right thing automatically via
`_get_or_create_session`.

---

## _get_or_create_session: the reconnect guard

```python
# boson-agent/packages/gateway/gateway/core.py, lines 261-265

def _get_or_create_session(self, session_id: str) -> SessionState:
    """Return existing session or create a new one."""
    if self._sessions.has(session_id):
        return self._sessions.get(session_id)
    return self._sessions.create(session_id, system_prompt=self._system_prompt)
```

This two-line method is the entire reconnect implementation. If the session
exists (the client dropped and reconnected with the same `session_id`), `get`
returns the existing `SessionState` with all its history intact. If the session
is new (first connection or the server was restarted), `create` initializes a
fresh one. There is no TTL, no tombstoning, no session migration — sessions
live in a `dict` in the `GatewayCore` instance and survive for the lifetime of
the process.

**Notice — server restart destroys all sessions:** `SessionStore._sessions` is
an in-memory `dict`. A process restart clears it. The `on_disconnect` handler
in `core.py` writes history to a JSON file under `agent_dir/history/`, but
there is no corresponding `on_connect` that reads from those files to restore
a session. Reconnect within a running process works perfectly; reconnect across
a server restart does not restore history. This is a known limitation of the
current architecture.

---

## What state survives a reconnect

When a client reconnects within the same process lifetime, it gets back a
`SessionState` object that still holds:

```python
# boson-agent/packages/gateway/gateway/schemas/session.py, lines 30-53

@dataclass
class SessionState:
    session_id: str
    messages: list[Message]          # full conversation history
    system_prompt: str               # agent system prompt
    active_skill: str | None         # current active skill
    active_stage: str | None         # current stage machine stage
    pending_compact: dict | None     # pending compaction job
    compact_in_progress: bool
    context_manager: Any             # ContextManager (shared ref with agent loop)
    conversation_api: Any            # ConversationAPI
    cancellation_flag: CancellationFlag   # reset at next turn start
    partial_buffer: PartialBuffer | None  # partial transcript state
    status_tracker: AgentStatusTracker
    history_lock: asyncio.Lock       # concurrency guard
```

All fields survive. The `cancellation_flag` will be reset at the next turn
start via `InterruptHandler.reset_cancellation(session)`. The `partial_buffer`
may hold stale state from an interrupted partial transcript — if the client
drops mid-partial, the buffer is never cleared. The next `user_message` from
the reconnected client will run `detect_and_handle_partial`, which may or may
not classify it as a partial depending on content overlap and the (now-large)
elapsed time. Since `timing_threshold_ms` defaults to 1000ms and a reconnect
could take several seconds, the timing gate will classify the new message as
`NEW_MESSAGE`, which is correct behavior.

---

## SharedHistory: shared reference survives reconnect

```python
# boson-agent/packages/gateway/gateway/session/history.py, lines 29-44

def create_context_manager(self) -> ContextManager:
    """Return the session's ContextManager, creating it once."""
    if self._session.context_manager is not None:
        return self._session.context_manager

    ctx = ContextManager(system_prompt=self._session.system_prompt)
    # Intentional direct assignment: share the same list object so that
    # gateway writes to session.messages are visible to the agent loop
    # through ctx._messages and vice-versa.
    ctx._messages = self._session.messages
    self._session.context_manager = ctx
    return ctx
```

`context_manager` is persisted on `SessionState` and only created once.
`create_context_manager` short-circuits on line 35 if one already exists.
This means the `ContextManager._messages` reference points at the same list
object across all turns — including turns after a reconnect. When the client
reconnects, `handle_message` calls `SharedHistory(session).create_context_manager()`,
which returns the existing manager. The agent loop gets a `ContextManager`
already pointing at the full history list with no re-initialization needed.

**Notice — `swap_compact` uses `session.messages.clear()` + `extend()`, not
replacement:** The compact operation (see [[excerpts/partial-transcript]] for
context on why) must update the list in-place rather than reassigning
`session.messages = new_list`, because `ctx._messages` holds a direct
reference to the old list object. If compaction replaced the list object,
the `ContextManager` would still point at the old (now-empty from the garbage
collector's perspective) list. The `clear()` + `extend()` pattern preserves
the reference while replacing the contents.

---

## The on_disconnect handler: history persistence

```python
# boson-agent/packages/gateway/gateway/core.py, lines 371-382

async def on_disconnect(self, session_id: str) -> None:
    """Save session history on client disconnect."""
    if not self._sessions.has(session_id): return
    session = self._sessions.get(session_id)
    if not session.messages: return
    hdir = self._agent_dir / "history"
    hdir.mkdir(exist_ok=True)
    path = hdir / f"gateway_{session_id}_{datetime.now():%Y%m%d_%H%M%S}.json"
    data = {"session_id": session_id, "timestamp": datetime.now().isoformat(),
            "message_count": len(session.messages),
            "messages": [m.model_dump() for m in session.messages]}
    path.write_text(json.dumps(data, indent=2, default=str))
```

On disconnect, the full message list is serialized to a timestamped JSON file.
This is a write-only audit log — history files accumulate and are never read
back by the gateway. The session itself is NOT removed from `_sessions` on
disconnect, which enables the implicit reconnect behavior: the session stays
alive in memory until the process terminates, waiting for the client to return.

**Notice — the session is not removed on disconnect:** `on_disconnect` writes
the history file but does not call `self._sessions.remove(session_id)`. This
is intentional for the reconnect use case but creates a memory leak for
long-running servers with many unique session IDs (each abandoned session
consumes memory indefinitely). A production deployment would need a TTL eviction
policy or explicit session cleanup.

Connection to universal pattern: the reconnect pattern is forced by the WebSocket
substrate — TCP connections drop; clients reconnect. Any stateful WebSocket
server must decide whether session identity is tied to the connection (session
dies on disconnect) or to a session_id (session survives, connection is
rebindable). Boson chooses session_id identity, which enables reconnect but
requires the in-memory session store to outlive individual connections.

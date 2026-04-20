---
chapter: ch-09
course: boson-agent
phase: read
excerpt_for: per-session-lock
created_at: "2026-04-19"
sources_cited:
  - "boson-agent/packages/gateway/gateway/schemas/session.py"
  - "boson-agent/packages/gateway/gateway/session/store.py"
  - "boson-agent/packages/gateway/gateway/server/websocket.py"
  - "boson-agent/docs/plan/v0_6/04-phase4-websocket-concurrency.md"
---

# Excerpt: Per-Session Lock and Turn Serialization

---

## SessionState: the lock field

```python
# boson-agent/packages/gateway/gateway/schemas/session.py, lines 30-53

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

`history_lock` is declared with `field(default_factory=asyncio.Lock)`. This
means every `SessionState` instance gets its own independent `asyncio.Lock` at
construction time. Two sessions on the same connection cannot share a lock.
Two turns of the same session contend on the same lock — which is exactly the
right granularity for preventing concurrent writes to `session.messages`.

**Notice — `asyncio.Lock` is not thread-safe:** This is a cooperative lock for
use inside a single asyncio event loop. If Gateway were ever run with
`asyncio.run_in_executor` for synchronous tools on a thread pool, a thread
appending to `session.messages` would bypass this lock entirely. The current
design assumes all writes to `session.messages` happen in coroutines on the
same event loop, which is true for the present implementation.

---

## SessionStore: isolation by construction

```python
# boson-agent/packages/gateway/gateway/session/store.py, lines 13-53

class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create(self, session_id: str, system_prompt: str = "") -> SessionState:
        if session_id in self._sessions:
            raise ValueError(f"Session already exists: {session_id!r}")
        session = SessionState(session_id=session_id, system_prompt=system_prompt)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id!r}")
        return self._sessions[session_id]

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_active(self) -> list[str]:
        return list(self._sessions.keys())

    def has(self, session_id: str) -> bool:
        return session_id in self._sessions
```

`create` raises `ValueError` if the session already exists. This is the
reconnect guard: a client that sends a `user_message` with an existing
`session_id` will find the session via `get`, not `create`. `GatewayCore`
uses `_get_or_create_session` which calls `has()` first, so reconnecting
clients transparently reuse the existing session with all its history intact.

**Notice — no lock on `_sessions` itself:** Concurrent calls to `create` for
the same `session_id` from two asyncio tasks would both pass the `has()` check
and race into `create`. In practice, the per-session task serialization in
`websocket.py` means only one task per session runs at a time, so this race
cannot occur under normal operation. It would be exposed only by direct
concurrent calls that bypass the WebSocket server.

---

## WebSocket server: the concurrency model

```python
# boson-agent/packages/gateway/gateway/server/websocket.py, lines 97-183

async def _handle_connection(
    self, websocket: websockets.server.ServerConnection
) -> None:
    """Handle a single client connection for its lifetime.

    v0.6: Reader loop spawns handler tasks instead of blocking.
    New user_message cancels in-progress handler for the same session.
    """
    remote = websocket.remote_address
    session_ids: set[str] = set()

    try:
        async for raw in websocket:
            # ... parse, validate ...
            session_ids.add(msg.session_id)

            # v0.6: Cancel in-progress handler for this session
            self._cancel_active_task(msg.session_id)

            # v0.6: Spawn handler as task (non-blocking reader loop)
            task = asyncio.create_task(
                self._process_message(websocket, msg.session_id, msg.content)
            )
            self._active_tasks[msg.session_id] = task

    finally:
        for sid in session_ids:
            self._cancel_silence_timer(sid)
            self._cancel_active_task(sid)
        if self._on_disconnect:
            for sid in session_ids:
                try:
                    await self._on_disconnect(sid)
                except Exception:
                    logger.exception("on_disconnect failed for session %s", sid)
```

The v0.6 design is "task-per-message with cancel-on-new-arrival." For each
incoming `user_message`, the server calls `_cancel_active_task(session_id)`
before spawning a new task. This means the asyncio `Task` running the previous
`handle_message + run_agent_loop` receives a cancellation at its next `await`
point — which is the next `await websocket.send(delta)` inside
`_process_message`. The streaming generator chain is then unwound via
`asyncio.CancelledError`, caught by the try/except in `_process_message` and
logged (not re-raised).

**Notice — the design in the doc vs. the code:** The v0.6 concurrency doc
(04-phase4-websocket-concurrency.md) describes a richer model with per-session
`asyncio.Queue` workers providing FIFO ordering of multiple messages. The
actual implementation in `websocket.py` uses a simpler model:
`_active_tasks[session_id]` stores only the most-recent task; a new arrival
cancels the old one immediately rather than queuing it. This means a second
message does not wait for the first to complete — it cancels it. The doc's
queue model would preserve both messages and process them sequentially; the
implemented model is "last writer wins" at the task level.

---

## Lock acquisition points in practice

The v0.6 doc specifies three lock sites: `core.handle_message` before user
message append, `pipeline.process` before staged injection commit, and
`router/executor._handle_inject`. Inspecting the actual `core.py` shows the
`history_lock` field exists on `SessionState` but the `async with
session.history_lock:` wrapper around `session.messages.append` in
`core.handle_message` is not present in the current code — the field is
defined, the design is documented, but the lock acquisition sites are the next
phase of integration. The primary concurrency protection in the current code is
the `_cancel_active_task` pattern, not the lock.

Connection to universal pattern: the per-session lock is the substrate-forced
answer to "how do you write to shared state from concurrent coroutines." The
asyncio event loop is single-threaded, but cooperative multitasking means a
coroutine can yield control mid-append if the list operation itself is not
atomic — and complex list mutations (extend, splice during compact) are not
atomic across yield points.

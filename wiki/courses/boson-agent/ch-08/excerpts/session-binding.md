---
chapter: ch-08
course: boson-agent
phase: read
excerpt_of: packages/gateway/gateway/session/store.py
created_at: "2026-04-19"
---

# Excerpt: Session Binding — `gateway/session/store.py` + `gateway/schemas/session.py`

One-line description: `SessionStore` is a plain Python dict wrapped in an
API that enforces create-once / fail-if-duplicate semantics; each entry is a
`SessionState` dataclass that carries the entire per-conversation mutable
state tree.

---

## SessionStore CRUD contract (lines 13–53)

```python
# packages/gateway/gateway/session/store.py, lines 13-53

class SessionStore:
    """Manages session lifecycle and isolation.

    Each session is identified by a unique session_id.
    Sessions are isolated — they do not share message lists.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create(self, session_id: str, system_prompt: str = "") -> SessionState:
        """Create and store a new session.

        Raises ValueError if a session with session_id already exists.
        """
        if session_id in self._sessions:
            raise ValueError(f"Session already exists: {session_id!r}")
        session = SessionState(session_id=session_id, system_prompt=system_prompt)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SessionState:
        """Return the session for session_id.

        Raises KeyError if not found.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id!r}")
        return self._sessions[session_id]

    def remove(self, session_id: str) -> None:
        """Remove the session for session_id. No-op if not found."""
        self._sessions.pop(session_id, None)

    def list_active(self) -> list[str]:
        """Return a list of all active session IDs."""
        return list(self._sessions.keys())

    def has(self, session_id: str) -> bool:
        """Return True if session_id exists in the store."""
        return session_id in self._sessions
```

`create` raises `ValueError` on a duplicate `session_id`. This is the
boundary enforcement: the Gateway never silently overwrites an existing
session. The caller (`GatewayCore._get_or_create_session`, core.py line 262)
calls `has()` first and only calls `create()` when the session does not yet
exist, so in practice the ValueError is a defensive contract violation
signal, not a user-facing error.

`get` raises `KeyError` when the session is missing. `remove` is a no-op
on a missing key. These are the two different error semantics: "fetch
something I believe exists" (`get`) versus "clean up something that may or
may not be there" (`remove`).

**Notice:** `SessionStore` holds no lock. Concurrent access from multiple
`asyncio` tasks is safe only because CPython's GIL makes `dict` operations
atomic, and because all Gateway code runs on a single-thread event loop.
The v0.6 `history_lock` added to `SessionState` (see below) is a per-session
`asyncio.Lock` for protecting multi-step message append sequences, not the
store's dict itself.

---

## How `GatewayCore` binds an incoming `session_id` to a session (core.py lines 261–265)

```python
# packages/gateway/gateway/core.py, lines 261-265

def _get_or_create_session(self, session_id: str) -> SessionState:
    """Return existing session or create a new one."""
    if self._sessions.has(session_id):
        return self._sessions.get(session_id)
    return self._sessions.create(session_id, system_prompt=self._system_prompt)
```

This is the binding moment: the first time a `session_id` appears on the
wire, a fresh `SessionState` is created and stored. Every subsequent
`user_message` with the same `session_id` retrieves the same object —
conversation history, stage state, partial buffer, and cancellation flag all
persist across turns because they live on the `SessionState` that was
created at turn 0.

---

## SessionState schema (schemas/session.py, lines 29–53)

```python
# packages/gateway/gateway/schemas/session.py, lines 29-53

@dataclass
class SessionState:
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

`SessionState` is the entire per-conversation world. Every field worth noting:

- `messages` — the canonical conversation history. GatewayCore appends to
  it directly (no defensive copies). The agent loop reads it via
  `SharedHistory` / `ContextManager` adapters.
- `active_stage` — which stage of the conversation state machine the session
  is currently in. `None` means not yet initialized. First turn sets it.
- `context_manager` / `conversation_api` — these are created once and
  persisted on the session (the "Architect/Critic fix" comment). If they
  were recreated each turn, any `pending_reminders` queued by hooks during
  turn N would be lost at turn N+1.
- `cancellation_flag` — a `CancellationFlag` checked at multiple points in
  `handle_message` and `run_agent_loop` to short-circuit streaming when a
  barge-in occurs.
- `history_lock` — an `asyncio.Lock` added in v0.6. Because the reader loop
  now spawns concurrent tasks, two tasks for the same session could both try
  to append to `messages` simultaneously. The lock serializes those appends.

**Notice:** `context_manager` and `conversation_api` are typed as `Any`
rather than their concrete types. This avoids a circular import — `session.py`
is in the `schemas` package, which must not import from `basement.context`
(a peer package). The `Any` type is a deliberate loose coupling at the
schema boundary.

Connection to universal pattern: session binding is step 1 of the pattern —
the server maps an opaque `session_id` string from the wire to a concrete
Python object that carries all state. Every subsequent step of the pattern
operates on that `SessionState` object.

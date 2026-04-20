---
chapter: ch-07
course: boson-agent
phase: read
excerpt_of: gateway/session/store.py — SessionStore
created_at: "2026-04-19"
---

# Excerpt: SessionStore — session lifecycle manager

**Source:** `boson-agent/packages/gateway/gateway/session/store.py`
**Class:** `SessionStore` (lines 13–53)

---

## Full class listing

```python
# boson-agent/packages/gateway/gateway/session/store.py, lines 13-53
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

    def has(self, session_id: str) -> bool:
        """Return True if session_id exists in the store."""
        return session_id in self._sessions
```

---

## Mechanical analysis

### Storage model

`SessionStore._sessions` is a plain Python dict mapping `session_id → SessionState`. The dict is owned by `GatewayCore` as `self._sessions = SessionStore()` — one store per server process, shared across all WebSocket connections.

There is no expiry, LRU eviction, or persistence. Sessions live until:
- `remove()` is explicitly called (currently only called internally in tests; the production path does not remove on disconnect — it calls `on_disconnect` to persist history then leaves the object in memory).
- The server process exits.

### `create` raises on duplicate

```python
if session_id in self._sessions:
    raise ValueError(f"Session already exists: {session_id!r}")
```

This is a hard error, not a no-op. The caller (`_get_or_create_session`) always calls `has()` before `create()`, so in practice this error should never fire in the production path. The guard exists to make bugs in caller code loudly visible rather than silently overwriting a live session.

### `get` raises on missing

`get()` raises `KeyError` if the session is not found. Again, the caller guards with `has()` first. The separation of `has + get` from `get-or-create` is intentional: it forces callers to decide the desired semantics explicitly rather than having the store silently create sessions.

### Isolation guarantee

The docstring states: "Sessions are isolated — they do not share message lists." This isolation is structural: each `SessionState` is constructed with `messages=field(default_factory=list)`, producing a unique list object per session. `SharedHistory` then assigns `ctx._messages = session.messages` — a reference to that session's unique list, never crossing sessions.

**Notice:** there is no session-level lock in `SessionStore` itself. Concurrent access to the dict from multiple asyncio coroutines is safe in CPython due to the GIL for dict reads, but write operations (create/remove) are not protected beyond the GIL. In practice, sessions are created once per WebSocket connection and the asyncio event loop is single-threaded, so this is not a hazard. Per-session concurrency (multiple frames from one connection) is handled by `SessionState.history_lock`.

---

## Relationship to GatewayCore

`GatewayCore` holds `SessionStore` privately:

```python
# boson-agent/packages/gateway/gateway/core.py, lines 45-46
self._sessions = SessionStore()
```

And exposes only two session-touching public methods:
- `get_session(session_id)` — used by the layer pipeline in `__main__.py` to access the session before calling `handle_message`
- `handle_message(session_id, content)` — the main entry point, which calls `_get_or_create_session` internally

This encapsulation means external code (WebSocket handlers, layer pipeline) can read session state but cannot create or destroy sessions except through the gateway's own lifecycle methods.

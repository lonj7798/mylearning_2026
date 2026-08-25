# boson-agent `gateway/server/` — the hand-rolled WebSocket transport (1,404 LOC)

<!-- slug: boson-gateway-server · type: boson · source: packages/gateway/gateway/server/ (websocket.py 734, access.py 374, protocol.py 114, interruption.py 95, history.py 70, __init__.py 17) -->

**Core Insight.** boson's "transport" is not a transport. `gateway/server/` is a *turn-arbitration engine* that happens to speak WebSocket: a 3-field JSON envelope, a per-session dispatch-generation protocol, a silence-timer endpointer, a principal-binding auth layer, and a debug/history projection. A Pipecat transport replaces only the socket plumbing and the envelope — roughly `start()`, `_handle_connection()`, and `protocol.py`. The other ~700 lines have **no Pipecat counterpart at the transport layer** and must be rebuilt above or beside the pipeline.

**Guideline.** Before adopting `FastAPIWebsocketTransport`, enumerate the state boson keeps *per session* outside the socket — there are ten dicts in `GatewayWebSocketServer.__init__`. Any of them you cannot map onto a Pipecat processor is migration work, not migration savings.

## Technical Details

### Wire protocol envelope

- **Exactly three string fields, both directions** (`protocol.py:15-31`):
  ```python
  @dataclass
  class ClientMessage:  session_id: str; type: str; content: str
  @dataclass
  class ServerMessage:  session_id: str; type: str; content: str = ""
  ```
  `serialize_server_message` (L110) emits `{"session_id", "type", "content"}`. There is **no** frame id, timestamp, sequence number, audio field, or binary path.
- `parse_client_message` (L69) rejects non-dict JSON, missing fields, and — per **Issue #43** — non-string fields, which "used to pass parsing and explode later in the reader-loop body (outside the parse try), tearing down the whole connection."
- **`VALID_CLIENT_TYPES`** (L33) = `{"user_message", "partial_transcript", "interrupt", "get_history"}`. Four types. That is the entire client vocabulary.
- **`VALID_SERVER_TYPES`** (L39) = `{"text_delta", "turn_end", "error", "interrupted", "stage_changed", "history"}` — but grepping every `ServerMessage(...)` construction yields only **four** emitted types: `error` (websocket.py:231, 240, 263, 500), `history` (:281), `text_delta` (:471), `turn_end` (:477).
- **`interrupted` and `stage_changed` are declared and never sent.** `agents/test-lina-gateway/client.py:113` has a live `elif data["type"] == "stage_changed":` branch that is dead code. Do not port a `stage_changed` message assuming it exists today — it would be new work.
- **`is_valid_session_id`** (L58): `[A-Za-z0-9][A-Za-z0-9_-]{0,127}`, `SESSION_ID_MAX_LENGTH = 128`. Deliberately narrow because "session IDs cross several persistence and routing boundaries."

### Session lifecycle

- `GatewayWebSocketServer.__init__` (`websocket.py:43-138`) keeps ten per-session maps: `_session_connections: dict[str, set[object]]`, `_session_owner` (Issue #40 — dispatch ownership is *separate* from authorization), `_session_timers`, `_partial_transcripts`, `_partial_transcript_owners`, `_partial_finalize_claims`, `_active_tasks`, `_active_started_at`, `_dispatch_locks`, `_dispatch_generations`.
- Teardown: `_teardown_connection_sessions` (:381) discards the socket, cancels only *its own* silence timer / dispatch, then calls `_on_disconnect(sid)` → `core.on_disconnect` (`core.py:605`) saves history and starts the idle-TTL clock (`session/store.py:19`, `DEFAULT_SESSION_TTL_SECONDS = 1800.0`). Eviction later calls back into `forget_session` (:140), which refuses to run while a socket is still connected ("Ignoring eviction callback for connected session %s").
- **Sessions survive disconnect on purpose** (Issue #29): the UI's same-id auto-reconnect resumes the conversation; boundedness comes from the TTL sweep, not from teardown.

### Turn dispatch — a generation protocol, not a mutex

- `_reserve_session_dispatch` (:515) bumps `_dispatch_generations[sid]` **before any await** — "event-loop tasks cannot interleave until an await, so incrementing the generation here records ordering across connection reader loops."
- `_replace_active_task` (:527) takes the per-session lock, re-checks its generation, `await self._cancel_active_task(session_id)`, re-checks **again**, and only then `asyncio.create_task(self._process_message(...))`.
- `_cancel_active_task` (:584) is the **v0.7.4 hotfix** that *awaits* the cancelled task, so `except CancelledError` has already stashed `session._pending_partial` before the successor reads session state.
- `_process_message` (:442) wraps the whole stream in `async with resolve_history_lock(...)` (**Issue #28**) and sets the `basement.session_context.current_session_id` contextvar, reset on every exit path including cancellation (**Issue #10**).

### Endpointing lives in the server

- `_start_silence_timer` (:616) sleeps `silence_timeout_ms / 1000` (default `2000`) then calls `_finalize_partial` (:661).
- `_finalize_partial` claims the buffered hypothesis into `_partial_finalize_claims`, re-runs `_allows_interruption`, then `await asyncio.shield(task)` so resetting the timer for a newer partial never cancels live generation.

### Access control (`access.py`, 374 L)

- `RequestAccess.process_request` (:75) serves `/play_data` or returns `401` with `WWW-Authenticate: Bearer realm="boson-gateway"`.
- Bearer via `Authorization` header, **or** — because browser WebSocket APIs cannot set headers — via a companion subprotocol `boson-bearer.<token>` (:296). `select_subprotocol` (:43) negotiates `"boson-gateway"`.
- Constructor invariants (`websocket.py:56-66`): `auth_token` ≥ 32 chars; a non-loopback host **requires** a token; enabling `customer_db_path` requires `allowed_origins`.
- `accepted_origins = (None, *allowed_origins)` — a *missing* `Origin` is reserved for native clients; browsers must match the allowlist.
- `/play_data` also accepts a per-process HMAC-SHA256 signed cookie `boson_gateway_play`, `PLAY_COOKIE_MAX_AGE_SECONDS = 300`, `HttpOnly; SameSite=Strict`, minted in `process_response` on a 101.
- **`SessionAccess.authorize(websocket, session_id, operation)`** (:338) binds a session to a *principal* (sha256 of the token, or a per-connection `secrets.token_urlsafe(32)` when tokenless). `get_history` on an unbound session is **refused** (:343-344). A tokenless loopback reconnect may take over a session only when `not session_connections[sid] and session_owner[sid] is None`.

### History endpoint

- `history.py:15` is one function: `serialize_history(get_session, session_id) -> str`, returning `{"messages": [...], "sentiment": {temperature, negative_streak, history}, "debug": {active_stage, active_skill, script_state, script_response, end_signals, turn_count, round_in_stage, escalate_count}}`.
- It **swallows every exception into `"[]"`**. It is a debug projection, not an API.

- **Migration angle:** this is the layer `FastAPIWebsocketTransport` / `WebsocketServerTransport` nominally replaces, and the honest accounting is *lossy*. Pipecat gives boson audio in/out, VAD, `InterruptionFrame` (`frames/frames.py:1142`), serializers, and turn strategies under `src/pipecat/turns/`. Pipecat gives boson **nothing** for: (a) all 374 L of `access.py` — no bearer/subprotocol auth, no origin allowlist, no principal-to-session binding, no signed play cookie; (b) `history.py`'s debug projection and the `history_lock` per-turn serialization; (c) the generation-based cancel-and-replace protocol (`_reserve_session_dispatch` / `_replace_active_task` / `_cancel_session_dispatch`) — `PipelineTask` cancels a bot turn but has no concept of *two sockets racing for the same session*; (d) session identity itself — Pipecat pipelines are per-connection, so boson's reconnect-and-resume (`SessionAccess.authorize` + the 1800 s idle TTL) must live in an outer FastAPI layer that owns the route and hands an accepted `WebSocket` to the transport. Concretely: `protocol.py` becomes a `FrameSerializer` subclass; `_start_silence_timer` / `_finalize_partial` move to `src/pipecat/turns/`; `access.py`, `history.py`, and the dispatch machinery are **kept, not deleted**. Budget ~700 L retained, ~700 L ported.

## Citation

boson-agent (private, Lina TMR), `packages/gateway` version `0.3.0`, commit `0a5e8ced2fd9e631dcbbe8c5f4adb68b89a4fafb` (2026-08-20), read 2026-08-25 at `/Users/jaewon/mywork_2026/Lina_2026/boson-agent-dev/boson-agent/packages/gateway/gateway/server/`. Pipecat comparison points read at `pipecat-ai/pipecat` commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (2026-08-25).

---
chapter: ch-08
course: boson-agent
phase: read
excerpt_of: docs/plan/v0_3/06-phase5-websocket.md + docs/plan/v0_4/07-phase6-websocket-e2e.md
created_at: "2026-04-19"
---

# Excerpt: Protocol Design History — v0.3 Phase 5 + v0.4 Phase 6

One-line description: The planning documents show how the protocol and server
grew from a minimal three-message vocabulary (v0.3) to a concurrent,
multi-feature system (v0.4/v0.6), and why each design decision was made
rather than alternatives.

---

## v0.3 Phase 5: the original minimal design (06-phase5-websocket.md)

```markdown
# docs/plan/v0_3/06-phase5-websocket.md, lines 57-128 (GREEN Implementation)

### websocket.py (~120 LOC)

class GatewayWebSocketServer:
    """WebSocket server that routes client messages to Gateway core.

    Per-connection flow:
    1. Client connects
    2. Receives JSON messages (ClientMessage)
    3. Routes to GatewayCore.handle_message()
    4. Streams ServerMessages back to client
    5. On disconnect: session preserved
    """

    def __init__(self, core, host: str = "localhost", port: int = 8765):
        self._core = core
        ...

    async def _handle_connection(self, websocket):
        """Handle a single client connection."""
        ...

    async def _send_response(self, websocket, session_id: str, text: str):
        """Stream text as text_delta messages, then turn_end."""
```

The v0.3 design was radically simple: the server held a direct reference to
`core` (a `GatewayCore` instance), not a `message_handler` callable. The
three message types were `user_message`, `text_delta`, `turn_end` — plus
`error`. There was no concurrency, no cancellation, no partial transcript
handling. The reader loop blocked on `handle_message()` completion before
accepting the next frame.

The quality gate (lines 132–138) is instructive about what the designers
considered essential:

```markdown
# docs/plan/v0_3/06-phase5-websocket.md, lines 132-138

- [ ] JSON protocol: send ClientMessage, receive ServerMessage
- [ ] Multiple concurrent sessions via session_id
- [ ] Invalid messages get error responses
- [ ] Disconnect preserves session
- [ ] No file exceeds 120 LOC
```

"Multiple concurrent sessions via session_id" was required from day one —
the payload-embedded session ID is a v0.3 decision, not a later addition.
The 120 LOC cap was a hard constraint that forced clean separation into
`protocol.py` (codec) and `websocket.py` (server) rather than one fat file.

**Notice:** The v0.3 `websocket.py` took `core` as a constructor argument
(a `GatewayCore` instance). By v0.6, this changed to `message_handler: Callable[[str, str], AsyncIterator[str]]` — a plain callable. This decoupling
means the server has zero knowledge of `GatewayCore`; it can be tested with
any async generator function. The architectural direction was toward stronger
separation of concerns.

---

## v0.4 Phase 6: the silence timer design rationale (07-phase6-websocket-e2e.md)

```markdown
# docs/plan/v0_4/07-phase6-websocket-e2e.md, lines 68-91

## Silence Timer Design

**Decision:** asyncio.Task per session. Created when first `partial_transcript`
arrives. Reset (cancelled + recreated) on each subsequent partial. When timer
fires (no new input within `silence_timeout_ms`), treats current partial as
final and sends to agent.

T=0ms:    partial arrives "well..."
          → Store in partial_buffer
          → Create silence_timer task (2000ms)

T=300ms:  partial arrives "well... hmm..."
          → Cancel existing timer
          → Replace in history
          → Create new silence_timer task (2000ms)

T=600ms:  partial arrives "well... I think..."
          → Cancel existing timer
          → Replace in history
          → Create new silence_timer task (2000ms)

T=2600ms: Timer fires (no new input for 2000ms)
          → Finalize: treat current partial as user_message
          → Send to core.handle_message() for agent processing

**Alternative (rejected):** Polling loop — wastes CPU, harder to cancel,
less precise timing.
```

The silence timer is the first place in the codebase where `asyncio.Task`
is used as a mechanism for time-based state transitions rather than just
for concurrency. The rejection of the polling loop alternative
(`while True: await asyncio.sleep(0.05); check_timeout()`) is worth
internalizing: polling requires an explicit cancellation check on every
iteration, while the task-per-session approach makes cancellation an
intrinsic property of `asyncio.Task.cancel()`.

---

## v0.4 new message types and backward compatibility

```markdown
# docs/plan/v0_4/07-phase6-websocket-e2e.md, lines 19-64

# Client -> Gateway (NEW)
{ "session_id": "s1", "type": "partial_transcript", "content": "well..." }

# Gateway -> Client (NEW)
{ "session_id": "s1", "type": "interrupted", "content": "I can help you wi" }
{ "session_id": "s1", "type": "stage_changed", "content": "explore" }
```

The v0.4 additions follow a strict backward-compatibility rule: new *client*
types (`partial_transcript`) are additive — a v0.3 client that never sends
`partial_transcript` sees no behavior change. New *server* types
(`interrupted`, `stage_changed`) are also additive — a v0.3 client that
ignores unknown message types will simply skip them.

The E2E test `test_v03_client_unchanged` (doc line 320–336) makes this
explicit: it asserts that a client sending only `user_message` and receiving
only `text_delta` + `turn_end` continues to work identically after all v0.4
server-side changes. This is the gateway's backward compatibility contract.

---

## The decoupling arc: v0.3 → v0.6

| Version | Constructor arg | Coupling |
|---------|----------------|---------|
| v0.3 | `core: GatewayCore` | Tight — server knows about GatewayCore |
| v0.4 | `message_handler: Callable` | Loose — server knows only the callable protocol |
| v0.6 | `message_handler: Callable[[str, str], AsyncIterator[str]]` | Fully typed — server is a generic streaming proxy |

The trajectory is consistent: each version moved the server further from
application logic and closer to a pure network adapter. By v0.6,
`GatewayWebSocketServer` knows nothing about rules, stages, sessions, or
tools — it only knows how to accept frames, call a function, and stream
the results back.

Connection to universal pattern: the design history shows that the
"universal pattern" in §1 of the chapter was not designed top-down — it
emerged incrementally as complexity was added. The minimal v0.3 loop is
already the correct shape; v0.4 and v0.6 add concurrency and features
without changing the fundamental structure.

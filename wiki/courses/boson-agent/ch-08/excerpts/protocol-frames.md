---
chapter: ch-08
course: boson-agent
phase: read
excerpt_of: packages/gateway/gateway/server/protocol.py
created_at: "2026-04-19"
---

# Excerpt: Protocol Frames — `gateway/server/protocol.py`

One-line description: A pure, sealed module that defines the complete wire
vocabulary between browser and gateway — two dataclasses, two sentinel sets,
one parser, one serializer. No I/O, no state.

---

## Dataclasses and type sets (lines 14–33)

```python
# packages/gateway/gateway/server/protocol.py, lines 14-33

@dataclass
class ClientMessage:
    """Message received from a WebSocket client."""
    session_id: str
    type: str
    content: str


@dataclass
class ServerMessage:
    """Message sent to a WebSocket client."""
    session_id: str
    type: str
    content: str = ""


VALID_CLIENT_TYPES = {"user_message", "partial_transcript", "get_history"}
VALID_SERVER_TYPES = {"text_delta", "turn_end", "error", "interrupted",
                      "stage_changed", "history"}
```

Both dataclasses share the same three fields: `session_id`, `type`,
`content`. This symmetry is intentional — the wire schema is identical in
both directions, which means the same JSON parser and serializer can handle
both sides with minimal branching. `ServerMessage.content` defaults to `""`
so that `turn_end` frames (which carry no payload) do not require a special
construction path — callers simply omit `content`.

`VALID_CLIENT_TYPES` and `VALID_SERVER_TYPES` are frozenset-like sentinels
checked by the WebSocket server's dispatch logic. Adding a new message type
to the protocol requires only updating these sets and the corresponding
handler branch — the dataclasses themselves never change.

**Notice:** There is no `Enum` or `Literal` type — just plain `str` fields
validated at parse time against the sets. This is a deliberate tradeoff:
dataclasses keep the wire layer dependency-free and trivially serializable
with `json.dumps`. Using `Enum` would require a custom encoder. The sentinel
sets provide the same exhaustiveness guarantee at the cost of slightly
weaker static typing.

---

## Parser (lines 41–60)

```python
# packages/gateway/gateway/server/protocol.py, lines 41-60

def parse_client_message(raw: str) -> ClientMessage:
    """Parse a raw JSON string into a ClientMessage.

    Raises:
        ValueError: If JSON is invalid or required fields are missing.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    missing = [f for f in ("session_id", "type", "content") if f not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return ClientMessage(
        session_id=data["session_id"],
        type=data["type"],
        content=data["content"],
    )
```

The parser does two things: JSON decode and field-presence check. It does
*not* validate `type` against `VALID_CLIENT_TYPES` — that check is
intentionally deferred to the WebSocket server's dispatch logic (websocket.py
line 122). This separation means `parse_client_message` is pure and
testable in isolation regardless of which types are "currently valid" —
the validity set can evolve without touching the parser.

**Notice:** The missing-field check at line 52 collects *all* missing fields
before raising, so the error message tells the client exactly which fields
are absent rather than failing on the first missing one. Small ergonomic
choice, large debugging win for integration clients.

---

## Serializer (lines 63–67)

```python
# packages/gateway/gateway/server/protocol.py, lines 63-67

def serialize_server_message(msg: ServerMessage) -> str:
    """Serialize a ServerMessage to a JSON string."""
    return json.dumps(
        {"session_id": msg.session_id, "type": msg.type, "content": msg.content}
    )
```

The serializer is three lines. It explicitly constructs the dict rather than
using `dataclasses.asdict()` — this guarantees field ordering and avoids any
future surprise if the dataclass gains internal fields that should not appear
on the wire. `json.dumps` with no `separators` argument produces
compact-but-readable JSON (with spaces after `:` and `,`), which is fine for
WebSocket text frames where framing overhead dominates wire cost.

Connection to universal pattern: `parse_client_message` is step 2 of the
pattern (decode inbound frame); `serialize_server_message` is step 5
(encode outbound frame). The protocol module is the codec layer that sits
between the raw WebSocket bytes and the typed Python objects used by all
other layers.

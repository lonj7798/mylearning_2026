---
chapter: ch-08
course: boson-agent
phase: read
excerpt_of: agents/demo-gateway/client.py + agents/test-lina-gateway/client.py
created_at: "2026-04-19"
---

# Excerpt: Client Usage Pattern — `demo-gateway/client.py` + `test-lina-gateway/client.py`

One-line description: Both clients are minimal Python scripts that expose
the protocol contract from the consumer side — they show exactly what a
correct WebSocket client must do to interoperate with the gateway, without
any framework magic.

---

## demo-gateway/client.py — the canonical send/receive loop (lines 57–84)

```python
# agents/demo-gateway/client.py, lines 57-84

# Send message
await ws.send(json.dumps({
    "session_id": session_id,
    "type": "user_message",
    "content": user_input,
}))

# Receive streaming response
print("Agent: ", end="", flush=True)
try:
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
        except asyncio.TimeoutError:
            print("\n[Timeout — no response]")
            break
        data = json.loads(raw)
        if data["type"] == "text_delta":
            print(data["content"], end="", flush=True)   # (A) stream to terminal
        elif data["type"] == "turn_end":
            print()  # newline
            break                                          # (B) stop condition
        elif data["type"] == "error":
            print(f"\n[Error: {data['content']}]")
            break
except (asyncio.CancelledError, KeyboardInterrupt):
    print("\n[Interrupted]")
    continue
```

The send side is a single `await ws.send(json.dumps({...}))` — the entire
user_message is one JSON frame. The receive side is an unbounded `while
True` loop that dispatches on `data["type"]`. This is the mirror image of
the server's `_process_message` loop: where the server's loop terminates by
sending `turn_end`, the client's loop terminates by *receiving* `turn_end`.

- **(A)** `end="", flush=True` on the `print` call is what makes streaming
  feel real-time in a terminal: each `text_delta` chunk is printed
  immediately without a trailing newline.
- **(B)** `break` on `turn_end` — this is the only valid termination
  condition for a normal turn. The client never tries to guess when the
  stream is "done" by timing out; it waits for the explicit protocol signal.

**Notice:** `session_id` is generated once per process run at line 23:
`session_id = str(uuid.uuid4())[:8]`. It is then stamped on every outgoing
frame. The connection itself (`async with websockets.connect(uri) as ws:`)
is held open for the entire interactive session — messages are multiplexed
over a single long-lived TCP connection, not opened/closed per message.

---

## test-lina-gateway/client.py — handling `stage_changed` frames (lines 82–104)

```python
# agents/test-lina-gateway/client.py, lines 82-104

async def _receive_response(ws):
    """Receive and print streaming response."""
    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                print("\n[Timeout — no response]")
                break
            data = json.loads(raw)
            if data["type"] == "text_delta":
                print(data["content"], end="", flush=True)
            elif data["type"] == "turn_end":
                print()
                break
            elif data["type"] == "stage_changed":
                stage = data.get("content", "")
                print(f"\n  [Stage → {stage}]", end="", flush=True)   # (C)
            elif data["type"] == "error":
                print(f"\n[Error: {data['content']}]")
                break
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\n[Interrupted]")
```

The Lina client extends the receive loop with one additional `elif` branch
for `stage_changed` at **(C)**. This is the client-side display logic for
the stage machine (covered in ch-07). The `stage_changed` frame carries the
name of the new stage as `content`; the client renders it inline as
`[Stage → explore]` without breaking the stream — it does not `break` out of
the loop, so subsequent `text_delta` frames continue to render after the
stage notification.

**Notice:** The Lina client also sends an initial synthetic message at
startup:

```python
# agents/test-lina-gateway/client.py, lines 44-47

await ws.send(json.dumps({
    "session_id": session_id,
    "type": "user_message",
    "content": "[call connected]",
}))
```

This is an application-level convention, not a protocol primitive. The
gateway has no concept of "call connected" — it processes `[call connected]`
as an ordinary user_message and lets the rule engine and stage machine decide
what to do with it (in Lina's case, the first-turn rules fire the agent
greeting). The protocol itself is connection-agnostic.

---

## What both clients reveal about the protocol contract

1. Session identity lives in the payload, not the URL. Both clients use
   `ws://localhost:8765` as the URI and stamp `session_id` on every
   frame — enabling future multi-session multiplexing over a single socket
   without server-side changes.

2. The client's receive loop must handle all `VALID_SERVER_TYPES` gracefully
   (or at least not crash on unknown types). New server message types
   (`interrupted`, `history`) are backwards-compatible: an older client that
   does not handle them will simply skip those frames and continue waiting.

3. The `turn_end` frame is the only guaranteed termination signal. Clients
   must not infer completion from a timeout or a lack of frames.

Connection to universal pattern: the client pattern is the inversion of
steps 2 and 5 — it sends step 2's inbound frame format and consumes step
5's outbound frame format, making the protocol contract observable from
both sides.

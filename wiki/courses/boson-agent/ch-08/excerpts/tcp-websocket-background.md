---
chapter: ch-08
course: boson-agent
phase: read
kind: background
title: "TCP and WebSocket — Protocol Background"
sources_cited:
  - RFC 793 (TCP)
  - RFC 6455 (WebSocket)
  - Python asyncio documentation
---

# TCP and WebSocket — Protocol Background

Supplementary reference for [[ch-08]]. If you already know TCP/WebSocket fundamentals, skip to the main `read.md`. Written as a grounding layer because boson-agent's Gateway sits directly on top of these protocols.

---

## 1. The protocol stack (what goes where)

```
┌─────────────────────────────────────────────────┐
│ Application layer                                │
│   HTTP, WebSocket, SSH, IRC, your-custom-protocol│  ← boson-agent lives here
├─────────────────────────────────────────────────┤
│ Transport layer                                  │
│   TCP (reliable stream), UDP (fire-and-forget)   │  ← "connection" concept
├─────────────────────────────────────────────────┤
│ Network layer                                    │
│   IP (v4/v6) — stateless packet routing          │
├─────────────────────────────────────────────────┤
│ Link layer                                       │
│   Ethernet, WiFi — physical signals              │
└─────────────────────────────────────────────────┘
```

Each layer adds a header to the payload below it. When Gateway does `await websocket.send("hello")`:
- WebSocket wraps `"hello"` in a frame (opcode + length + mask + payload)
- TCP wraps the frame in a segment (source port, dest port, sequence number, etc.)
- IP wraps the segment in a packet (source IP, dest IP, TTL)
- Link wraps the packet in an Ethernet/WiFi frame with MAC addresses

The reverse happens at the other end. Each layer only talks to the layer directly above and below — layered concern separation at the protocol level.

---

## 2. TCP — what a "connection" actually is

### 2.1 IP alone is not enough

IP gives you "deliver this packet to address X." Nothing more:
- No delivery guarantee (packet may be dropped by a router)
- No ordering (packet B may arrive before packet A)
- No deduplication (same packet may arrive twice due to retransmission)
- No flow control (sender may overwhelm receiver)

For "send a message and know it arrived, in order, exactly once", you need a protocol on top. TCP is that protocol.

### 2.2 The 3-way handshake (connection establishment)

```
Client                          Server
  │                               │
  ├────── SYN (seq=X) ──────────→│         "I want to talk. My seq starts at X."
  │                               │
  │←────  SYN-ACK (seq=Y, ack=X+1)┤         "OK, my seq starts at Y. I got up to X."
  │                               │
  ├────── ACK (ack=Y+1) ─────────→│         "Got it. Connection open."
  │                               │
  │═══════ connection live ═══════│
```

After these three packets, **both sides agree they are in a connected state**. Each side has allocated buffer memory and initialized sequence numbers. Data transfer begins.

### 2.3 What the "connection" IS, physically

A TCP connection is **shared state** between two endpoints. The tuple that identifies it:

```
(source_ip, source_port, dest_ip, dest_port)
```

Both sides maintain:
- **Sequence numbers** — which byte we've sent / received next
- **Send/receive buffers** — in-flight data waiting to be ACKed / delivered to app
- **Window size** — how much more data we can send without ACK (flow control)
- **Connection state** — ESTABLISHED, FIN_WAIT, TIME_WAIT, etc.

When you call `socket.close()`, both sides exchange FIN packets to tear down these buffers. If one side crashes without closing, the other side eventually times out (TCP keepalive) or gets RST on next write.

### 2.4 Four reliability guarantees TCP provides

1. **Delivery** — if a packet is lost, TCP retransmits until ACKed or timeout
2. **Ordering** — if packet 5 arrives before packet 4, TCP buffers 5 until 4 arrives, then delivers in order
3. **Deduplication** — retransmitted packets that arrived twice get dropped
4. **Flow control** — receiver advertises its window size; sender never exceeds it

Application just sees "a stream of bytes arrives in order". All the messy packet management is hidden.

### 2.5 The "stream" abstraction

TCP doesn't preserve message boundaries. If you call `send("hello")` and `send("world")`, the receiver might get `recv() → "hello"` then `recv() → "world"`, OR `recv() → "helloworld"` in one call, OR `recv() → "hel"` then `recv() → "loworld"`. It's a byte stream, not a message stream.

**This is why WebSocket exists.** HTTP used TCP and had to add its own `Content-Length` header to know where one request ends and the next begins. WebSocket adds proper message framing on top of TCP's stream.

---

## 3. WebSocket — TCP with framing and two-way messages

### 3.1 Why WebSocket exists

HTTP is request-response: client asks, server replies, done. For a chat app or a streaming LLM response, you need **server-initiated messages** and **long-lived connections**. Before WebSocket (2011), people hacked this with:
- Long polling (keep HTTP request open, server replies when data ready — but still one-shot per request)
- Server-Sent Events (one-directional server → client only)
- Comet (various long-poll tricks)

WebSocket replaced all of these. It's a **proper two-way channel over a single long-lived TCP connection**.

### 3.2 The WebSocket handshake

A WebSocket connection starts as an HTTP request, then "upgrades":

```
Client                                     Server
  │                                          │
  │  (TCP 3-way handshake completes)         │
  │                                          │
  ├─ HTTP GET /ws HTTP/1.1                 ─→│
  │  Upgrade: websocket                      │
  │  Connection: Upgrade                     │
  │  Sec-WebSocket-Key: <base64 random>      │
  │  Sec-WebSocket-Version: 13               │
  │                                          │
  │←─ HTTP/1.1 101 Switching Protocols     ──┤
  │   Upgrade: websocket                     │
  │   Connection: Upgrade                    │
  │   Sec-WebSocket-Accept: <hash of key>    │
  │                                          │
  │══════ WebSocket frames flow both ways ═══│
```

After the 101 response, the same TCP connection stops speaking HTTP and starts speaking WebSocket frames. It is the **same** TCP connection — no new handshake, no new socket. The protocol on top changed.

### 3.3 WebSocket frames

Each message is wrapped in a frame:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-------+-+-------------+-------------------------------+
|F|R|R|R| opcode|M| Payload len |    Extended payload length    |
|I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
|N|V|V|V|       |S|             |   (if payload len==126/127)   |
| |1|2|3|       |K|             |                               |
+-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
|     Extended payload length continued, if payload len == 127  |
+ - - - - - - - - - - - - - - - +-------------------------------+
|                               |Masking-key, if MASK set to 1  |
+-------------------------------+-------------------------------+
|                     Payload Data                              |
+---------------------------------------------------------------+
```

Key opcodes:
- `0x1` text frame (UTF-8 string)
- `0x2` binary frame
- `0x8` close frame
- `0x9` ping
- `0xA` pong

Each WebSocket frame restores **message boundaries** that TCP's stream lost. When boson-agent sends `{"type": "text_delta", "text": "hi"}` as a JSON string, it goes in one text frame — receiver reads exactly one frame, gets exactly one JSON string.

---

## 4. How Python `asyncio` maps to TCP

`asyncio` abstracts the OS socket API behind coroutines:

```python
# TCP server side (simplified)
async def handle_client(reader, writer):
    data = await reader.read(1024)     # ← yields until bytes arrive on socket
    writer.write(b"response")
    await writer.drain()               # ← yields until send buffer has space

server = await asyncio.start_server(handle_client, "0.0.0.0", 8000)
```

- `await reader.read(...)` = "suspend this coroutine until the OS tells us bytes arrived on this TCP socket"
- `await writer.drain()` = "suspend until the OS accepts more bytes into the send buffer"
- Each connection is **one coroutine** that lives as long as the TCP connection is open

**WebSocket libraries (`websockets`, `aiohttp`, `starlette`) build on top:**
- They do the HTTP-to-WebSocket upgrade handshake
- They parse/build WebSocket frames
- They expose `await ws.send(...)` and `await ws.recv()` APIs that look like TCP reader/writer but operate at frame boundaries

So when boson-agent's Gateway accepts a WebSocket connection, underneath:
1. A TCP socket was opened by the OS when the client connected
2. The HTTP upgrade handshake ran once
3. A coroutine was spawned to handle this one connection
4. The coroutine now loops `await ws.recv()` → dispatch to `handle_message` → stream events back → `await ws.send(...)`

When the client disconnects (closes browser tab, network drops, explicit close), the TCP connection is torn down, `ws.recv()` raises `ConnectionClosed`, and the coroutine exits.

---

## 5. What this means for ch-08 concepts

Mapping to boson-agent terms:

| boson-agent term | Protocol reality |
|---|---|
| "WebSocket connection" | One TCP connection + WebSocket framing on top |
| "Session" (from ch-07) | Conceptual grouping of turns — one per connection in boson-agent |
| `session_id` | Identifier Gateway uses to route a connection to the right state |
| "Reconnect" (ch-09) | New TCP handshake + new WebSocket upgrade + lookup of old session_id |
| "Disconnect" | TCP FIN/RST or timeout → WebSocket close frame → coroutine exits |
| `text_delta` wire message | One WebSocket text frame containing JSON |
| "Barge-in" (ch-09) | Client-initiated message frame arriving while server is still sending frames — NOT a new connection |

### Key insight — one TCP connection = one session lifespan (in boson-agent)

boson-agent binds `session_id` to the WebSocket connection. If the TCP connection drops:
- The session lives in `SessionStore` (in-memory dict on the Gateway)
- But no connection is serving it → no input can arrive, no output can go out
- On reconnect: client presents the old session_id, Gateway looks it up in the store, attaches the new WebSocket to the old SessionState

This is why [[ch-09]] has to talk about reconnect and session rebind — the TCP connection's lifetime and the session's lifetime are **not** the same, even though they usually coincide.

### Why this matters for code reading

When you read `packages/gateway/gateway/server/websocket.py` in ch-08, you'll see:
- `async def _handle_connection(websocket)` — this is the per-connection coroutine, spawned on each new TCP connection after handshake
- `async for message in websocket` — this iterates incoming frames, each `await`ing on the TCP socket
- `await websocket.send(frame)` — this `await` can block on the OS send buffer if the client is slow to receive (TCP flow control bubbling up)

All the async/await points in that file map directly to actual TCP socket operations underneath. The async generator chain from ch-07 now has its endpoint: the final `await websocket.send(...)` hands bytes to the OS kernel, which hands them to the TCP stack, which sends them in a packet.

---

## 6. One-sentence summary

**TCP** is a reliable ordered byte-stream between two endpoints, identified by a `(src_ip, src_port, dst_ip, dst_port)` tuple and maintained as shared state on both sides. **WebSocket** is a message-framing protocol layered on top of TCP, with an HTTP-based upgrade handshake, enabling long-lived bidirectional communication. boson-agent's Gateway uses one WebSocket (= one TCP connection) per session, where the session state outlives any single connection and can be rebound on reconnect.

---
chapter: ch-08
course: boson-agent
phase: read
excerpt_of: packages/gateway/gateway/server/websocket.py
created_at: "2026-04-19"
---

# Excerpt: WebSocket Server — `gateway/server/websocket.py`

One-line description: `GatewayWebSocketServer` is the network boundary of the gateway — it accepts raw WebSocket frames, parses them into typed `ClientMessage` objects, and fans out to per-session `asyncio.Task` workers that stream `ServerMessage` frames back to the wire.

---

## Startup and endpoint wiring (lines 60–70)

```python
# packages/gateway/gateway/server/websocket.py, lines 60-70

async def start(self) -> None:
    """Start the server and run forever (blocking)."""
    async with websockets.serve(
        self._handle_connection, self._host, self._port
    ) as server:
        self._server = server
        logger.info(
            "Gateway WebSocket server listening on %s:%d",
            self._host, self._port,
        )
        await asyncio.get_running_loop().create_future()
```

`websockets.serve` registers `_handle_connection` as the coroutine that is
spawned for every new TCP connection. The final
`await asyncio.get_running_loop().create_future()` creates a Future that
never resolves, which parks the coroutine forever — this is the idiomatic
way to make an `async with` block block indefinitely without a
`while True: await asyncio.sleep(…)` polling loop.

**Notice:** `start_background()` (lines 72–83) skips the
`create_future()` trick and instead uses the plain `await
websockets.serve(...)` API (which returns without blocking), then reads the
OS-assigned port back from `server.sockets[0].getsockname()[1]`. This is
what the E2E tests use — they need to bind, then interact, then
`server.stop()`. The two modes share identical handler code but differ only
in how the event loop is kept alive.

Connection to universal pattern: this is step 1 of the pattern — the server
binds once and multiplexes all subsequent per-connection logic through
`_handle_connection`.

---

## `_handle_connection`: reader loop + task dispatch (lines 97–192)

```python
# packages/gateway/gateway/server/websocket.py, lines 97-178

async def _handle_connection(
    self, websocket: websockets.server.ServerConnection
) -> None:
    remote = websocket.remote_address
    logger.debug("Client connected: %s", remote)
    session_ids: set[str] = set()

    try:
        async for raw in websocket:               # (A) suspend here per frame
            try:
                msg = parse_client_message(str(raw))
            except ValueError as exc:
                error_payload = serialize_server_message(
                    ServerMessage(session_id="", type="error", content=str(exc))
                )
                await websocket.send(error_payload)
                continue

            session_ids.add(msg.session_id)

            if msg.type not in VALID_CLIENT_TYPES:
                ...
                continue

            # --- get_history branch (lines 134-160) omitted for brevity ---

            if msg.type == "partial_transcript":   # (B) voice/STT path
                async for _ in self._message_handler(msg.session_id, msg.content):
                    pass
                self._start_silence_timer(msg.session_id, websocket)
                continue

            self._cancel_silence_timer(msg.session_id)

            # v0.6: Cancel in-progress handler for this session
            self._cancel_active_task(msg.session_id)  # (C) barge-in

            # v0.6: Spawn handler as task (non-blocking reader loop)
            task = asyncio.create_task(           # (D) fan-out
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

Walk through the key points:

- **(A)** `async for raw in websocket` is the WebSocket receive loop. Each
  iteration suspends until a frame arrives on the wire. Because this is a
  plain `await` under the covers, the event loop can interleave other tasks
  (handler tasks, timers) while waiting. The reader is **never blocked** by
  handler completion — that is the critical v0.6 change.

- **(B)** `partial_transcript` is the voice/STT path added in v0.4. Partial
  frames are forwarded to the message handler (which buffers them in
  `session.partial_buffer`) and then a silence timer is armed. No response
  is streamed for partial frames — the `async for _ in …: pass` drains the
  generator without emitting anything.

- **(C)** Before spawning a new handler, the existing handler task for that
  `session_id` is cancelled. This is the barge-in mechanism: if the LLM is
  mid-stream and the user sends a new `user_message`, the old
  `_process_message` coroutine is cancelled at its next `await` point and a
  fresh one starts.

- **(D)** `asyncio.create_task(...)` schedules `_process_message` as an
  independent coroutine on the event loop. The reader loop immediately
  returns to its `async for` to listen for the next frame — it does not
  `await` the task.

**Notice:** A single `websocket` object can carry messages for *multiple*
`session_id` values — see `session_ids: set[str] = set()`. The protocol
deliberately puts `session_id` inside the JSON payload, not in the URL path.
This means one long-lived TCP connection can multiplex conversations —
useful for voice clients that hold a single socket for an entire call and
switch logical sessions without reconnecting.

---

## `_process_message`: the token-streaming path (lines 194–232)

```python
# packages/gateway/gateway/server/websocket.py, lines 194-232

async def _process_message(
    self,
    websocket: websockets.server.ServerConnection,
    session_id: str,
    content: str,
) -> None:
    try:
        async for chunk in self._message_handler(session_id, content):  # (E)
            delta = serialize_server_message(
                ServerMessage(
                    session_id=session_id, type="text_delta", content=chunk
                )
            )
            await websocket.send(delta)                                  # (F)

        turn_end = serialize_server_message(
            ServerMessage(session_id=session_id, type="turn_end")
        )
        await websocket.send(turn_end)                                   # (G)
    except asyncio.CancelledError:                                       # (H)
        logger.info("Handler cancelled for session %s (new message arrived)", session_id)
    except Exception as exc:
        logger.exception("message_handler raised: %s", exc)
        try:
            error_payload = serialize_server_message(
                ServerMessage(
                    session_id=session_id, type="error", content=str(exc)
                )
            )
            await websocket.send(error_payload)
        except Exception:
            pass  # websocket may be closed
    finally:
        self._active_tasks.pop(session_id, None)
```

- **(E)** `async for chunk in self._message_handler(session_id, content)` is
  the central await chain. `_message_handler` is `GatewayCore.handle_message`
  — an async generator. Each `chunk` is a text string that was yielded from
  deep inside `run_agent_loop` as a `TextDelta.text` value. The `async for`
  suspends at every `yield` point in the generator chain.

- **(F)** For each chunk, the coroutine constructs a `text_delta` JSON frame
  and `await websocket.send(delta)`. This is the final hop: the string token
  that came from the LLM provider is now on the wire.

- **(G)** After the generator is exhausted (turn complete), a `turn_end`
  frame is sent. The client's receive loop uses this as the stop signal.

- **(H)** `asyncio.CancelledError` is caught and *not re-raised*. When a
  barge-in cancels this task (step C above), the cancellation is swallowed
  here, preventing it from propagating to the reader loop. This is the
  "fail-open" contract for interruption: the gateway never sends a partial
  error frame — the stream simply stops.

**Notice:** Error handling at line 220 wraps every unhandled exception and
attempts to send an `error` frame before giving up. But the outermost
`except Exception: pass` at line 229 silently drops send failures — because
by the time an unhandled exception propagates, the WebSocket may already be
closed. This is intentional fail-open semantics: never let a send failure in
error-reporting itself crash the server.

Connection to universal pattern: `_process_message` is the implementation
of steps 4 and 5 of the pattern — consume the async generator and translate
each yielded string into a wire frame.

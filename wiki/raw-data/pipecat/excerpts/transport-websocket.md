# The WebSocket transport — a socket plus a serializer slot, and nothing else

<!-- slug: transport-websocket · type: source · source: src/pipecat/transports/websocket/{fastapi.py,server.py,client.py} -->

**Core Insight.** Pipecat's WebSocket transport is deliberately protocol-agnostic: it owns the socket, the audio clock, and connect/disconnect events — and delegates *every byte on the wire* to a pluggable `FrameSerializer`. That single `serializer` field is the seam that turns one transport into a Twilio bot, a Telnyx bot, or a custom-protocol bot.

**Guideline.** Use `FastAPIWebsocketTransport` for anything real (you keep your own ASGI app, auth, and routing); use `SingleClientWebsocketServerTransport` only for local dev. Everything above the frame layer — sessions, auth, history, reconnection, multi-tenancy — you still write yourself.

## Technical Details

- Three modules, all under `src/pipecat/transports/websocket/`: `fastapi.py` (707 L), `server.py` (716 L), `client.py` (559 L). `__init__.py` is empty (0 L) — import from the concrete module.

**`FastAPIWebsocketTransport`** (`fastapi.py:611`) — the production path.
- `__init__(self, websocket: WebSocket, params: FastAPIWebsocketParams, input_name=None, output_name=None)`. It receives an **already-accepted** FastAPI `WebSocket`. You own the route, the handshake, and any auth before this line.
- Raises `ValueError(f"WebSocket connection rejected: origin '{origin}' not allowed")` at construction when `params.allowed_origins` is set and the Origin header fails `is_origin_allowed` (line 649-652). The docstring is explicit: *"The caller is responsible for closing the WebSocket in that case."*
- `FastAPIWebsocketParams(TransportParams)` (line 59) adds exactly six fields:
  ```python
  add_wav_header: bool = False
  serializer: FrameSerializer | None = None
  session_timeout: int | None = None
  fixed_audio_packet_size: int | None = None
  allowed_origins: list[str] = Field(default_factory=default_allowed_origins)
  ws_close_timeout: float = 0.5     # _WS_CLOSE_TIMEOUT_DEFAULT
  ```
  `ws_close_timeout` exists specifically so "a dead or half-closed peer (e.g. a telephony call already torn down on the provider's side)" cannot stall pipeline shutdown.
- Only three events: `on_client_connected`, `on_client_disconnected`, `on_session_timeout`.

**The serializer hook — both directions.**
- Setup: `await self._params.serializer.setup(setup)` is called from *both* `FastAPIWebsocketInputTransport.setup()` and `...OutputTransport.setup()` (`fastapi.py:461`; `server.py:152`, `server.py:382`). The serializer therefore learns `setup.audio_in_sample_rate` — that is how telephony serializers know the pipeline rate to resample to.
- Inbound (`fastapi.py:372-389`), verbatim shape:
  ```python
  async for message in self._client.receive():
      if not self._params.serializer: continue
      frame = await self._params.serializer.deserialize(message)
      if not frame: continue
      if isinstance(frame, InputAudioRawFrame):        await self.push_audio_frame(frame)
      elif isinstance(frame, InputTransportMessageFrame):
          await self.broadcast_frame(InputTransportMessageFrame, message=frame.message)
      else:                                            await self.push_frame(frame)
  ```
  **No serializer ⇒ every inbound message is silently dropped.** Same for outbound: `_write_frame` returns `False` early if `self._params.serializer` is None (`fastapi.py:568`).
- Outbound: `_write_frame()` calls `serializer.serialize(frame)` and sends the payload. With `fixed_audio_packet_size` set, binary payloads are buffered in `self._audio_send_buffer` and emitted in exact-size packets, remainder preserved. The in-code example: *"e.g. 640 for 20ms @ 16kHz PCM16 mono"*.

**The audio clock — the non-obvious part.**
- `setup()` computes `self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2` (`fastapi.py:456`, `server.py:379`). With defaults (`audio_out_sample_rate=24000`, `audio_out_10ms_chunks=4`) that is a 40 ms chunk and a 20 ms send interval.
- `write_audio_frame()` sends, then `await self._write_audio_sleep()`. The comment states the reason plainly: *"since this is just a network connection we would be sending it to quickly. Instead, we want to block to emulate an audio device."* A WebSocket has no playback clock, so the transport fabricates one with `time.monotonic()` bookkeeping (`server.py:506-515`).
- Interruptions reset it: `process_frame` on `InterruptionFrame` writes the frame through the serializer and sets `self._next_send_time = 0` (`server.py:435-437`).

**`SingleClientWebsocketServerTransport`** (`server.py:518`) — dev only.
- `__init__(self, params: SingleClientWebsocketServerParams, host: str = "localhost", port: int = 8765, ...)`. It runs its own `websockets.asyncio.server.serve`.
- Hard single-client: a second connection is rejected with `await websocket.close(code=1013, reason="Server already has a connected client")` (`server.py:255`). Docstring: *"well suited for local development and single-session bots, but not for serving multiple concurrent clients."*
- Renamed in 1.4.0. `WebsocketServerTransport`, `WebsocketServerParams`, `WebsocketServerCallbacks`, `WebsocketServerInputTransport`, `WebsocketServerOutputTransport` are all `@deprecated`, removal in 2.0.0.
- Shutdown uses a refcount (`_server_refs`, `acquire_server()` / `release_server()`) so the output side can flush a goodbye TTS after the input side has seen the `EndFrame`.
- `WebsocketClientTransport` (`client.py:479`) is the mirror image — Pipecat *dialing out* to someone else's WebSocket.

**What the transport does NOT give you** (checked, not assumed): no authentication beyond an Origin allowlist; no session store or session ID; no conversation history; no reconnect/resume; no multi-tenant routing; no per-connection concurrency control beyond one pipeline per transport instance; no protocol — `session_timeout` is a bare `asyncio.sleep` that fires one callback (`fastapi.py:398-401`).

- **Migration angle:** this is the direct analogue of boson's `packages/gateway/gateway/server/` — 1,404 LOC total (`websocket.py` 734, `access.py` 374, `protocol.py` 114, `interruption.py` 95, `history.py` 70, `__init__.py` 17), fronted by `GatewayWebSocketServer` (`websocket.py:35`). The honest split: `FastAPIWebsocketTransport` replaces roughly the socket-plumbing part of `websocket.py` (`start`, `_handle_connection`, connection teardown) and `protocol.py` becomes a `FrameSerializer` subclass. It replaces **nothing** in `access.py` (374 L of auth), `history.py`, or the session-dispatch machinery — `_reserve_session_dispatch`, `_replace_active_task`, `_cancel_session_dispatch`, `_start_silence_timer`, `_finalize_partial` have no Pipecat counterpart at the transport layer and must be rebuilt as processors, or kept as an outer layer that owns the FastAPI route and hands an accepted `WebSocket` to the transport. Note the direct collision: boson's `_start_silence_timer` / `_finalize_partial` is endpointing logic living in the server; on Pipecat that belongs in `src/pipecat/turns/`, not here. Budget the migration as "keep ~700 L, port ~700 L", not "delete 1,404 L".

## Citation

pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (2026-08-25), read 2026-08-25.
Paths: `src/pipecat/transports/websocket/fastapi.py`, `server.py`, `client.py`; boson-agent `packages/gateway/gateway/server/` (read-only).

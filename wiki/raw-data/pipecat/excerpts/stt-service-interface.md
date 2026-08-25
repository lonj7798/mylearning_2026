# The Pipecat STT Service Base Class

<!-- slug: stt-service-interface · type: source · source: src/pipecat/services/stt_service.py -->

**Core Insight.** Pipecat's STT base class is not a transcription wrapper — it is a *latency
contract*. `STTService` splits providers into two shapes (continuous streaming vs
VAD-segmented request/response), redefines "TTFB" as *speech-end → final transcript*
rather than request → first byte, and broadcasts that number downstream in an
`STTMetadataFrame` so the turn-taking layer can size its own timers. The base class
itself never emits an interim transcript; that is entirely the provider's job.

**Guideline.** When wiring an STT service, decide first whether it is streaming
(`STTService` / `WebsocketSTTService`) or segmented (`SegmentedSTTService`), then supply a
measured `ttfs_p99_latency` for your deployment. Every downstream turn-stop timer is
derived from that value, so a wrong default silently mistunes end-of-turn detection.

## Technical Details

- Three base classes in `stt_service.py` (1041 L): `class STTService(AIService)` (L51),
  `class SegmentedSTTService(STTService)` (L797),
  `class WebsocketSTTService(STTService, WebsocketService)` (L929).
- Constructor (L94): `__init__(*, audio_passthrough=True, sample_rate=None,
  stt_ttfb_timeout=2.0, ttfs_p99_latency=None, keepalive_timeout=None,
  keepalive_interval=5.0, settings=None, **kwargs)`.
- The one abstract method (L334): `async def run_stt(self, audio: bytes) ->
  AsyncGenerator[Frame | None, None]`.
- **How audio reaches it.** `process_frame` (L465) matches `AudioRawFrame` →
  `process_audio_frame(frame, direction)` (L421) → `await
  self.process_generator(self.run_stt(frame.audio))` (L463); then, if
  `_audio_passthrough`, the same frame is re-pushed downstream (L479-480) so TTS/recording
  processors still see it. Transports deliver `InputAudioRawFrame` (frames.py L1449) or
  `UserAudioRawFrame` (L1495, carries `user_id`); services with no per-user track get
  `self._user_id = ""` (L452).
- **Streaming vs segmented.** Streaming: every ~20 ms chunk goes straight to the socket.
  Segmented (L901): audio accumulates into `self._audio_buffer`; while the user is silent
  the buffer is trimmed to `self._audio_buffer_size_1s = self.sample_rate * 2` (L837) —
  a 1-second pre-roll covering VAD's detection lag. On `VADUserStoppedSpeakingFrame`,
  `_handle_user_stopped_speaking` (L876) wraps the buffer with `pcm_to_wav(...)` (L890)
  and calls `run_stt` once. Local models override `wants_wav_segments` (L840) to `False`
  to avoid the 44-byte WAV header being read as samples.
- **Interim vs final frames — the real class names.** `TranscriptionFrame(TextFrame)`
  (frames.py L450) with fields `user_id, timestamp, language, result, finalized: bool =
  False`; `InterimTranscriptionFrame(TextFrame)` (frames.py L476) with the same fields
  *minus* `finalized`. **`stt_service.py` contains zero references to
  `InterimTranscriptionFrame`** — 25 provider modules construct it directly (e.g.
  `deepgram/stt.py` L772). The base class only *intercepts* `TranscriptionFrame` in
  `push_frame` (L519) to stamp `finalized` and close the TTFB window.
  `SegmentedSTTService.push_frame` (L850) forces `frame.finalized = True` on every
  transcript, since one segment yields exactly one result.
- **Two-phase finalize.** `request_finalize()` (L209) sets `_finalize_requested`;
  `confirm_finalize()` (L221) promotes it to `_finalize_pending`, and the next
  `TranscriptionFrame` gets `finalized = True`. Deepgram drives this from its
  `from_finalize` field (`deepgram/stt.py` L746-748) after sending `Finalize` on
  `VADUserStoppedSpeakingFrame`.
- **Redefined TTFB.** `_handle_vad_user_stopped_speaking` (L627) computes
  `speech_end_time = frame.timestamp - frame.stop_secs` (L645) and starts the metric
  there, then arms `_ttfb_timeout_handler` for `stt_ttfb_timeout=2.0 s` (L654) to catch
  the late final transcript. Docstring (L64-70): *"A streaming STT reports latency through
  TTFB — speech end to final transcript — and not through processing metrics."*
- **Metadata broadcast.** `service_metadata_frame() -> STTMetadataFrame` (L559) publishes
  `ttfs_p99_latency`; `supports_ttfs` (L548) returns `False` for turn-based services where
  the server owns the boundary (overridden in `cartesia/turns/stt.py` L185 and
  `deepgram/flux/stt.py` L249). Missing value → `DEFAULT_TTFS_P99 = 1.0` plus a warning.
- **`stt_latency.py`** (69 L) — measured P99 speech-end→final-transcript seconds, *all
  measured with `VADParams.stop_secs=0.2`*: `DEEPGRAM 0.35`, `SONIOX 0.35`,
  `ELEVENLABS_REALTIME 0.41`, `ASSEMBLYAI 0.42`, `GRADIUM 0.62`, `SPEECHMATICS 0.74`,
  `CARTESIA 0.81`, `GLADIA 1.49`, `GROQ 1.54`, `GOOGLE 1.57`, `AZURE 1.80`,
  `AWS_TRANSCRIBE 1.90`, `OPENAI 2.01`, `XAI 2.14`; `WHISPER`/`NVIDIA` fall back to 1.0.
  Re-measure with <https://github.com/pipecat-ai/stt-benchmark>.
- **`websocket_service.py`** (413 L): `class WebsocketService(ABC)` (L84) with abstract
  `_connect_websocket` / `_disconnect_websocket` / `_receive_messages`.
  `_try_reconnect(max_retries=3, report_error=None)` (L191) uses
  `exponential_backoff_time(attempt)`; `QuickFailureTracker` (L128) detects the
  handshake-succeeds-then-immediately-closes case (bad API key) that backoff cannot fix
  and gives up permanently. `WS_CLOSE_TIMEOUT = 2.0` (L50) caps the closing handshake.
- **VAD-aware reconnect** is STT-specific: `_request_reconnect()` (L733) refuses to
  reconnect while `_can_reconnect is False` (set on `VADUserStartedSpeakingFrame`, L608)
  and defers to `UserStoppedSpeakingFrame` (L613). Audio arriving during the reconnect is
  buffered in `_reconnect_audio_buffer` and replayed (L718-721) — no words lost.
- Keepalive: a task sends `0.1 s` of PCM silence (`_KEEPALIVE_SILENCE_DURATION`, L48) after
  `keepalive_timeout` of no audio; it counts toward billed usage (L1034-1036).
- **Migration angle:** This replaces *nothing* in boson-agent — a grep for
  `deepgram|whisper|speech_to_text|vad` across
  `/Users/jaewon/mywork_2026/Lina_2026/boson-agent-dev/boson-agent/packages/**/*.py`
  returns **zero hits**. STT is a net addition, not a port. The collision is at the
  protocol edge: `packages/gateway/gateway/server/protocol.py` L32-37 declares
  `VALID_CLIENT_TYPES = {"user_message", "partial_transcript", "interrupt",
  "get_history"}`, i.e. the *client* sends already-transcribed text. Moving to Pipecat
  deletes `partial_transcript` from the wire and re-creates it server-side as
  `InterimTranscriptionFrame`; `user_message` becomes a finalized `TranscriptionFrame`.
  `gateway/server/websocket.py` (734 L) is the module that dissolves into a Pipecat
  transport + `STTService` pair.

## Citation

pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (2026-08-25);
`src/pipecat/services/stt_service.py`, `src/pipecat/services/websocket_service.py`,
`src/pipecat/services/stt_latency.py`, `src/pipecat/frames/frames.py`.
Read 2026-08-25.

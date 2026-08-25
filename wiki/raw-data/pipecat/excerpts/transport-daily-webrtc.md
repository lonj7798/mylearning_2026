# Daily and SmallWebRTC transports — the WebRTC media path

<!-- slug: transport-daily-webrtc · type: source · source: src/pipecat/transports/{base_transport.py,base_input.py,base_output.py,daily/transport.py,smallwebrtc/} -->

**Core Insight.** A Pipecat transport is not a server — it is a *pair of FrameProcessors* (`input()` and `output()`) that sit at the two ends of the pipeline. `BaseTransport` declares only those two abstract methods; everything else (media handling, room lifecycle, event fan-out) lives in a provider client the transport wraps. Swapping Daily for SmallWebRTC changes the connection object and the params class, not the pipeline.

**Guideline.** Choose the transport for the *connection topology* you need (SFU room vs. direct peer vs. socket), configure media entirely through `TransportParams`, and wire VAD/turn-taking **separately** — in this version of Pipecat the transport does not own the VAD analyzer.

## Technical Details

- `BaseTransport(BaseObject)` — `src/pipecat/transports/base_transport.py:96`, only 137 lines total. Two abstract methods:
  ```python
  @abstractmethod
  def input(self) -> FrameProcessor: ...
  @abstractmethod
  def output(self) -> FrameProcessor: ...
  ```
  Constructor is `__init__(self, *, name=None, input_name=None, output_name=None)`.
- `TransportParams(BaseModel)` — `base_transport.py:25`. Media config is flat and declarative. Real defaults: `audio_out_enabled=False`, `audio_out_channels=1`, `audio_out_bitrate=96000`, `audio_out_10ms_chunks=4`, `audio_out_end_silence_secs=2`, `audio_out_auto_silence=True`, `audio_out_write_timeout_secs=10.0`, `audio_in_enabled=False`, `audio_in_channels=1`, `audio_in_stream_on_start=True`, `audio_in_passthrough=True`, `video_out_width=1024`, `video_out_height=768`, `video_out_framerate=30`. Sample rates default to `None` and resolve at setup.
- **Sample-rate resolution.** `BaseInputTransport.setup()` (`base_input.py:126`) does `self._sample_rate = self._params.audio_in_sample_rate or setup.audio_in_sample_rate`; `BaseOutputTransport.setup()` (`base_output.py:130`) mirrors it. `FrameProcessorSetup` defaults (`processors/frame_processor.py:106-107`) are `audio_in_sample_rate = 16000`, `audio_out_sample_rate = 24000`. So an unconfigured pipeline runs 16 kHz in / 24 kHz out.
- **Output chunking math** (`base_output.py:135-136`), verbatim:
  ```python
  audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
  self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks
  ```
- **⚠️ VAD is NOT attached to the transport.** `TransportParams` has **no** `vad_analyzer` field and `BaseInputTransport` never references one. Grep across `src/pipecat` finds `vad_analyzer` only in:
  - `LLMUserAggregatorParams.vad_analyzer: VADAnalyzer | None = None` — `processors/aggregators/llm_response_universal.py:175`
  - `VADProcessor(vad_analyzer=..., speech_activity_period=0.2, audio_idle_timeout=1.0)` — `processors/audio/vad_processor.py:41`
  - `VADController` — `audio/vad/vad_controller.py:31`

  The canonical wiring in the repo's own examples is `LLMContextAggregatorPair(..., user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))` (e.g. `examples/function-calling/function-calling-cerebras.py:114`). Turn boundaries come from strategies under `src/pipecat/turns/user_start/` — `VADUserTurnStartStrategy`, `TranscriptionUserTurnStartStrategy`, `MinWordsUserTurnStartStrategy`, `WakePhraseUserTurnStartStrategy`. **If a course outline says "attach the VAD analyzer to the transport params", it is describing an older Pipecat.**
- The only VAD code *inside* a transport is `WebRTCVADAnalyzer(VADAnalyzer)` at `daily/transport.py:203`, which wraps `Daily.create_native_vad(reset_period_ms=VAD_RESET_PERIOD_MS, sample_rate=..., channels=1)`. It is a `VADAnalyzer` *implementation* you pass to the aggregator — nothing in the transport consumes it (grep: single definition, zero call sites).
- **Daily** — `DailyTransport(BaseTransport)` at `daily/transport.py:2279`, signature `__init__(self, room_url: str, token: str | None, bot_name: str, params: DailyParams | None = None, input_name=None, output_name=None)`. `DailyParams(TransportParams)` (line 320) adds `api_url="https://api.daily.co/v1"`, `api_key=""`, `audio_in_user_tracks=True`, `camera_out_enabled=True`, `microphone_out_enabled=True`, `dialin_settings: DailyDialinSettings | None`, `transcription_enabled=False`, `transcription_settings=DailyTranscriptionSettings()` (Deepgram `nova-2-general`, `interim_results: True`).
- **Room lifecycle (Daily).** `DailyTransportClient.join()` (line 817) first calls `update_subscription_profiles({"base": {"camera": "unsubscribed", "screenVideo": "unsubscribed"}})` — video is off by default "for performance reasons" — then joins, extracts `participants.local.id` and `meetingSession.id`, fires `on_joined`, sets `_joined_event`, and flushes queued messages. ~29 event handlers are registered, including `on_first_participant_joined`, `on_participant_left(participant, reason)`, `on_dialin_ready(sip_endpoint)`, `on_dialout_answered`, `on_dtmf_event`. Per-participant audio: `capture_participant_audio(participant_id, callback, audio_source="microphone", sample_rate=16000, callback_interval_ms=20)` (line 1157).
- **SmallWebRTC** — `SmallWebRTCTransport(BaseTransport)` at `smallwebrtc/transport.py:951`, signature `__init__(self, webrtc_connection: SmallWebRTCConnection, params: TransportParams, ...)`. It takes plain `TransportParams` (no subclass). `SmallWebRTCConnection(ice_servers=None, connection_timeout_secs=60)` (`connection.py:245`) wraps aiortc's `RTCPeerConnection` and registers `app-message`, `track-started`, `track-ended`, `connecting`, `connected`, `disconnected`, `closed`, `failed`, `new`. Only three transport-level events exist: `on_client_connected`, `on_client_disconnected`, `on_app_message`.
- **Media clocking.** SmallWebRTC output uses `RawAudioTrack(sample_rate, auto_silence=True)` (`transport.py:76`) which enforces 10 ms granularity: `_bytes_per_10ms = (sample_rate * 10 // 1000) * 2`, and `add_audio_bytes()` **raises `ValueError("Audio bytes must be a multiple of 10ms size.")`** otherwise. Input reads 20 ms at a time (`read_audio_frame`, line 375, with a 2.0 s recv timeout).
- **Full transport inventory** at this commit — `daily`, `smallwebrtc`, `websocket`, `livekit`, `local`, `moq`, `whatsapp`, `vonage` (a *video connector*, `VonageVideoConnectorTransport`), plus avatar vendors `heygen`, `lemonslice`, `tavus`. There is **no `twilio/` and no `telnyx/` transport directory** — see `[[transport-telephony]]`.
- **Migration angle:** boson-agent has no media path at all today; `gateway/server/websocket.py` speaks a text protocol over `websockets.ServerConnection`. Daily/SmallWebRTC are therefore *additive*, not replacements — they would supply the audio ingress boson never had. For Lina TMR (Korean insurance tele-sales over PSTN) neither is the production path: WebRTC transports are the right choice for the *browser-based agent console and QA/eval harness*, while the live call rides `[[transport-websocket]]` + `[[transport-telephony]]`. The load-bearing consequence for boson: barge-in currently reasons over partial *text* transcripts in `gateway/interrupt/detector.py`; on Pipecat that logic collides with `VADUserTurnStartStrategy` / `LLMUserAggregatorParams.vad_analyzer`, which is aggregator-level, not transport-level. Plan the interrupt subsystem port against `src/pipecat/turns/`, not against transport params.

## Citation

pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (2026-08-25), read 2026-08-25.
Paths: `src/pipecat/transports/base_transport.py`, `base_input.py`, `base_output.py`, `daily/transport.py` (3,065 L), `smallwebrtc/transport.py` (1,085 L), `smallwebrtc/connection.py` (825 L), `src/pipecat/processors/audio/vad_processor.py`, `src/pipecat/turns/`.

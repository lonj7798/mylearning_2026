# Silero VAD — VADAnalyzer, VADParams, and the four-state speech machine
<!-- slug: vad-silero · type: source · source: src/pipecat/audio/vad/vad_analyzer.py, silero.py, vad_controller.py -->

**Core Insight.** Pipecat's VAD is deliberately split in three: a *model* (`SileroOnnxModel`, per-chunk confidence), a *hysteresis state machine* (`VADAnalyzer`, four states with frame counters), and an *event emitter* (`VADController`, edge-triggered callbacks). Only the state machine turns a noisy per-32ms probability into "the user is speaking", and it does so by requiring `start_secs` of continuous confidence to enter SPEAKING and `stop_secs` of continuous silence to leave. Every barge-in latency number in a voice bot is a direct consequence of those two floats.

**Guideline.** Tune barge-in by moving `VADParams.start_secs` / `stop_secs`, not by touching `confidence`. `confidence` and `min_volume` are an **AND** gate applied per-chunk; the timing params are what buy you noise immunity. And in current Pipecat the analyzer is passed to `LLMUserAggregatorParams(vad_analyzer=...)` (or a standalone `VADProcessor`) — **not** to the transport.

## Technical Details

- **Module constants** (`vad_analyzer.py` L25-28) — the real defaults, verbatim:
  ```python
  VAD_CONFIDENCE = 0.7
  VAD_START_SECS = 0.2
  VAD_STOP_SECS = 0.2
  VAD_MIN_VOLUME = 0.6
  ```
  `VADParams(BaseModel)` (L47-60) has exactly four fields: `confidence: float = VAD_CONFIDENCE`, `start_secs: float = VAD_START_SECS`, `stop_secs: float = VAD_STOP_SECS`, `min_volume: float = VAD_MIN_VOLUME`. Pydantic, not dataclass.
- **`VADState(Enum)`** (L31-44): `QUIET = 1`, `STARTING = 2`, `SPEAKING = 3`, `STOPPING = 4`.
- **`VADAnalyzer(ABC)`** (L63). `__init__(self, *, sample_rate: int | None = None, params: VADParams | None = None)`. Abstract: `num_frames_required(self) -> int` and `voice_confidence(self, buffer: bytes) -> float`. Runs the model off-loop: `self._executor = ThreadPoolExecutor(max_workers=1)` (L92), and `analyze_audio` (L178) is `await loop.run_in_executor(self._executor, self._run_analyzer, buffer)`.
- **Where the seconds become frame counts** — `set_params` (L151-171):
  ```python
  self._vad_frames = self.num_frames_required()
  self._vad_frames_num_bytes = self._vad_frames * self._num_channels * 2
  vad_frames_per_sec = self._vad_frames / self.sample_rate
  self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
  self._vad_stop_frames  = round(self._params.stop_secs  / vad_frames_per_sec)
  ```
  At 16 kHz with 512-sample chunks, `vad_frames_per_sec = 0.032 s`, so `start_secs=0.2` → **7 consecutive chunks** (`round(6.25)`), same for stop. Calling `set_params` resets `_vad_state` to `QUIET` but deliberately **not** the volume tracker (comment L166-168: "the rolling window and its smoothing follow the audio stream, which is continuous across parameter changes").
- **The gate is an AND** (`_run_analyzer` L211): `speaking = confidence >= self._params.confidence and volume >= self._params.min_volume`. Volume is `AudioVolumeTracker` output passed through `exp_smoothing(..., self._smoothing_factor)` with `self._smoothing_factor = 0.2` (L87).
- **State machine** (`_run_analyzer` L213-246), exactly as written:
  - speaking & QUIET → `STARTING`, `_vad_starting_count = 1`
  - speaking & STARTING → `_vad_starting_count += 1`
  - speaking & STOPPING → back to **`SPEAKING`**, `_vad_stopping_count = 0` (a mid-pause resume never leaves SPEAKING)
  - not speaking & STARTING → back to `QUIET` (false start discarded, no event ever fired)
  - not speaking & SPEAKING → `STOPPING`, `_vad_stopping_count = 1`
  - not speaking & STOPPING → `_vad_stopping_count += 1`
  - after the chunk loop: `STARTING` + `count >= _vad_start_frames` → `SPEAKING`; `STOPPING` + `count >= _vad_stop_frames` → `QUIET`.
- **`SileroVADAnalyzer(VADAnalyzer)`** (`silero.py` L130). `num_frames_required()` returns `512 if self.sample_rate == 16000 else 256` (L191-197). `set_sample_rate` raises `ValueError` for anything but 16000/8000 (L184). `voice_confidence` does `np.frombuffer(buffer, np.int16).astype(np.float32) / 32768.0` then `self._model(audio_float32, self.sample_rate)[0]`. Model state is reset every `_MODEL_RESET_STATES_TIME = 5.0` seconds (L23, L216-220) — "memory will keep growing otherwise". Bare `except` returns `0` confidence on error.
- **`SileroOnnxModel`** (`silero.py` L34): ONNX Runtime, `inter_op_num_threads = 1`, `intra_op_num_threads = 1`, `force_onnx_cpu=True` by default, model bundled at `pipecat.audio.vad.data/silero_vad.onnx`. Context carry-over of 64 samples @16 kHz / 32 @8 kHz (L101, L123).
- **`VADController(BaseObject)`** (`vad_controller.py` L31). `__init__(self, vad_analyzer, *, speech_activity_period: float = 0.2, audio_idle_timeout: float = 1.0)`. Five events, all `sync=True`: `on_speech_started`, `on_speech_stopped`, `on_speech_activity`, `on_push_frame`, `on_broadcast_frame`. `_handle_vad` (L176) fires **only on transitions into terminal states** — `STARTING`/`STOPPING` are explicitly filtered out (L179-183), so the intermediate states are never observable outside the analyzer. `_audio_idle_handler` (L192) forces `SPEAKING → QUIET` + `on_speech_stopped` if no `InputAudioRawFrame` arrives for `audio_idle_timeout` (mic mute mid-utterance).
- **Where the analyzer is plugged in — the outline's assumption is stale.** `TransportParams` (`transports/base_transport.py` L25-93) has **no `vad_analyzer` and no `vad_enabled` field**; it stops at `audio_in_filter`, `audio_in_stream_on_start`, `audio_in_passthrough`. CHANGELOG (L4402): *"⚠️ Removed `vad_analyzer` and `turn_analyzer` parameters from `TransportParams` and all transport input classes... VAD and turn detection are now handled entirely by `LLMUserAggregator`."* The two live mount points are:
  1. `LLMUserAggregatorParams.vad_analyzer: VADAnalyzer | None = None` (`llm_response_universal.py` L175) — the aggregator builds its own `VADController` at L750-761 with `audio_idle_timeout=self._params.audio_idle_timeout` (default `1.0`, L172).
  2. `VADProcessor(vad_analyzer=SileroVADAnalyzer(), speech_activity_period=0.2, audio_idle_timeout=1.0)` (`processors/audio/vad_processor.py` L26-65) — a standalone `FrameProcessor` used when there is no LLM aggregator (e.g. transcription-only bots; ~18 examples under `examples/transcription/`).
- **Canonical wiring** (`examples/getting-started/06-voice-agent.py` L75-89):
  ```python
  user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
      context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))
  pipeline = Pipeline([transport.input(), stt, user_aggregator, llm, tts,
                       transport.output(), assistant_aggregator])
  ```
- **Emitted frames are not the turn frames.** The controller's `on_speech_started` broadcasts `VADUserStartedSpeakingFrame(start_secs=...)` (`frames.py` L1226, also carries `timestamp: float = field(default_factory=time.time)`), *not* `UserStartedSpeakingFrame`. A turn-start strategy converts one into the other — `VADUserTurnStartStrategy.process_frame` (`turns/user_start/vad_user_turn_start_strategy.py` L31-33) matches `VADUserStartedSpeakingFrame` → `await self.trigger_user_turn_started()`. Default start strategies are `[VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()]`; default stop is `[TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3())]` (`turns/user_turn_strategies.py` L27-50). So **Silero decides "there is voice"; the strategy decides "this is a turn"** — two separate layers.
- **Migration angle:** boson-agent has no VAD at all — barge-in is decided from *text* by `PartialDetector` (`packages/gateway/gateway/interrupt/detector.py`, 82 L, `overlap_chars=10`, `timing_threshold_ms=1000`, `silence_timeout_ms=2000`) plus a `CompositePolicy` of `DurationPolicy(min_ms=500)` / `WordFilterPolicy` (`policy.py`). Adopting Pipecat replaces that *timing half* wholesale: `DurationPolicy.min_ms=500` is superseded by `VADParams.start_secs` (0.2 s at 16 kHz ≈ 7 chunks), and `silence_timeout_ms=2000` by `stop_secs` plus a stop strategy. What it does **not** replace is `WordFilterPolicy` — Pipecat has no "ignore 네/응 backchannels" filter in the VAD layer; that semantic gate must be re-implemented as a custom `BaseUserTurnStartStrategy` (e.g. alongside `MinWordsUserTurnStartStrategy`, `turns/user_start/min_words_user_turn_start_strategy.py`). Also note the 8 kHz telephony path is supported natively (`num_frames_required()` → 256), but `start_secs`/`stop_secs` then map to 32 ms chunks as well, so the frame counts are identical — the tuning does not need to change for Lina TMR's SIP audio.

## Citation
pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (2026-08-25; CHANGELOG head `[1.7.0] - 2026-08-01`). Files: `src/pipecat/audio/vad/vad_analyzer.py`, `src/pipecat/audio/vad/silero.py`, `src/pipecat/audio/vad/vad_controller.py`, `src/pipecat/processors/audio/vad_processor.py`, `src/pipecat/transports/base_transport.py`. Read 2026-08-25.

# Pipecat Frame Taxonomy — System vs Data vs Control
<!-- slug: frame-taxonomy · type: source · source: src/pipecat/frames/frames.py -->

**Core Insight.** Pipecat's entire concurrency model is encoded in *which base class a frame inherits from*, not in any scheduler. `SystemFrame` means "jump the queue and survive interruption"; `DataFrame` and `ControlFrame` mean "stay in arrival order and get thrown away on interruption." Barge-in works because user audio and interruption signals are `SystemFrame`s while bot speech is `DataFrame`s — the priority is structural, declared once at class-definition time.

**Guideline.** When you write a custom frame, pick the base class by asking "must this survive an interruption and arrive out-of-band?" If yes → `SystemFrame`. If it carries payload the bot is producing → `DataFrame`. If it is ordered control that should die with a cancelled turn → `ControlFrame`. If it is control that must *never* be dropped (shutdown, settings commit) → `ControlFrame, UninterruptibleFrame`.

## Technical Details

- **Four base classes** (`frames.py` L64-138), all `@dataclass`:
  - `Frame` (L65) — root. Fields are all `field(init=False)`, assigned in `__post_init__` (L91): `id` (`obj_id()`), `name` (`f"{self.__class__.__name__}#{obj_count(self)}"`), `pts: int | None`, `broadcast_sibling_id: int | None`, `metadata: dict[str, Any]`, `transport_source: str | None`, `transport_destination: str | None`.
  - `SystemFrame(Frame)` (L105) — verbatim docstring: *"A frame that takes higher priority than other frames. System frames are handled in order and are not affected by user interruptions."*
  - `DataFrame(Frame)` (L116) — *"processed in order ... Data frames are cancelled by user interruptions."*
  - `ControlFrame(Frame)` (L128) — same ordering as data, *"contains control information such as update settings or to end the pipeline after everything is flushed. Control frames are cancelled by user interruptions."*
- **`UninterruptibleFrame`** (L147) is **not a `Frame` subclass** — it is a bare `@dataclass` mixin. Docstring: frames with it *"are preserved during interruptions: they remain in internal queues and any task processing them will not be cancelled."* Used as `class EndFrame(ControlFrame, UninterruptibleFrame)`.
- **Why `SystemFrame` is out-of-band:** the *reason* lives in the processor, not here — `FrameProcessorQueue` is an `asyncio.PriorityQueue` with `START_PRIORITY = 1`, `SYSTEM_PRIORITY = 10`, `DEFAULT_PRIORITY = 20`, and the input task processes `SystemFrame`s inline while everything else is re-queued into a second, cancellable queue. See [[frame-processor]].
- **Payload mixins**, also non-`Frame` dataclasses: `AudioRawFrame` (L161) — `audio: bytes`, `sample_rate: int`, `num_channels: int`, `num_frames: int = field(default=0, init=False)` computed as `int(len(self.audio) / (self.num_channels * 2))` (assumes 16-bit PCM). `ImageRawFrame` (L181) — `image: bytes`, `size: tuple[int, int]`, `format: str | None`.

### The frames a voice bot actually touches

| Frame | Line | Base | Real fields |
|---|---|---|---|
| `InputAudioRawFrame` | 1449 | `SystemFrame, AudioRawFrame` | inherited; mic audio from transport |
| `UserAudioRawFrame` | 1495 | `InputAudioRawFrame` | `+ user_id: str = ""` |
| `OutputAudioRawFrame` | 201 | `DataFrame, AudioRawFrame` | audio headed to the speaker |
| `TTSAudioRawFrame` | 241 | `OutputAudioRawFrame` | `+ context_id: str \| None` |
| `TranscriptionFrame` | 450 | `TextFrame(DataFrame)` | `text`, `user_id: str`, `timestamp: str`, `language: Language \| None`, `result: Any \| None`, `finalized: bool = False` |
| `InterimTranscriptionFrame` | 476 | `TextFrame` | `text`, `user_id`, `timestamp`, `language`, `result` — no `finalized` |
| `TextFrame` | 303 | `DataFrame` | `text: str`; `skip_tts`, `includes_inter_frame_spaces`, `append_to_context` all `init=False`, set in `__post_init__` |
| `LLMTextFrame` | 343 | `TextFrame` | no new fields; sets `includes_inter_frame_spaces = True` |
| `AggregatedTextFrame` | 387 | `TextFrame` | `aggregated_by: AggregationType \| str`, `context_id`, `raw_text`, `will_be_spoken` (init=False) |
| `TTSTextFrame` | 417 | `AggregatedTextFrame` | the sentence TTS is about to speak |
| `LLMFullResponseStartFrame` / `EndFrame` | 2059 / 2074 | `ControlFrame` | `skip_tts: bool \| None` (init=False) |
| `TTSStartedFrame` | 2210 | `ControlFrame` | `context_id: str \| None`, `append_to_context: bool = True` |
| `TTSStoppedFrame` | 2231 | `ControlFrame` | `context_id: str \| None` |
| `InterruptionFrame` | 1142 | `SystemFrame` | **no fields** — pure signal |
| `UserStartedSpeakingFrame` / `UserStoppedSpeakingFrame` | 1154 / 1165 | `SystemFrame` | no fields; the *turn* boundary |
| `VADUserStartedSpeakingFrame` | 1226 | `SystemFrame` | `start_secs: float = 0.0`, `timestamp: float = field(default_factory=time.time)` |
| `VADUserStoppedSpeakingFrame` | 1241 | `SystemFrame` | `stop_secs: float = 0.0`, `timestamp` |
| `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame` | 1282 / 1293 | `SystemFrame` | no fields |
| `StartFrame` | 924 | `SystemFrame` | `audio_in_sample_rate: int = 16000`, `audio_out_sample_rate: int = 24000`, `enable_metrics`, `enable_tracing`, `enable_usage_metrics`, `report_only_initial_ttfb`, `tracing_context` — **all seven deprecated since 1.8.0**; `__getattribute__` (L982) fires `warn_deprecated_read` on read. Config moved to `FrameProcessorSetup.setup()`. |
| `EndFrame` | 1899 | `ControlFrame, UninterruptibleFrame` | `reason: Any \| None` — graceful drain |
| `StopFrame` | 1923 | `ControlFrame, UninterruptibleFrame` | no fields; stop the pipeline, keep processors alive |
| `CancelFrame` | 999 | `SystemFrame` | `reason: Any \| None` — stop *now*, skip queued frames |
| `ErrorFrame` | 1016 | `SystemFrame` | `error: str`, `fatal: bool = False` (deprecated 1.8.0), `processor`, `exception`, `category: ErrorCategory \| None` |

- **The `EndFrame` vs `CancelFrame` split is the graceful/hard shutdown pair**: `EndFrame` is a `ControlFrame` so it arrives *after* everything queued ahead of it; `CancelFrame` is a `SystemFrame` so it arrives *ahead* of them.
- **Not found where expected:** there is no `StartInterruptionFrame` or `StopInterruptionFrame` in this tree (the names our COLLECTION-PLAN assumed). A `grep` across `src/pipecat/` returns zero hits. The single frame is `InterruptionFrame`, broadcast in *both* directions by `FrameProcessor.broadcast_interruption()`. Likewise there is no `BotInterruptionFrame`.
- `ProposedUserStartedSpeakingFrame` (L1256) / `ProposedUserStoppedSpeakingFrame` (L1270) exist as an explicit *proposal* tier: a provider-side turn signal that a `UserTurnStartStrategy` must ratify into a real `UserStartedSpeakingFrame`. Useful when a Korean STT provider reports its own endpoints.
- **Migration angle:** this taxonomy directly replaces boson-agent's flat string protocol in `packages/gateway/gateway/server/protocol.py` — `VALID_CLIENT_TYPES = {"user_message", "partial_transcript", "interrupt", "get_history"}` and `VALID_SERVER_TYPES = {"text_delta", "turn_end", "error", "interrupted", "stage_changed", "history"}`. Ten string constants become ~150 typed frame classes, and boson's `"partial_transcript"` splits into the real distinction `InterimTranscriptionFrame` vs `TranscriptionFrame(finalized=...)`. Boson's `"interrupt"` client message becomes `InterruptionFrame`. **Collides**: boson's `gateway/interrupt/cancellation.py` `CancellationFlag` is a hand-rolled cooperative flag doing what `SystemFrame` priority + `UninterruptibleFrame` do declaratively — one of them has to go. **Untouched**: everything boson keeps *inside* the LLM turn — `basement/loop/agent_loop.py`, the tool router, `gateway/layers/pipeline.py` `LayerPipeline` — has no frame equivalent and would live behind a single custom `FrameProcessor`.

## Citation
pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (post-1.7.0 / pre-1.8.0 main, latest CHANGELOG entry `[1.7.0] - 2026-08-01`), read 2026-08-25. Repo path: `src/pipecat/frames/frames.py` (2415 lines).

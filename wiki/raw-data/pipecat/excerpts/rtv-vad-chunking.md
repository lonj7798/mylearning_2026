# realtime_voice VAD, chunking, and the playout ledger
<!-- slug: rtv-vad-chunking · type: boson · source: packages/realtime_voice/realtime_voice/{vad/energy.py,vad/silero.py,chunking.py,ledger.py} -->

**Core Insight.** realtime_voice's VAD is a *frame-counting* hysteresis machine where Pipecat's is a *seconds-based* one — same shape, different unit, and the frame unit silently couples endpointing latency to whatever the transport happens to send. But downstream of VAD, boson has two components Pipecat has no equivalent of: a Korean-aware phrase chunker with a 1→2→tail batching schedule, and an `AudioTextPlayoutLedger` that maps text spans onto sample positions so a mid-phrase barge-in can recover *the character prefix the customer actually heard*.

**Guideline.** Frame-count thresholds are a latency bug waiting to happen — `min_silence_frames=6` is 120 ms at 20 ms frames and 600 ms at 100 ms frames, and nothing in the code asserts the frame duration. Convert to seconds (as Pipecat does) before tuning against recorded Korean dental audio. Conversely, do not throw away the ledger: it is the only implementation of "history records what was *audible*, not what was *generated*" that survives being ported.

## Technical Details

### VAD — two implementations, identical state machine

- `EnergyVADConfig` (`vad/energy.py` L15): `speech_rms: float = 500.0`, `min_speech_frames: int = 2`, `min_silence_frames: int = 4`. `EnergyVAD.rms()` (L104) is pure-Python `math.sqrt(sum(s*s)/n)` over an `array("h")` — no numpy, no model. Docstring: *"RMS hysteresis VAD intended for fallback and deterministic tests."*
- `SileroVADConfig` (`vad/silero.py` L21): `threshold: float = 0.5`, `min_speech_frames: int = 2`, `min_silence_frames: int = 6`. `SileroVAD.process` raises `ValueError("SileroVAD requires 16 kHz mono PCM")` for anything but 16 kHz mono (L58) — **no 8 kHz telephony path**, unlike Pipecat's `num_frames_required() -> 512 if 16000 else 256`. `from_pretrained()` (L44) lazily imports `silero_vad.load_silero_vad`; the model runs via `await asyncio.to_thread(call_model)` (L123) — the same off-loop idea as Pipecat's `ThreadPoolExecutor(max_workers=1)`, one thread per call instead of a pinned worker.
- **Parameter-for-parameter against Pipecat's `VADParams`** (see [[vad-silero]]):

  | | realtime_voice | Pipecat |
  |---|---|---|
  | confidence gate | `SileroVADConfig.threshold = 0.5` | `VADParams.confidence = 0.7` |
  | volume gate | **absent** | `VADParams.min_volume = 0.6` (ANDed with confidence) |
  | speech onset | `min_speech_frames = 2` (frames) | `start_secs = 0.2` → 7 chunks @16 kHz |
  | speech offset | `min_silence_frames = 6` (frames) | `stop_secs = 0.2` → 7 chunks |
  | states | 2 (`self._speaking` bool) | 4 (`VADState.QUIET/STARTING/SPEAKING/STOPPING`) |
  | chunk size | whatever the transport sends | fixed `num_frames_required()` = 512 @16 kHz |
  | model state reset | `reset()` calls `model.reset_states()` if present | forced every `_MODEL_RESET_STATES_TIME = 5.0` s |
  | idle-mic timeout | **absent** | `VADController(audio_idle_timeout=1.0)` |
  | backchannel filter | **absent** here (lives in Gateway `WordFilterPolicy`) | absent |

  realtime_voice's threshold is *looser* (0.5 vs 0.7) and it has no `min_volume` AND-gate, so it will false-trigger more on room noise; its 2-state machine also cannot discard a false start silently the way Pipecat's `STARTING → QUIET` transition does — a 2-frame blip becomes a real `SPEECH_STARTED`, which in `VoiceSession._on_speech_started` **immediately advances the generation and cancels the assistant**. That is the sharpest correctness delta in the whole comparison.
- Both emit `endpoint_latency_ms = self._silence_frames * frame.duration_seconds * 1000` (energy.py L79, silero.py L89) — the only self-measured latency in the VAD layer.
- **Chunking of *input* audio does not exist here.** Audio arrives already framed by `InboundAudioPump` (see [[rtv-webrtc-transport]]), which emits one `AudioFrame` per PyAV resampler output; the VADs consume whatever size that is. `VoiceSessionConfig.vad_prefix_frames = 5` keeps a `deque(maxlen=5)` of pre-speech frames (`session.py` L131) that is replayed into ASR on `SPEECH_STARTED` (L296-299) — a pre-roll buffer Pipecat does not expose as a tunable.

### `chunking.py` — `KoreanPhraseChunker` (283 L), no Pipecat counterpart

- `__init__(*, min_chars=12, max_chars=60, hard_max_chars=None, batch_max_chars=320, adaptive_batching=True)`. When `hard_max_chars is None` it resolves to `min(batch_max_chars, max_chars * 2)` (L56-60).
- Docstring L28-34: *"Adaptive mode emits the first complete sentence immediately, batches the next two complete sentences, then holds the remaining response as one final group until `flush`. `max_chars` is a soft latency target rather than an immediate cut point."* `_batch_phase` 0 → single sentence (time-to-first-audio), 1 → pairs, 2 → bounded tail (`_accept_adaptive` L115-149).
- `_BoundaryKind` StrEnum: `SENTENCE / SOFT / OVERLONG / FINAL_TAIL`. `_STRONG_END = frozenset(".!?。！？\n")`, `_SOFT_END = frozenset(",，;；:")`, `_CLOSING_PUNCTUATION = frozenset("\"'”’)]}」』】")`.
- Real Korean/CJK handling Pipecat's text aggregators do not attempt: `_is_safe_period` (L255) refuses to split `1.5`, `...`, or ASCII identifiers (`gpt-4.1`, hostnames) — comment L266-269: *"A dot between ASCII token characters belongs to a model name, hostname, abbreviation, or identifier rather than ending a Korean sentence."* `_is_numeric_separator` (L277) protects `1,000`. `_INTERNAL_TAG = re.compile(r"\[(?:interruption|system|tool|objection|customer|assistant)[^\]]*\]")` strips Gateway control tags from spoken text while `start_char`/`end_char` keep the *source* span intact.

### `ledger.py` — `AudioTextPlayoutLedger` (110 L)

- Tracks four dicts keyed by `GenerationId`: `_phrases` (list of `PhrasePlayout(request, sample_start, sample_end, complete)`), `_by_phrase`, `_next_sample`, `_played_sample`.
- `begin(request)` / `append(request, sample_count) -> (start, end)` / `finish(request)` build the sample map as TTS streams; `acknowledge(generation_id, played_sample)` moves the client cursor with `max(current, played_sample)` (monotonic, so a late ack cannot rewind).
- `audible_text()` (L74) is the payoff: it walks phrases until the cursor, and for the partially-played phrase does `ratio = (cursor - sample_start) / (sample_end - sample_start)` then `text[:int(len(text) * ratio)]`. **This is a linear character-per-sample approximation, not a word-timestamp alignment** — Pipecat gets the same guarantee for free by placing the assistant aggregator *after* `transport.output()` and pacing it with word-timestamped `TTSTextFrame`s ([[interruption-cascade]]). Boson's is less accurate mid-word, but works with a TTS that emits no timestamps at all.
- `playout_complete()` (L98) — all phrases `complete` **and** `played_sample >= queued_samples` — is what lets `_cancel_generation` distinguish "customer interrupted me" from "I finished and they replied" (`session.py` L502-507, `semantic_interrupt` flag).
- **Migration angle:** the VAD half is a straight replacement — delete `vad/energy.py` and `vad/silero.py`, mount `SileroVADAnalyzer()` on `LLMUserAggregatorParams(vad_analyzer=...)`, and gain the 4-state machine, `min_volume`, 8 kHz, and `audio_idle_timeout` that boson lacks. The chunking and ledger halves are the opposite: `KoreanPhraseChunker` has no Pipecat equivalent (Pipecat's TTS services split on their own sentence heuristics and have no 1→2→tail schedule and no Korean numeric/identifier guards), and `AudioTextPlayoutLedger` would become redundant *only if* the chosen TTS emits word timestamps. If it does not, port the ledger as a custom `FrameProcessor` sitting beside `transport.output()`. Keep `WordFilterPolicy` in Gateway either way — neither system filters backchannels.

## Citation
boson-agent (private), branch `voice-chat-dev`, commit `034ce4ca09a2f109e6c248a43bc989f8d26a6abf` (2026-07-29). Paths: `packages/realtime_voice/realtime_voice/vad/energy.py` (111 L), `vad/silero.py` (139 L), `chunking.py` (283 L), `ledger.py` (110 L). Compared against pipecat-ai/pipecat commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`, read 2026-08-25.

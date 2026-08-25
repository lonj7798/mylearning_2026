# The Voice-to-Voice Latency Budget, as Pipecat Actually Measures It
<!-- slug: latency-budget-voice · type: source · source: src/pipecat/services/stt_latency.py, src/pipecat/observers/user_bot_latency_observer.py, src/pipecat/processors/metrics/frame_processor_metrics.py, src/pipecat/turns/user_stop/ -->

**Core Insight.** The user→bot interval is not one number, it is a sum of stages Pipecat times
separately, and the two biggest terms are *not* the LLM. They are the endpointing wait
(`VADParams.stop_secs` plus the STT's P99 time-to-final-segment) and the TTS's time-to-first-*audible*-
audio. Pipecat encodes the first as per-provider constants shipped in the source tree, and the
second as `TTFA = TTFB + leading_silence`, precisely because both are routinely mistaken for zero.

**Guideline.** Budget the turn as `stop_secs + max(0, TTFS_p99 - stop_secs) + LLM TTFB (+ thinking
time) + text-aggregation + TTS TTFA + transport`, and pick your STT by its `*_TTFS_P99` constant,
not by its streaming-token latency. Where the repo gives no target number — and it gives none —
say so rather than quoting a threshold as a Pipecat claim.

## Technical Details
- **Stage 1, endpointing.** `VADParams` defaults (`audio/vad/vad_analyzer.py:25-28`):
  `VAD_CONFIDENCE = 0.7`, `VAD_START_SECS = 0.2`, `VAD_STOP_SECS = 0.2`, `VAD_MIN_VOLUME = 0.6`.
  `VADUserStoppedSpeakingFrame` carries `timestamp` and `stop_secs`; consumers reconstruct the true
  speech end as `frame.timestamp - frame.stop_secs`
  (`user_bot_latency_observer.py:249`, `stt_service.py:645`).
- **Stage 2, STT finalization — the repo's only measured benchmark table.**
  `src/pipecat/services/stt_latency.py` defines P99 **TTFS** ("time from when speech ends to when
  the final transcript is received"), all "measured with `VADParams.stop_secs=0.2`":
  `DEFAULT_TTFS_P99 = 1.0`, `DEEPGRAM = 0.35`, `SONIOX = 0.35`, `ELEVENLABS_REALTIME = 0.41`,
  `ASSEMBLYAI = 0.42`, `GRADIUM = 0.62`, `SPEECHMATICS = 0.74`, `CARTESIA = 0.81`,
  `TOGETHER = 1.00`, `SARVAM_REALTIME = 1.00` (marked provisional), `SARVAM = 1.17`,
  `GLADIA = 1.49`, `GROQ = 1.54`, `GOOGLE = 1.57`, `SMALLEST = 1.59`, `OPENAI_REALTIME = 1.66`,
  `AZURE = 1.80`, `MISTRAL = 1.89`, `AWS_TRANSCRIBE = 1.90`, `ELEVENLABS = 2.01`,
  `OPENAI = 2.01`, `FAL = 2.07`, `XAI = 2.14`. `NVIDIA` and `WHISPER` fall back to
  `DEFAULT_TTFS_P99` because they "run locally and should be replaced with measured values".
  Re-measure with <https://github.com/pipecat-ai/stt-benchmark> and pass
  `DeepgramSTTService(api_key=..., ttfs_p99_latency=0.45)`. Turn-based services
  (`CartesiaTurnsSTTService`, `DeepgramFluxSTTService`) override `STTService.supports_ttfs → False`.
- **How the number is spent.** `STTService.service_metadata_frame` emits
  `STTMetadataFrame(service_name, ttfs_p99_latency)` at start; turn-stop strategies consume it.
  `SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.6)` runs two timers and computes
  `effective_stt_wait = max(0.0, self._stt_timeout - self._stop_secs)` (line 223), short-circuited
  by `TranscriptionFrame.finalized=True`. It warns if `stop_secs != VAD_STOP_SECS` (the benchmark
  assumption) or if `stop_secs >= stt_timeout`. `TurnAnalyzerUserTurnStopStrategy` uses the same
  safety net behind a turn-detection model, and `wait_for_transcript=False` takes transcripts "off
  the latency critical path" for realtime/S2S services.
- **STT's own TTFB is redefined.** `STTService.__init__(stt_ttfb_timeout: float = 2.0, ...)`:
  "STT 'TTFB' differs from traditional TTFB … Since STT receives continuous audio, we measure from
  when the user stops speaking to when the final transcript arrives." Implementation:
  `_handle_vad_user_stopped_speaking` calls `start_ttfb_metrics(start_time=frame.timestamp - frame.stop_secs)`,
  then `_ttfb_timeout_handler` sleeps `stt_ttfb_timeout` and calls
  `stop_ttfb_metrics(end_time=self._last_transcript_time)` (`stt_service.py:627-678`). Streaming STT
  reports **no** `ProcessingMetricsData` by design; `SegmentedSTTService` does report both.
- **Stage 3, LLM.** `TTFBMetricsData.value` ends at the *first output of any kind* —
  `stop_ttfb_metrics` docstring: "including content a caller never sees, such as an LLM's
  reasoning… Events that merely acknowledge the request (an HTTP response head, a stream-open or
  keepalive event) carry no output and must not stop it." The user-visible figure is
  `TTFATMetricsData(ttfat = ttfb + thinking_time)`, ended at the first answer token or at the first
  tool call. A turn that answers from a tool result "reports twice — once for the call, once for the
  answer built from its result."
- **Stage 4, aggregation.** `TextAggregationMetricsData` = "time from the first LLM token to the
  first complete sentence, representing the latency cost of sentence aggregation in the TTS
  pipeline" (`tts_service.py:771, 791, 1101`). It is a distinct, nameable slice of the budget.
- **Stage 5, TTS.** `TTFAMetricsData(ttfa, ttfb, leading_silence)`, computed in
  `FrameProcessorMetrics.process_ttfa_metrics` by buffering PCM and calling
  `detect_speech_onset(buffer, sample_rate, num_channels)`; `silence_duration = onset / sample_rate`,
  `ttfa = last_ttfb + silence_duration`. Buffer capped at `_TTFA_MAX_BUFFER_SECONDS = 3.0`; past
  that it logs "no onset within 3s of audio; not reporting". Warning in the model docstring: `ttfb`
  is mirrored inside `TTFAMetricsData`, "it is not a separate measurement, so don't aggregate both".
- **Where the numbers surface.** `UserBotLatencyObserver` measures
  `VADUserStoppedSpeakingFrame → BotStartedSpeakingFrame` and emits `on_latency_measured(seconds)`
  plus `on_latency_breakdown(LatencyBreakdown)`. `LatencyBreakdown` fields:
  `ttfb: list[TTFBBreakdownMetrics(processor, model, start_time, duration_secs)]`,
  `text_aggregation: TextAggregationBreakdownMetrics | None`, `user_turn_start_time`,
  `user_turn_secs` ("includes VAD silence detection, STT finalization, and any turn analyzer wait"),
  `function_calls: list[FunctionCallMetrics(function_name, start_time, duration_secs)]`, and a
  `chronological_events()` helper that sorts every sub-metric by `start_time`. Accumulators are
  cleared on `InterruptionFrame` so "stale metrics from cancelled LLM/TTS cycles" are discarded.
  A separate one-shot `on_first_bot_speech_latency` measures `ClientConnectedFrame → first
  BotStartedSpeakingFrame` (i.e. greeting cold start), abandoned if the user speaks first.
- **No perceptual target numbers exist in this repo.** `README.md` says only "Ultra-low latency
  interaction"; `AGENTS.md`, `CLAUDE.md` (which is just `@AGENTS.md`) and `docs/` (Sphinx config
  only) state none. The one latency constant in the eval harness is a *test* timeout, not a target:
  `evals/harness.py:113 DEFAULT_EVENT_TIMEOUT_MS = 60000`, explicitly "generous … rather than
  failing on latency", with per-expectation `within_ms` in scenario YAML.
  **External knowledge, not a repo claim:** the commonly cited conversational targets — roughly
  ~200 ms as the natural human turn-taking gap and ~800 ms voice-to-voice as the "feels live"
  ceiling — come from the wider voice-UX literature and vendor marketing, not from pipecat.
  Do not attribute them to this source.
- **Migration angle:** boson-agent measures none of this. Its only latency-ish quantity is
  `elapsed_ms` since the agent started streaming, threaded from `bootstrap.py:222`
  (`(_time.monotonic() - started_at) * 1000`) into `InterruptHandler.check_barge_in(content, policy,
  elapsed_ms)` and `PartialDetector.is_likely_partial_by_timing(elapsed_ms)` — a barge-in heuristic,
  not a budget. Because boson has no server-side STT/TTS/VAD, the migration *adds* stages 1, 2 and 5
  to the budget: today's boson turn starts at a client-delivered text partial, so the entire
  `stop_secs + TTFS_p99` term (0.55 s–2.3 s depending on provider) is new cost that must be
  reclaimed elsewhere. The rule/layer pipeline (`gateway/layers/`, `gateway/rules/`) sits serially
  between transcript and LLM and is currently untimed — it should be wrapped in a processor that
  calls `start_processing_metrics()`/`stop_processing_metrics()` so it appears in
  `LatencyBreakdown`. Korean STT on 8 kHz telephony audio has **no** entry in `stt_latency.py`; the
  Lina TMR provider's TTFS must be benchmarked and passed as `ttfs_p99_latency=` before any budget
  claim is credible. Untouched: boson's compact/summarization path runs off-turn and does not enter
  this budget, provided it stays off the critical path.

## Citation
pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (release 1.7.0, 2026-08-01),
read 2026-08-25. Paths: `src/pipecat/services/stt_latency.py`, `src/pipecat/services/stt_service.py`,
`src/pipecat/observers/user_bot_latency_observer.py`,
`src/pipecat/processors/metrics/frame_processor_metrics.py`,
`src/pipecat/turns/user_stop/`, `src/pipecat/audio/vad/vad_analyzer.py`, `src/pipecat/evals/harness.py`.
boson-agent read-only at `/Users/jaewon/mywork_2026/Lina_2026/boson-agent-dev/boson-agent`.

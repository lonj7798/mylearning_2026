# Endpointing: What Actually Decides the User Stopped Talking

<!-- slug: endpointing-turn-boundary · type: module · source: src/pipecat/turns/ + src/pipecat/audio/turn/base_turn_analyzer.py -->

**Core Insight.** In Pipecat "the user stopped talking" is not a VAD event — it is a *negotiated
verdict* from a chain of pluggable stop strategies, each of which can veto. VAD silence is one
input; the default strategy runs an ONNX end-of-turn model over the audio, then still waits for the
STT to confirm it has nothing more to send. The boundary is deliberately decoupled from every
component that could claim to own it (VAD, STT, LLM), so any can be swapped without redefining a turn.

**Guideline.** Choose the stop strategy by asking *what evidence you trust*: VAD-only
(`SpeechTimeoutUserTurnStopStrategy`) for cheap CPU and predictable latency; a turn analyzer
(`TurnAnalyzerUserTurnStopStrategy`, the default) when trailing-off speech costs you false cut-ins;
`ExternalUserTurnStopStrategy` when the STT provider already does server-side endpointing. Never tune
`VADParams.stop_secs` alone — the built-in STT P99 latencies assume `stop_secs=0.2`, and the code warns you.

## Technical Details

- **Container.** `UserTurnStrategies` (`turns/user_turn_strategies.py` L54): `start:
  list[BaseUserTurnStartStrategy] | None`, `stop: list[BaseUserTurnStopStrategy] | None`. Defaults
  (L27, L43): `start = [VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()]`,
  `stop = [TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]`.
  **Wiring point:** `LLMUserAggregatorParams` (`processors/aggregators/llm_response_universal.py`
  L120) exposes `user_turn_strategies`, `user_mute_strategies`, `user_turn_stop_timeout: float = 5.0`,
  `user_idle_timeout: float = 0`, `vad_analyzer`, `audio_idle_timeout: float = 1.0`.
- **Start strategies** (`turns/user_start/`): `VADUserTurnStartStrategy` (35 L, fires on
  `VADUserStartedSpeakingFrame`, returns `ProcessFrameResult.STOP`);
  `TranscriptionUserTurnStartStrategy(use_interim=True)` — the soft-speaker fallback when VAD misses;
  `MinWordsUserTurnStartStrategy(min_words, use_interim=True)` — requires `min_words` **only while
  the bot is speaking**, drops to 1 word otherwise (L108);
  `WakePhraseUserTurnStartStrategy(phrases, timeout=10.0, single_activation=False)`;
  `ExternalUserTurnStartStrategy(enable_interruptions=True)`; `KrispVivaIP*`.
  **Stop strategies** (`turns/user_stop/__init__.py`): `SpeechTimeout*`, `TurnAnalyzer*`, `External*`,
  `ExternalUserTurnCompletionStopStrategy`, `LLMTurnCompletion*`, `DeferredUserTurnStopStrategy` + `deferred()`.

### VAD-only: `SpeechTimeoutUserTurnStopStrategy` (328 L)

`__init__(*, user_speech_timeout: float = 0.6, wait_for_transcript: bool = True)`. Two independent
timers armed at `VADUserStoppedSpeakingFrame`; **both** must finish and (unless
`wait_for_transcript=False`) ≥1 transcript must exist. `user_speech_timeout` is the policy floor —
the grace window in which the user may resume. `stt_timeout` is the safety net:
`effective_stt_wait = max(0.0, self._stt_timeout - self._stop_secs)` (L223), where `_stt_timeout`
comes from `STTMetadataFrame.ttfs_p99_latency` (L160), short-circuited when
`TranscriptionFrame.finalized` arrives (L236-244). An `InterimTranscriptionFrame` clears
`_transcript_finalized` (L181): more audio is in flight, so an earlier finalize no longer covers the utterance.

### Semantic: `TurnAnalyzerUserTurnStopStrategy` (364 L) + `BaseTurnAnalyzer`

`__init__(*, turn_analyzer: BaseTurnAnalyzer, wait_for_transcript: bool = True)`. `BaseTurnAnalyzer`
(`audio/turn/base_turn_analyzer.py`, 137 L) is a small ABC: `speech_triggered`, `params`,
`append_audio(buffer: bytes, is_speech: bool) -> EndOfTurnState`, `async analyze_end_of_turn() ->
tuple[EndOfTurnState, MetricsData | None]`, `clear()`. `EndOfTurnState` is `COMPLETE = 1` /
`INCOMPLETE = 2`. Two analyzer shapes in one strategy: **batch** analyzers (`BaseSmartTurn`) return
`COMPLETE` from `append_audio` only on silence timeout and are otherwise queried at each VAD stop;
**streaming** analyzers (`KrispVivaTurn`) decide frame-by-frame inside `append_audio` (L205-216).

`SmartTurnParams` (`smart_turn/base_smart_turn.py`): `stop_secs = 3`, `pre_speech_ms = 500`,
`max_duration_secs = 8`. **Note the asymmetry** — the analyzer's own silence fallback is 3 s, but the
model is *consulted* at every VAD stop (0.2 s), so the ML verdict is the normal path.
`LocalSmartTurnAnalyzerV3` loads the bundled `smart-turn-v3.2-cpu.onnx` via `onnxruntime`
(`cpu_count: int = 1`), computes Whisper log-mel features, resamples to `_MODEL_SAMPLE_RATE = 16000`
with `soxr` — it is **audio-only** and never reads transcript text. The STT wait is anchored to an
absolute deadline so inference time does not eat it:
`stt_deadline = frame.timestamp - frame.stop_secs + self._stt_timeout` (L236).

### Controller, veto, and surrounding layers

- `UserTurnController(*, user_turn_strategies, user_turn_stop_timeout: float = 5.0)`
  (`user_turn_controller.py` L36) runs start then stop strategies in list order, breaking on
  `ProcessFrameResult.STOP` (`turns/types.py`). `_trigger_user_turn_stop` (L354) drops a second stop
  for a closed turn (L358) and **refuses to finalize while `self._user_speaking`** (L367) — a late
  LLM ✓ that resolves after the user resumed is stale. A watchdog
  (`_user_turn_stop_timeout_task_handler`, L385) force-stops a stuck turn.
- `UserTurnProcessor(FrameProcessor)` (`user_turn_processor.py` L32, `__init__(*,
  user_turn_strategies=None, user_turn_stop_timeout=5.0, user_idle_timeout=0)`) broadcasts
  `UserStartedSpeakingFrame` (L205) / `UserStoppedSpeakingFrame` (L223) and calls
  `broadcast_interruption()` (L210). A `ProposedUserStartedSpeakingFrame` is *consumed*, not
  forwarded, when a local strategy resolves it (L162-170), so no downstream resolver decides the same
  turn twice. `UserIdleController(user_idle_timeout=0)` and the mute strategies
  (`BaseUserMuteStrategy.process_frame(frame) -> bool`; `Always`, `FirstSpeech`, `FunctionCall`,
  `MuteUntilFirstBotComplete`) sit alongside; mute drops audio *before* the STT sees it
  (`stt_service.py` L437-438).
- LLM-gated: `FilterIncompleteUserTurnStrategies` wraps detectors in `deferred(...)` and appends
  `LLMTurnCompletionUserTurnStopStrategy`. Markers (`user_turn_completion_mixin.py` L38-40): `✓`
  complete, `○` incomplete-short, `◐` incomplete-long;
  `UserTurnCompletionConfig(incomplete_short_timeout=5.0, incomplete_long_timeout=10.0)`. Only `✓`
  finalizes. VAD defaults (`audio/vad/vad_analyzer.py` L25-28): `confidence 0.7`, `start_secs 0.2`,
  `stop_secs 0.2`, `min_volume 0.6`; both timing strategies warn if `stop_secs != 0.2` or
  `stop_secs >= stt_timeout` (safety net collapses to 0 s).
- **Expected but absent:** no `endpointing.py`, no `EndpointingConfig`, no single "turn detector"
  class. "Endpointing" appears only in provider settings (Deepgram's `endpointing`, Speechmatics'
  `TurnDetectionMode`). Also absent: any *text*-based semantic turn model — smart-turn-v3 is
  acoustic, so it carries no Korean-language risk but gets no help from Korean sentence-final endings.
- **Migration angle:** boson-agent has **no equivalent** — turns arrive pre-cut as `user_message` /
  `partial_transcript` JSON (`gateway/server/protocol.py` L32-37). Its two analogues are text
  heuristics: `interrupt/detector.py` `PartialDetector(overlap_chars=10, timing_threshold_ms=1000,
  silence_timeout_ms=2000)`, whose `should_finalize(elapsed_since_last_ms)` is a **2000 ms
  text-silence timer** — the entire boson end-of-turn mechanism; and `interrupt/policy.py`
  `BargeInPolicy` (`AlwaysPolicy`, `DurationPolicy(min_ms=500)`, `WordFilterPolicy`), a turn-*start*
  decision made on text. Mapping: `BargeInPolicy` → start strategies (`WordFilterPolicy` ≈
  `MinWordsUserTurnStartStrategy`); `PartialDetector.should_finalize` → stop strategies. The headline
  is quantitative: 2000 ms of text silence becomes `0.6 s` + smart-turn-v3, roughly 1.2–1.4 s less
  dead air per turn. `interrupt/handler.py` (151 L) and `interrupt/cancellation.py` (187 L) are
  superseded by `broadcast_interruption()`; `interrupt/fillers.py` (40 L) has no Pipecat equivalent
  and needs a custom processor.

## Citation

pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (2026-08-25); all of
`src/pipecat/turns/`, plus `audio/turn/base_turn_analyzer.py`,
`audio/turn/smart_turn/base_smart_turn.py`, `audio/vad/vad_analyzer.py`. boson-agent read-only at
`/Users/jaewon/mywork_2026/Lina_2026/boson-agent-dev/boson-agent`. Read 2026-08-25.

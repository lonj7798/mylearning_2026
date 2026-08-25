# RTVI, Observers and Metrics — What Pipecat Instruments Out of the Box
<!-- slug: rtvi-observability · type: source · source: src/pipecat/observers/, src/pipecat/metrics/metrics.py, src/pipecat/processors/metrics/, src/pipecat/processors/frameworks/rtvi/, src/pipecat/utils/tracing/ -->

**Core Insight.** Pipecat's instrumentation is not a logging convention bolted on top — it is a
second, read-only plane over the frame graph. `BaseObserver` sees every frame transfer without
being in the pipeline, services emit typed `MetricsData` as `MetricsFrame`s, and the same
`MetricsFrame` stream feeds console logs, OpenTelemetry spans, Sentry transactions, and the
wire protocol the client speaks. You instrument by *subscribing*, not by editing processors.

**Guideline.** Turn on `PipelineParams(enable_metrics=True, enable_usage_metrics=True)` from day
one and read latency off `UserBotLatencyObserver.on_latency_breakdown` — a typed per-service
breakdown of one user→bot cycle. Never hand-roll timing inside a processor: the metric classes
already exist and every downstream consumer keys off them.

## Technical Details
- **Observer contract** (`observers/base_observer.py:90`): `BaseObserver(BaseObject)` with four
  hooks — `on_process_frame(FrameProcessed)`, `on_push_frame(FramePushed)`,
  `on_processor_setup(ProcessorSetUp)`, `on_pipeline_started()`. Event dataclasses:
  `FramePushed(source, destination, frame, direction, timestamp)`,
  `FrameProcessed(processor, frame, direction, timestamp)`,
  `ProcessorSetUp(processor, started_at_ns, finished_at_ns)`. `timestamp` is pipeline-clock ns.
- **Metric classes** (`metrics/metrics.py`), all Pydantic, base `MetricsData(processor, model)`:
  `TTFBMetricsData(value)`, `TTFAMetricsData(ttfa, ttfb, leading_silence)` — time-to-first-*audible*-
  audio, `TTFATMetricsData(ttfat, ttfb, thinking_time)` — time to first *answer* token (reasoning
  excluded), `ProcessingMetricsData(value)`, `TextAggregationMetricsData(value)` — "time from the
  first LLM token to the first complete sentence", `LLMUsageMetricsData(value: LLMTokenUsage)`,
  `STTUsageMetricsData(value: STTUsage(audio_seconds))`, `TTSUsageMetricsData(value: int chars)`,
  `TurnMetricsData(is_complete, probability, e2e_processing_time_ms)`. `SmartTurnMetricsData` is
  `@deprecated` since 0.0.104, removal 2.0.0.
- **Who emits them**: `FrameProcessorMetrics` (`processors/metrics/frame_processor_metrics.py:31`)
  owns the timers — `start_ttfb_metrics`, `stop_ttfb_metrics`, `cancel_ttfb_metrics`,
  `process_ttfa_metrics`, `stop_ttfat_metrics`, `start/stop_processing_metrics`,
  `start_llm_usage_metrics`, `start_stt_usage_metrics`, `start_tts_usage_metrics`,
  `start/stop_text_aggregation_metrics`. Each returns a `MetricsFrame`. `FrameProcessor` wraps them
  (`frame_processor.py:504-573`) and gates on `can_generate_metrics() and self.metrics_enabled` —
  `can_generate_metrics()` returns `False` on the base class, so **only services that opt in
  report**. Services call the wrappers, e.g. `openai/base_llm.py:445` `start_ttfb_metrics()`,
  `:504` `stop_ttfb_metrics()`, `:513` `stop_ttfat_metrics()` on a tool-call delta.
- **Bundled observers** (`observers/`): `TurnTrackingObserver(max_frames=100,
  turn_end_timeout_secs=2.5)` — turn 1 starts at `StartFrame`, later turns at
  `UserStartedSpeakingFrame`, ends on `BotStoppedSpeakingFrame` + 2.5 s timer or barge-in; events
  `on_turn_started(n)` / `on_turn_ended(n, duration, was_interrupted)`.
  `UserBotLatencyObserver` — events `on_latency_measured(seconds)`,
  `on_latency_breakdown(LatencyBreakdown)`, `on_first_bot_speech_latency(seconds)`.
  `StartupTimingObserver(processor_types=None)` — events `on_startup_timing_report`
  (`StartupTimingReport(start_time, total_duration_secs, processor_timings)` where each
  `ProcessorStartupTiming(processor_name, start_offset_secs, duration_secs, setup_duration_secs)`)
  and `on_transport_timing_report` (`TransportTimingReport(bot_connected_secs,
  client_connected_secs)`). `observers/loggers/` holds `MetricsLogObserver(include_metrics=...)`,
  `LLMLogObserver`, `DebugLogObserver(frame_types={Frame: (Processor, FrameEndpoint)})`,
  `TranscriptionLogObserver`.
- **Auto-wiring in `PipelineWorker.__init__`** (`pipeline/worker.py:422-482`): `enable_turn_tracking=True`
  by default appends a `TurnTrackingObserver`; `enable_tracing=True` additionally creates a
  `UserBotLatencyObserver` **and** a `TurnTraceObserver`; `enable_rtvi=True` (default) constructs
  `RTVIProcessor()` and appends `rtvi.create_rtvi_observer(params=rtvi_observer_params)`. So RTVI
  observability is on unless you turn it off.
- **RTVI** (`processors/frameworks/rtvi/`): `RTVIProcessor` (processor.py:50) handles inbound client
  messages; `RTVIObserver` (observer.py:197) converts frames to outbound messages.
  `RTVIObserver._handle_metrics(MetricsFrame)` (observer.py:839) buckets metrics into a dict with
  keys `"ttfb"`, `"ttfa"`, `"ttfat"`, `"processing"`, `"tokens"`, `"stt_usage"`, `"characters"` and
  sends `RTVI.MetricsMessage(type="metrics", data=metrics)`. `RTVIObserverParams.metrics_enabled`
  defaults to `True`; `audio_level_period_secs: float = 0.15`.
- **OpenTelemetry** (`utils/tracing/`, not in the brief but present): `setup_tracing(service_name,
  exporter, console_export)`, `is_tracing_available()`, `TurnTraceObserver` creates one span per
  turn under a conversation span and sets `turn.user_bot_latency_seconds`; service spans come from
  `@traced_llm`, `@traced_stt`, `@traced_tts`, `traced_gemini_live(op)`, `traced_openai_realtime(op)`.
- **Sentry**: `processors/metrics/sentry.py:26` `SentryMetrics(FrameProcessorMetrics)` — pass per
  service as `metrics=SentryMetrics()` (see `examples/observability/observability-sentry-metrics.py:69,77,82`).
- **`examples/observability/`** holds exactly three files: `observability-observer.py` (193 L, custom
  `BaseObserver` + `LLMLogObserver` + `DebugLogObserver`, `enable_metrics=True`),
  `observability-sentry-metrics.py` (146 L), `observability-heartbeats.py` (57 L,
  `PipelineParams(enable_heartbeats=True)` + `@worker.event_handler("on_heartbeat_timeout")`).
  There is **no** dashboard, exporter, or metrics-aggregation example in the repo — collection is
  provided, aggregation is left to you.
- Heartbeat/idle knobs: `HEARTBEAT_SECS = 1.0`, `HEARTBEAT_MONITOR_SECS = 10.0`,
  `IDLE_TIMEOUT_SECS = 300` (`pipeline/worker.py:91-100`).
- **Migration angle:** boson-agent has essentially no instrumentation plane. The only timing code is
  `packages/gateway/gateway/debug/log_decorator.py` (`time.perf_counter()` around a call, printed as
  `[TRACE …] EXIT (…ms)`) and the ad-hoc `elapsed_ms` threaded into barge-in policy
  (`core.py:166 should_interrupt(session_id, content, elapsed_ms)`,
  `interrupt/detector.py:53 is_likely_partial_by_timing`). Migrating means **deleting** the trace
  decorator's role as a latency tool and adopting `UserBotLatencyObserver` +
  `MetricsLogObserver`; boson's `elapsed_ms` barge-in input maps onto the observer's
  `on_latency_breakdown` / `BotStartedSpeakingFrame` timeline rather than a hand-rolled clock.
  `gateway/server/protocol.py` — boson's own client wire format — collides directly with RTVI:
  RTVI already carries transcripts, bot-speaking, interruption, function-call and metrics events, so
  either boson's protocol becomes an `RTVIObserverParams`-configured projection or the two coexist
  and duplicate. `gateway/layers/status.py` (`AgentStatusTracker`, generating/settling/idle) is the
  boson analogue of `TurnTrackingObserver` and is a replace candidate. Untouched: boson's
  rules/layers engines emit no metrics today, so instrumenting them requires new
  `FrameProcessor.start_processing_metrics()` calls in whatever processor wraps them.

## Citation
pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (latest release in
`CHANGELOG.md`: 1.7.0, 2026-08-01), read 2026-08-25.
Paths: `src/pipecat/observers/`, `src/pipecat/metrics/metrics.py`,
`src/pipecat/processors/metrics/`, `src/pipecat/processors/frameworks/rtvi/`,
`src/pipecat/utils/tracing/`, `examples/observability/`.
boson-agent read-only at `/Users/jaewon/mywork_2026/Lina_2026/boson-agent-dev/boson-agent`.

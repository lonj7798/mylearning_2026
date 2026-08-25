# PipelineTask / PipelineRunner are now PipelineWorker / WorkerRunner
<!-- slug: pipeline-task-runner · type: source · source: src/pipecat/pipeline/worker.py + src/pipecat/workers/runner.py -->

**Core Insight.** The two files the course outline points at are empty shims. As of 1.3.0 Pipecat
renamed its runnable unit from "task" to "worker": the real implementation is `PipelineWorker`
in `src/pipecat/pipeline/worker.py` (1,506 L) and `WorkerRunner` in
`src/pipecat/workers/runner.py` (550 L). `PipelineTask` and `PipelineRunner` still exist purely as
`@deprecated` subclasses scheduled for removal in 2.0.0. The worker owns *lifecycle*
(start/end/cancel/idle/heartbeat); the runner owns *process concerns* (signals, the bus, ending
when every worker is done).

**Guideline.** Write new code against `PipelineWorker` + `WorkerRunner`, and drive shutdown through
exactly one path: `stop_when_done()` to drain, `cancel()` to abandon. Never `asyncio.cancel` a
pipeline from outside — the worker has its own cooperative cancel that waits for the `CancelFrame`
to traverse the pipeline.

## Technical Details
- **Where the shims point.** `src/pipecat/pipeline/task.py` (29 L) is a pure re-export:
  `from pipecat.pipeline.worker import (IdleFrameObserver, PipelineParams, PipelineTask,
  PipelineTaskParams, PipelineWorker)`. `src/pipecat/pipeline/runner.py` (37 L) does
  `from pipecat.workers.runner import WorkerRunner` and defines
  `class PipelineRunner(WorkerRunner): pass` (L27–37) under
  `@deprecated("... deprecated since 1.3.0 and will be removed in 2.0.0. Use `WorkerRunner` instead.")`.
  In `worker.py`, `class PipelineTask(PipelineWorker)` is at L1482 and
  `class PipelineTaskParams(WorkerParams)` at L1498 — both deprecated 1.3.0.
- **Params.** `PipelineParams(BaseModel)` (worker.py L163–195) carries what is broadcast to
  processors via `StartFrame`: `audio_in_sample_rate=16000`, `audio_out_sample_rate=24000`,
  `enable_heartbeats=False`, `enable_metrics=False`, `enable_usage_metrics=False`,
  `heartbeats_period_secs=HEARTBEAT_SECS`, `heartbeats_monitor_secs=HEARTBEAT_MONITOR_SECS`,
  `report_only_initial_ttfb=False`, `send_initial_empty_metrics=True`, `start_metadata={}`.
  Module constants (L91–100): `HEARTBEAT_SECS = 1.0`, `HEARTBEAT_MONITOR_SECS = 10.0`,
  `IDLE_TIMEOUT_SECS = 300`, `CANCEL_TIMEOUT_SECS = 20.0`, `SETUP_TIMEOUT_SECS = 20.0`,
  `START_TIMEOUT_SECS = 20.0`. Everything else is a **constructor kwarg**, not a param field —
  `PipelineWorker.__init__` (L273–303) is keyword-only after `pipeline` and takes `active`,
  `app_resources`, `bridged`, `cancel_on_idle_timeout=True`, `cancel_runner_on_idle_timeout=True`,
  `cancel_timeout_secs`, `clock`, `conversation_id`, `enable_tracing`, `enable_turn_tracking=True`,
  `enable_rtvi=True`, `idle_timeout_frames=(BotSpeakingFrame, UserSpeakingFrame)`,
  `idle_timeout_secs`, `name`, `observers`, `processor_unusable_policy`, `params`,
  `rtvi_processor`, `setup_timeout_secs`, `start_timeout_secs`, `task_manager`.
- **What the constructor builds.** The user pipeline is re-wrapped:
  `self._pipeline = Pipeline(processors, source=source, sink=self._sink)` (L549) where `processors`
  is `[self._rtvi, pipeline]` when RTVI was auto-created, else `[pipeline]` (L548). With
  `bridged is not None` a further `Pipeline([edge_source, pipeline, edge_sink])` wrap adds bus
  edges (L522–537). So a user's `Pipeline` sits **two or three envelopes deep** at runtime.
- **Lifecycle** (`run(params: WorkerParams)`, L748–791): `_setup_within_timeout` (bounded by
  `setup_timeout_secs`, L1104–1121; on timeout fires `on_setup_timeout` and cleans up without ever
  pushing a frame) → `_create_tasks()` → `_wait_for_pipeline_finished()`. The engine is
  `_process_push_queue` (L1205–1246): it constructs the `StartFrame`, queues it, then
  `_wait_for_pipeline_start` blocks on `_pipeline_start_event` with `start_timeout_secs`; on
  timeout it logs "being blocked somewhere?", fires `on_pipeline_timeout` and refuses to run. Then
  the loop `frame = await self._push_queue.get()` → `self._pipeline.queue_frame(frame)`, and on
  `CancelFrame | EndFrame | StopFrame` it calls `_wait_for_pipeline_end(frame)` and exits.
- **queue_frame** (L793–808) is direction-split:
  `DOWNSTREAM → await self._push_queue.put(frame)` (enters at the head),
  otherwise `await self._sink.queue_frame(frame, direction)` (enters at the tail).
  `queue_frames` (L810–829) accepts `Iterable | AsyncIterable`.
  `flush_pipeline(timeout: float = 5.0) -> bool` (L831–855) injects a `PipelineFlushFrame` whose
  round-trip (sink bounces it upstream, source sets its event) proves the pipeline drained;
  returns `False` on timeout instead of hanging.
- **Shutdown paths.** `stop_when_done()` (L730–737) = `queue_frame(EndFrame())`.
  `cancel(*, reason=None)` (L739–746) → `_cancel` (L973–984), which sets `_cancelled`, unblocks
  `_pipeline_start_event`, and queues `CancelFrame(reason=reason)`. The asymmetry is explicit in
  `_wait_for_pipeline_end` (L1063–1091): a `CancelFrame` wait is bounded by `cancel_timeout_secs`
  (20 s) and degrades to `on_pipeline_timeout`; an `EndFrame` wait is **unbounded** — "Ending
  flushes what is queued, so cutting the wait short would drop the audio the EndFrame exists to
  play out." Upstream `EndWorkerFrame` / `CancelWorkerFrame` / `StopWorkerFrame` /
  `InterruptionWorkerFrame` are translated in `_source_push_frame` (L1268–1286); the interruption
  case deliberately bypasses `_push_queue` and goes straight to `self._pipeline.queue_frame`
  because the push task may be blocked awaiting an end frame.
- **Idle handling.** `IdleFrameObserver` (L106–140) sets an `asyncio.Event` on `StartFrame` or any
  of `idle_timeout_frames`. `_idle_monitor_handler` (L1401–1415) waits on that event with
  `timeout=idle_timeout_secs`; on `TimeoutError` → `_idle_timeout_detected` (L1417–1441): fires
  `on_idle_timeout`, and if `cancel_on_idle_timeout` cancels the worker, and if
  `cancel_runner_on_idle_timeout` also calls `BaseWorker.cancel(self, ...)` so a `BusCancelMessage`
  brings down every other root worker. Default idle timeout is 300 s.
- **Error policy.** `ProcessorUnusablePolicy` (L143–160): `CONTINUE` (default) / `END` / `CANCEL`,
  applied once per processor in `_handle_unusable_processor` (L1299–1318).
- **WorkerRunner** (`workers/runner.py` L83): `__init__` (L109–171) takes `name`, `bus` (defaults to
  in-process `AsyncQueueBus`), `handle_sigint=True`, `handle_sigterm=False`, `force_gc=False`,
  `check_dangling_tasks=True`, `task_manager` (`loop` deprecated 1.5.0). `add_workers(*workers)`
  (L199–235) attaches each worker to bus+registry and starts it. `run(worker=None, *, auto_end=True)`
  (L237–319) awaits `_shutdown_event`; with `auto_end=True` the runner ends when every root worker
  finishes, then cancels stragglers, cleans up, and stops the bus. `end(reason)` (L332–348) sends
  `BusEndWorkerMessage` per root worker; `cancel(reason)` (L350–366) sends `BusCancelWorkerMessage`;
  both are idempotent on `_shutdown_event`. SIGINT/SIGTERM route to `_sig_cancel` →
  `cancel(reason="interrupt signal")` (L529–537). Passing a worker to `run()` is deprecated 1.3.0.
- **Migration angle:** `PipelineWorker` + `WorkerRunner` together replace boson-agent's
  `gateway/server/websocket.py` `GatewayWebSocketServer` (734 L) as the session lifecycle owner —
  specifically `start()`/`start_background()`/`stop()` (L162–213), `_handle_connection` (L215),
  `_teardown_connection_sessions` (L381), `_replace_active_task` (L527),
  `_cancel_session_dispatch` (L555), `_cancel_active_task` (L584) and the silence timer
  (`_on_silence` L627, `_cancel_silence_timer_and_wait` L643). The silence timer maps cleanly onto
  `idle_timeout_secs` + `on_idle_timeout`; per-session task replacement maps onto
  `cancel()`/`stop_when_done()`. **Collision:** boson currently runs one asyncio task per session
  inside one server process, whereas the Pipecat model is one `PipelineWorker` per call attached to
  a `WorkerRunner`; multi-session hosting requires `auto_end=False` (documented at runner.py
  L263–266 for exactly the FastAPI case) or one runner per call. Untouched: `basement/loop`
  and `basement/tools` — worker lifecycle sits strictly above them.

## Citation
pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (v1.7.0 line, changelog head
dated 2026-08-01), read 2026-08-25. Repo paths: `src/pipecat/pipeline/worker.py`,
`src/pipecat/workers/runner.py`; shims at `src/pipecat/pipeline/task.py`,
`src/pipecat/pipeline/runner.py`.

# Process Model, Concurrency and Cold Start — What Pipecat Actually Ships for Deployment
<!-- slug: deployment-scaling · type: source · source: src/pipecat/runner/, src/pipecat/workers/, src/pipecat/cli/, src/pipecat/cli/templates/server/ -->

**Core Insight.** Pipecat ships a *development* runner, not a production orchestrator. Its unit of
isolation is the **worker** (`BaseWorker` / `PipelineWorker` under a `WorkerRunner`), not the OS
process; the bundled runner multiplexes every session as an `asyncio.Task` inside one
`uvicorn.run(app, ...)` process. Scale-out is delegated — to a container platform (the generated
`Dockerfile` + `pcc-deploy.toml` target Pipecat Cloud) or to a network bus (`RedisBus` / `PgmqBus`).

**Guideline.** Do not treat `pipecat.runner.run.main()` as your production entrypoint. Decide
explicitly whether a session is a process (container-per-session, the Pipecat Cloud model) or a task
(one server, N concurrent pipelines), and measure cold start with `StartupTimingObserver` first —
connection setup happens in each processor's `setup()` and is the whole of your first-audio delay.

## Technical Details
- **The runner is labelled as dev.** `runner/run.py` docstring: "This development runner executes
  Pipecat bots and provides the supporting infrastructure they need"; `_print_dev_runner_banner()`
  prints `ᓚᘏᗢ PIPECAT DEVELOPMENT RUNNER` and links `docs.pipecat.ai/pipecat/deployment/overview`.
  There are **no deployment docs in the repo** — `docs/` is Sphinx scaffolding only.
- **Bot entrypoint contract.** "All bots must implement a `bot(runner_args)` async function as the
  entry point." Session args are `RunnerArguments` subclasses (`runner/types.py:140`):
  `DailyRunnerArguments(room_url, token)`, `WebSocketRunnerArguments(websocket, transport_type)`,
  `SmallWebRTCRunnerArguments(webrtc_connection)`, `LiveKitRunnerArguments(room_name, url, token)`,
  `MOQRunnerArguments`, `VonageRunnerArguments`, `EvalRunnerArguments(host="localhost", port=7860)`.
  `__post_init__` defaults `handle_sigint=False`, `handle_sigterm=False`,
  `pipeline_idle_timeout_secs=300`; every session carries a `session_id`.
- **How a session starts — one process, one event loop.** `main()` ends at
  `uvicorn.run(app, host=args.host, port=args.port)` (`run.py:1999`) — no `workers=`, no reload, so
  a single process. Each transport route mints `session_id = str(uuid.uuid4())` and calls
  `_start_bot_session(bot_module.bot(runner_args))` (`run.py:821, 845, 909, 1284, 1392`), which is:

      def _start_bot_session(coro) -> asyncio.Task:      # run.py:215
          """Run a bot in the background, holding a reference until it finishes."""
          task = asyncio.create_task(coro)
          _bot_sessions.add(task)                        # module-level set, run.py:212
          task.add_done_callback(_bot_sessions.discard)
          return task

  Its comment names the failure it prevents: "the event loop only holds a weak reference to a task,
  so one that nothing else references can be collected while it is still running." WebRTC differs —
  `POST /start` only registers `active_sessions[session_id] = body`, and the bot launches later from
  the offer handler via `background_tasks.add_task(bot_module.bot, runner_args)` (`run.py:1002-1023`).
  So **concurrency = concurrent asyncio tasks on one loop**: no process pool, no worker count, no
  admission control, no per-session CPU isolation anywhere in `runner/`. `POST /start` (`run.py:643`)
  returns `StartBotResult(sessionId, iceConfig, dailyRoom, dailyToken, url, wsUrl, token, moq)`, and
  a `/sessions/{session_id}/{path}` route exists to "Mimic Pipecat Cloud's proxy".
- **The real runtime unit: workers** (`src/pipecat/workers/`). `BaseWorker` (1565 L) owns
  activation, end/cancel, bus subscription and job RPC; `PipelineWorker` wraps a user pipeline
  (`PipelineTask` is a deprecated 1.3.0 alias). `WorkerRunner` (`workers/runner.py:83`) owns the
  shared `WorkerBus` + `WorkerRegistry` and SIGINT/SIGTERM handling: `WorkerRunner(name, bus,
  handle_sigint=True, handle_sigterm=False, force_gc=False, check_dangling_tasks=True, loop,
  task_manager)`, then `add_workers(*workers)` / `run()`. Scaling note from its docstring:
  `auto_end=True` ends the runner when every root worker finishes — right for one-shot bot
  processes — while "long-lived hosts that add and remove workers over many sessions (e.g. a FastAPI
  server), pass `auto_end=False`". `PipelineRunner` and `run(worker)` are deprecated (1.3.0).
- **Cross-process scale-out is the bus, not the runner.** `bus/local/async_queue.py`
  (`AsyncQueueBus`, the in-process default) vs `bus/network/{redis.py, pgmq.py, pgmq_backends.py}`
  (`RedisBus`, `PgmqBus`). `examples/multi-worker/README.md`: "Distributed bus — Same patterns, but
  workers run in separate processes (or machines)", runnable as `distributed-handoff/redis-handoff/`
  and `.../pgmq-handoff/`. `workers/proxy/websocket/` (`WebSocketProxyServer(BaseWorker)`) gives
  point-to-point forwarding with "No shared bus required".
- **Lifecycle safety valves** (`pipeline/worker.py:91-100`): `IDLE_TIMEOUT_SECS = 300`,
  `CANCEL_TIMEOUT_SECS = 20.0`, `SETUP_TIMEOUT_SECS = 20.0`, `START_TIMEOUT_SECS = 20.0`,
  `HEARTBEAT_SECS = 1.0`, `HEARTBEAT_MONITOR_SECS = 10.0`, plus `PipelineWorker(...,
  idle_timeout_frames=(BotSpeakingFrame, UserSpeakingFrame), cancel_on_idle_timeout=True,
  cancel_runner_on_idle_timeout=True, processor_unusable_policy=ProcessorUnusablePolicy.CONTINUE)`.
  These are the framework's only cost controls: an abandoned call self-terminates after 5 min of
  no speech.
- **Cold start is `setup()`, not `StartFrame`.** `StartupTimingObserver.setup()` starts its clock
  "before any processor has been set up"; `ProcessorStartupTiming` splits `setup_duration_secs`
  (connect, auth, model load) out of `duration_secs`; `StartupTimingReport.total_duration_secs` is
  a span, not a sum — "Processors are set up concurrently". `TransportTimingReport` adds
  `bot_connected_secs` (SFU only) and `client_connected_secs`; end-to-end greeting cold start is
  `UserBotLatencyObserver.on_first_bot_speech_latency`.
- **Packaging**: `cli/main.py` registers exactly two commands — `init` and `eval`. `pipecat init`
  scaffolds from `cli/templates/server/`: `bot_cascade.py.jinja2` / `bot_realtime.py.jinja2`,
  `Dockerfile.jinja2` (`FROM dailyco/pipecat-base:latest`, `uv sync --locked --no-install-project
  --no-dev`, then `COPY ./bot.py bot.py`) and `pcc-deploy.toml.jinja2`, whose entire contents are
  `agent_name`, `secret_set`, `agent_profile = "agent-2x"` (when video in/out, else `"agent-1x"`),
  an optional `[krisp_viva] audio_filter = "tel"`, and `[scaling] min_agents = 1`. That
  `min_agents = 1` line is the **entirety** of the repo's scaling configuration surface; sizing and
  warm-pool floor are Pipecat Cloud concepts, not framework code. Expected-but-absent: no Kubernetes
  manifests, no autoscaling logic, no load-shedding, no session-count limit, no graceful-drain
  helper beyond `WorkerRunner.end()`.
- **Migration angle:** boson-agent is already the "one long-lived process, many sessions" shape, so
  it does not collide with the Pipecat *runner* — it collides with `WorkerRunner`.
  `packages/gateway/gateway/__main__.py` builds one `GatewayCore(config)` that "owns process-scoped
  resources (including MCP subprocesses)", discovers rules/layers/stages once, and hands them to
  `GatewayWebSocketServer` (`server/websocket.py:35`), whose `start()` wraps a single
  `websockets.serve(...)`; sessions are multiplexed inside via `_handle_connection`,
  `_reserve_session_dispatch`, `_replace_active_task`, `_cancel_session_dispatch`, `forget_session`.
  That bookkeeping is what `PipelineWorker` + `WorkerRunner(auto_end=False)` **replaces**:
  `_cancel_active_task` → worker cancel, `_start_silence_timer` → `idle_timeout_secs`/
  `idle_timeout_frames`. `server/websocket.py` + `server/protocol.py` are replace candidates
  (Pipecat transport + RTVI). `GatewayCore`'s process-scoped MCP subprocess ownership is **keep**,
  but must be re-hosted above the worker so it is not re-spawned per session — a real risk if Lina
  moves to container-per-call on Pipecat Cloud, where `min_agents` warm pool and per-agent MCP
  startup become cold-start cost. Untouched: rules/layers/stage discovery is startup-time config
  loading and ports as-is into `bot(runner_args)` — but then lands on the per-session cold-start
  path `StartupTimingObserver` measures. Open ch-10 decision: telephony arrives at a webhook, so
  container-per-session needs a warm pool sized to concurrent-call peak that boson's single-process
  model needs none of — a cost/architecture trade, not a correctness one.

## Citation
pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (release 1.7.0, 2026-08-01),
read 2026-08-25. Paths: `src/pipecat/runner/run.py` (2003 L), `runner/types.py`, `workers/runner.py`,
`workers/base_worker.py`, `pipeline/worker.py`, `bus/`, `cli/main.py`, `cli/templates/server/`,
`examples/multi-worker/README.md`. boson-agent read-only at
`/Users/jaewon/mywork_2026/Lina_2026/boson-agent-dev/boson-agent`
(`packages/gateway/gateway/__main__.py`, `core.py`, `server/websocket.py`).

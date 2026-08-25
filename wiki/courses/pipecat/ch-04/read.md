---
title: "What Runs When You Call worker.run(): Queues, Tasks, and Out-of-Band Priority"
chapter: ch-04
phase: read
course: pipecat
sources:
  - theory-out-of-band-priority
  - pipeline-task-runner
  - frame-processor
  - canonical-voice-bot
  - deployment-scaling
figure: figures/one-call-runtime.html
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
---

# What Runs When You Call `worker.run()`

## 왜 이 챕터인가

[[ch-01/read]] gave you the composition law — `link()` is two pointer assignments, `Pipeline._link_processors` is a fold, nothing validates anything. [[ch-02/read]] gave you the narrow waist — `Frame` is the one type everything agrees on, and the price is a sum type that only grows. [[ch-03/read]] characterised what you already shipped in `realtime_voice`: a closed union of six frozen dataclasses supervised by one 561-line `VoiceSession`.

All three of those chapters are about **structure at rest**. None of them tells you what happens at 14:03:22 on a Tuesday when a Korean customer picks up, says "여보세요", listens to eleven seconds of Lina's opening, and hangs up in the middle of your fourth sentence.

That is this chapter. It is deliberately not an API tour. There is exactly one running example — **one Lina sales call** — and every constant, method and queue in the chapter earns its place by answering a question that call raises:

- The customer hangs up mid-sentence. What tears down, how fast, and what happens to the 3 seconds of Korean TTS audio already sitting in the output queue?
- The bot finishes its closing line and you want the call to end cleanly. Why is *that* wait unbounded when the hang-up wait is capped at 20 seconds?
- The customer says nothing for four minutes. Nothing happens. Why? And is that the behaviour you want on a sales dial?
- Your STT provider's websocket handshake stalls at connect time. Does the call start anyway, half-wired?

By the end you own a named deliverable: **the process / session / worker topology for the Lina host**. Later chapters consume it. [[ch-05/read]] plugs a transport into it, [[ch-10/read]] injects flow-manager frames into it, [[ch-12/read]] puts your rule layers inside it. None of them re-derive it.

A note on how to read the numbers. Every Pipecat line number below was opened at commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`. Where something in the excerpt library disagrees with the tree, I say so in the text rather than quietly picking one. Everything about `boson-agent` and `realtime_voice` comes from the excerpt files ([[boson-gateway-server]], [[rtv-pipeline-session]], [[boson-interrupt-subsystem]]) and is attested there, not here.

---

## 1. One call, four exits

Here is the whole chapter in one table. Read it now; you will not understand it yet. Come back to it at the end and it should read as obvious.

| What the customer does | Frame that starts the exit | Where the wait is bounded | The constant |
|---|---|---|---|
| Setup: STT/TTS provider connect | (none yet — no frame has been pushed) | `_setup_within_timeout`, `worker.py:1104-1121` | `SETUP_TIMEOUT_SECS = 20.0` |
| Start: `StartFrame` must traverse the whole chain | `StartFrame` | `_wait_for_pipeline_start`, `worker.py:1039-1061` | `START_TIMEOUT_SECS = 20.0` |
| Hangs up mid-sentence | `CancelFrame` | `_wait_for_pipeline_end`, `worker.py:1063-1095` | `CANCEL_TIMEOUT_SECS = 20.0` |
| Bot finishes its closing line | `EndFrame` | **unbounded, on purpose** | — |
| Goes silent | (absence of frames) | `_idle_monitor_handler`, `worker.py:1401-1415` | `IDLE_TIMEOUT_SECS = 300` |

All six constants are declared in one block at the top of the worker module. Nothing is hidden in a config file:

**`src/pipecat/pipeline/worker.py:91-100`**
```python
HEARTBEAT_SECS = 1.0
HEARTBEAT_MONITOR_SECS = 10.0

IDLE_TIMEOUT_SECS = 300

CANCEL_TIMEOUT_SECS = 20.0

SETUP_TIMEOUT_SECS = 20.0

START_TIMEOUT_SECS = 20.0
```

Two things about this block are worth naming immediately.

First, `IDLE_TIMEOUT_SECS = 300` is **five minutes**. On a Korean outbound insurance dial, thirty seconds of silence is already a lost customer and sixty is a customer who put the phone down on the table. The framework's default is off by an order of magnitude for your use case. That is not a bug — Pipecat's default is tuned for a browser demo where an abandoned tab should not burn GPU forever. It means you must retune it, and §9 shows the exact kwarg.

Second, there is no `END_TIMEOUT_SECS`. The graceful path has no timeout at all, and §8 shows you the comment in the source that says why in as many words. Memorize the asymmetry now: **the violent path is bounded, the graceful path is unbounded.** Almost everyone guesses it the other way round.

---

## 2. The vocabulary, corrected first

Before anything else: most Pipecat material you will find on the web — blog posts, YouTube walkthroughs, the majority of Stack Overflow answers, and a large fraction of LLM-generated Pipecat code — is written against `PipelineTask` and `PipelineRunner`. Those names still work. They are also both deprecated, and code written against them is a 2.0.0 migration you are choosing to take on.

The real names are `PipelineWorker` and `WorkerRunner`.

**`src/pipecat/pipeline/worker.py:1478-1482`**
```python
@deprecated(
    "`PipelineTask` is deprecated since 1.3.0 and will be removed in 2.0.0. "
    "Use `PipelineWorker` instead."
)
class PipelineTask(PipelineWorker):
```

**`src/pipecat/pipeline/worker.py:1493-1498`**
```python
@deprecated(
    "`PipelineTaskParams` is deprecated since 1.3.0 and will be removed in 2.0.0. "
    "Use `WorkerParams` instead."
)
@dataclass
class PipelineTaskParams(WorkerParams):
```

`src/pipecat/pipeline/runner.py` is 37 lines total, and the last fourteen of them are this:

**`src/pipecat/pipeline/runner.py` (tail)**
```python
    "`PipelineRunner` is deprecated since 1.3.0 and will be removed in 2.0.0. "
    "Use `WorkerRunner` instead."
)
class PipelineRunner(WorkerRunner):
    """Deprecated alias for :class:`~pipecat.workers.runner.WorkerRunner`.

    .. deprecated:: 1.3.0
        Use :class:`~pipecat.workers.runner.WorkerRunner` instead.
        Will be removed in 2.0.0. The :class:`PipelineRunner` now runs workers
        (of which :class:`~pipecat.pipeline.worker.PipelineWorker` is one kind),
        not just pipelines.
    """

    pass
```

`src/pipecat/pipeline/task.py` is 29 lines and contains no logic at all — it is a pure re-export of five names from `pipecat.pipeline.worker`:

**`src/pipecat/pipeline/task.py:15-21`**
```python
from pipecat.pipeline.worker import (
    IdleFrameObserver,
    PipelineParams,
    PipelineTask,
    PipelineTaskParams,
    PipelineWorker,
)
```

The rename is not cosmetic and it is not undocumented. `AGENTS.md` states the reason in one sentence:

**`AGENTS.md:82`**
```
Terminology note: a "worker" is a runnable unit, "task" now refers only to
asyncio tasks, and cross-worker RPC uses "jobs" and "job groups".
```

That matters for you specifically, because this chapter is *about* asyncio tasks. If "task" still meant "the runnable pipeline unit," half the sentences below would be ambiguous — "the process task is cancelled but the task keeps running" is unreadable. With the rename it is precise: a **worker** is the session-scoped runnable; a **task** is an `asyncio.Task`; there are two of those per processor and you are about to meet both.

The division of labour, stated once so you can stop wondering:

- **`PipelineWorker(BaseWorker)`** (`worker.py:198`) owns **lifecycle** — setup, start, end, cancel, idle detection, heartbeats. One per call.
- **`WorkerRunner(BaseObject, BusSubscriber)`** (`workers/runner.py:83`) owns **process concerns** — the bus, the registry, SIGINT/SIGTERM, and the question of when the *process* should stop. One per host process.

That second line is the one that decides your deployment shape, so it gets its own section.

---

## 3. The host: `WorkerRunner`, and why `auto_end=False` is mandatory for Lina

Lina is not a demo bot. Lina is a FastAPI process that sits up, accepts websocket connections from many customers over a working day, and must still be alive at 18:00 after the last call of the afternoon has hung up.

`WorkerRunner` has exactly two entry points, and its own class docstring names them:

**`src/pipecat/workers/runner.py:91-100`**
```
    Two entry points:

    - :meth:`add_workers(*workers)` — register one or more workers on the
      runner's bus and start them in the background. Workers run
      concurrently and remaining workers are cancelled when the runner
      ends.
    - :meth:`run` — block until the runner ends. By default
      (``auto_end=True``) the runner ends once every root worker has
      finished; pass ``auto_end=False`` to keep the runner up until
      :meth:`end` / :meth:`cancel` is called.
```

The `run()` docstring spells out the Lina case by name. This is not me extrapolating from framework mechanics — the framework tells you:

**`src/pipecat/workers/runner.py:245-266`**
```
        By default (``auto_end=True``), the runner ends once every root
        worker has finished — so a single-pipeline bot naturally ends
        when its pipeline does. Multi-worker bots whose helpers run
        forever (e.g. waiting for bus messages) end by calling
        :meth:`end` / :meth:`cancel` from an event handler (typically on
        transport disconnect). For long-lived hosts that add and remove
        workers over many sessions (e.g. a FastAPI server), pass
        ``auto_end=False`` so the runner does not exit when no workers
        are left.

        Args:
            worker: Optional worker to run.

                .. deprecated:: 1.3.0
                    Register the worker with :meth:`add_workers` before
                    calling ``run()`` instead.
                    Will be removed in 2.0.0.

            auto_end: When ``True`` (the default), the runner ends once
                every root worker has finished. When ``False``, the
                runner blocks until :meth:`end` or :meth:`cancel` is
                called.
```

"For long-lived hosts that add and remove workers over many sessions (e.g. a FastAPI server)" is a literal description of the Lina host. If you leave the default, here is precisely what happens, in the actual code:

**`src/pipecat/workers/runner.py:441-459`**
```python
    async def _run_worker(self, worker: BaseWorker) -> None:
        """Drive a registered worker to completion."""
        try:
            params = WorkerParams(task_manager=self.task_manager)
            await worker.run(params)
        except asyncio.CancelledError:
            pass
        finally:
            # End the runner once every root worker has finished. The
            # current worker's task is still "running" (we're inside its
            # body), so exclude it from the check.
            if self._auto_end and worker.parent is None:
                others_running = any(
                    e.runner_task is not None and not e.runner_task.done()
                    for e in self._entries.values()
                    if e.worker.parent is None and e.worker is not worker
                )
                if not others_running:
                    self._shutdown_event.set()
```

Trace it against a real morning. 09:14, one customer on the line, one `PipelineWorker` running. Customer hangs up. `_run_worker`'s `finally` fires. `others_running` is `False` because there is exactly one root worker and it is the one finishing. `self._shutdown_event.set()`. Back in `run()`, `await self._shutdown_event.wait()` returns, the runner cancels stragglers, calls `cleanup()`, stops the bus, and returns. **Your FastAPI host has exited at 09:14 because one customer hung up.** Every subsequent call in the day hits a dead port.

That is the failure the docstring at `:250-253` is warning about, and it is a one-word fix. The top strip of the chapter figure lets you watch it happen: connect and disconnect simulated customers with `auto_end=False` and the runner stays up while workers come and go; flip it to `True` and the whole host disappears the moment the last call ends.

→ **[one-call-runtime.html](figures/one-call-runtime.html)** — open it now and drive it alongside §3 through §9. Do the three experiments in order: flip `auto_end` in the top strip, inject a `SystemFrame` in the middle panel and watch it overtake twice, then fire all four exit buttons in the bottom panel and compare the `CancelFrame` and `EndFrame` traces on the shared clock. The bounded/unbounded asymmetry in §7–§8 is a picture in that panel and a sentence here; look at the picture first.

### 3.1 What `add_workers` actually does, and the thing it does not have

**`src/pipecat/workers/runner.py:218-235`**
```python
        for worker in workers:
            if worker.name in self._entries:
                logger.error(
                    f"WorkerRunner '{self}': worker '{worker.name}' already exists, skipping"
                )
                continue
            # ``attach`` is async because it also subscribes the worker
            # to the bus — eager subscription is required so workers
            # added later are listening before earlier workers emit
            # their first messages.
            await worker.attach(registry=self._registry, bus=self._bus, worker_runner=self)
            await self._registry.watch(worker.name, self._on_local_worker_ready)
            entry = _WorkerEntry(worker=worker)
            self._entries[worker.name] = entry
            logger.debug(f"WorkerRunner '{self}': added worker '{worker.name}'")

            if self._running:
                await self._start_worker(entry)
```

Three facts you need for the deliverable in §13, and one of them is bad news.

**(a) Names must be unique, and a collision is a silent no-op with a log line.** If `worker.name` is already in `self._entries`, the worker is *skipped* — not started, not raised on. If you name your workers after something that can repeat (a customer phone number, a CRM lead id that gets re-dialled), a re-dial to the same number quietly never starts. Default names are safe: `BaseObject.__init__` does `self._name = name or f"{self.__class__.__name__}#{obj_count(self)}"` (`utils/base_object.py:72`), and `obj_count` is a per-class `itertools.count` under a lock (`utils/utils.py:33-42`), so unnamed workers get `PipelineWorker#0`, `#1`, `#2` … If you *do* want a readable name, use the per-connection session UUID, never the phone number.

**(b) Adding while running starts immediately.** `if self._running: await self._start_worker(entry)`. So the FastAPI websocket route can construct a `PipelineWorker` and `await runner.add_workers(w)` at any moment after startup and it just runs. This is the mechanic that makes one long-lived runner viable.

**(c) There is no `remove_workers`.** I grepped for it:

```
$ grep -rn "remove_worker" src/ examples/
(no output)
```

Zero hits across `src/` and `examples/`. `self._entries` is a `dict[str, _WorkerEntry]` that only ever grows. A finished worker's entry stays in it — `_run_worker`'s `finally` block checks `entry.runner_task.done()` but never deletes the entry. Over a full day of Lina calls, that dict accumulates one entry per call, each holding a reference to a `PipelineWorker`, which holds a reference to the whole `Pipeline`, which holds every processor and every closed provider client.

The excerpt at [[pipeline-task-runner]] describes the topology as "one `PipelineWorker` per call added via `add_workers` and removed on disconnect." The first half is right; **the second half has no API behind it in this commit.** I am telling you plainly rather than smoothing it: if you run one long-lived `WorkerRunner` for a whole trading day, you are accumulating dead worker entries and you will have to deal with it yourself — either by reaching into `runner._entries` (private, and it will break), or by pooling runners, or by recycling the process on a schedule. §13 makes a call.

### 3.2 The shutdown verbs on the runner

Three methods, all idempotent on the same `_shutdown_event`:

**`src/pipecat/workers/runner.py:332-366`**
```python
    async def end(self, reason: str | None = None) -> None:
        """Gracefully end all running workers.

        Idempotent; subsequent calls are ignored.

        Args:
            reason: Optional human-readable reason for ending.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"WorkerRunner '{self}': ending gracefully (reason={reason})")
        self._shutdown_event.set()
        for name, entry in self._entries.items():
            if entry.worker.parent is None:
                await self._bus.send(
                    BusEndWorkerMessage(source=self.name, target=name, reason=reason)
                )

    async def cancel(self, reason: str | None = None) -> None:
        """Immediately cancel all running workers.

        Idempotent; subsequent calls are ignored.

        Args:
            reason: Optional human-readable reason for cancelling.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"WorkerRunner '{self}': cancelling (reason={reason})")
        self._shutdown_event.set()
        for name, entry in self._entries.items():
            if entry.worker.parent is None:
                await self._bus.send(
                    BusCancelWorkerMessage(source=self.name, target=name, reason=reason)
                )
```

Read the `for` loop carefully: **both of these hit every root worker on the runner.** `runner.cancel()` is not "cancel this call." It is "cancel every call currently on this host."

The canonical example in the repo does exactly that, and it is correct *there* because that example is one process per call:

**`examples/getting-started/06-voice-agent.py:116-119`**
```python
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()
```

Copy that line into a multi-tenant Lina host and one customer hanging up drops every other customer on the box. The per-call verb is `worker.cancel()`, not `runner.cancel()`. This is the single most consequential copy-paste hazard in the whole framework's example set, and it is invisible unless you read `runner.py:350-366`.

Signals route to the runner-wide path, which is what you want for a container SIGTERM:

**`src/pipecat/workers/runner.py:529-537`**
```python
    def _sig_handler(self) -> None:
        if not self._sig_task:
            self._sig_task = asyncio.create_task(self._sig_cancel())

    async def _sig_cancel(self) -> None:
```

with the handlers installed at `:515` (`loop.add_signal_handler(signal.SIGINT, ...)`) and `:524` (SIGTERM), gated on `handle_sigint=True` / `handle_sigterm=False` from the constructor (`:114-115`). Note the default: **SIGINT yes, SIGTERM no.** Kubernetes and most container runtimes send SIGTERM on eviction. If you deploy Lina in a container and leave `handle_sigterm=False`, a rolling restart kills the process without any of the graceful `BusEndWorkerMessage` path running. Set `handle_sigterm=True`.

---

## 4. Inside one processor: two queues, two tasks

Now zoom all the way in, from the host to a single processor. This is the mechanic that makes every later chapter legible — [[ch-06/read]]'s turn boundary, [[ch-08/read]]'s barge-in cascade, [[ch-12/read]]'s rule layers all depend on knowing exactly what happens between "a frame arrives" and "your `process_frame` runs."

The one-sentence version: **every `FrameProcessor` owns two asyncio queues drained by two asyncio tasks, and `SystemFrame`s only ever touch the first of each.**

### 4.1 The priority queue

**`src/pipecat/processors/frame_processor.py:132-171`**
```python
class FrameProcessorQueue(asyncio.PriorityQueue):
    """A priority queue for the frames arriving at a frame processor.

    Frames are dequeued in three tiers: the `StartFrame` first, then
    `SystemFrame`, then data and control frames. Frames of the same tier keep
    their arrival order.

    """

    START_PRIORITY = 1
    SYSTEM_PRIORITY = 10
    DEFAULT_PRIORITY = 20

    def __init__(self):
        """Initialize the FrameProcessorQueue."""
        super().__init__()
        # Counts every frame enqueued, which keeps frames of the same tier in
        # arrival order and stops the queue from ever having to compare frames.
        self.__counter = 0

    async def put(self, item: tuple[Frame, FrameDirection, FrameCallback | None]):
        """Put an item into the priority queue.

        The `StartFrame` outranks every other frame and `SystemFrame` frames
        outrank data and control frames.

        Args:
            item: The frame to enqueue, with its direction and callback.

        """
        frame, _, _ = item
        if isinstance(frame, StartFrame):
            priority = self.START_PRIORITY
        elif isinstance(frame, SystemFrame):
            priority = self.SYSTEM_PRIORITY
        else:
            priority = self.DEFAULT_PRIORITY

        self.__counter += 1
        await super().put((priority, self.__counter, item))
```

Three tiers, assigned by `isinstance`, nothing configurable. The `__counter` is doing two jobs at once and the comment names both: it preserves FIFO **within** a tier, and it "stops the queue from ever having to compare frames." That second job is load-bearing. `asyncio.PriorityQueue` uses `heapq`, which compares tuples element-wise; if two entries tie on priority, Python moves to the next element. Frames are `@dataclass` instances without ordering, so a tie would raise `TypeError: '<' not supported between instances of 'TTSAudioRawFrame' and 'TTSAudioRawFrame'`. The monotonic counter guarantees element two never ties, so element three — the frame — is never compared. It is a two-line defence against a crash that would only show up under load, when two frames of the same tier happen to be resident simultaneously. Which, in a voice pipeline, is always.

The `StartFrame` getting its own tier above `SystemFrame` looks like over-engineering until you read `queue_frame`:

**`src/pipecat/processors/frame_processor.py:700-728`**
```python
    async def queue_frame(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
        callback: FrameCallback | None = None,
    ):
        """Queue a frame for processing.

        Args:
            frame: The frame to queue.
            direction: The direction of frame flow.
            callback: Optional callback to call after processing.
        """
        # If we are cancelling we don't want to process any other frame.
        if self._cancelling:
            return

        if self._enable_direct_mode:
            await self.__process_frame(frame, direction, callback)
            return

        await self.__input_queue.put((frame, direction, callback))

        # Nothing drains the queue until the StartFrame arrives, so a processor
        # never acts on a frame before it has been started. Frames pushed
        # between setup and the StartFrame simply wait, and the StartFrame is
        # dequeued ahead of them.
        if isinstance(frame, StartFrame):
            self.__create_input_task()
```

The input task does not exist until the `StartFrame` arrives. So there is a window — between `setup()` (where your STT service opens its websocket) and the `StartFrame` — during which frames can be *enqueued* but nothing drains them. If a provider client pushes an error frame during connect, it sits in the queue. When the `StartFrame` finally arrives it is dequeued **first**, at priority 1, ahead of everything that was already waiting. That is the entire reason for a third tier: it guarantees a processor is started before it acts on anything, even things that arrived earlier in wall-clock time.

Three things about `queue_frame` you should file away now because they bite later:

1. **`self._cancelling` is a silent drop.** After a `CancelFrame`, `queue_frame` returns without enqueueing and without logging. Frames vanish. §7 shows where `_cancelling` is set.
2. **`_enable_direct_mode` skips both queues.** More on this in §4.4.
3. **`queue_frame` is the *only* entry point.** `push_frame` does not call a neighbour's `process_frame` — it calls the neighbour's `queue_frame`. Pushing is enqueueing on the neighbour, always. That is what makes each hop an independent scheduling point, and it is why `link()` can be three lines.

### 4.2 The split — the whole trick, in nine lines

**`src/pipecat/processors/frame_processor.py:1287-1313`**
```python
    async def __input_frame_task_handler(self):
        """Handle frames from the input queue.

        It only processes system frames. Other frames are queue for another task
        to execute.

        """
        while True:
            (frame, direction, callback) = await self.__input_queue.get()

            if self.__should_block_system_frames and self.__input_event:
                logger.trace(f"{self}: system frame processing paused")
                await self.__input_event.wait()
                self.__input_event.clear()
                self.__should_block_system_frames = False
                logger.trace(f"{self}: system frame processing resumed")

            if isinstance(frame, SystemFrame):
                await self.__process_frame(frame, direction, callback)
            elif self.__process_queue:
                await self.__process_queue.put((frame, direction, callback))
            else:
                raise RuntimeError(
                    f"{self}: __process_queue is None when processing frame {frame.name}"
                )

            self.__input_queue.task_done()
```

Look at the `if/elif`. A `SystemFrame` is **executed right here, on the input task**. Everything else is *relayed* — put onto a second queue and forgotten about.

The second queue has its own task:

**`src/pipecat/processors/frame_processor.py:1315-1333`**
```python
    async def __process_frame_task_handler(self):
        """Handle non-system frames from the process queue."""
        while True:
            self.__process_current_frame = None

            (frame, direction, callback) = await self.__process_queue.get()

            self.__process_current_frame = frame

            if self.__should_block_frames and self.__process_event:
                logger.trace(f"{self}: frame processing paused")
                await self.__process_event.wait()
                self.__process_event.clear()
                self.__should_block_frames = False
                logger.trace(f"{self}: frame processing resumed")

            await self.__process_frame(frame, direction, callback)

            self.__process_queue.task_done()
```

Now count the overtaking. A `SystemFrame` arriving at a processor that currently has 75 audio frames queued gets ahead of them **twice**:

1. **On the way in**, by priority. `FrameProcessorQueue.put` gives it 10; the audio frames are at 20. The input task pops it next regardless of arrival order.
2. **By never entering the slow queue at all.** The 75 audio frames are in `__process_queue`, waiting on `__process_frame_task_handler`, which is busy `await`ing inside your TTS handler. The system frame does not join that line. It executes on the input task, which is idle because its only job is popping and routing.

That is why an `InterruptionFrame` reaches every processor in single-digit milliseconds while three seconds of audio is still resident. It is not a fast path bolted on; it is a *different task*.

[[theory-out-of-band-priority]] traces the same line back to GStreamer, which drew it in 2001: downstream events there are either "in-band (serialised with the buffer flow)" or "out-of-band (travelling through the pipeline instantly … skipping ahead of buffers being processed or queued in the pipeline)". `SEGMENT`, `CAPS`, `TAG` and `EOS` are serialised; `FLUSH_START` is out-of-band. Buffers:events maps onto DataFrame:SystemFrame, and EOS-is-serialised maps onto `EndFrame`-is-a-`ControlFrame` — which is §8. Pipecat did not invent this; it re-derived it, and it vendors real GStreamer graphs in `src/pipecat/processors/gstreamer/pipeline_source.py:39` (`class GStreamerPipelineSource(FrameProcessor)`).

And the pause machinery is two-tiered for the same reason. `pause_processing_frames()` gates the process task via `__process_event`; `pause_processing_system_frames()` gates the input task via `__input_event`. You can freeze data flow while leaving control flow live, which is exactly what you want if you are, say, holding audio while a tool call resolves but still want a barge-in to land.

### 4.3 The cancellable half

The asymmetry has a purpose beyond latency: **the process task is disposable and the input task is not.**

**`src/pipecat/processors/frame_processor.py:1130-1150`**
```python
    async def _start_interruption(self):
        """Start handling an interruption by cancelling current tasks."""
        try:
            current_is_uninterruptible = isinstance(
                self.__process_current_frame, UninterruptibleFrame
            )
            if current_is_uninterruptible:
                # The frame currently being processed is uninterruptible, so we
                # must not cancel it. Just flush non-uninterruptible frames from
                # the queue; any uninterruptible ones will be kept and processed
                # after the current frame finishes.
                self.__reset_process_queue()
            else:
                # Cancel and re-create the process task. Previously this branch
                # was skipped when the queue contained an uninterruptible frame,
                # which caused slow non-uninterruptible frames to block
                # interruptions. Uninterruptible queued frames are safe here
                # because __create_process_task calls __reset_process_queue
                # internally, which always preserves them.
                await self.__cancel_process_task()
                self.__create_process_task()
```

`await self.__cancel_process_task()` kills the coroutine **mid-`await`**, wherever it happens to be inside your `process_frame`. Then a fresh task and a fresh queue. The input task is untouched, so the processor is still listening for the *next* system frame while the data half is being rebuilt.

The selective flush is `FrameQueue.reset()`:

**`src/pipecat/utils/frame_queue.py:84-95`**
```python
    def reset(self) -> None:
        """Remove all non-UninterruptibleFrame items, keeping uninterruptible ones."""
        kept: asyncio.Queue = asyncio.Queue()
        while not self.empty():
            item = self.get_nowait()
            if isinstance(self._frame_getter(item), UninterruptibleFrame):
                kept.put_nowait(item)
            self.task_done()
        while not kept.empty():
            item = kept.get_nowait()
            self.put_nowait(item)
            kept.task_done()
```

Drain everything, re-enqueue only the `UninterruptibleFrame`s. A half-spoken Korean TTS response evaporates; a queued `EndFrame` survives, because `EndFrame` carries the mixin:

**`src/pipecat/frames/frames.py:1899-1910`**
```python
class EndFrame(ControlFrame, UninterruptibleFrame):
    """Frame indicating pipeline has ended and should shut down.

    Indicates that a pipeline has ended and frame processors and pipelines
    should be shut down. If the transport receives this frame, it will stop
    sending frames to its output channel(s) and close all its threads. Note,
    that this is a control frame, which means it will be received in the order it
    was sent.

    This frame is marked as UninterruptibleFrame to ensure it is not lost when
    an InterruptionFrame is processed. Terminal frames must survive interruption
    to guarantee proper pipeline shutdown.
```

The `_uninterruptible_count` bookkeeping is kept O(1) by overriding `_put`/`_get` (`frame_queue.py:72-81`), so `has_uninterruptible` is a comparison, not a scan.

> **One thing that is wrong in the source, said plainly.** `has_queued_frame`'s docstring at `frame_processor.py:1244-1249` claims the check "is O(distinct enqueued types) with no queue scanning." It delegates to `FrameQueue.has_frame`, whose body is `for item in self._queue: if isinstance(...): return True` (`frame_queue.py:64-67`) — a linear scan of the deque. The O(1) claim is true of `has_uninterruptible` (the counter) and false of `has_frame`. It does not matter at voice-pipeline queue depths, but do not build a hot loop on that docstring.

What the flush does *downstream* — to the TTS service's aggregation buffer, to the audio clock, to Twilio's playout buffer that you cannot reach — is [[ch-08/read]]'s subject. It is taught once, there, where you will already own the output queue. Here you only need: **flush is selective, and it is triggered by an out-of-band signal.**

### 4.4 The third path, and where the framework itself uses it

`_enable_direct_mode` bypasses both queues. It appears six times in `frame_processor.py` (`:243, 717, 784, 1206, 1224, 1233`) and every use is the same shape: return immediately, or process inline.

**`src/pipecat/processors/frame_processor.py:1222-1229`**
```python
    def __create_process_task(self):
        """Create the non-system frame processing task."""
        if self._enable_direct_mode:
            return

        if not self.__process_frame_task:
            self.__reset_process_task()
            self.__process_frame_task = self.create_task(self.__process_frame_task_handler())
```

This is not a debug switch. Grep says the framework's own structural processors are all in direct mode:

```
$ grep -rn "enable_direct_mode=True" src/pipecat/
src/pipecat/pipeline/service_switcher.py:388
src/pipecat/pipeline/service_switcher.py:395
src/pipecat/pipeline/sync_parallel_pipeline.py:76
src/pipecat/pipeline/sync_parallel_pipeline.py:109
src/pipecat/pipeline/pipeline.py:36
src/pipecat/pipeline/pipeline.py:72
src/pipecat/pipeline/pipeline.py:113
src/pipecat/tests/utils.py:103
```

`pipeline.py:36` is `PipelineSource`, `:72` is `PipelineSink`, and `:113` is `Pipeline` itself:

**`src/pipecat/pipeline/pipeline.py:99-121`**
```python
    def __init__(
        self,
        processors: Sequence[FrameProcessor],
        *,
        source: FrameProcessor | None = None,
        sink: FrameProcessor | None = None,
    ):
        """Initialize the pipeline with a list of processors.

        Args:
            processors: Sequence of frame processors to connect in sequence.
            source: An optional pipeline source processor.
            sink: An optional pipeline sink processor.
        """
        super().__init__(enable_direct_mode=True)

        # Add a source and a sink queue so we can forward frames upstream and
        # downstream outside of the pipeline.
        self._source = source or PipelineSource(self.push_frame, name=f"{self}::Source")
        self._sink = sink or PipelineSink(self.push_frame, name=f"{self}::Sink")
        self._processors: list[FrameProcessor] = [self._source, *processors, self._sink]

        self._link_processors()
```

This is a genuinely important consequence of [[ch-01/read]]'s law that a `Pipeline` is itself a `FrameProcessor`. If the container had its own queues and tasks, then nesting a pipeline inside a pipeline — which the worker does, twice, see §6.3 — would add a queue hop and two tasks per level of nesting. Instead: **containers add zero queueing.** `Pipeline.process_frame` just routes into `self._source.queue_frame(...)` or `self._sink.queue_frame(...)` (`pipeline.py:192-195`), and the source/sink are themselves direct-mode. Every queue in the system belongs to a *leaf* processor that does real work. Composition is free at runtime, not just at authoring time.

The trade is stated in [[frame-processor]] and it is real: in direct mode the ordering guarantees do not apply, because there is no queue to order. Do not set it on a processor you wrote.

---

## 5. Two things that are true regardless of how good your code is

These are the two theory paragraphs this chapter owns, and they are the argument [[theory-out-of-band-priority]] makes from first principles. Everything else in the course cites them; they are not re-derived anywhere else.

### 5.1 Control latency is queue depth over drain rate

Take the concrete case first.

Lina is halfway through a sentence: *"고객님, 이 상품은 65세까지 갱신 없이 보장이 되고요, 지금 가입하시면…"* Your Korean TTS vendor streamed that whole 3-second utterance to you over its websocket in about 400 ms — vendors do that, because their job is to be fast, and the audio is smaller than the network is wide.

Where is that audio right now? In the output transport's queue:

**`src/pipecat/transports/base_output.py:690`**
```python
                self._audio_queue = FrameQueue()
```

drained by a **clock-paced** task, `_clock_task_handler` (`base_output.py:1079`) — i.e. drained at realtime, because that is the only rate at which audio can be played. And it is written out in chunks whose size is set here:

**`src/pipecat/transports/base_output.py:132-136`**
```python
        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks
```

with the default set at **`src/pipecat/transports/base_transport.py:72`**:
```python
    audio_out_10ms_chunks: int = 4
```

Four ten-millisecond chunks = **40 ms of PCM per written chunk**. So 3 seconds of Korean audio is roughly 75 chunks resident in `_audio_queue`, plus whatever the carrier already holds downstream of you.

Now suppose the customer barges in — "아니 잠깐만요" — and your stop signal is an ordinary frame that queues behind that audio. The queue drains at realtime. Your stop signal is behind 3 seconds of realtime audio. **It arrives 3 seconds late.** Not because your handler is slow. Your handler could be a single `return` statement and it would still arrive 3 seconds late.

The formula, now that you have seen it happen: a strictly ordered pipe has one channel. A control message enqueued at time *t* is delivered only after everything enqueued before *t* is drained. With *N* items resident and a sink draining at realtime rate *r*:

```
control latency = N / r
```

and *r* is fixed by physics — it is the rate at which sound leaves a speaker. The only free variable is *N*. You do not make this better by optimising code. You make it better by not being in that queue.

Note what falls out of this: `audio_out_10ms_chunks` is not a buffering tuning knob, it is an **interrupt-granularity** knob, and the source comment says so — "This will help with interruption handling." Smaller chunks mean the write loop checks for cancellation more often, at the cost of more syscalls. That is the actual trade, and it is one of the numbers [[ch-11/read]] will spend when it builds the latency budget.

That single paragraph is why a priority tier exists at all. Everything in §4 is a consequence of *N/r*.

### 5.2 You cannot back-pressure a live microphone

Back-pressure is a consumer telling a producer "slow down." The canonical statement is the Reactive Streams spec (v1.0.4, 2022-05-26): the purpose is "asynchronous stream processing with non-blocking backpressure," where backpressure exists "in order to allow the queues which mediate between threads to be bounded."

Pipecat's queues are not bounded. Here is the grep, and it is the whole evidence:

```
$ grep -n maxsize src/pipecat/processors/frame_processor.py \
                  src/pipecat/utils/frame_queue.py \
                  src/pipecat/transports/base_input.py \
                  src/pipecat/transports/base_output.py
$ echo $?
1
```

Zero hits across all four files. `FrameProcessorQueue.__init__` calls a bare `super().__init__()` (`frame_processor.py:146-151`), `FrameQueue.__init__` likewise (`frame_queue.py:43`), and the inbound audio queue is:

**`src/pipecat/transports/base_input.py:265`**
```python
            self._audio_in_queue = asyncio.Queue()
```

No `maxsize`. No demand signal anywhere in the framework.

This is not an oversight, and the argument is physical rather than architectural. **Blocking the producer does not pause the speaker.** If your STT processor stops reading, the microphone keeps capturing, the carrier keeps sending, and the audio buffers *somewhere else* — in the OS socket buffer, in the transport library, in Twilio's servers. You have not reduced the work; you have moved the queue somewhere you cannot see it and cannot flush it, and latency grows monotonically for the rest of the call.

A realtime media pipeline has exactly two real options: **drop** or **flush**. Pipecat chooses selective flush, triggered by the out-of-band signal — that is `_start_interruption` → `FrameQueue.reset()` from §4.3, keeping the `UninterruptibleFrame`s.

Your own `realtime_voice` made the other available choice, and it is worth naming precisely because you already know how it behaves in production. Per [[rtv-pipeline-session]], `VoiceSessionConfig` bounds every queue — `ingress_queue_size=64`, `event_queue_size=256`, `phrase_queue_size=8`, `audio_queue_size=32` — and applies **three different overflow policies**: ingress is reject-on-overflow (`push_audio` raises `QueueOverflowError("ingress queue full; frame rejected instead of adding latency")`), while the phrase and audio queues backpressure. The class docstring states the reasoning: *"Ingress uses reject-on-overflow so a transport cannot silently extend user turn latency."*

Put the two designs side by side without scoring them, because scoring is [[ch-13/read]]'s job:

- **`realtime_voice`**: bounded ingress, **drop** on overflow, chosen so that latency cannot silently grow. The dropped frame is gone; the transport learns about it via an exception.
- **Pipecat**: unbounded queues, **flush** on an out-of-band signal, chosen so that a barge-in can evacuate the whole pipeline at once. Nothing is dropped until something says to drop it.

Both are answers to *N/r*. `realtime_voice` caps *N* at admission. Pipecat lets *N* grow and empties it on command. What each one *does* is the fact; which one fits Lina is a question you will answer with evidence you do not have yet.

---

## 6. Exit one — setup and start

Now walk the four exits in order. Start with the one nobody thinks about until it fires at 2 a.m.

`worker.run(params)` is the whole lifecycle in one method:

**`src/pipecat/pipeline/worker.py:748-791`**
```python
    async def run(self, params: WorkerParams):
        """Start and manage the pipeline execution until completion or cancellation.

        Args:
            params: Configuration parameters for pipeline execution.
        """
        if self.has_finished():
            return

        try:
            # Setup processors.
            if not await self._setup_within_timeout(params):
                # Nothing was pushed into the pipeline, so there is nothing to
                # drain: release whatever was set up and give up.
                await self._cleanup(cleanup_pipeline=True)
                return

            # Create the worker's tasks and wait for the push task, which
            # feeds frames to the very beginning of our pipeline (i.e. to
            # our controlled source processor).
            await self._create_tasks()

            try:
                # Wait for pipeline to finish.
                await self._wait_for_pipeline_finished()
            except asyncio.CancelledError:
                logger.debug(f"Pipeline worker {self} got cancelled from outside...")
                # We have been cancelled from outside, let's just cancel everything.
                await self._cancel()
                # Wait again for pipeline to finish. This time we have really
                # cancelled, so it should really finish.
                await self._wait_for_pipeline_finished()
                # Re-raise in case there's more cleanup to do.
                raise
        finally:
            # We can reach this point for different reasons:
            #
            # 1. The pipeline worker has finished (try case).
            # 2. By an asyncio worker cancellation (except case).
            logger.debug(f"Pipeline worker {self} is finishing...")
            await self._cancel_tasks()
            self._print_dangling_tasks()
            self._finished = True
            logger.debug(f"Pipeline worker {self} has finished")
```

Three stages: `_setup_within_timeout` → `_create_tasks` → `_wait_for_pipeline_finished`.

Note the `except asyncio.CancelledError` block. If you `asyncio.cancel` the worker from outside — the obvious thing to reach for — the worker does **not** just die. It calls its own `_cancel()`, waits for the `CancelFrame` to traverse the pipeline, and only then re-raises. That is the cooperative shutdown [[pipeline-task-runner]] warns you not to bypass. It works, but you get the 20-second bound from §7 layered under an asyncio cancellation, which is a confusing thing to debug. Drive shutdown through `stop_when_done()` or `cancel()` and you never see it.

### 6.1 `SETUP_TIMEOUT_SECS` — the exit with no frame

**`src/pipecat/pipeline/worker.py:1104-1121`**
```python
    async def _setup_within_timeout(self, params: WorkerParams) -> bool:
        """Set up the pipeline worker and all processors, bounded by a timeout.

        Returns:
            Whether everything was set up. A processor that blocks while being
            set up never lets the pipeline start, so setting up is abandoned
            once ``setup_timeout_secs`` elapses.
        """
        try:
            await asyncio.wait_for(self._setup(params), timeout=self._setup_timeout_secs)
            return True
        except TimeoutError:
            logger.error(
                f"{self}: timeout setting the pipeline up "
                "(a processor blocked while connecting?), stopping the pipeline."
            )
            await self._call_event_handler("on_setup_timeout")
            return False
```

`setup()` is where processors connect: STT opens its websocket to the vendor, TTS authenticates, the LLM client builds its session. If your Korean STT provider's endpoint is having a bad minute and the connect hangs, this is the wall the call hits — after 20 seconds, `on_setup_timeout` fires and `run()` goes straight to `_cleanup(cleanup_pipeline=True)` and returns.

The comment in `run()` names the important property: *"Nothing was pushed into the pipeline, so there is nothing to drain."* No `StartFrame` was constructed, no processor was started, no frame ever entered a queue. This exit path is the only one where the pipeline is torn down without a single frame having traversed it. That is why it needs its own constant instead of reusing the cancel path.

For Lina: 20 seconds is a long time to hold a live carrier connection open playing nothing. If your telephony provider is streaming silence to a customer for 20 seconds after "여보세요", the call is already lost. `setup_timeout_secs` is a constructor kwarg (`worker.py:299`); consider 5.

### 6.2 `START_TIMEOUT_SECS` — the `StartFrame` must reach the end

Setup succeeded, so `_create_tasks()` spawns the single push task (`worker.py:986-989`), which runs the engine:

**`src/pipecat/pipeline/worker.py:1205-1246`**
```python
    async def _process_push_queue(self):
        """Process frames from the push queue and send them through the pipeline.

        This is the worker that runs the pipeline for the first time by sending
        a StartFrame and by pushing any other frames queued by the user. It runs
        until the worker is cancelled or stopped (e.g. with an EndFrame).
        """
        self._maybe_start_idle_task()

        # Processors read the pipeline configuration from FrameProcessorSetup,
        # but the deprecated StartFrame fields carry it until they are removed,
        # so that a processor still reading one gets the configured value.
        start_frame = StartFrame(
            audio_in_sample_rate=self._params.audio_in_sample_rate,
            audio_out_sample_rate=self._params.audio_out_sample_rate,
            enable_metrics=self._params.enable_metrics,
            enable_tracing=self._enable_tracing,
            enable_usage_metrics=self._params.enable_usage_metrics,
            report_only_initial_ttfb=self._params.report_only_initial_ttfb,
            tracing_context=self._tracing_context,
        )
        start_frame.metadata = self._create_start_metadata()
        await self._pipeline.queue_frame(start_frame)

        # Wait for the pipeline to be started before pushing any other frame.
        running = await self._wait_for_pipeline_start(start_frame)

        if running and self._params.enable_metrics and self._params.send_initial_empty_metrics:
            await self._pipeline.queue_frame(self._initial_metrics_frame())

        # A pipeline that never started can't process anything we push into it,
        # so skip straight to cleanup.
        cleanup_pipeline = True
        while running:
            frame = await self._push_queue.get()
            await self._pipeline.queue_frame(frame)
            if isinstance(frame, (CancelFrame, EndFrame, StopFrame)):
                await self._wait_for_pipeline_end(frame)
            running = not isinstance(frame, (CancelFrame, EndFrame, StopFrame))
            cleanup_pipeline = not isinstance(frame, StopFrame)
            self._push_queue.task_done()
        await self._cleanup(cleanup_pipeline)
```

This is a nine-line `while` loop and it is the entire engine. Pop from `_push_queue`, hand to the pipeline, and if the frame was terminal, wait for it to come out the far end and stop.

The `StartFrame` gets the same treatment as everything else — `queue_frame`, then wait — except that the wait is bounded:

**`src/pipecat/pipeline/worker.py:1039-1061`**
```python
    async def _wait_for_pipeline_start(self, frame: Frame) -> bool:
        """Wait for the specified start frame to reach the end of the pipeline.

        Returns:
            Whether the pipeline started. A pipeline that doesn't start within
            ``start_timeout_secs`` is torn down, since nothing pushed into it
            afterwards would be processed.
        """
        logger.debug(f"{self}: Starting. Waiting for {frame} to reach the end of the pipeline...")
        try:
            await asyncio.wait_for(
                self._pipeline_start_event.wait(), timeout=self._start_timeout_secs
            )
        except TimeoutError:
            logger.error(
                f"{self}: timeout waiting for {frame} to reach the end of the pipeline "
                "(being blocked somewhere?), stopping the pipeline."
            )
            await self._call_event_handler("on_pipeline_timeout", frame)
            return False
        self._pipeline_start_event.clear()
        logger.debug(f"{self}: {frame} reached the end of the pipeline, pipeline is now ready.")
        return True

```

`"being blocked somewhere?"` is the log line you will grep for. It means: one of your processors received the `StartFrame` and did not push it onward within 20 seconds. Because `process_frame` for a `StartFrame` runs on the **input task** (§4.2 — `StartFrame` is a `SystemFrame`, `frames.py:924`), a processor that does slow work in its `StartFrame` branch blocks the whole start. The [[frame-processor]] guideline is exactly this: never do slow work in `process_frame()` for a system frame.

And the sibling failure mode, which is worse because it is silent: if your custom processor overrides `process_frame` and forgets `await super().process_frame(frame, direction)`, the base implementation never runs, `__start()` never fires, `__create_process_task()` never happens — and your processor silently never handles a single data frame. It will not error. It will just be deaf. When you write the rule-layer processor in [[ch-12/read]], that one line is the difference between working and mysteriously doing nothing.

### 6.3 Your pipeline is not the pipeline that runs

The list you hand to `PipelineWorker` is not what executes. It is re-wrapped, once or twice, on the way in.

**`src/pipecat/pipeline/worker.py:522-537`**
```python
        if bridged is not None:
            edge_source = _BusEdgeProcessor(
                worker=self,
                direction=FrameDirection.UPSTREAM,
                bridges=bridged,
                exclude_frames=exclude_frames,
                name=f"{self}::EdgeSource",
            )
            edge_sink = _BusEdgeProcessor(
                worker=self,
                direction=FrameDirection.DOWNSTREAM,
                bridges=bridged,
                exclude_frames=exclude_frames,
                name=f"{self}::EdgeSink",
            )
            pipeline = Pipeline([edge_source, pipeline, edge_sink])
```

**`src/pipecat/pipeline/worker.py:543-549`**
```python
        source = PipelineSource(self._source_push_frame, name=f"{self}::Source")
        self._sink = PipelineSink(self._sink_push_frame, name=f"{self}::Sink")
        # Only prepend the RTVIProcessor if we created it ourselves. When the
        # user already placed it inside their pipeline we must not insert it
        # again or it will appear twice in the frame chain.
        processors = [self._rtvi, pipeline] if prepend_rtvi else [pipeline]
        self._pipeline = Pipeline(processors, source=source, sink=self._sink)
```

So with `enable_rtvi=True` (the default, `worker.py:289`) your seven-processor list is actually running as:

```
Pipeline([ RTVIProcessor, Pipeline([ your 7 ]) ], source=Source, sink=Sink)
```

and with `bridged` set, one more envelope. Your `Pipeline` sits two or three levels deep at runtime.

This is [[ch-01/read]]'s associativity law being cashed by the framework itself. The worker nests your pipeline inside another pipeline and the frame semantics are unchanged, because `Pipeline` is a `FrameProcessor` and — per §4.4 — the container adds no queue. If nesting cost a queue hop, `enable_rtvi=True` would silently add latency to every call and nobody would use it. It costs nothing, so it is on by default.

Practical consequence for Lina: when you read a Pipecat log and see a processor named `PipelineWorker#0::Source` or an `RTVIProcessor` you never constructed, nothing is wrong. And when [[ch-11/read]] counts hops for the latency budget, the count is not seven.

---

## 7. Exit two — the customer hangs up mid-sentence

14:03:22. Lina is four words into "그럼 제가 자세한 안내를 문자로…" and the line goes dead. Your transport's disconnect handler fires. What do you call?

**`src/pipecat/pipeline/worker.py:739-746`**
```python
    async def cancel(self, *, reason: str | None = None):
        """Request the running pipeline to cancel.

        Args:
            reason: Optional reason to indicate why the pipeline is being cancelled.
        """
        if not self._finished:
            await self._cancel(reason=reason)
```

Look at the signature: `async def cancel(self, *, reason: str | None = None)`. The `*` makes `reason` **keyword-only**. `await worker.cancel("hangup")` is a `TypeError`, not a cancel-with-reason. Given that `WorkerRunner.cancel(self, reason=None)` at `runner.py:350` is *positional-or-keyword*, the two are not call-compatible, and the mistake is easy to make when you are refactoring between them. Always write `reason=`.

**`src/pipecat/pipeline/worker.py:973-984`**
```python
    async def _cancel(self, *, reason: str | None = None):
        """Internal cancellation logic for the pipeline worker.

        Args:
            reason: Optional reason to indicate why the pipeline is being cancelled.
        """
        if not self._cancelled:
            logger.debug(f"Cancelling pipeline worker {self}")
            self._cancelled = True
            if not self._pipeline_start_event.is_set():
                self._pipeline_start_event.set()
            await self.queue_frame(CancelFrame(reason=reason))
```

Three moves. Set `_cancelled`. **Unblock the start event** — this is what lets you cancel a call that is still stuck in `_wait_for_pipeline_start`, i.e. a customer who hangs up during a slow provider handshake. Then queue a `CancelFrame`.

`CancelFrame` is a `SystemFrame` (`frames.py:999`), so it takes the out-of-band path from §4.2 at every processor: priority 10, executed inline on the input task, ahead of the 75 queued audio chunks. At each processor it lands in the base `process_frame` dispatch and calls `__cancel`:

**`src/pipecat/processors/frame_processor.py:1099-1106`**
```python
    async def __cancel(self, frame: CancelFrame):
        """Handle the cancel frame to stop processor operation.

        Args:
            frame: The cancel frame.
        """
        self._cancelling = True
        await self.__cancel_process_task()
```

`self._cancelling = True` — and now recall §4.1: from this moment `queue_frame` returns early and every subsequent frame at that processor is **silently dropped**. Then the process task is killed mid-`await`. The queued Korean audio does not play out. That is the point: the customer is gone, the audio has nowhere to go, and holding the session open to drain it is pure cost.

The wait is bounded:

**`src/pipecat/pipeline/worker.py:1063-1095`**
```python
    async def _wait_for_pipeline_end(self, frame: Frame):
        """Wait for the specified frame to reach the end of the pipeline."""

        async def wait_for_cancel():
            try:
                await asyncio.wait_for(
                    self._pipeline_end_event.wait(), timeout=self._cancel_timeout_secs
                )
                logger.debug(f"{self}: {frame} reached the end of the pipeline.")
            except TimeoutError:
                logger.warning(
                    f"{self}: timeout waiting for {frame} to reach the end of the pipeline (being blocked somewhere?)."
                )
                await self._call_event_handler("on_pipeline_timeout", frame)
            finally:
                await self._call_event_handler("on_pipeline_finished", frame)

        logger.debug(f"{self}: Closing. Waiting for {frame} to reach the end of the pipeline...")

        if isinstance(frame, CancelFrame):
            await wait_for_cancel()
        else:
            # Ending flushes what is queued, so cutting the wait short would
            # drop the audio the EndFrame exists to play out. A processor that
            # could hold it up watches for that itself.
            await self._pipeline_end_event.wait()
            logger.debug(f"{self}: {frame} reached the end of the pipeline, pipeline is closing.")

        self._pipeline_end_event.clear()

        # We are really done. Setting ``_finished_event`` makes
        # ``BaseWorker.wait()`` resolve for callers awaiting this worker.
        self._finished_event.set()
```

`CancelFrame` → `wait_for_cancel()` → `asyncio.wait_for(..., timeout=self._cancel_timeout_secs)`, defaulting to `CANCEL_TIMEOUT_SECS = 20.0`. If some processor's `CancelFrame` handler wedges — a vendor SDK that swallows `CancelledError`, a `close()` that blocks on a dead socket — the worker gives up after 20 seconds, fires `on_pipeline_timeout`, and **still** fires `on_pipeline_finished` in the `finally`. The session ends either way. That degradation is why the bound exists: a hung teardown must not pin a worker forever on a host that is taking new calls all day.

For Lina, 20 seconds of held resources after the customer is already gone is generous but survivable. If you find yourself at 200 concurrent calls, `cancel_timeout_secs` (`worker.py:283`) is a knob, and a 5-second bound with an alert on `on_pipeline_timeout` is a defensible posture.

---

## 8. Exit three — the bot finishes its closing line

Different situation, same call. The customer has said "네, 그럼 문자로 보내주세요," Lina is delivering the closing line, and *you* want to end the call cleanly when she is done. The customer is still on the line. The audio matters.

**`src/pipecat/pipeline/worker.py:730-737`**
```python
    async def stop_when_done(self):
        """Schedule the pipeline to stop after processing all queued frames.

        Sends an EndFrame to gracefully terminate the pipeline once all
        current processing is complete.
        """
        logger.debug(f"Task {self} scheduled to stop when done")
        await self.queue_frame(EndFrame())
```

One line of behaviour: queue an `EndFrame`. And an `EndFrame` is deliberately **not** a `SystemFrame`:

**`src/pipecat/frames/frames.py:1899`**
```python
class EndFrame(ControlFrame, UninterruptibleFrame):
```

`ControlFrame` (`frames.py:128`), not `SystemFrame` (`frames.py:105`). It gets `DEFAULT_PRIORITY = 20` in `FrameProcessorQueue.put`. It goes into `__process_queue` with the audio. It queues **behind** every chunk of Lina's closing line, at every processor, and arrives at the output transport only after the last sample has been written.

That is the design. The `EndFrame` docstring says it in the source: *"Note, that this is a control frame, which means it will be received in the order it was sent."* It is in-band **because being in-band is the feature.** If it overtook the audio, it would tear down the transport while Lina was mid-word and the customer would hear a click.

And the `UninterruptibleFrame` mixin from §4.3 is what makes that survivable: if a barge-in fires while the `EndFrame` is still queued, `FrameQueue.reset()` flushes the audio and **keeps** the `EndFrame`. The docstring says why: *"Terminal frames must survive interruption to guarantee proper pipeline shutdown."* Without the mixin, an interrupt during teardown would delete your shutdown signal and the worker would hang until something else killed it.

Then the wait. Go back to `_wait_for_pipeline_end` in §7 and read the `else` branch:

```python
        else:
            # Ending flushes what is queued, so cutting the wait short would
            # drop the audio the EndFrame exists to play out. A processor that
            # could hold it up watches for that itself.
            await self._pipeline_end_event.wait()
```

Bare `await`. No `wait_for`. No timeout. **Unbounded, forever, by design**, and the comment states the reasoning: a timeout here would drop the audio that the `EndFrame` exists to play out, which defeats the entire purpose of the graceful path.

So, the asymmetry, stated as the thing to memorize:

| | frame class | priority tier | queue it enters | wait |
|---|---|---|---|---|
| `cancel()` | `CancelFrame(SystemFrame)` | 10 | none — inline on input task | **bounded**, `cancel_timeout_secs` = 20 s |
| `stop_when_done()` | `EndFrame(ControlFrame, UninterruptibleFrame)` | 20 | `__process_queue`, behind the audio | **unbounded** |

The violent path is bounded; the graceful path is not. The reason is not symmetry or taste — it is that the violent path has nothing left to protect and the graceful path exists *only* to protect something. Every intuition that says "surely the graceful shutdown is the one with the timeout" is inverted here.

The operational consequence for the Lina host is direct and you should design for it: **`stop_when_done()` can hang forever.** If a processor never pushes the `EndFrame` onward — a TTS service stuck awaiting a vendor socket that will never reply — that worker never finishes, its entry never leaves `_entries` (§3.1), and nothing in the framework times it out. Your host-level defence is a watchdog: `asyncio.wait_for(worker.wait(), timeout=...)` around the graceful path, falling back to `worker.cancel(reason="drain timeout")`. Pipecat does not ship that. §13 puts it in the topology.

There is also a probe if you want to know whether the pipeline actually drained without ending it:

**`src/pipecat/pipeline/worker.py:831-855`**
```python
    async def flush_pipeline(self, timeout: float = 5.0) -> bool:
        """Flush all in-flight frames from the pipeline and wait for it to drain.

        Pushes a :class:`~pipecat.frames.frames.PipelineFlushFrame` downstream;
        the sink bounces it back upstream and the source sets its event once it
        completes the round-trip, signalling that every frame queued ahead of it
        has been processed. The probe is injected straight into the pipeline so
        it bypasses any ``queue_frame`` override (e.g. tool-call deferral).

        Args:
            timeout: Seconds to wait before giving up. On timeout a warning is
                logged and ``False`` is returned rather than blocking forever
                (e.g. if a processor swallows the probe).

        Returns:
            True if the pipeline drained, False if the wait timed out.
        """
        event = asyncio.Event()
        await self._pipeline.queue_frame(PipelineFlushFrame(event=event))
        try:
            await asyncio.wait_for(event.wait(), timeout)
            return True
        except TimeoutError:
            logger.warning(f"{self}: pipeline flush timed out after {timeout}s")
            return False
```

Note the contrast in engineering posture within one file: this method **does** have a timeout and returns `False` rather than hanging, and the docstring says why — "rather than blocking forever (e.g. if a processor swallows the probe)." The framework knows how to bound a wait. It chose not to bound the `EndFrame` one.

The round trip works because of the second out-of-band axis, direction. The sink bounces the probe **upstream**, and the worker's source catches it:

**`src/pipecat/pipeline/worker.py:1259-1266`**
```python
        if isinstance(frame, PipelineFlushFrame):
            # The flush probe completed its round-trip (down to the sink, back up
            # to the source). Everything queued ahead of it has been processed;
            # release whoever is awaiting it.
            logger.debug(f"{self}: flush probe reached source — pipeline drained")
            if frame.event:
                frame.event.set()
            return
```

That upstream return path is also how the pipeline asks the worker to shut itself down. `_source_push_frame` (`worker.py:1248-1297`) translates upstream worker frames into downstream lifecycle frames — `EndWorkerFrame` → `EndFrame`, `CancelWorkerFrame` → `CancelFrame`, `StopWorkerFrame` → `StopFrame`. And one of them bypasses the push queue entirely, with a comment that is a small lesson in its own right:

**`src/pipecat/pipeline/worker.py:1280-1286`**
```python
        elif isinstance(frame, InterruptionWorkerFrame):
            # Tell the worker we should interrupt the pipeline. Note that we are
            # bypassing the push queue and directly queue into the
            # pipeline. This is in case the push worker is blocked waiting for a
            # pipeline-ending frame to finish traversing the pipeline.
            logger.debug(f"{self}: received interruption worker frame upstream {frame}")
            await self._pipeline.queue_frame(InterruptionFrame())
```

Read the reason: the push task might be sitting in that unbounded `EndFrame` wait. If an interruption had to go through `_push_queue`, it could never arrive during a drain. So it goes straight into the pipeline. That is the *out-of-band principle applied one level up* — same argument as §5.1, different queue.

---

## 9. Exit four — the customer goes silent

The customer said "여보세요," heard Lina's opening, and then… nothing. No hangup. The line is open. Four minutes pass.

Nothing happens. Here is why, and here is the number that is wrong for you.

The plumbing has three parts. First, an observer that notices activity:

**`src/pipecat/pipeline/worker.py:106-140`**
```python
class IdleFrameObserver(BaseObserver):
    """Idle timeout observer.

    This observer waits for specific frames being generated in the pipeline. If
    the frames are generated the given asyncio event is set. If the event is not
    set it means the pipeline is probably idle.

    """

    def __init__(self, *, idle_event: asyncio.Event, idle_timeout_frames: tuple[type[Frame], ...]):
        """Initialize the observer.

        Args:
            idle_event: The event to set if the idle timeout frames are being pushed.
            idle_timeout_frames: A tuple with the frames that should set the event when received
        """
        super().__init__()
        self._idle_event = idle_event
        self._idle_timeout_frames = idle_timeout_frames
        self._processed_frames = set()

    async def on_push_frame(self, data: FramePushed):
        """Callback executed when a frame is pushed in the pipeline.

        Args:
            data: The frame push event data.
        """
        # Skip already processed frames
        if data.frame.id in self._processed_frames:
            return

        self._processed_frames.add(data.frame.id)

        if isinstance(data.frame, StartFrame) or isinstance(data.frame, self._idle_timeout_frames):
            self._idle_event.set()
```

It is installed automatically, but only if a timeout is configured:

**`src/pipecat/pipeline/worker.py:500-506`**
```python
        self._idle_event = asyncio.Event()
        self._idle_monitor_task: asyncio.Task | None = None
        if self._idle_timeout_secs:
            idle_frame_observer = IdleFrameObserver(
                idle_event=self._idle_event,
                idle_timeout_frames=idle_timeout_frames,
            )
            observers.append(idle_frame_observer)
```

Second, a monitor task that waits on that event with a timeout:

**`src/pipecat/pipeline/worker.py:1401-1415`**
```python
    async def _idle_monitor_handler(self):
        """Monitor pipeline activity and detect idle conditions.

        Tracks frame activity and triggers idle timeout events when the
        pipeline hasn't received relevant frames within the timeout period.

        Note: Heartbeats are excluded from idle detection.
        """
        running = True
        while running:
            try:
                await asyncio.wait_for(self._idle_event.wait(), timeout=self._idle_timeout_secs)
                self._idle_event.clear()
            except TimeoutError:
                running = await self._idle_timeout_detected()
```

Third, the action:

**`src/pipecat/pipeline/worker.py:1417-1441`**
```python
    async def _idle_timeout_detected(self) -> bool:
        """Handle idle timeout detection and optional cancellation.

        Returns:
            Whether the pipeline worker should continue running.
        """
        # If we are cancelling, just exit the worker.
        if self._cancelled:
            return False

        logger.warning("Idle timeout detected.")
        await self._call_event_handler("on_idle_timeout")
        if not self._cancel_on_idle_timeout:
            return True

        logger.warning("Idle pipeline detected, cancelling pipeline worker...")
        await self.cancel(reason="idle timeout")
        if self._cancel_runner_on_idle_timeout:
            logger.warning("...and cancelling the runner.")
            # ``BaseWorker.cancel`` sends ``BusCancelMessage`` on the bus
            # so the runner broadcasts cancellation to every other root
            # worker too. This worker's pipeline is already cancelling
            # from the call above.
            await BaseWorker.cancel(self, reason="idle timeout")
        return False
```

### 9.1 The defaults, and both of the things wrong with them for Lina

**`src/pipecat/pipeline/worker.py:281-292`** (excerpt of the keyword-only constructor signature, `:273-303`)
```python
        cancel_on_idle_timeout: bool = True,
        cancel_runner_on_idle_timeout: bool = True,
        cancel_timeout_secs: float = CANCEL_TIMEOUT_SECS,
        check_dangling_tasks: bool = True,
        clock: BaseClock | None = None,
        conversation_id: str | None = None,
        enable_tracing: bool = False,
        enable_turn_tracking: bool = True,
        enable_rtvi: bool = True,
        exclude_frames: tuple[type[Frame], ...] | None = None,
        idle_timeout_frames: tuple[type[Frame], ...] = (BotSpeakingFrame, UserSpeakingFrame),
        idle_timeout_secs: float | None = IDLE_TIMEOUT_SECS,
```

**Problem one: 300 seconds.** Five minutes of dead air on an outbound Korean insurance dial. Thirty seconds is already a lost customer; sixty seconds and they have put the handset on the table and walked away. Five minutes is five minutes of telephony minutes, STT websocket, TTS session and an LLM context you are paying to keep warm for a customer who is gone. The default is not calibrated for your product and there is no reading of it that makes it right.

**Problem two, and this one is a genuine hazard: `cancel_runner_on_idle_timeout=True`.** Read `_idle_timeout_detected` again. On idle, it cancels the worker — fine — and then calls `BaseWorker.cancel(self, ...)`, which the comment tells you "sends `BusCancelMessage` on the bus so the runner broadcasts cancellation to every other root worker too."

On a Lina host with `auto_end=False` and thirty concurrent calls, **one silent customer takes down the other twenty-nine.** This is the same shape as the `runner.cancel()` hazard in §3.2 and it is more insidious, because it fires without anyone calling anything: a customer who says nothing for five minutes triggers a host-wide cancel. If you take one line away from this section, take this: on a multi-session host, `cancel_runner_on_idle_timeout=False` is mandatory.

The retune is a constructor kwarg, not a patch, and not a subclass:

```python
worker = PipelineWorker(
    pipeline,
    idle_timeout_secs=45.0,                       # sales-call dead air, not a browser demo
    cancel_on_idle_timeout=True,                  # end this call
    cancel_runner_on_idle_timeout=False,          # NEVER on a multi-session host
    idle_timeout_frames=(BotSpeakingFrame, UserSpeakingFrame),
)
```

### 9.2 What resets the clock — the part that is easy to get wrong

`idle_timeout_frames=(BotSpeakingFrame, UserSpeakingFrame)` does not mean "reset when a turn happens." Both frames are emitted **periodically while speech is in progress**, on a 200 ms cadence.

**`src/pipecat/transports/base_output.py:459-463`**
```python
            # Last time a BotSpeakingFrame was pushed.
            self._bot_speaking_frame_time = 0
            # How often a BotSpeakingFrame should be pushed (value should be
            # greater than the audio chunks to have any effect).
            self._bot_speaking_frame_period = 0.2
```

**`src/pipecat/transports/base_output.py:774-781`**
```python
        async def _bot_currently_speaking(self):
            """Handle bot speaking event."""
            await self._bot_started_speaking()

            diff_time = time.time() - self._bot_speaking_frame_time
            if diff_time >= self._bot_speaking_frame_period:
                await self._transport.broadcast_frame(BotSpeakingFrame)
                self._bot_speaking_frame_time = time.time()
```

and on the user side, from the VAD processor's own docstring — *"`UserSpeakingFrame`: Pushed periodically while speech is detected"* (`processors/audio/vad_processor.py:34`) — with `speech_activity_period: float = 0.2` (`:45`) and the push at `:86`.

`BotSpeakingFrame`'s docstring names the exact reason it is on the default list:

**`src/pipecat/frames/frames.py:1304-1311`**
```python
class BotSpeakingFrame(SystemFrame):
    """Frame indicating the bot is currently speaking.

    Emitted upstream and downstream by the BaseOutputTransport while the bot is
    still speaking. This can be used, for example, to detect when a user is
    idle. That is, while the bot is speaking we don't want to trigger any user
    idle timeout since the user might be listening.
    """
```

So the clock measures **true mutual silence**: neither party producing audio, for `idle_timeout_secs` continuously. A customer listening to a 90-second Lina monologue is not idle, because `BotSpeakingFrame` fires five times a second the whole way through. That is the right semantics, and it means 45 seconds is a real 45 seconds of nobody talking — genuinely dead air, not a long pause in a busy call.

Which also tells you how to *narrow* the tuple. If you want the timer to measure "the customer has not spoken," drop `BotSpeakingFrame` and pass `idle_timeout_frames=(UserSpeakingFrame,)`. Then a Lina monologue does not reset the clock and a customer who has stopped responding is detected even while the bot is still talking. That is a different product decision — arguably the right one for detecting an abandoned handset — and it is a one-tuple change, not a code change.

One boundary condition worth knowing: `IdleFrameObserver.on_push_frame` also sets the event on `StartFrame`, so the clock starts at pipeline start rather than at construction; and `_maybe_start_idle_task()` is called from the top of `_process_push_queue` (`worker.py:1212`), so no idle monitoring exists during setup. A provider that hangs during connect is `SETUP_TIMEOUT_SECS`'s problem (§6.1), not the idle timer's.

For the record on how this maps to what you already have: per [[boson-gateway-server]], boson's endpointer is `_start_silence_timer` (`websocket.py:616`), which sleeps `silence_timeout_ms / 1000` (default `2000`) then calls `_finalize_partial` (`:661`). That timer is a **turn-boundary** device operating at 2 seconds — it decides when the user's utterance is over. Pipecat's idle timeout at 300 seconds is a **session-abandonment** device. They are two different clocks answering two different questions, and mapping one onto the other directly would be a category error; the boson silence timer's counterpart lives in `src/pipecat/turns/`, which is [[ch-06/read]]'s territory.

---

## 10. The direction split on `queue_frame`, and why later chapters need it

One more mechanism, small, and it is the hinge for two later chapters.

**`src/pipecat/pipeline/worker.py:793-808`**
```python
    async def queue_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        """Queue a single frame to be pushed through the pipeline.

        Downstream frames are pushed from the beginning of the pipeline.
        Upstream frames are pushed from the end of the pipeline.

        Args:
            frame: The frame to be processed.
            direction: The direction to push the frame. Defaults to downstream.
        """
        if direction == FrameDirection.DOWNSTREAM:
            await self._push_queue.put(frame)
        else:
            await self._sink.queue_frame(frame, direction)
```

Two different injection points depending on direction:

- **DOWNSTREAM** → `self._push_queue`, which is drained by `_process_push_queue` and fed into `self._pipeline.queue_frame(...)`. The frame enters at the **head** and traverses every processor.
- **anything else** → `self._sink.queue_frame(frame, direction)`. The frame enters at the **tail** and travels backwards.

Note the second branch is `else`, not `elif direction == UPSTREAM`, and `FrameDirection` has exactly two members (`frame_processor.py:60-69`), so the behaviour is total.

Why this matters downstream in the course:

- [[ch-10/read]]: `FlowManager` lives **outside** the pipeline and drives it by injecting frames. Which end a frame enters from decides which processors see it, and therefore whether a node transition is observed by the aggregators before or after the LLM.
- [[ch-12/read]]: the transition race. Two injections in flight from different directions do not have a global order — §4.2 gave you per-processor ordering only. There is no global ordering in this system, and the direction split is where that stops being an abstraction.
- [[ch-09/read]]: the assistant aggregator sits after `transport.output()` and must push context **backwards** to the LLM. Upstream traffic is not an exotic case; it is how the conversation loop closes.

Also on the worker's convenience wrapper: `queue_frames` (`worker.py:810-829`) accepts either an `Iterable` or an `AsyncIterable` and simply loops, calling `queue_frame` per item. It adds no batching and no atomicity — frames from a `queue_frames` call can interleave with frames from elsewhere.

---

## 11. The canonical pipeline, read as a data-dependency chain

You have the runtime. Now here is the shape everything runs in, and the reason to read it as a **dependency graph** rather than as a call sequence.

**`examples/getting-started/06-voice-agent.py:81-91`**
```python
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )
```

Per [[canonical-voice-bot]] this exact seven-item list appears verbatim in `06a-voice-agent-local.py` (L69–73), `07-function-calling.py` (L110–114) and across `examples/voice/voice-*.py` — it is the house pattern, not one example's choice.

Every position is fixed by **what evidence exists only at that point**:

1. `transport.input()` — sole producer of inbound audio. Nothing upstream exists.
2. `stt` — needs raw audio, produces text. Must sit between audio and anything that reasons over text.
3. `user_aggregator` — needs transcriptions; on turn end it writes the user message into the shared `LLMContext` and pushes a context frame downstream. **That downstream push is the LLM trigger**, so this must be immediately upstream of the LLM.
4. `llm` — consumes the context frame, emits streaming text.
5. `tts` — needs text, produces audio. After the LLM, before the output transport.
6. `transport.output()` — the only component that knows what was *actually played*.
7. `assistant_aggregator` — needs `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame` at real playback boundaries. Put it before `transport.output()` and it commits text the customer never heard.

That last one is the position that surprises people and the one that matters most for a sales script. A barge-in truncates the assistant turn at what was *reached*, not at what was generated — which is the difference between your CRM logging "I offered the 65-세 renewal option" and the customer actually having heard it.

And the entry point — the thing that is genuinely counter-intuitive on first contact:

**`examples/getting-started/06-voice-agent.py:107-114`**
```python
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])
```

**Nothing runs until a frame is queued.** The pipeline is constructed, the worker is added, `runner.run()` is awaited, the `StartFrame` has traversed every processor and every processor is started and listening — and the bot says nothing. A voice agent that must speak first, which every outbound sales call does, produces its first word only because `queue_frames([LLMRunFrame()])` fired from a transport event handler. The pipeline is machinery; the conversation is something you start.

For Lina that is where your opening script goes, and per [[boson-gateway-server]] it is also where the session-identity question lands — Pipecat pipelines are per-connection, so boson's reconnect-and-resume behaviour (session survives disconnect, 1800 s idle TTL, `SessionAccess.authorize`) has to live in an outer FastAPI layer that owns the route. §13 puts that layer in the topology explicitly.

**`ParallelPipeline` exists** (`src/pipecat/pipeline/parallel_pipeline.py:24`, `class ParallelPipeline(BasePipeline)`) and this course does not use it.

---

## 12. What the framework does *not* give you

Short section, because [[deployment-scaling]] carries the detail and [[ch-13/read]] does the accounting. But the deliverable in §13 is only honest if you know the boundary.

The bundled runner is labelled development in its own source: `runner/run.py`'s banner prints `ᓚᘏᗢ PIPECAT DEVELOPMENT RUNNER`. It multiplexes every session as an `asyncio.Task` on one loop:

**`src/pipecat/runner/run.py:215-220`**
```python
def _start_bot_session(coro) -> asyncio.Task:
    """Run a bot in the background, holding a reference until it finishes."""
    task = asyncio.create_task(coro)
    _bot_sessions.add(task)
    task.add_done_callback(_bot_sessions.discard)
    return task
```

with the module-level set at `:212` existing for a reason the comment states at `:209-211`: *"the event loop only holds a weak reference to a task, so one that nothing else references can be collected while it is still running."* That is a real bug class you would otherwise hit yourself when you spawn per-session tasks from a FastAPI route — hold a strong reference.

And the process model is one process, full stop: `uvicorn.run(app, host=args.host, port=args.port)` at `run.py:1999`, with no `workers=` argument. So concurrency is concurrent asyncio tasks on one event loop. There is no process pool, no admission control, no per-session CPU isolation, and no session-count limit anywhere in `runner/`.

Add that to §3.1's missing `remove_workers` and §8's unbounded drain and the shape of your remaining work is clear: **Pipecat gives you a session runtime, not a host.** The host is yours.

---

## 13. Deliverable: the process / session / worker topology for the Lina host

This is what the chapter was for. It is a decision, not a survey, and later chapters build on it rather than re-open it.

### 13.1 The topology

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ONE OS PROCESS  ·  uvicorn, ONE event loop, no workers= arg              │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ FastAPI app (yours — owns routes, auth, session identity)          │  │
│  │   · GET /ws/{session_id}   — accepts, authorises, then hands the   │  │
│  │     accepted WebSocket to a Pipecat transport                      │  │
│  │   · session store + reconnect/resume + idle TTL   (kept, not       │  │
│  │     replaced — Pipecat has no counterpart; see §11)                │  │
│  │   · process-scoped resources: MCP subprocesses, DB pool, HTTP      │  │
│  │     clients — created ONCE at startup, above the workers           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ ONE WorkerRunner, created at startup, run with auto_end=False      │  │
│  │   WorkerRunner(handle_sigint=True, handle_sigterm=True)            │  │
│  │   asyncio.create_task(runner.run(auto_end=False))   ← held ref     │  │
│  │                                                                    │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │  │
│  │   │PipelineWorker│  │PipelineWorker│  │PipelineWorker│  … per call │  │
│  │   │  call A      │  │  call B      │  │  call C      │             │  │
│  │   │  Pipeline(7) │  │  Pipeline(7) │  │  Pipeline(7) │             │  │
│  │   │  ~2 asyncio  │  │              │  │              │             │  │
│  │   │  tasks per   │  │              │  │              │             │  │
│  │   │  processor   │  │              │  │              │             │  │
│  │   └──────────────┘  └──────────────┘  └──────────────┘             │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

Stated as rules:

1. **One OS process per host.** Not one per call. `uvicorn.run(app, ...)` with no `workers=`. Scale out by running more containers, not more loops in one container.
2. **One `WorkerRunner`, created once at application startup**, with `auto_end=False`. This is non-negotiable per §3 — the default exits your host when the first customer hangs up.
3. **One `PipelineWorker` per call**, constructed in the websocket route, registered with `await runner.add_workers(worker)`. Default (auto-generated) names, or the per-connection session UUID; never a phone number (§3.1a).
4. **Every session is an `asyncio.Task` on the same loop.** `add_workers` does this for you via `_start_worker`. You do not create the task; you must not `asyncio.cancel` it (§6).
5. **Per-call shutdown is `worker.*`, never `runner.*`.** `worker.cancel(reason=...)` for a hangup; `worker.stop_when_done()` for a clean close. `runner.cancel()` and `runner.end()` are host-wide (§3.2) and belong only in your process shutdown handler.
6. **Process-scoped resources live above the runner.** MCP subprocesses, DB pools, HTTP clients: FastAPI startup, not `PipelineWorker.__init__`, and passed down through `app_resources` (`worker.py:279`), which the constructor docstring says the framework "passes through untouched" and exposes as `worker.app_resources`. Per [[deployment-scaling]], boson's `GatewayCore` process-scoped MCP subprocess ownership is a **keep**, and re-hosting it above the worker is the whole point — otherwise it is re-spawned per call.

### 13.2 The constants, decided

| Knob | Framework default | Lina | Why |
|---|---|---|---|
| `WorkerRunner(auto_end=...)` | `True` | **`False`** | §3 — otherwise the host exits when the first call ends |
| `WorkerRunner(handle_sigterm=...)` | `False` | **`True`** | §3.2 — container eviction sends SIGTERM |
| `idle_timeout_secs` | `300` | **`45`** | §9 — 5 min of dead air on a sales dial is 5 min of burned minutes |
| `cancel_on_idle_timeout` | `True` | `True` | end the abandoned call |
| `cancel_runner_on_idle_timeout` | `True` | **`False`** | §9.1 — otherwise one silent customer kills every concurrent call |
| `idle_timeout_frames` | `(BotSpeakingFrame, UserSpeakingFrame)` | keep, or narrow to `(UserSpeakingFrame,)` | §9.2 — narrowing measures "customer stopped responding" instead of "mutual silence" |
| `setup_timeout_secs` | `20.0` | **`5.0`** | §6.1 — 20 s of silence after "여보세요" is a lost call |
| `start_timeout_secs` | `20.0` | `20.0` | leave; a trip here is a bug in your processor, not a tunable |
| `cancel_timeout_secs` | `20.0` | `20.0` initially | §7 — revisit under concurrency, alert on `on_pipeline_timeout` |
| `enable_rtvi` | `True` | `True` | §6.3 — costs no queue hop, and [[ch-11/read]] wants the observability |
| `processor_unusable_policy` | `CONTINUE` | decide in [[ch-13/read]] | needs the provider-failover evidence from [[ch-05/read]] and [[ch-07/read]] |

### 13.3 The three things you must build yourself

Named now so they are budgeted, not discovered.

1. **A drain watchdog.** `stop_when_done()` is unbounded (§8). Wrap it: `asyncio.wait_for(worker.wait(), timeout=N)` and fall back to `worker.cancel(reason="drain timeout")`. The framework will not do this and there is no kwarg for it.
2. **Worker-entry hygiene.** There is no `remove_workers` (§3.1c). `WorkerRunner._entries` grows one entry per call for the process lifetime, each holding a live reference to a finished `PipelineWorker` and its whole `Pipeline`. Pick one: recycle the process on a call-count or wall-clock schedule; or pool several runners and retire them; or accept the growth after measuring it for a full day at your real call volume. Do not reach into `_entries` — it is private and it will move.
3. **The outer session layer.** Per [[boson-gateway-server]], Pipecat pipelines are per-connection and give you nothing for principal-to-session binding, the origin allowlist, the bearer/subprotocol auth, the history projection, or the generation-based cancel-and-replace protocol that handles two sockets racing for one session. That layer owns the route and hands an accepted socket to the transport. Pipecat has no counterpart for any of it, so it stays where it is.

### 13.4 The sketch

Not runnable — the transport is [[ch-05/read]]'s subject — but every line below is a decision from the table above.

```python
# --- startup, once per process ---
@app.on_event("startup")
async def _startup():
    app.state.resources = await build_process_resources()   # MCP, DB, HTTP — ONCE
    app.state.runner = WorkerRunner(handle_sigint=True, handle_sigterm=True)
    # hold a strong reference; see run.py:209-211
    app.state.runner_task = asyncio.create_task(
        app.state.runner.run(auto_end=False)                # §3
    )

# --- per call ---
@app.websocket("/ws/{session_id}")
async def _call(ws: WebSocket, session_id: str):
    await authorize(ws, session_id)                          # yours; §13.3.3
    transport = build_transport(ws)                          # ch-05
    worker = PipelineWorker(
        build_lina_pipeline(transport),                      # ch-11, ch-12
        name=f"call-{session_id}",                           # unique; §3.1a
        app_resources=app.state.resources,                   # §13.1.6
        idle_timeout_secs=45.0,                              # §9.1
        cancel_runner_on_idle_timeout=False,                 # §9.1 — mandatory
        setup_timeout_secs=5.0,                              # §6.1
    )
    await app.state.runner.add_workers(worker)               # starts immediately

    @transport.event_handler("on_client_connected")
    async def _connected(transport, client):
        await worker.queue_frames([LLMRunFrame()])           # §11 — Lina speaks first

    @transport.event_handler("on_client_disconnected")
    async def _disconnected(transport, client):
        await worker.cancel(reason="customer hangup")        # NOT runner.cancel(); §3.2

    await worker.wait()                                      # or the drain watchdog
```

Three lines in that sketch are the ones that would have been wrong if you had copied `06-voice-agent.py`: `auto_end=False`, `cancel_runner_on_idle_timeout=False`, and `worker.cancel()` instead of `runner.cancel()`. Each of them turns a working demo into a host that drops every concurrent customer. That is the value of having read `runner.py` rather than the README.

---

## 다음 챕터로

What this chapter hands forward, named so later chapters can cite it instead of re-deriving it:

- **The `N/r` arithmetic** (§5.1) — control latency is queue depth over drain rate, and the only free variable is *N*. [[ch-08/read]] spends this to explain why the interruption cascade must tear down the output queue, the audio clock and the carrier's playout buffer rather than just stopping the producer. [[ch-11/read]] spends it again in the latency budget, where `audio_out_10ms_chunks = 4` reappears as an interrupt-granularity term.
- **The two-queue / two-task model per processor** (§4) — priority tiers 1/10/20, system frames inline on the input task, everything else on a cancellable process task, `FrameQueue.reset()` keeping the `UninterruptibleFrame`s. [[ch-06/read]] needs it to explain why a VAD-originated turn signal arrives ahead of buffered audio; [[ch-08/read]] needs it for the cascade; [[ch-12/read]] needs it because your rule-layer processor will be cancelled mid-`await` and must reset its own accumulators.
- **The bounded/unbounded asymmetry** (§7, §8) — `CancelFrame` is a `SystemFrame` with a 20 s bound, `EndFrame` is a `ControlFrame, UninterruptibleFrame` with no bound at all, and the reason is that one path has nothing to protect and the other exists only to protect something.
- **The direction split on `worker.queue_frame`** (§10) — downstream enters at the head via `_push_queue`, everything else enters at the tail via `_sink`. [[ch-10/read]]'s `FlowManager` injection and [[ch-12/read]]'s transition race are both illegible without it.
- **The Lina host topology** (§13) — one process, one `WorkerRunner(auto_end=False)`, one `PipelineWorker` per call, every session an `asyncio.Task` on the same loop, per-call shutdown through `worker.*` only, and three named pieces of work the framework does not ship.

[[ch-05/read]] fills the one slot this chapter deliberately left empty: `transport.input()` and `transport.output()`, positions 1 and 6 of the canonical chain. It answers where the audio actually comes from — WebRTC, a raw websocket, or a telephony carrier whose entire protocol difference reduces to a `FrameSerializer` — and it is the first chapter where boson's `gateway/server/` has a concrete Pipecat counterpart to sit beside.

Open questions parked here for later, so they are not lost:

- **Container-per-call vs task-per-call.** §13 chose task-per-call for the Lina host. [[deployment-scaling]] notes the counter-pressure: telephony arrives at a webhook, so a container-per-session model needs a warm pool sized to concurrent-call peak. That is a cost/architecture trade, not a correctness one, and it belongs in [[ch-13/read]].
- **`processor_unusable_policy`.** `CONTINUE` / `END` / `CANCEL` (`worker.py:143-160`), applied once per processor. Choosing needs the provider-failover evidence from [[ch-05/read]] and [[ch-07/read]].
- **Where boson's tool_use/tool_result repair goes.** Per [[boson-interrupt-subsystem]], `InterruptionFrame` truncates a turn but never synthesizes a `ToolResultBlock`. §4.3 told you the process task dies mid-`await`; the repair is processor state you write and reset yourself. [[ch-09/read]] owns it.

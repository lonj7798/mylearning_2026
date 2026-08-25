# FrameProcessor — Two Queues, Two Tasks, and the Physics of Barge-In
<!-- slug: frame-processor · type: source · source: src/pipecat/processors/frame_processor.py -->

**Core Insight.** Every Pipecat processor runs **two** asyncio tasks over **two** queues. The input task drains a `PriorityQueue` and executes `SystemFrame`s *inline, immediately*; everything else it shovels into a second queue drained by a separate, **cancellable** task. Barge-in is therefore not a feature — it is the arithmetic consequence of that split: an `InterruptionFrame` outranks the queued bot speech at every processor simultaneously, and handling it cancels the task that was producing that speech.

**Guideline.** Never do slow work in `process_frame()` for a `SystemFrame` — it runs on the input task and blocks every subsequent system frame at that processor. And always call `await super().process_frame(frame, direction)` in a subclass, or `StartFrame` never creates the process task and your processor silently never handles data frames.

## Technical Details

- **`FrameDirection(Enum)`** (L60-69): exactly two members, `DOWNSTREAM = 1`, `UPSTREAM = 2`. Docstring: *"DOWNSTREAM: Frames flowing from input to output. UPSTREAM: Frames flowing back from output to input."*
- **`link()`** (L671) is a doubly-linked list, nothing more:
  ```python
  def link(self, processor: FrameProcessor):
      self._next = processor
      processor._prev = self
      logger.debug(f"Linking {self} -> {self._next}")
  ```
  There is no central scheduler or broker. A "pipeline" is just `_prev`/`_next` pointers.
- **`push_frame(frame, direction=FrameDirection.DOWNSTREAM)`** (L1004) fires `on_before_push_frame`, calls `__internal_push_frame`, fires `on_after_push_frame`. `__internal_push_frame` (L1160) resolves the neighbor by direction and — critically — calls **`await self._next.queue_frame(frame, direction)`** (L1182) or `await self._prev.queue_frame(...)` (L1194). Pushing is *enqueueing on the neighbor*, never a direct call.
- **`process_frame(frame, direction)`** (L820) is the base implementation subclasses override. It notifies the observer, then handles exactly five lifecycle cases:
  ```python
  if isinstance(frame, StartFrame):            await self.__start(frame)
  elif isinstance(frame, InterruptionFrame):   await self._start_interruption()
                                               await self.stop_all_metrics()
  elif isinstance(frame, CancelFrame):         await self.__cancel(frame)
  elif isinstance(frame, (FrameProcessorPauseFrame, FrameProcessorPauseUrgentFrame)):  await self.__pause(frame)
  elif isinstance(frame, (FrameProcessorResumeFrame, FrameProcessorResumeUrgentFrame)): await self.__resume(frame)
  ```

### The mechanism that makes system frames jump ahead

1. **The priority queue.** `FrameProcessorQueue(asyncio.PriorityQueue)` (L132-183) with three tiers:
   ```python
   START_PRIORITY = 1
   SYSTEM_PRIORITY = 10
   DEFAULT_PRIORITY = 20
   ```
   `put()` (L152) assigns `START_PRIORITY` for `StartFrame`, `SYSTEM_PRIORITY` for any `SystemFrame`, else `DEFAULT_PRIORITY`, then pushes `(priority, self.__counter, item)` — a monotonic `__counter` both preserves arrival order *within* a tier and "stops the queue from ever having to compare frames" (dataclass frames are not orderable).
2. **`queue_frame()`** (L700) is the only entry point. It returns early if `self._cancelling`; if `_enable_direct_mode` it processes inline; otherwise `await self.__input_queue.put((frame, direction, callback))`. The `StartFrame` is what spawns the input task (L727) — comment: *"Nothing drains the queue until the StartFrame arrives, so a processor never acts on a frame before it has been started."*
3. **The split, verbatim** — `__input_frame_task_handler` (L1287):
   ```python
   while True:
       (frame, direction, callback) = await self.__input_queue.get()
       ...
       if isinstance(frame, SystemFrame):
           await self.__process_frame(frame, direction, callback)
       elif self.__process_queue:
           await self.__process_queue.put((frame, direction, callback))
   ```
   **This is the whole trick.** System frames are executed on the input task; data/control frames are merely relayed to `self.__process_queue`, a `FrameQueue(frame_getter=lambda item: item[0])` drained by `__process_frame_task_handler` (L1315) on a *different* task, `self.__process_frame_task`.
4. **Interruption cancels the second task, never the first.** `_start_interruption()` (L1130):
   ```python
   current_is_uninterruptible = isinstance(self.__process_current_frame, UninterruptibleFrame)
   if current_is_uninterruptible:
       self.__reset_process_queue()          # drain, keep uninterruptible items
   else:
       await self.__cancel_process_task()    # kill the task mid-await
       self.__create_process_task()          # fresh task + fresh queue
   ```
   `FrameQueue.reset()` (`src/pipecat/utils/frame_queue.py` L84) drains every item and re-inserts only the `UninterruptibleFrame` ones. So a half-spoken TTS response evaporates while a queued `EndFrame` survives.
5. **`broadcast_interruption()`** (L1017) — the public trigger:
   ```python
   self.__reset_process_task()
   await self.stop_all_metrics()
   await self.broadcast_frame(InterruptionFrame)
   ```
   `broadcast_frame(frame_cls, **kwargs)` (L1038) builds **two** instances, cross-links them via `broadcast_sibling_id`, then pushes one DOWNSTREAM and one UPSTREAM. The interruption thus races outward in both directions from wherever VAD fired, and at each hop it lands at priority 10 ahead of the priority-20 audio already queued.

### Ordering guarantees you can rely on

- Within a tier, strict FIFO (the `__counter`). Across tiers, `StartFrame` > `SystemFrame` > data/control, **per processor** — there is no global ordering.
- `_enable_direct_mode=True` (constructor kwarg, L226) bypasses both queues entirely: `queue_frame` calls `__process_frame` directly and both `__create_input_task`/`__create_process_task` return immediately. Ordering guarantees do not apply.
- Pausing is two-tiered too: `pause_processing_frames()` / `resume_processing_frames()` gate the *process* task via `__process_event`; `pause_processing_system_frames()` / `resume_processing_system_frames()` gate the *input* task via `__input_event`. `pause_processing_all_frames_until(ready, timeout=PAUSE_UNTIL_READY_TIMEOUT_SECS)` where `PAUSE_UNTIL_READY_TIMEOUT_SECS = 5.0` (L188).
- `INPUT_TASK_CANCEL_TIMEOUT_SECS = 3` (L192) guards `__cancel_input_task` against a library that swallows `CancelledError`.
- `CancelFrame` → `__cancel()` (L1099) sets `self._cancelling = True` then cancels the process task; from then on `queue_frame` drops everything.
- `has_queued_frame(frame_type)` (L1244) lets a processor ask whether e.g. an `EndFrame` is already waiting, delegating to `FrameQueue.has_frame`.
- **Migration angle:** this replaces boson-agent's entire barge-in subsystem. `packages/gateway/gateway/interrupt/cancellation.py` — `CancellationFlag` (a bool checked between agent-loop iterations, `check()` raising `CancellationError`) plus `cancel_before_llm()`, `cancel_during_streaming(partial_text)`, `cancel_during_tool(tool_name, arguments)` — is a hand-rolled, *cooperative* version of `_start_interruption()`. Pipecat's is *preemptive*: `cancel_task` kills the coroutine mid-`await`, so there is no "check between iterations" latency floor. `packages/gateway/gateway/interrupt/detector.py` `PartialDetector(overlap_chars=10, timing_threshold_ms=1000, silence_timeout_ms=2000)` — inferring barge-in from *text overlap between partial transcripts* — is obviated: Pipecat fires interruption from VAD on the audio stream before any STT text exists, which is the whole point of moving voice server-side. **Collides hard**: `gateway/layers/pipeline.py` `LayerPipeline.process()` is a second, incompatible pipeline abstraction (`async for` over layers with an `ACTION_PRIORITY` table); it cannot become a Pipecat `Pipeline` and must be wrapped inside one custom `FrameProcessor`. **Untouched**: `server/protocol.py` message framing survives only if boson keeps its own WebSocket client — otherwise a Pipecat transport + serializer replaces `server/websocket.py` wholesale.

## Citation
pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (post-1.7.0 main), read 2026-08-25. Repo paths: `src/pipecat/processors/frame_processor.py` (1333 lines), `src/pipecat/utils/frame_queue.py`.

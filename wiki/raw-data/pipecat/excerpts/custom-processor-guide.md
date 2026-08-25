# Writing a Custom FrameProcessor — the escape hatch

<!-- slug: custom-processor-guide · type: source · source: src/pipecat/processors/frame_processor.py, src/pipecat/processors/aggregators/sentence.py, gated.py, gated_llm_context.py, src/pipecat/processors/filters/ -->

**Core Insight.** A custom processor is four lines of ceremony around one `if isinstance(...)` chain. Subclass `FrameProcessor`, `await super().process_frame(frame, direction)` first, then decide per frame whether to forward, swallow, buffer, rewrite, or emit something new. Everything Pipecat ships — filters, aggregators, gates, even the context aggregators — is that same shape, which means arbitrary orchestration logic can be inserted into a Pipecat pipeline without touching or forking any service.

**Guideline.** Always call `await super().process_frame(frame, direction)` as the first statement, and always let `SystemFrame` (and `StartFrame`/`EndFrame`/`CancelFrame`) through unless you are certain otherwise. The default for anything you don't recognize is `await self.push_frame(frame, direction)` — forwarding in the direction it arrived. A processor that silently drops an unrecognized frame breaks pipeline lifecycle and interruption.

## Technical Details

- **Minimal shape.** `FrameProcessor.__init__(self, *, name=None, enable_direct_mode=False, metrics=None, **kwargs)` (`frame_processor.py:222`) — keyword-only, so subclasses take `**kwargs` and pass them up. The only method you must override is `async def process_frame(self, frame: Frame, direction: FrameDirection)`.
- **Why `super().process_frame` is mandatory.** The base implementation (`frame_processor.py:820-847`) is not a no-op: it notifies the observer (`observer.on_process_frame(FrameProcessed(...))`) and runs lifecycle bookkeeping — `StartFrame` → `self.__start(frame)`, `InterruptionFrame` → `self._start_interruption()` + `stop_all_metrics()`, `CancelFrame` → `self.__cancel(frame)`, pause/resume frames → `__pause`/`__resume`. Skip it and your processor never starts its internal tasks, never clears state on interruption, and is invisible to observers. It does **not** push the frame — forwarding is entirely your job.
- **System frames bypass the queue.** Each processor holds two queues (`__input_queue`, `__process_queue`, L259-288). System frames are processed immediately; data and control frames go to the second queue (`__input_frame_task_handler`, L1309-1312). This is why every well-behaved processor forwards `SystemFrame` unconditionally *before* its own logic — `GatedAggregator` does it at `aggregators/gated.py:61-63` with the comment "We must not block system frames."
- **Complete template — `SentenceAggregator`** (`processors/aggregators/sentence.py`, the whole class, 46 lines including docstrings). Transform + buffer + emit, in one file:
  ```python
  class SentenceAggregator(FrameProcessor):
      """Aggregates text frames into complete sentences.

      Frame input/output::

          TextFrame("Hello,") -> None
          TextFrame(" world.") -> TextFrame("Hello, world.")
      """

      def __init__(self):
          super().__init__()
          self._aggregation = ""

      async def process_frame(self, frame: Frame, direction: FrameDirection):
          await super().process_frame(frame, direction)

          # We ignore interim description at this point.
          if isinstance(frame, InterimTranscriptionFrame):
              return

          if isinstance(frame, TextFrame):
              self._aggregation += frame.text
              if match_endofsentence(self._aggregation):
                  await self.push_frame(TextFrame(self._aggregation))
                  self._aggregation = ""
          elif isinstance(frame, EndFrame):
              if self._aggregation:
                  await self.push_frame(TextFrame(self._aggregation))
              await self.push_frame(frame)
          else:
              await self.push_frame(frame, direction)
  ```
  All four moves are visible: **filter** (`return` on `InterimTranscriptionFrame`), **buffer** (`self._aggregation += ...`), **inject a new frame** (`push_frame(TextFrame(...))` — a frame it constructed, not the one it received), **flush on lifecycle** (`EndFrame`), **passthrough default** (the `else`).
- **Frame-level API you get for free** (`frame_processor.py`): `await self.push_frame(frame, direction=FrameDirection.DOWNSTREAM)` (L1004); `await self.broadcast_frame(frame_cls, **kwargs)` (L1038) which builds two instances, cross-links them via `broadcast_sibling_id`, and pushes one each way; `await self.broadcast_interruption()` (L1017); `await self.push_error(error_msg, exception=None, fatal=False, category=None)` (L864); `self.has_queued_frame(frame_type)` (L1244) for O(1) "is another one of these already waiting?"; `self.pipeline_worker` (L440) → `.app_resources`; `self.link(processor)` (L671). Lifecycle hooks to override: `async setup(self, setup: FrameProcessorSetup)` (L636, always `await super().setup(setup)`) and `async cleanup(self)` (L655). Background work uses `self.create_task(coro)` / `await self.cancel_task(task)`, never raw `asyncio.create_task`.
- **Gating pattern, buffered** — `GatedAggregator(gate_open_fn, gate_close_fn, start_open, direction=DOWNSTREAM)` (`aggregators/gated.py:20`). Holds `self._accumulator: list[tuple[Frame, FrameDirection]]`; when the gate flips open it pushes the opening frame then drains the accumulator (L80-84). Gate functions are plain predicates over a frame.
- **Gating pattern, latest-wins** — `GatedLLMContextAggregator(*, notifier: BaseNotifier, start_open=False)` (`aggregators/gated_llm_context.py:14`). Keeps only `self._last_context_frame` (L56, overwriting) and releases it from a background task started in `_start()`: `self._gate_task = self.create_task(self._gate_task_handler())` (L68), which loops on `await self._notifier.wait()`. Correct model for "hold the LLM turn until my external condition fires, and don't replay stale turns."
- **Filtering patterns** (`processors/filters/`, 6 files): `FunctionFilter(filter: Callable[[Frame], Awaitable[bool]], direction=DOWNSTREAM, filter_system_frames=False)` (`function_filter.py:21`) — `_should_passthrough_frame` (L57) hard-codes the three exemptions (wrong direction, `StartFrame`/`EndFrame`/`CancelFrame`, and `SystemFrame` unless opted in) before consulting the predicate. `NullFilter` (`null_filter.py:18-48`) is the 31-line floor: forward `SystemFrame` and `EndFrame`, drop everything else. Also `IdentityFilter` (45 L), `FrameFilter` (53 L), `WakeCheckFilter` (142 L), `WakeNotifierFilter` (58 L).
- **Injection into the LLM turn.** To add a message rather than a frame, a processor holds the `LLMContext` reference and calls `context.add_message(...)`, or pushes `LLMMessagesAppendFrame(messages, run_llm=...)` / `LLMMessagesUpdateFrame` / `LLMMessagesTransformFrame` at an aggregator — see [[llm-service-context]]. To force a turn, push `LLMRunFrame()`.
- **Migration angle:** This is what lets boson's gateway survive intact. `packages/gateway/gateway/stage/machine.py` (101 L) + `stage/context.py` (86 L) become a `FrameProcessor` sitting between the user aggregator and the LLM: it inspects `TranscriptionFrame`/`LLMContextFrame`, mutates its own stage state, and injects the stage prompt via `context.add_message({"role": "developer", ...})` or swaps tools with `LLMSetToolsFrame` — no Pipecat class is subclassed or forked. `packages/gateway/gateway/rules/engine.py` (184 L) + `rules/check.py` (42 L) + `rules/registry.py` (63 L) map onto the `FunctionFilter` predicate shape (`Callable[[Frame], Awaitable[bool]]`) when they gate, and onto the `SentenceAggregator` transform shape when they rewrite. `packages/gateway/gateway/layers/pipeline.py` (396 L) is the awkward one: it is a *second* pipeline abstraction with its own ordering, and porting it as a single monolithic processor keeps it working but forfeits the per-processor queueing, observability, and interruption semantics Pipecat gives each stage — the honest port splits its four layers (`01-filler-filter`, `02-analyzer`, `03-orchestrator`, `04-committer`, per `agents/*/layers/`) into four linked `FrameProcessor`s. `packages/gateway/gateway/script/` (517 L) maps cleanly onto `GatedLLMContextAggregator`: a scripted purchase flow is exactly "hold the context frame until the script says the turn may proceed," and its latest-wins buffering is the right semantics for a barged-in script step. Nothing here requires changing Pipecat; all of it is composition.

## Citation

pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`, read 2026-08-25. Repo path `wiki/raw-data/pipecat/pipecat-src/`. boson-agent read-only at `/Users/jaewon/mywork_2026/Lina_2026/boson-agent-dev/boson-agent`.

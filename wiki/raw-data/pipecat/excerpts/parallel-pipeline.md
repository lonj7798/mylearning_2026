# ParallelPipeline — fan-out by reference, merge by first-arrival dedup
<!-- slug: parallel-pipeline · type: source · source: src/pipecat/pipeline/parallel_pipeline.py -->

**Core Insight.** `ParallelPipeline` fans the *same frame object* into every branch and merges
outputs on a first-come-first-served basis, deduplicated by `frame.id`. It guarantees ordering for
exactly three frames — `StartFrame`, `EndFrame`, `CancelFrame` — and nothing else. Any other
cross-branch ordering you think you have is an accident of timing.

**Guideline.** Use `ParallelPipeline` only when branches produce *disjoint* frame types and their
relative order does not matter. If order across branches matters, use `SyncParallelPipeline`
(`sync_parallel_pipeline.py`, which adds a `SyncFrame` sentinel and a `FrameOrder.PIPELINE` mode)
instead. And never mutate a frame inside a branch — every branch holds the same instance.

## Technical Details
- File is 208 lines; single class `ParallelPipeline(BasePipeline)` at L24.
- **Construction** (L33–76): `def __init__(self, *args)` — each positional arg must be a `list`
  (`TypeError` otherwise, L60–61) and zero args raises
  `Exception("ParallelPipeline needs at least one argument")` (L47–48). For each list it builds a
  private envelope and a real `Pipeline`:
  ```python
  source = PipelineSource(self._parallel_push_frame, name=f"{self}::Source{num_pipelines}")
  sink = PipelineSink(self._pipeline_sink_push_frame, name=f"{self}::Sink{num_pipelines}")
  pipeline = Pipeline(processors, source=source, sink=sink)
  ```
  So each branch is a normal `Pipeline` whose escape hatches are rewired to the parallel merge
  logic. `processors` and `entry_processors` both return `self._pipelines` (L82–106).
- **Not direct mode.** `super().__init__()` is called with no `enable_direct_mode` and the comment
  at L43–44 explains why: *"We don't set it to direct mode because we use frame pausing and that
  requires queues."* Unlike `Pipeline`, this adds a queue hop.
- **Fan-out** (`process_frame`, L133–164) is a plain loop with no copying:
  `for p in self._pipelines: await p.queue_frame(frame, direction)`. Every branch receives the
  **same object**, not a clone. Mutating a frame in one branch is visible in the others.
- **Lifecycle barrier.** Before fanning out a `StartFrame | EndFrame | CancelFrame` it sets
  `self._frame_counter[frame.id] = len(self._pipelines)`, `self._synchronizing = True`, and calls
  `pause_processing_system_frames()` + `pause_processing_frames()` (L156–160). The in-source
  rationale (L142–155) is worth quoting: a fast branch leaking `StartFrame` early lets other
  branches receive data frames before their own start; a leaked `EndFrame` makes the output
  transport shut down "while other branches still have frames to flush, causing lost output"; a
  leaked `CancelFrame` makes `PipelineWorker` "consider cancellation complete prematurely."
- **Merge / dedup** (`_parallel_push_frame`, L166–178):
  ```python
  if frame.id not in self._seen_ids:
      self._seen_ids.add(frame.id)
      if self._synchronizing:
          self._buffered_frames.append((frame, direction))
      else:
          await self.push_frame(frame, direction)
  ```
  This is the whole merge policy. A frame that entered all N branches unchanged is emitted **once**,
  by whichever branch reaches the sink first — the other N−1 copies are silently dropped. Frames
  *created* inside a branch have fresh ids and always pass.
- **Barrier release** (`_pipeline_sink_push_frame`, L180–202): each branch's arrival decrements
  `_frame_counter[frame.id]`; at zero it clears `_synchronizing` and drains
  `_buffered_frames` — with `StartFrame` pushed **before** the flush and `EndFrame`/`CancelFrame`
  **after** it (L193–198), then resumes both pause channels. `_flush_buffered_frames` (L204–208)
  pops FIFO.
- **Ordering caveats, concretely.**
  1. Only three frame types are synchronized; every data/control frame races.
  2. The merged stream interleaves branches in arrival order — a slow LLM branch and a fast
     passthrough branch will interleave nondeterministically run to run.
  3. Dedup is by identity of `frame.id`, so a *shared* frame is attributed to no particular branch;
     you cannot tell which branch emitted it downstream.
  4. `self._seen_ids` is a `set` created at L52 and **never cleared** (only reads at L173 and an
     add at L174 — those are the only three occurrences in the file). Over a long-lived session it
     grows once per distinct frame id. For a multi-hour voice call this is unbounded growth worth
     measuring before adopting `ParallelPipeline` on a hot audio path.
  5. Because branches share instances, per-branch state must live in the processors, not the frame.
- **Migration angle:** this is the mechanism boson-agent's `gateway/layers/` voting phase would
  have to be rebuilt on — and it is a poor fit. `LayerPipeline` (`layers/pipeline.py` L54) runs
  layers to collect *competing actions* and then arbitrates them by `ACTION_PRIORITY`
  (`filter > respond > inject > orchestration > pass`, L41–52), which requires (a) waiting for
  every layer and (b) a total order over the results. `ParallelPipeline` provides neither: it
  merges on first arrival and drops duplicates. The honest mapping is `SyncParallelPipeline` with
  `FrameOrder.PIPELINE` (branch-definition order, `sync_parallel_pipeline.py` L32–48) as the
  *transport* for a fan-out, with the arbitration itself written as a dedicated merge
  `FrameProcessor` placed after it — Pipecat has no built-in equivalent of `_resolve_actions`.
  Legitimate uses in the Lina migration are narrower: running a Korean STT branch alongside a
  keyword/DTMF branch off the same audio, or teeing audio to a recording/analytics branch.
  Untouched: `gateway/rules/` engines can port unchanged as the *body* of a branch processor.

## Citation
pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (v1.7.0 line, changelog head
dated 2026-08-01), read 2026-08-25. Repo path: `src/pipecat/pipeline/parallel_pipeline.py`
(208 L), contrasted with `src/pipecat/pipeline/sync_parallel_pipeline.py`.

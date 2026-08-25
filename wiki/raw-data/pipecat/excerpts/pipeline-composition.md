# Pipeline — linear composition with a source/sink envelope
<!-- slug: pipeline-composition · type: source · source: src/pipecat/pipeline/pipeline.py -->

**Core Insight.** A Pipecat `Pipeline` is *not* a scheduler or a router — it is a doubly-linked
list of `FrameProcessor`s wrapped in a source/sink envelope. All routing intelligence lives in
the processors themselves; the pipeline only decides who is next (`_next`) and who is previous
(`_prev`). Because a `Pipeline` is itself a `FrameProcessor`, pipelines nest with no special case.

**Guideline.** Compose behaviour by *ordering processors*, not by writing dispatch logic. If you
need a frame to reach a component, put the component where that frame already flows. Never expect
the pipeline to fan out, filter, or reorder for you — it does none of those things.

## Technical Details
- File is 202 lines. Three classes: `PipelineSource` (L21–52), `PipelineSink` (L55–88),
  `Pipeline(BasePipeline)` (L91–202).
- Constructor (L99–121):
  `Pipeline(processors: Sequence[FrameProcessor], *, source: FrameProcessor | None = None, sink: FrameProcessor | None = None)`.
  It builds `self._processors = [self._source, *processors, self._sink]` (L119) then calls
  `self._link_processors()` (L121). Defaults are
  `PipelineSource(self.push_frame, name=f"{self}::Source")` and
  `PipelineSink(self.push_frame, name=f"{self}::Sink")` (L117–118) — both bound to the *pipeline's
  own* `push_frame`, which is how a frame escapes the pipeline into the parent chain.
- Linking is four lines (L197–202):
  ```python
  def _link_processors(self):
      prev = self._processors[0]
      for curr in self._processors[1:]:
          prev.link(curr)
          prev = curr
  ```
  `FrameProcessor.link()` (`processors/frame_processor.py` L671–679) is just
  `self._next = processor; processor._prev = self`. Push resolution happens in
  `__internal_push_frame` (L1160–1194): DOWNSTREAM → `await self._next.queue_frame(...)`,
  UPSTREAM → `await self._prev.queue_frame(...)`. Frames never "go" anywhere else.
- **Expected but absent:** the `_link_processors` docstring says "Link all processors in sequence
  **and set their parent**", but the body never sets a parent, and `FrameProcessor` has no
  `parent`/`_parent` attribute at all (`grep -n "_parent\b" processors/frame_processor.py` →
  no hits). The only `parent` in the codebase is `BaseWorker.parent` (`workers/base_worker.py`
  L271). The docstring is stale; do not build on a processor-level parent link.
- Direction handling is asymmetric and deliberate. `PipelineSource.process_frame` (L39–52) sends
  UPSTREAM frames to the injected `_upstream_push_frame` callback and DOWNSTREAM frames to the
  next processor; `PipelineSink.process_frame` (L75–88) mirrors it. `Pipeline.process_frame`
  (L183–195) injects at the correct end: DOWNSTREAM → `self._source.queue_frame(...)`,
  UPSTREAM → `self._sink.queue_frame(...)`.
- `Pipeline`, `PipelineSource` and `PipelineSink` all pass `enable_direct_mode=True`
  (L113, L36, L72). Direct mode (`frame_processor.py` L717–719) skips the internal input queue and
  processes the frame inline in the caller's task — the envelope adds no queue hop or latency.
  Regular processors (direct mode off) queue, and their input task is only created when the
  `StartFrame` arrives (L723–728), so nothing is processed before start.
- Compound-processor protocol: `processors` (L127–137) returns the full list *including* source and
  sink; `entry_processors` (L139–151) returns `[self._source]`; `processors_with_metrics`
  (L153–167) recurses into nested pipelines collecting `p.can_generate_metrics()`.
- `setup()` / `cleanup()` (L169–181) delegate to `BasePipeline._setup_processors` /
  `_cleanup_processors` (`base_pipeline.py` L26–81), which run **concurrently** via
  `asyncio.gather`. A processor that raises during setup is reported with
  `push_error(..., force_treat_as_permanent=True)` and the rest of the pipeline still starts —
  setup is never retried, so that processor stays half-built for the session.
- **Migration angle:** this replaces boson-agent's `gateway/layers/pipeline.py`
  (`LayerPipeline`, 396 L) as the composition primitive — but the two are not equivalent and the
  swap is lossy. `LayerPipeline.process()` runs N layers as a *voting* phase, resolves competing
  actions by `ACTION_PRIORITY` (`filter=0 > respond=1 > inject=2 > orchestration=3 > pass=4`), then
  commits in array order and can short-circuit the LLM entirely. Pipecat has no voting, no action
  priority, and no rollback: a processor either passes a frame on or it doesn't. Porting means
  rewriting each Lina layer as a `FrameProcessor` that either swallows a frame (= `filter`),
  emits its own (= `respond`), or mutates and forwards (= `inject`) — and reimplementing the
  priority arbitration as *processor order*, which is strictly less expressive because order is
  static while `ACTION_PRIORITY` resolves per-turn. `gateway/core.py` `GatewayCore._handle_message`
  (L199–430) has no Pipecat counterpart at all; its responsibilities scatter across processors.
  Untouched: `basement/` (loop, tools, metatool, skills) sits *inside* what would become a single
  LLM-service processor and needs no restructuring for this piece.

## Citation
pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (v1.7.0 line, changelog
head dated 2026-08-01), read 2026-08-25. Repo path: `src/pipecat/pipeline/pipeline.py`,
with `src/pipecat/pipeline/base_pipeline.py` and `src/pipecat/processors/frame_processor.py`.

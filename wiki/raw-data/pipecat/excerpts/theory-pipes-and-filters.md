# Pipes and Filters — why "lego block" is a named style, and what the uniform interface costs

<!-- slug: theory-pipes-and-filters · type: theory · source: src/pipecat/processors/frame_processor.py, src/pipecat/pipeline/pipeline.py, src/pipecat/pipeline/base_pipeline.py, src/pipecat/pipeline/parallel_pipeline.py, src/pipecat/processors/filters/identity_filter.py, src/pipecat/processors/filters/null_filter.py -->

**Core Insight.** The "lego block" feeling is not a metaphor and not Pipecat's invention — it is the **Pipes and Filters** architectural style, and the single property that produces it is the **uniform interface**. Garlan & Shaw state the mechanism exactly: pipe-and-filter systems "support reuse: **any two filters can be hooked together, provided they agree on the data that is being transmitted between them**." Pipecat makes that proviso trivially true by giving *every* processor the identical signature — `async def process_frame(self, frame: Frame, direction: FrameDirection)` (`frame_processor.py:820`) — so agreement on the transmitted data is universal by construction. You can splice a processor into position N for exactly one reason: at the type level, position N has no type. The corollary is the price: uniformity moves the ordering constraints *out* of the type system, where the compiler enforced them, and into your head, where it does not.

**Guideline.** Before splicing a processor into position N, check four things the interface cannot check for you: (1) does it call `await super().process_frame(frame, direction)` first — unconditionally, before any early return; (2) does it re-push every frame class it does not consume, in the direction that frame arrived; (3) does the frame it needs actually exist *upstream* of N, and does anything downstream still need what it swallows; (4) is it a `FrameProcessor` at all, or one of the three unrelated classes in this repo that also define a method named `process_frame`.

## Technical Details

- **Lineage, verified.** McIlroy, Bell Labs internal memo, **11 October 1964**: *"We should have some ways of coupling programs like garden hose—screw in another segment when it becomes necessary to massage data in another way."* Implemented by Ken Thompson in Unix, **January 1973**. Named and analysed as an architectural style by **Garlan & Shaw**, *An Introduction to Software Architecture*, CMU-CS-94-166 (Jan 1994); also in *Advances in Software Engineering and Knowledge Engineering* Vol. I, World Scientific, **1993**. Catalogued as a pattern in **POSA Vol. 1** — Buschmann, Meunier, Rohnert, Sommerlad, Stal, *Pattern-Oriented Software Architecture: A System of Patterns*, Wiley, **1996** (ISBN 0-471-95869-7): *"provides a structure for systems that process a stream of data. Each processing step is encapsulated in a filter component. Data are passed through pipes between adjacent filters."* McIlroy's "screw in another segment" **is** the learner's "add a process in the middle" — same sentence, 62 years apart.

- **The two style invariants, and Pipecat's compliance.** Garlan & Shaw: *"filters must be independent entities: in particular, they should not share state with other filters. Another important invariant is that filters do not know the identity of their upstream and downstream filters."* Pipecat enforces invariant 2 structurally: `FrameProcessor.link()` (`frame_processor.py:671`) sets only `self._next = processor; processor._prev = self`, and `push_frame()` (`frame_processor.py:1004`) delegates to `__internal_push_frame`, which does `await self._next.queue_frame(frame, direction)` — it names a *slot*, never a class. A processor literally cannot reference its neighbour by type.

- **The uniform interface is exactly two methods.** Read one, write one:
  ```python
  async def process_frame(self, frame: Frame, direction: FrameDirection)          # :820
  async def push_frame(self, frame: Frame,
                       direction: FrameDirection = FrameDirection.DOWNSTREAM)     # :1004
  ```
  `FrameDirection` is a two-valued `Enum` — `DOWNSTREAM = 1`, `UPSTREAM = 2` (`frame_processor.py:60-70`). That is the whole contract. `Frame` (`frames/frames.py:65`) is the universal type with three branches: `SystemFrame` (:105), `DataFrame` (:116), `ControlFrame` (:128). **131** `async def process_frame` overrides exist across **117** files in `src/` — and every one of them presents this same shape, which is why substitutability holds.

- **`link()` is the whole splicing story — and it validates nothing.** `Pipeline.__init__` (`pipeline.py:99-121`) does `self._processors = [self._source, *processors, self._sink]` then `self._link_processors()`, which is five lines (`pipeline.py:197-202`):
  ```python
  prev = self._processors[0]
  for curr in self._processors[1:]:
      prev.link(curr)
      prev = curr
  ```
  A fold over a list with `link` as the operator. There is **no type check, no capability negotiation, no ordering assertion anywhere in `src/pipecat/pipeline/`** — I grepped for one; it does not exist. "Splice at position N" is `list.insert(N, p)` on the argument to `Pipeline(...)`, and it always type-checks.

## The algebra (real class names, not hand-waving)

The style gives you a set (all `FrameProcessor`s), an operator (`link`), and two distinguished
elements that actually ship in this repo. Written as laws, with the class that witnesses each:

| Law | Witness in `src/` |
|---|---|
| `p ∘ Identity = Identity ∘ p = p` | `IdentityFilter` (`filters/identity_filter.py:17`) |
| `p ∘ Null ≈ Null` (data frames only) | `NullFilter` (`filters/null_filter.py:18`) |
| `(a ∘ b) ∘ c = a ∘ (b ∘ c)` | `Pipeline(BasePipeline(FrameProcessor))` (`pipeline.py:91`) |
| `a ∥ b` (fan-out, weak merge) | `ParallelPipeline` (`parallel_pipeline.py:24`) |
| `if cond then p else pass` | `FunctionFilter` (`filters/function_filter.py:21`) |

The operator is **not commutative** — `stt ∘ llm ≠ llm ∘ stt` — and nothing in the type system
records that. Hold onto this; it is the entire content of "The price" below.

- **Identity element.** `IdentityFilter` (`processors/filters/identity_filter.py:17`) — its entire body is `await super().process_frame(frame, direction); await self.push_frame(frame, direction)` (:44-45). Inserting it at any position is a provable no-op. Its docstring says it exists *"when testing `ParallelPipeline` to create pipelines that pass through frames"* — the repo uses it as an identity in `tests/test_frame_processor.py:49` and `tests/test_base_worker.py:89`.
- **Zero element.** `NullFilter` (`processors/filters/null_filter.py:18`) — `if isinstance(frame, (SystemFrame, EndFrame)): await self.push_frame(...)` (:47-48), otherwise the frame dies. Everything after it in the chain is annihilated for data purposes. Note it is a *near*-zero, not a true zero: it must leak `SystemFrame` and `EndFrame` or the pipeline cannot start, interrupt, or shut down. That leak is where the algebra meets reality.
- **Associativity via nesting — verified in the type hierarchy.** `class BasePipeline(FrameProcessor)` (`pipeline/base_pipeline.py:19`) and `class Pipeline(BasePipeline)` (`pipeline/pipeline.py:91`). **A pipeline is a processor.** So `Pipeline([a, Pipeline([b, c])])` and `Pipeline([Pipeline([a, b]), c])` are both legal and behave alike: nesting only inserts a `PipelineSource`/`PipelineSink` pair (`pipeline.py:21`, `:55`) whose `process_frame` is a two-arm `match direction:` that forwards. Grouping is free — that is what associativity buys. Corroborated in [[pipeline-composition]].
- **Pipecat itself depends on that law.** Not a theoretical nicety: `PipelineWorker` wraps *your* pipeline in another one at `pipeline/worker.py:537`:
  ```python
  pipeline = Pipeline([edge_source, pipeline, edge_sink])
  ```
  The variable `pipeline` is rebound to a `Pipeline` containing itself as a middle element. If `Pipeline` were not a `FrameProcessor`, this line does not compile. The framework is splicing into *your* composition using the same operation you use — the clearest possible evidence that "add a process in the middle" is the primitive, not a convenience.
- **Substitutability is what makes processors unit-testable.** `pipecat/tests/utils.py:182` is the entire harness: `pipeline = Pipeline([source, processor, sink])`, where `source`/`sink` are `QueuedFrameProcessor`s draining into `asyncio.Queue`s (:171-180). *Any* processor can be dropped into that middle slot. A test rig that is generic over every component in the system is only possible because position N has no type — testability is a downstream consequence of the uniform interface, not a separate design effort.
- **Parallel combinator.** `ParallelPipeline(BasePipeline)` (`pipeline/parallel_pipeline.py:24`), `def __init__(self, *args)` where each arg is a `list` of processors; it builds one `Pipeline(processors, source=source, sink=sink)` per branch (:73). It is *not* a clean product type — merge is first-arrival dedup by `frame.id` with ordering guaranteed for only three frames; see [[parallel-pipeline]].
- **Conditional combinator.** `FunctionFilter` (`processors/filters/function_filter.py:21`) gates on a caller-supplied predicate via `_should_passthrough_frame(frame, direction)` (:57) — the "if" of the algebra.

## The price

- **Every processor must tolerate every frame it does not care about.** A filter that only understands its own frames is not substitutable — it is a filter that works in exactly one position. Uniformity therefore imposes a tax on *every* author, and Pipecat collects it in one mandatory line, first in every override:
  ```python
  async def process_frame(self, frame: Frame, direction: FrameDirection):
      await super().process_frame(frame, direction)     # not optional
  ```
  The base implementation (`frame_processor.py:820-848`) is not bookkeeping — it is the processor's entire lifecycle. It notifies the observer, then dispatches:
  - `StartFrame → self.__start(frame)`, and `__start` (`:1091`) does exactly one thing: `self.__create_process_task()`.
  - `InterruptionFrame → self._start_interruption()` + `stop_all_metrics()` — barge-in.
  - `CancelFrame → self.__cancel(frame)`; plus the pause/resume frames.
- **What breaks when an author forgets it.** Skip the super call and `StartFrame` never creates the process task, so the processor's non-system queue is never drained. **It accepts frames forever and emits nothing** — a silent black hole mid-pipeline, with no exception and no warning. Two sibling failures from the same root:
  - An early `return` *before* re-pushing an unrecognised frame starves everything downstream of it.
  - `EndFrame` is a `ControlFrame` (`frames.py:1899`), so it rides the same queue as data. Swallow it and it never reaches the transport — shutdown hangs rather than crashing. Note `NullFilter` explicitly re-pushes `EndFrame` for this reason; even the *zero element* cannot afford to be a true zero.
- **This contract is unwritten.** I grepped every `*.md` in the repo for `super().process_frame` and found **zero hits** — `AGENTS.md` mentions `process_frame` only under Observers (:78). The rule is enforced by 107 `await super().process_frame` calls in `src/` and by nothing else. (This corrects the claim in [[pipecat-design-philosophy]] that `AGENTS.md` states it.)
- **Uniform in name is not uniform in type.** Three unrelated hierarchies in this repo define a method called `process_frame`, and only one is the pipeline interface: `BaseAudioFilter.process_frame(self, frame: FilterControlFrame)` (`audio/filters/base_audio_filter.py:50`, no `direction`), `BaseUserTurnStartStrategy.process_frame(self, frame: Frame) -> ProcessFrameResult | None` (`turns/user_start/base_user_turn_start_strategy.py:164`, *returns* a value), and `LLMContextSummarizer(BaseObject).process_frame(self, frame: Frame)` (`processors/aggregators/llm_context_summarizer.py:144`). These are not splice-able into a `Pipeline`; the name collision is a trap.
- **"Any processor anywhere" is a type-level truth and a semantic lie.** Nothing stops you writing:
  ```python
  Pipeline([transport.input(), tts, llm, transport.output()])   # links, starts, runs, wrong
  ```
  `_link_processors` accepts it happily. It is simply wrong: `tts` receives no `TextFrame`, because the LLM that produces them sits *downstream* of it. Nothing raises. The bot is silent, and the silence has no error attached to it.
  - The ordering constraint is real and total — STT before LLM before TTS — it just lives nowhere in the code. It is encoded only in which `isinstance` branch each processor happens to test.
  - This is the style working as designed, not a Pipecat bug. Erasing position from the interface is *precisely* the move that makes any two filters connectable; Garlan & Shaw's third liability names the bill: pipe-and-filter systems *"may force a lowest common denominator on data transmission, resulting in added work for each filter to parse and unparse its data."*
  - Pipecat's `Frame` union **is** that lowest common denominator. The `isinstance` checks scattered through 131 `process_frame` overrides **are** the parse work. You did not eliminate the type discipline; you paid for flexibility by moving it from compile time to runtime, and from one place to 131 places.
- **The style's known weakness is Pipecat's actual use case.** Garlan & Shaw: *"because of their transformational character, pipe and filter systems are typically not good at handling interactive applications. This problem is most severe when incremental display updates are required."* A voice agent is that hard case. Pipecat's answer is two structural additions to the classic style: a second flow direction (`UPSTREAM`) so downstream stages can signal back, and a priority queue that runs `SystemFrame`s inline ahead of buffered data so barge-in preempts queued speech ([[frame-processor]]). Read those as *repairs to a style that does not natively support interactivity*, not as decoration.

## The contrast: boson-agent has no seam to splice into

- boson's turn is one function: `async def run_agent_loop(runtime: AgentRuntime, user_input: str) -> AsyncIterator[StreamEvent]` (`packages/basement/basement/loop/agent_loop.py:176`), a 561-line body around `while turn_count < runtime.config.max_turns:` (:207-209) that ends at `break  # Done — text response means end of turn` (:363). Adding a step means editing that `while` body. **There is no position N.** A turn is an atomic call, and function call is the *opposite* connector from a pipe: the caller names the callee (violating Garlan & Shaw's invariant 2) and they share the `runtime` object outright (violating invariant 1).
- **State the trade honestly.** What boson buys is a single, totally-ordered, greppable control flow: every state transition in a turn is on one screen, cancellation semantics are explicit (`cancellation_flag` read at exactly two sites, :344 and :513), and the failure mode is a stack trace, not a frame vanishing at an unknown hop. What it costs is exactly the property the learner wants: no seam. Pipecat inverts both. You gain splice-ability and pay in *distributed* control flow — after the migration, the answer to "why did nothing come out" is no longer a traceback but a frame-flow trace across N processors, which is why `FramePushed`/`FrameProcessed` observers exist at all ([[rtvi-observability]]).
- **Migration angle:** the honest port is not loop-to-pipeline, it is loop-to-*two*-things. boson's turn mechanics (stream → collect tool_uses → execute → append result → re-prompt) are already Pipecat's default pipeline and come free. boson's turn *policy* — `max_turns`, cancellation reconciliation, `<system-reminder>` injection, the Korean sales stage machine — has **no home** in the style: it is cross-cutting state, and Garlan & Shaw's invariant 1 says filters must not share state. So each policy becomes either its own `FrameProcessor` holding that state privately (the way `LLMContext` aggregators do), or a `ParallelPipeline` branch, or it is consciously dropped. Do that classification *before* writing any processor — the first splice is cheap, but a policy you smeared across four processors' `isinstance` checks is the hardest thing in this style to take back out. Practical first move: port one boson stage as a single `FrameProcessor` and prove you can insert and remove it from position N without touching its neighbours. If you cannot, it is shared state masquerading as a filter.

## Closing guideline — the position-N checklist

The type system checks nothing at a splice site, so these are the checks you own. Run them in
order; the first three are about the processor, the last two about the position.

1. **Lifecycle.** Is `await super().process_frame(frame, direction)` the first statement, reached unconditionally on every path? An early `return` above it is the black-hole bug (`frame_processor.py:1091`).
2. **Transparency.** For every frame class the processor does *not* consume, does it `push_frame` onward in **the direction the frame arrived**? Compare against `IdentityFilter` (`identity_filter.py:44-45`) — that is the reference implementation of "tolerate what you don't care about."
3. **Identity of the base class.** Is it actually a `FrameProcessor`? Three other hierarchies define `process_frame` with different arities and return types; none of them link.
4. **Upstream supply.** Does the frame it consumes get *produced* by something before position N? Putting a `TextFrame` consumer above the LLM is a legal, silent no-op.
5. **Downstream demand.** Does anything after position N still need what this processor swallows or rewrites? Removal is the same question in reverse — this is why splicing *out* is as risky as splicing *in*, despite feeling safer.

If a candidate processor fails 1-3, it is a bug. If it fails 4-5, the processor is fine and the
*position* is wrong. Keeping those two diagnoses apart is most of the debugging skill this style
demands, because the interface reports neither.

## Citation

- pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`, v1.7.0 (CHANGELOG 2026-08-01), read 2026-08-25. Paths: `src/pipecat/processors/frame_processor.py` (:60-70, :671, :820-848, :1004, :1091), `src/pipecat/pipeline/pipeline.py` (:91, :99-121, :197-202), `src/pipecat/pipeline/base_pipeline.py` (:19), `src/pipecat/pipeline/parallel_pipeline.py` (:24, :73), `src/pipecat/processors/filters/identity_filter.py` (:17, :44-45), `src/pipecat/processors/filters/null_filter.py` (:18, :47-48).
- Garlan, D. & Shaw, M., *An Introduction to Software Architecture*, CMU-CS-94-166, Carnegie Mellon University, Jan 1994; also in *Advances in Software Engineering and Knowledge Engineering* Vol. I (Ambriola & Tortora, eds.), World Scientific, 1993. §3.1 "Pipes and Filters", pp. 6-8. https://www.cse.msu.edu/~cse870/Materials/Design/intro_softarch-Garlan-Shaw.pdf
- Buschmann, F., Meunier, R., Rohnert, H., Sommerlad, P., Stal, M., *Pattern-Oriented Software Architecture, Volume 1: A System of Patterns*, Wiley, 1996, ISBN 0-471-95869-7 — "Pipes and Filters" architectural pattern.
- McIlroy, M. D., Bell Labs internal memo, 11 Oct 1964 ("garden hose" proposal); pipes implemented by K. Thompson, Unix, Jan 1973. https://www.nokia.com/bell-labs/about/dennis-m-ritchie/mdmpipe.html
- boson-agent: `packages/basement/basement/loop/agent_loop.py` (:176, :207-209, :344, :363, :513) via [[boson-agent-loop]].

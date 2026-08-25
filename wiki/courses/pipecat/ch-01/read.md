---
title: "Pipes and Filters: The Uniform Interface That Makes Splicing Possible"
chapter: ch-01
phase: read
course: pipecat
sources:
  - theory-pipes-and-filters
  - frame-processor
  - pipeline-composition
  - parallel-pipeline
  - processor-vocabulary
  - pipecat-design-philosophy
  - boson-agent-loop
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
verified: 2026-08-25
---

# ch-01 — Pipes and Filters: The Uniform Interface That Makes Splicing Possible

## 왜 이 챕터인가

You said the thing you want out of Pipecat is the lego-block feeling: *"I can add a process in
the middle or remove it very easily."* That sentence is the entire subject of this chapter, and
the first thing worth knowing is that it is not a feeling and not a Pipecat invention. It is a
named architectural style with a 1964 origin memo, a 1993/94 formal write-up, and a precisely
stated cost. Pipecat is one implementation of it, and the implementation is small enough that
you can hold all of it in your head by the end of this page.

The chapter answers three questions in order:

1. **What exactly makes splicing legal?** — one method signature, one write verb, one two-valued
   enum, and a `link()` whose body is three statements.
2. **What algebra do you get for free, and does it actually ship?** — identity, zero,
   associativity, parallel, conditional, higher-order — each with a class in `src/` that
   witnesses it, not a diagram.
3. **What does the flexibility cost?** — two distinct taxes, both of which produce *silent*
   failures, which is why the chapter ends in a checklist rather than a summary.

Then it contrasts all of that against `boson-agent`, where a turn is one 561-line function and
there is no position N to splice into. That contrast is not a scoring exercise. Both designs are
described by what they *do*; nothing in ch-01 through ch-12 ranks them. [[ch-13/read]] is the
only place anything is scored, and it will be scored against evidence these chapters supply.

One process note, because it matters for how you read the rest: **every number in this chapter
was re-measured against the tree at commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` while
writing it.** Two counts in the source excerpt did not survive that re-measurement. I flag them
inline rather than quietly using the corrected value, because knowing *which* claims about a
codebase are fragile is part of the skill this chapter is teaching.

---

## 1. The style has a name, and the name comes with a bill

### 1.1 McIlroy's sentence is your sentence

Doug McIlroy, Bell Labs internal memo, 11 October 1964:

> We should have some ways of coupling programs like garden hose — screw in another segment when
> it becomes necessary to massage data in another way.

Ken Thompson implemented it in Unix in January 1973. Garlan & Shaw named and analysed it as an
architectural style in *An Introduction to Software Architecture* (CMU-CS-94-166, Jan 1994; also
in *Advances in Software Engineering and Knowledge Engineering* Vol. I, World Scientific, 1993).
POSA Vol. 1 (Buschmann et al., Wiley, 1996) catalogued it as a pattern: *"provides a structure
for systems that process a stream of data. Each processing step is encapsulated in a filter
component. Data are passed through pipes between adjacent filters."*

"Screw in another segment when it becomes necessary to massage data in another way" **is** "add
a process in the middle." Same sentence, 62 years apart. You are not asking Pipecat for a
feature; you are asking for the defining property of a style it already implements. All the
citations here come from [[theory-pipes-and-filters]].

### 1.2 The reuse claim, stated exactly

Garlan & Shaw's reuse claim is worth reading with care, because the proviso is where all the
engineering is:

> [Pipe-and-filter systems] support reuse: **any two filters can be hooked together, provided
> they agree on the data that is being transmitted between them.**

Two filters compose iff they agree on the transmitted data. That is a *conditional*. Most
systems fail it — a function that returns `dict[str, float]` does not compose with one that
takes `AudioChunk`. Pipecat's move is to make the antecedent **trivially, universally true** by
giving every processor the identical signature over a single universal type. Agreement is not
negotiated at a splice site; it is guaranteed by construction because there is only one shape.

### 1.3 The two style invariants, and Pipecat's compliance

Garlan & Shaw also state two invariants that a system must satisfy to *be* pipe-and-filter:

> Filters must be independent entities: in particular, they should not share state with other
> filters. Another important invariant is that filters do not know the identity of their
> upstream and downstream filters.

Hold on to invariant 2 in particular. Pipecat enforces it *structurally* — a processor cannot
name its neighbour by type even if it wants to, because the neighbour is reachable only through
a slot, `self._next` / `self._prev`, which is populated by an external linker. We will see the
three lines that do it in §3. Invariant 1 is enforced by convention only, and §11 is where it
starts to bite.

---

## 2. The uniform interface is exactly two methods

### 2.1 Read one, write one

`src/pipecat/processors/frame_processor.py:820`

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
```

`src/pipecat/processors/frame_processor.py:1004`

```python
    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame to the next processor in the pipeline.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await self._call_event_handler("on_before_push_frame", frame)

        await self.__internal_push_frame(frame, direction)

        await self._call_event_handler("on_after_push_frame", frame)
```

That is the contract. One read verb, one write verb. Everything else in the 1,333-line
`frame_processor.py` is machinery serving those two.

### 2.2 The direction axis has exactly two values

`src/pipecat/processors/frame_processor.py:60-69`

```python
class FrameDirection(Enum):
    """Direction of frame flow in the processing pipeline.

    Parameters:
        DOWNSTREAM: Frames flowing from input to output.
        UPSTREAM: Frames flowing back from output to input.
    """

    DOWNSTREAM = 1
    UPSTREAM = 2
```

Two members. Not three, not an open registry. This is Pipecat's first structural deviation from
the classic style: a canonical pipe is unidirectional, and Pipecat adds a return path so that a
downstream stage can signal a stage that ran before it. That second direction is what makes
`push_error(...)` reach the application and what lets an assistant aggregator re-prompt an LLM
that sits *upstream* of it. Read `UPSTREAM` as a repair to the style, not as decoration —
Garlan & Shaw explicitly name interactive applications as the style's weak case, and a voice
agent is that case.

### 2.3 The transmitted data is one union type

`src/pipecat/frames/frames.py:64-65` and `:104-138`

```python
@dataclass
class Frame:
    """Base frame class for all frames in the Pipecat pipeline.
    ...
    """

    id: int = field(init=False)
    name: str = field(init=False)
    pts: int | None = field(init=False)
    broadcast_sibling_id: int | None = field(init=False)
    metadata: dict[str, Any] = field(init=False)
    transport_source: str | None = field(init=False)
    transport_destination: str | None = field(init=False)
```

```python
@dataclass
class SystemFrame(Frame):
    """System frame class for immediate processing.

    A frame that takes higher priority than other frames. System frames are
    handled in order and are not affected by user interruptions.
    """


@dataclass
class DataFrame(Frame):
    """Data frame class for processing data in order. ..."""


@dataclass
class ControlFrame(Frame):
    """Control frame class for processing control information in order. ..."""
```

`Frame` at `:65`, then three branches: `SystemFrame` at `:105`, `DataFrame` at `:116`,
`ControlFrame` at `:128`. That three-way split is the whole subject of [[ch-02/read]] — the
narrow waist — and it is also the thing that makes barge-in arithmetic rather than a feature
(the priority tiers in [[ch-04/read]]). For ch-01 you only need one consequence: **the
transmitted data is a single open union, so "agreement on the transmitted data" is universal.**

### 2.4 How many things actually present this shape — and where the excerpt is wrong

[[theory-pipes-and-filters]] says: *"131 `async def process_frame` overrides exist across 117
files in `src/` — and every one of them presents this same shape."* The first half is right; the
second half is not, and the error matters because the *whole* claim of the style rests on shape
uniformity.

Measured at this commit:

| Measurement | Command | Value |
|---|---|---|
| `async def process_frame` definitions in `src/` | `grep -rn "async def process_frame" src/ \| wc -l` | **131** |
| files containing one | `grep -rln ... \| wc -l` | **117** |
| definitions with the pipeline signature `(self, frame: Frame, direction: FrameDirection)` | `grep -rn "...` | **101** |
| files containing one of those | | **87** |
| definitions with a *different* signature | | **30** |
| files containing one of those | | **30** |

87 + 30 = 117, and no file contains both kinds. So of the 131 definitions, **101 are the
pipeline interface** (100 overrides plus the base implementation at `frame_processor.py:820`),
and **30 belong to seven unrelated class hierarchies that merely happen to use the same method
name**:

| Hierarchy | Base class | Signature | Anchor |
|---|---|---|---|
| audio input filters | `BaseAudioFilter(ABC)` | `process_frame(self, frame: FilterControlFrame)` — no `direction` | `audio/filters/base_audio_filter.py:18`, method `:50` |
| audio mixers | `BaseAudioMixer(ABC)` | `process_frame(self, frame: MixerControlFrame)` | `audio/mixers/base_audio_mixer.py:18`, method `:51` |
| VAD control | `VADController(BaseObject)` | `process_frame(self, frame: Frame)` | `audio/vad/vad_controller.py:31`, method `:122` |
| turn-start strategies | `BaseUserTurnStartStrategy(BaseObject)` | `-> ProcessFrameResult \| None` — **returns a value** | `turns/user_start/base_user_turn_start_strategy.py:39`, method `:164` |
| turn-stop strategies | `BaseUserTurnStopStrategy(BaseObject)` | `-> ProcessFrameResult \| None` | `turns/user_stop/base_user_turn_stop_strategy.py:38`, method `:171` |
| user-mute strategies | `BaseUserMuteStrategy(BaseObject)` | `-> bool` | `turns/user_mute/base_user_mute_strategy.py:14`, method `:46` |
| context summarizer | `LLMContextSummarizer(BaseObject)` | `process_frame(self, frame: Frame)` | `processors/aggregators/llm_context_summarizer.py:57`, method `:144` |

None of these link into a `Pipeline`. `BaseUserTurnStartStrategy.process_frame` even *returns*
something, which is structurally impossible for a pipe — a filter writes by pushing, not by
returning. The excerpt's own guideline names "three unrelated classes"; it is seven hierarchies
and 30 definitions.

**Why this is worth two paragraphs instead of a footnote.** `grep -rn "process_frame"` is the
first thing you will do when you start porting Lina, and roughly a quarter of the hits are not
the interface you are looking for. Worse, three of those seven hierarchies (`turns/*`) are the
ones you will be reading hardest in [[ch-06/read]], where the turn-boundary strategy chain
lives — so the collision happens exactly where you will be least able to afford it. The
uniform-interface claim survives (101 definitions of one shape is still a uniform interface),
but "uniform in name" and "uniform in type" are different properties, and the repo only has the
second one if you filter the grep.

---

## 3. `link()` is the whole splicing story, and it validates nothing

Here is the function that makes lego-block composition true.

`src/pipecat/processors/frame_processor.py:671-679`

```python
    def link(self, processor: FrameProcessor):
        """Link this processor to the next processor in the pipeline.

        Args:
            processor: The processor to link to.
        """
        self._next = processor
        processor._prev = self
        logger.debug(f"Linking {self} -> {self._next}")
```

Three statements: a pointer assignment at `:677`, a back-pointer assignment at `:678`, and a
debug log at `:679`. There is no type check. No capability negotiation. No "can you accept what
I emit" handshake. No ordering assertion. A "pipeline" in Pipecat is a doubly-linked list, and
`link` is `insert`.

Notice what `link` does *not* store: any information about what either processor consumes or
produces. That is invariant 2 enforced in code — after linking, neither processor can name the
other by class, only reach it through a slot.

The write path confirms it. `push_frame` delegates to `__internal_push_frame`:

`src/pipecat/processors/frame_processor.py:1160-1194` (elided)

```python
    async def __internal_push_frame(self, frame: Frame, direction: FrameDirection):
        ...
            if direction == FrameDirection.DOWNSTREAM and self._next:
                logger.trace(f"Pushing {frame} downstream from {self} to {self._next}")
                ...
                await self._next.queue_frame(frame, direction)
            elif direction == FrameDirection.UPSTREAM and self._prev:
                logger.trace(f"Pushing {frame} upstream from {self} to {self._prev}")
                ...
                await self._prev.queue_frame(frame, direction)
```

`await self._next.queue_frame(frame, direction)` at `:1182`, `await self._prev.queue_frame(...)`
at `:1194`. **Pushing is enqueueing on the neighbour, never a direct call.** The processor names
a slot; the slot is filled by whoever ran `link` last. This is also why "why did nothing come
out" is a different debugging problem here than in a call stack — §11 comes back to that.

> **Open [`figures/splice-algebra.html`](figures/splice-algebra.html) now, before §4.** Drag a
> processor into position N and watch `link()` expand into exactly these three statements while
> the fold runs; the point of the interaction is to convince yourself, by trying to break it,
> that there is no fourth statement anywhere that could have rejected your arrangement.

---

## 4. `Pipeline` is a fold with `link` as the operator

`src/pipecat/pipeline/pipeline.py:99-121`

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

`src/pipecat/pipeline/pipeline.py:197-202`

```python
    def _link_processors(self):
        """Link all processors in sequence and set their parent."""
        prev = self._processors[0]
        for curr in self._processors[1:]:
            prev.link(curr)
            prev = curr
```

Four lines of body (`:199-202`) under a def at `:197`. It is a left fold over a list with `link`
as the operator and no accumulator other than the previous element. **There is no type check, no
capability negotiation and no ordering assertion anywhere in `src/pipecat/pipeline/`.** I looked
for one. `grep -n "raise\|assert " pipeline.py base_pipeline.py` returns exactly one hit and it
is a comment about teardown being best-effort. The only `raise`s in the whole package's core
composition path are in `parallel_pipeline.py` and they check *arity and container type*, not
semantics:

`src/pipecat/pipeline/parallel_pipeline.py:47-48, 60-61`

```python
        if len(args) == 0:
            raise Exception("ParallelPipeline needs at least one argument")
```

```python
                raise TypeError(f"ParallelPipeline argument {processors} is not a list")
```

"Is it a list" — not "does branch 2 emit what the merge expects."

### 4.1 A stale docstring you should not build on

Read `:198` again: *"Link all processors in sequence **and set their parent**."* The body never
sets a parent. `FrameProcessor` has no `parent` or `_parent` attribute at all —
`grep -c "_parent\b" src/pipecat/processors/frame_processor.py` returns **0**. The only `parent`
in the codebase is `BaseWorker.parent`, a property at `src/pipecat/workers/base_worker.py:271`,
which is about the worker topology of [[ch-04/read]], not about processors. The docstring is a
leftover. If you were planning to walk from a processor up to its containing pipeline — you
cannot, and this is the first of several places in this course where the prose in the repo
describes a thing the code does not do.

### 4.2 So "splice at position N" is literally `list.insert`

Put §3 and §4 together and the lego claim becomes mechanical:

```python
# before
Pipeline([transport.input(), stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator])

# after — a rule processor spliced between STT and the user aggregator
Pipeline([transport.input(), stt, korean_honorific_rule, user_aggregator, llm, tts,
          transport.output(), assistant_aggregator])
```

The argument is a `Sequence[FrameProcessor]`. Splicing is `list.insert(N, p)`. Removing is
`list.pop(N)`. **It always type-checks, for every `p`, at every `N`.** That is the property you
wanted, stated precisely — and stated precisely, it should already make you slightly nervous,
because "always type-checks" and "always correct" are very far apart. §8 is where that gap gets
a name.

---

## 5. The algebra, with witnesses that actually ship

A style gives you a set and an operator. Pipecat's set is all `FrameProcessor`s; its operator is
`link`. What makes this more than a metaphor is that the distinguished elements of the algebra
are not diagrams — they are classes in `src/` you can import.

| Law | Witness in `src/` |
|---|---|
| `p ∘ Identity = Identity ∘ p = p` | `IdentityFilter` — `processors/filters/identity_filter.py:17` |
| `p ∘ Null ≈ Null` (data frames only) | `NullFilter` — `processors/filters/null_filter.py:18` |
| `(a ∘ b) ∘ c = a ∘ (b ∘ c)` | `Pipeline(BasePipeline(FrameProcessor))` — `pipeline/pipeline.py:91`, `pipeline/base_pipeline.py:19` |
| `a ∥ b` (fan-out, weak merge) | `ParallelPipeline` — `pipeline/parallel_pipeline.py:24` |
| `if cond then p else pass` | `FunctionFilter` — `processors/filters/function_filter.py:21` |
| higher-order: a pipeline parameterized by a strategy | `ServiceSwitcher` — `pipeline/service_switcher.py:247` |

The operator is **not commutative**. `stt ∘ llm ≠ llm ∘ stt`. Nothing in the type system records
that. Hold that thought until §8; it is the entire content of "the price."

### 5.1 Identity — and a fact about it that the repo does not advertise

`src/pipecat/processors/filters/identity_filter.py:17-23, 37-45`

```python
class IdentityFilter(FrameProcessor):
    """A pass-through filter that forwards all frames without modification.

    This filter acts as a transparent passthrough, allowing all frames to flow
    through unchanged. It can be useful when testing `ParallelPipeline` to
    create pipelines that pass through frames (no frames should be repeated).
    """
```

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an incoming frame by passing it through unchanged.

        Args:
            frame: The frame to process and forward.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
```

Two statements. Inserting this at any position is a *provable* no-op, and it is the reference
implementation of "tolerate what you don't care about": it forwards the frame in **the direction
the frame arrived**, not in a hardcoded direction.

Now the fact. `grep -rn "IdentityFilter" src tests` returns **76** hits. `grep -rn
"IdentityFilter" src/` returns **1** — its own class definition. All 75 other references are in
`tests/`. The identity element of Pipecat's composition algebra is used exclusively by the test
suite; no shipping code path composes with it. That is not a criticism, it is a data point about
what "the algebra" is for: it exists so composition is *reasoned about* and *tested*
compositionally, not because production pipelines need a no-op stage.

### 5.2 Zero — and why it is deliberately a *near*-zero

`src/pipecat/processors/filters/null_filter.py:18-24, 38-48`

```python
class NullFilter(FrameProcessor):
    """A filter that blocks all frames except system and end frames.

    This processor acts as a null filter, preventing frames from passing
    through the pipeline while still allowing essential system and end
    frames to maintain proper pipeline operation.
    """
```

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, only allowing system and end frames through.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, (SystemFrame, EndFrame)):
            await self.push_frame(frame, direction)
```

An absorbing zero would swallow *everything*. This one cannot. `:47-48` leaks `SystemFrame` and
`EndFrame`, and it has to: without `SystemFrame`, `StartFrame` never reaches the downstream half
of the pipeline, so nothing downstream ever creates its process task (§7.2), and
`InterruptionFrame` never propagates so barge-in dies at the filter. Without `EndFrame`, the
transport never learns to shut down and the session hangs instead of crashing.

**This is where the algebra meets reality, and it is the most important structural fact in the
chapter.** Pipecat's composition operator is annihilating on the *data plane* and transparent on
the *control plane*. You cannot algebraically sever a Pipecat pipeline. `StartFrame`, `EndFrame`,
`CancelFrame` and `InterruptionFrame` survive every stock filter by construction — and the same
escape hatch appears verbatim one file over:

`src/pipecat/processors/filters/frame_filter.py:36-41`

```python
    def _should_passthrough_frame(self, frame):
        """Determine if a frame should pass through the filter."""
        if isinstance(frame, self._types):
            return True

        return isinstance(frame, (EndFrame, SystemFrame))
```

So `FrameFilter(types=())` *is* `NullFilter()`, and the tests use exactly that spelling rather
than the class: `tests/test_filters.py:61`, `tests/test_pipeline.py:751`, `:1042`, `:1065`.
Meanwhile `grep -rn "NullFilter" src tests examples` returns **1** hit — its own definition at
`null_filter.py:18`. Nobody builds a `NullFilter` because they need one. It exists to close the
algebra. That is a deliberate authorial act and it tells you the authors were thinking in these
terms.

### 5.3 Associativity is a type fact, and Pipecat depends on it in its own code

`src/pipecat/pipeline/base_pipeline.py:19-24`

```python
class BasePipeline(FrameProcessor):
    """Base class for all pipeline implementations."""

    def __init__(self, **kwargs):
        """Initialize the base pipeline."""
        super().__init__(**kwargs)
```

`src/pipecat/pipeline/pipeline.py:91-97`

```python
class Pipeline(BasePipeline):
    """Main pipeline implementation that connects frame processors in sequence.

    Creates a linear chain of frame processors with automatic source and sink
    processors for external frame handling. Manages processor lifecycle and
    provides metrics collection from contained processors.
    """
```

`Pipeline` is a `BasePipeline` is a `FrameProcessor`. **A pipeline is a processor.** Therefore
`Pipeline([a, Pipeline([b, c])])` and `Pipeline([Pipeline([a, b]), c])` are both legal, and both
behave the same — nesting only inserts a `PipelineSource` (`pipeline.py:21`) /`PipelineSink`
(`:55`) pair whose `process_frame` is a two-arm direction switch that forwards. Grouping is
free. That is exactly what associativity buys you.

This is not a theoretical nicety. Pipecat's own runtime splices into *your* composition using
the same operation you use:

`src/pipecat/pipeline/worker.py:537`

```python
            pipeline = Pipeline([edge_source, pipeline, edge_sink])
```

The local `pipeline` is rebound to a new `Pipeline` containing itself as the middle element. If
`Pipeline` were not a `FrameProcessor`, that line does not even parse as valid typing. The
framework adds a process in the middle of your pipeline by calling the same constructor you
called. There is no stronger evidence available that "splice at position N" is the primitive and
not a convenience.

And the envelope is not free-floating overhead — `Pipeline`, `PipelineSource` and `PipelineSink`
all construct with `enable_direct_mode=True` (`pipeline.py:113`, `:36`, `:72`), which per
`frame_processor.py:717-719` skips the internal input queue and processes inline in the caller's
task. Nesting costs you no queue hop. Contrast [[parallel-pipeline]]: `ParallelPipeline`
deliberately does *not* use direct mode, and says so in a comment at `parallel_pipeline.py:43-44`
— *"We don't set it to direct mode because we use frame pausing and that requires queues."*

### 5.4 Parallel is a real combinator with a real caveat

`src/pipecat/pipeline/parallel_pipeline.py:24, 33-50`

```python
class ParallelPipeline(BasePipeline):
    """Pipeline that processes frames through multiple sub-pipelines concurrently.
```

```python
    def __init__(self, *args):
        """Initialize the parallel pipeline with processor lists.

        Args:
            *args: Variable number of processor lists, each becoming a parallel branch.

        Raises:
            Exception: If no processor lists are provided.
            TypeError: If any argument is not a list of processors.
        """
        # We don't set it to direct mode because we use frame pausing and that
        # requires queues.
        super().__init__()

        if len(args) == 0:
            raise Exception("ParallelPipeline needs at least one argument")
```

Each positional arg is a `list` that becomes one real `Pipeline` with rewired escape hatches.
For ch-01 the thing to record is that `∥` is *not* a clean product type: fan-out passes the
**same frame object** to every branch (no copy), and the merge is first-arrival dedup by
`frame.id` with ordering synchronised for exactly three frame types. [[parallel-pipeline]] has
the details and [[ch-04/read]] will need them. Do not reach for it as your default composition
tool.

### 5.5 The conditional

`src/pipecat/processors/filters/function_filter.py:57-71, 73-85`

```python
    def _should_passthrough_frame(self, frame, direction):
        """Check if a frame should pass through without filtering."""
        # Always passthrough frames in the wrong direction
        if self._direction and direction != self._direction:
            return True

        # Always passthrough lifecycle frames
        if isinstance(frame, (StartFrame, EndFrame, CancelFrame)):
            return True

        # If not filtering system frames, passthrough all other system frames
        if not self._filter_system_frames and isinstance(frame, SystemFrame):
            return True

        return False
```

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame through the filter.

        Args:
            frame: The frame to process.
            direction: The direction the frame is moving in the pipeline.
        """
        await super().process_frame(frame, direction)

        passthrough = self._should_passthrough_frame(frame, direction)
        allowed = await self._filter(frame)
        if passthrough or allowed:
            await self.push_frame(frame, direction)
```

Three things to notice, all at `:82-84`.

1. The control-plane escape hatch appears a *third* time, now with `CancelFrame` and `StartFrame`
   named explicitly (`:64`). Same design commitment as §5.2.
2. `await self._filter(frame)` at `:83` is **unconditional** — it runs even when `passthrough`
   is already `True` and the answer will be discarded. A side-effecting predicate therefore sees
   every `StartFrame`, `EndFrame`, `CancelFrame` and `SystemFrame` it will never gate. If you
   port a Lina rule into a `FunctionFilter` predicate and that rule mutates state, it will fire
   on frames you did not intend to route it. Make the predicate pure.
3. `direction` defaults to `DOWNSTREAM`, so an upstream frame passes unfiltered unless you set
   `direction=None`.

### 5.6 Higher-order composition: a pipeline parameterized by a strategy

This is the witness that goes furthest beyond "list of steps," and it is the one worth studying
before you design anything for Lina.

`src/pipecat/pipeline/service_switcher.py:247`

```python
class ServiceSwitcher(ParallelPipeline, Generic[StrategyType]):
    """Parallel pipeline that routes frames to one active service at a time.
```

`src/pipecat/pipeline/service_switcher.py:267-279`

```python
    def __init__(
        self,
        services: list[FrameProcessor],
        strategy_type: type[StrategyType] = ServiceSwitcherStrategyManual,
    ):
        """Initialize the service switcher with a list of services and a switching strategy.

        Args:
            services: List of frame processors to switch between.
            strategy_type: The strategy class to use for switching between services.
                Defaults to ``ServiceSwitcherStrategyManual``.
        """
        _strategy = strategy_type(services)
        super().__init__(*self._make_pipeline_definitions(services, _strategy))
```

`src/pipecat/pipeline/service_switcher.py:369-397`

```python
    @staticmethod
    def _make_pipeline_definition(
        service: FrameProcessor, strategy: ServiceSwitcherStrategy
    ) -> Any:
        async def filter(_: Frame) -> bool:
            return service == strategy.active_service

        # Layout: Filter → Service → Filter
        #
        # filter_system_frames: we want to run filter functions also on system
        # frames.
        #
        # enable_direct_mode: filter functions are quick so we don't need
        # additional tasks.
        return [
            FunctionFilter(
                filter=filter,
                direction=FrameDirection.DOWNSTREAM,
                filter_system_frames=True,
                enable_direct_mode=True,
            ),
            service,
            FunctionFilter(
                filter=filter,
                direction=FrameDirection.UPSTREAM,
                filter_system_frames=True,
                enable_direct_mode=True,
            ),
        ]
```

Read what this is doing. `ServiceSwitcher` takes a *list of processors* and a *strategy class*,
and **generates** a `ParallelPipeline` whose branches are each `[FunctionFilter, service,
FunctionFilter]` — the filters closing over a predicate that consults the strategy object at
frame time. Composition is being computed, not written. The three strategies that ship are
`ServiceSwitcherStrategy` (`:31`), `ServiceSwitcherStrategyManual` (`:148`) and
`ServiceSwitcherStrategyFailover` (`:180`).

**Framework-extension move, and this is the one I want you to actually keep.** A Lina call has a
provider risk you already carry: a Korean STT or TTS vendor degrades mid-call and you have no
in-band failover. `ServiceSwitcherStrategyFailover` is that shape, already built, generic over
`FrameProcessor` — meaning it does not care that the members are STT services. Any two
processors that occupy the same position in a pipeline can be switcher members. So: two Korean
TTS vendors behind one switcher; or a "cheap model / careful model" pair of LLM services
switched by a strategy that reads deal stage; or — the interesting one — an
`InsuranceComplianceStrategy` whose `active_service` picks between a permissive rule processor
and a strict one depending on whether the customer has already been read the mandatory
disclosure. You get per-frame routing without writing any routing code, because the routing is a
predicate compiled into a filter pair. Note the ceiling too: a switcher activates **one** member
at a time. It is a selector, not an arbiter, and [[ch-12/read]] is where the difference between
those two starts to hurt.

---

## 6. Substitutability is why every processor is unit-testable

`src/pipecat/tests/utils.py:169-190`

```python
    received_up = asyncio.Queue()
    received_down = asyncio.Queue()
    source = QueuedFrameProcessor(
        queue=received_up,
        queue_direction=FrameDirection.UPSTREAM,
        ignore_start=ignore_start,
    )
    sink = QueuedFrameProcessor(
        queue=received_down,
        queue_direction=FrameDirection.DOWNSTREAM,
        ignore_start=ignore_start,
    )

    pipeline = Pipeline([source, processor, sink])

    worker = PipelineWorker(
        pipeline,
        cancel_on_idle_timeout=False,
        enable_rtvi=enable_rtvi,
        observers=observers,
        params=pipeline_params,
    )
```

`Pipeline([source, processor, sink])` at `:182`. That is the entire test harness for the whole
framework. `source` and `sink` are `QueuedFrameProcessor`s draining into `asyncio.Queue`s, and
*any* processor drops into the middle slot: an LLM service, a filter, a transport output, a
nested pipeline, a thing you wrote this morning.

A test rig that is generic over every component in a 60-provider framework is only possible
because **position N has no type**. Testability here is a downstream consequence of the uniform
interface, not a separate design effort. Note also what the harness proves and what it does not:
`run_test()` asserts *frame plumbing* — which frames came out, in which direction. It cannot
assert that the arrangement was semantically right, which is §8's problem, and `AGENTS.md:214-223`
says as much when it points you at `pipecat.evals` for behaviour instead.

---

## 7. The price, part 1: the transparency tax and one mandatory line

### 7.1 Every processor must tolerate every frame it does not care about

A filter that only understands its own frames is not substitutable — it is a filter that works
in exactly one position, which defeats the whole style. So uniformity taxes *every* author, and
here is what paying it looks like in a real, small processor:

`src/pipecat/processors/aggregators/sentence.py:40-63`

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and aggregate text into complete sentences.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
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

The `else` at `:62-63` is the tax: for every frame class this processor has no opinion about,
push it onward in the direction it arrived. That single arm is what keeps `SentenceAggregator`
insertable at any position.

Read the rest of it as a worked example of how much can be non-obvious in 24 lines:

- **`:50-51` is a swallow, not a passthrough.** `InterimTranscriptionFrame` hits a bare `return`.
  Splice this above anything that renders interim transcripts and the interim transcripts
  disappear, with no error. That is a *deliberate* swallow, but nothing at the splice site tells
  you it exists.
- **`:56` and `:60` drop `direction`.** `push_frame(TextFrame(...))` uses the default
  `DOWNSTREAM`. For an aggregator that only ever sees downstream text this is fine; as a pattern
  to copy into a Lina rule processor that must also handle upstream frames, it is a bug
  generator.
- **`:56` and `:60` also construct a fresh bare `TextFrame`.** If the input was a
  `TranscriptionFrame` (a `TextFrame` subclass carrying `user_id` and `timestamp`), the output is
  a plain `TextFrame` and that metadata is gone. Subclass identity does not survive this
  processor. [[processor-vocabulary]] flags the identical hazard in
  `StatelessTextTransformer` (`text_transformer.py:48`) and tells you to write a custom
  processor rather than reuse it — that advice applies here too.

### 7.2 The mandatory first line, and exactly what it buys

Every one of those overrides begins the same way:

```python
await super().process_frame(frame, direction)     # not optional
```

Here is why. The base implementation is not bookkeeping — **it is the processor's lifecycle.**

`src/pipecat/processors/frame_processor.py:820-847`

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        observer = self._setup.observer if self._setup else None
        if observer:
            data = FrameProcessed(
                processor=self,
                frame=frame,
                direction=direction,
                timestamp=self.get_clock().get_time(),
            )
            await observer.on_process_frame(data)

        if isinstance(frame, StartFrame):
            await self.__start(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._start_interruption()
            await self.stop_all_metrics()
        elif isinstance(frame, CancelFrame):
            await self.__cancel(frame)
        elif isinstance(frame, (FrameProcessorPauseFrame, FrameProcessorPauseUrgentFrame)):
            await self.__pause(frame)
        elif isinstance(frame, (FrameProcessorResumeFrame, FrameProcessorResumeUrgentFrame)):
            await self.__resume(frame)
```

Observer notification, then five lifecycle cases, ending at `await self.__resume(frame)` on
`:847`. Note what is *not* here: it does not push the frame. Calling `super()` never forwards
anything; forwarding is always your job.

Now follow `StartFrame`:

`src/pipecat/processors/frame_processor.py:1091-1097`

```python
    async def __start(self, frame: StartFrame):
        """Handle the start frame to initialize processor state.

        Args:
            frame: The start frame containing initialization parameters.
        """
        self.__create_process_task()
```

`__start`'s entire body is one call, at `:1097`.

`src/pipecat/processors/frame_processor.py:1222-1229`

```python
    def __create_process_task(self):
        """Create the non-system frame processing task."""
        if self._enable_direct_mode:
            return

        if not self.__process_frame_task:
            self.__reset_process_task()
            self.__process_frame_task = self.create_task(self.__process_frame_task_handler())
```

### 7.3 What actually happens when an author forgets it — traced, not asserted

I traced this rather than repeating the excerpt, because the failure mode is the single most
expensive thing a new Pipecat author can do and the mechanism is worth being exact about.

Each processor runs **two** tasks over **two** queues — the structure [[frame-processor]] calls
"the physics of barge-in," and which [[ch-04/read]] and [[ch-08/read]] take apart properly. Here
we need only enough of it to follow one bug. The *input* task is created by `queue_frame`, not by
`process_frame`:

`src/pipecat/processors/frame_processor.py:713-728`

```python
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

So the *input* task starts regardless of whether you called `super()`. The input task then
splits system frames from everything else:

`src/pipecat/processors/frame_processor.py:1287-1313`

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

The chain is now exact:

1. Your override omits `await super().process_frame(...)`.
2. `StartFrame` arrives → `queue_frame` creates the **input** task (`:728`) → the input task
   dequeues it → it is a `SystemFrame` → `__process_frame` → your override runs → and
   `__start()` is **never reached**.
3. `__create_process_task()` is therefore never called, so `self.__process_frame_task` stays
   `None` and `__process_frame_task_handler` (`:1315`) never runs.
4. Every subsequent `DataFrame` and `ControlFrame` still lands in `self.__process_queue` at
   `:1307`, because that queue is constructed in `__init__` at `:284`, is never assigned `None`,
   and `FrameQueue(asyncio.Queue)` defines neither `__len__` nor `__bool__` — so
   `elif self.__process_queue:` is always truthy and the `RuntimeError` at `:1309-1311` is
   unreachable.
5. Result: **the processor accepts frames forever and emits nothing.** A silent black hole,
   mid-pipeline, with no exception, no warning, no log line.

I want you to notice step 4 specifically. There *is* a defensive `raise` sitting right there,
and it cannot fire. If it could, forgetting `super()` would produce a loud crash naming the
processor and the frame. Instead the guard protects against a condition that never occurs while
the condition that does occur is silent. That is not a Pipecat bug, but it is a good example of
where a framework's diagnostics are and are not aimed.

Two sibling failures from the same root, both silent:

- An early `return` *above* the super call, or above a re-push, starves everything downstream.
  `SentenceAggregator:50-51` is a controlled instance of this; an accidental one looks
  identical.
- `EndFrame` is `class EndFrame(ControlFrame, UninterruptibleFrame)` (`frames/frames.py:1899`),
  so it rides the *data* queue, not the system queue. Swallow it and it never reaches the
  transport: shutdown hangs rather than crashing. This is precisely why `NullFilter` re-pushes
  it (§5.2) — even the zero element cannot afford to be a true zero.

### 7.4 The contract is unwritten, and enforced by 100 call sites

`grep -rn "super().process_frame" --include="*.md" .` across the entire repository returns
**zero** hits. Not in `AGENTS.md`, not in `CONTRIBUTING.md`, not in `README.md`. `AGENTS.md`
mentions `process_frame` only in the Observers bullet. ([[pipecat-design-philosophy]] initially
recorded this rule as stated in `AGENTS.md`; [[theory-pipes-and-filters]] corrects it, and the
grep confirms the correction.)

So what enforces it? Only the pattern. I checked with an AST pass rather than grep, because grep
over-counts here:

```
overrides of the pipeline signature (excluding the base): 100
  reaching the base via `await super().process_frame(...)`:  99
  reaching the base via an explicit unbound call:             1
  not reaching the base at all:                               0
```

Across 86 files. Compliance is total — and the one exception is instructive rather than a lapse:

`src/pipecat/extensions/voicemail/voicemail_detector.py:163-185`

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and control gate state based on notifier signals.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await FrameProcessor.process_frame(self, frame, direction)

        # Gate logic: open gate allows all frames, closed gate filters frames
        if self._gate_opened:
            await self.push_frame(frame, direction)
        elif isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
            # Only allow speaking frames if conversation was NOT detected (i.e., voicemail case)
            # This prevents the UserContextAggregator from issuing a warning about no aggregation
            # to push.
            if not self._conversation_detected:
                await self.push_frame(frame, direction)
        elif isinstance(frame, (SystemFrame, EndFrame, StopFrame)):
            # Always allow system frames through
            # This includes the UserStartedSpeakingFrame and UserStoppedSpeakingFrame
            # which are used to detect voicemail timing.
            await self.push_frame(frame, direction)
```

`ClassifierGate(NotifierGate)` at `:121`. Its parent `NotifierGate.process_frame` (`:91`) has its
own push logic that `ClassifierGate` must *not* run — so `:170` reaches past the parent to the
grandparent with an explicit unbound call, `await FrameProcessor.process_frame(self, frame,
direction)`. It is the only such call in the tree.

**Restate the rule correctly, because this example proves the usual phrasing is wrong:** the
contract is not "call `super()`." It is **"`FrameProcessor.process_frame` must run, exactly
once, before you do anything else."** When you subclass a processor that already overrides
`process_frame`, `super()` gets you the parent's behaviour too — and that may be exactly what
you do not want.

Also note what `grep` would have told you: `grep -rn "await super().process_frame" src/ | wc -l`
returns **107**, but 7 of those 107 belong to the `turns/user_start`, `turns/user_stop` and
`turns/user_mute` strategy hierarchies from §2.4 — different base class, different method,
coincidentally identical string. The pipeline-interface count is 100. [[theory-pipes-and-filters]]
reports 107 as the enforcement count; the enforcement count is 100.

---

## 8. The price, part 2: "any processor anywhere" is a type-level truth and a semantic lie

Here is the canonical, working voice bot that ships in the repo:

`examples/getting-started/06-voice-agent.py:81-91`

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

Seven elements, and the order is load-bearing in at least four independent ways: STT must precede
the aggregator that consumes transcriptions; the aggregator must precede the LLM that consumes
its context; the LLM must precede the TTS that consumes its text; and `assistant_aggregator`
sits **after** `transport.output()` so that what gets recorded as the assistant's turn is what
was actually emitted to the wire, not what the LLM produced.

Now write this instead:

```python
Pipeline([transport.input(), tts, llm, transport.output()])   # links, starts, runs, wrong
```

`_link_processors` accepts it happily. `PipelineWorker` starts it. Every processor sets up. No
exception is raised at construction, at link time, at start time, or at any point during the
call. And it is simply wrong: `tts` receives no `TextFrame`, because the LLM that produces them
sits *downstream* of it. **The bot is silent, and the silence has no error attached to it.**

The ordering constraint is real and total — STT before LLM before TTS — and it lives nowhere in
the code. It is encoded only in which `isinstance` branch each processor happens to test, in a
comment (`# LLM`, `# TTS`), and in the examples directory. `AGENTS.md:60` describes `Pipeline` in
five words — *"Chains processors together"* — and says nothing about order. There is no
"a processor must not block," no "must not hold state across turns," no ordering rule anywhere
in the documented contract.

**This is the style working as designed, not a Pipecat defect.** Erasing position from the
interface is *precisely* the move that makes any two filters connectable. Garlan & Shaw's third
liability names the bill directly:

> [Pipe-and-filter systems] may force a lowest common denominator on data transmission,
> resulting in added work for each filter to parse and unparse its data.

Pipecat's `Frame` union **is** that lowest common denominator. The `isinstance` checks scattered
through 100 `process_frame` overrides **are** the parse work. You did not eliminate type
discipline by adopting this style. You moved it from compile time to runtime, and from one place
to a hundred places. [[ch-02/read]] is about whether that trade is worth it and what the union
looks like from the inside.

Two corollaries you should carry into every splice decision:

1. **Splicing *out* is exactly as dangerous as splicing *in*, and feels safer.** Removing a
   processor is "does anything downstream still need what this produced or preserved" — the same
   unanswerable-by-the-compiler question, asked in reverse. The lego intuition makes removal feel
   like it obviously cannot break anything. It obviously can.
2. **The failure signature is silence, not a crash.** In a text system a wrong arrangement
   usually throws. In a voice system a wrong arrangement produces a call where nobody says
   anything, and by the time you notice, the customer has hung up.

---

## 9. The position-N checklist

The type system checks nothing at a splice site, so these are the checks you own. Run them in
order. The first three are about the **processor**; the last two are about the **position**.

1. **Lifecycle.** Does `FrameProcessor.process_frame` run, exactly once, as the first statement,
   on every path including early returns? Usually that means `await super().process_frame(frame,
   direction)`; when your parent is itself a processor whose behaviour you are replacing, it may
   mean the unbound form at `voicemail_detector.py:170`. Failing this is the silent black hole of
   §7.3.
2. **Transparency.** For every frame class the processor does *not* consume, does it `push_frame`
   onward in **the direction the frame arrived**? The reference implementation is
   `identity_filter.py:44-45`. The reference *anti*-pattern is a bare `return`
   (`sentence.py:50-51`) or a `push_frame` that drops `direction` (`sentence.py:56`).
3. **Identity of the base class.** Is it actually a `FrameProcessor`? Seven other hierarchies in
   this repo define a method named `process_frame` with different arities and return types
   (§2.4). None of them link. This one is easy to get wrong precisely because the method name
   matches.
4. **Upstream supply.** Is the frame it consumes actually *produced* by something before position
   N? A `TextFrame` consumer above the LLM is legal, silent, and useless.
5. **Downstream demand.** Does anything after position N still need what this processor swallows
   or rewrites? Ask it in both directions: for a splice-in and for a splice-out.

**The diagnostic discipline:** if a candidate fails 1-3, the processor is broken. If it fails
4-5, the processor is fine and the *position* is wrong. Keeping those two diagnoses apart is
most of the debugging skill this style demands, because the interface reports neither, and the
observable symptom — nothing came out — is identical for both.

The figure's two failure presets exist to train exactly this discrimination: **'Wrong order'** is
a position failure with a working processor, and **'Forgot super()'** is a processor failure in a
correct position. They produce the same silence. Run both in
[`figures/splice-algebra.html`](figures/splice-algebra.html) and make yourself say which category
each one is *before* you open the side panel.

---

## 10. The contrast: `boson-agent` has no seam to splice into

Everything above assumes there *is* a position N. Your own agent does not have one, and the
comparison is sharper for it. All facts in this section come from [[boson-agent-loop]], read from
the private repo at commit `0a5e8ced2fd9e631dcbbe8c5f4adb68b89a4fafb`; they are not checkable
against the Pipecat clone.

### 10.1 A turn is one function

`packages/basement/basement/loop/agent_loop.py:176` — via [[boson-agent-loop]]

```python
async def run_agent_loop(runtime: AgentRuntime, user_input: str) -> AsyncIterator[StreamEvent]:
```

561 lines, built around a single loop:

`packages/basement/basement/loop/agent_loop.py:207-209` — via [[boson-agent-loop]]

```python
turn_count = 0
while turn_count < runtime.config.max_turns:
```

ending at:

`packages/basement/basement/loop/agent_loop.py:363` — via [[boson-agent-loop]]

```python
        break  # Done — text response means end of turn
```

One function simultaneously owns turn bounding, message-list mutation, provider streaming, tool
dispatch, hook firing, and history repair after cancellation. Adding a step means editing that
`while` body. **There is no position N.** A turn is an atomic call.

### 10.2 A function call is the opposite connector from a pipe

Check `run_agent_loop` against the two style invariants from §1.3:

| Garlan & Shaw invariant | Pipecat | boson-agent |
|---|---|---|
| Filters do not know the identity of their neighbours | `link()` fills `_next`/`_prev` slots; `push_frame` names a slot, never a class (`frame_processor.py:1182`, `:1194`) | The caller names the callee directly: `async for event in run_agent_loop(runtime, content)` at `gateway/core.py:323` |
| Filters do not share state | Each processor holds its own state; frames carry data | `ctx = runtime.context_manager`, `api = runtime.conversation_api`, `hooks = runtime.hook_registry` at `:184-186` — the whole world aliased into three locals, shared outright |

Both invariants are violated, and that is the definition of not being a pipe-and-filter system.
It is a **call-and-return** system, which is a different named style with a different set of
properties. Neither statement is a criticism; they are structural facts, and they predict
different things.

### 10.3 What each connector actually gives you

This is the part worth being precise about, because the honest comparison is not
"seam vs. no seam."

**What the call-and-return structure produces in boson-agent:**

- One totally-ordered, greppable control flow. Every state transition in a turn is readable on
  one screen, top to bottom.
- Explicit cancellation semantics at exactly two sites: `cancellation_flag` is read at `:344`
  (after a tool batch) and `:513` (after one tool completes). You can point at both. The excerpt
  also records the consequence of *where* those sites are: the flag is never checked between
  `TextDelta`s, so a cooperative cancel cannot stop token generation, only the next re-prompt —
  `gateway/interrupt/cancellation.py:171` says so in a comment.
- The failure mode is a stack trace. "Why did nothing come out" is answered by a traceback that
  names a line.
- Turn *policy* has an obvious home. `max_turns` is a loop condition. Exhaustion has a
  `while/else` at `:365` that yields a user-visible message.

**What the pipe-and-filter structure produces in Pipecat:**

- A seam at every position, with the splice cost measured in §3-§4: two pointer assignments.
- Distributed control flow. "Why did nothing come out" is answered by a frame-flow trace across
  N processors, which is why `FramePushed` / `FrameProcessed` observer hooks exist at all
  (`frame_processor.py:829`, `:1174`, `:1186`) and why [[ch-11/read]] is a whole chapter about
  the observer plane.
- Turn policy has **no obvious home**. Pipecat states no turn limit anywhere; `max_turns` must
  be rebuilt as a counting `FrameProcessor` or consciously dropped.
- Cross-cutting state is structurally discouraged. Invariant 1 says filters must not share state,
  so anything that is genuinely global to a turn has to become a processor's private state, a
  frame payload, or something outside the pipeline entirely — which is exactly the design that
  [[ch-10/read]] finds in `FlowManager`.

That is the whole comparison ch-01 makes. Two connectors, two sets of consequences, no verdict.
The second counter-design — your own already-shipped `realtime_voice` — is [[ch-03/read]]'s
subject and is deliberately not previewed here, because a comparison is only worth making once
its baseline has been established in detail.

### 10.4 The migration shape this implies

`run_agent_loop` has no Pipecat counterpart to port *into*. It dissolves, and it dissolves into
two piles that behave very differently:

- **Mechanics** — stream → collect `tool_uses` → execute → append result → re-prompt. This is
  Pipecat's default pipeline and it comes free. [[ch-09/read]] shows the exact frame topology.
- **Policy** — `max_turns`, cancellation reconciliation, `<system-reminder>` injection, the
  Korean sales stage machine. **None of this has a home in the style.** It is cross-cutting
  state, and the style's first invariant forbids exactly that.

Do the classification *before* writing a single processor. The first splice is cheap; a policy
you have smeared across four processors' `isinstance` branches is the hardest thing in this style
to take back out — because taking it out is a splice-*out*, which by §8's corollary is the
dangerous direction.

**A concrete first move, and a real test.** Port exactly one Lina sales stage as a single
`FrameProcessor`. Then prove — with `pipecat.tests.utils.run_test()` and the generic harness from
§6 — that you can `list.insert` it and `list.pop` it from position N without touching either
neighbour, and that the frame stream is byte-identical to the un-spliced run when the stage is
inert. If you cannot make that test pass, what you have is shared state wearing a filter's
signature, and you have learned it for the cost of one processor instead of at integration time.

---

## 11. Three framework-extension moves for Lina

Mechanics you can describe are worth less than mechanisms you can aim. Here are three, in
increasing order of how much of the chapter they use.

**(1) The `IdentityFilter` regression harness.** §5.1 established that the identity element ships
and is exercised 75 times in `tests/`. Use it as a *diff oracle* for your own splices: run a
scripted Korean call through `Pipeline([...])`, then through `Pipeline([..., IdentityFilter(), ...])`
with the identity inserted at the position you are about to put a real rule processor, and assert
the two frame streams are identical. If they are not, your harness is nondeterministic and no
splice you make afterwards will be diagnosable. This is cheap and it catches the class of bug —
timing-dependent frame ordering — that will otherwise waste a week in [[ch-08/read]].

**(2) The near-zero as a compliance kill-switch.** §5.2 showed that the stock filters cannot sever
the control plane: `SystemFrame` and `EndFrame` always leak. That is normally described as a
limitation. Aimed at your domain it is a *guarantee* you can build on. A Korean insurance
tele-sales call has utterances the agent must never produce — an unapproved product claim, a
guarantee of returns. A `FunctionFilter` that blocks `TextFrame`/`TTSSpeakFrame` on a predicate
gives you a hard mute, and the algebra guarantees the call still starts, still interrupts, and
still terminates cleanly while muted, because `StartFrame`, `InterruptionFrame` and `EndFrame`
route around your filter by construction (`function_filter.py:60-71`). You get "silence the bot
without killing the session" for free, which is a property you would otherwise have to design.
Note the §5.5 trap while you build it: your predicate runs on frames it will never gate, so keep
it pure.

**(3) `ServiceSwitcher` as a rule-strength selector.** §5.6 is the highest-leverage class in this
chapter for you, because it demonstrates that a Pipecat composition can be *computed from a
policy object* instead of written as a literal list. Your rule layers today arbitrate per turn;
`ServiceSwitcher` arbitrates per frame, but only over a set of alternatives that occupy the same
position. The mapping that works is not "layers → switcher" but "one decision point → one
switcher": disclosure-given vs. not, first-call vs. follow-up, high-value vs. standard lead. Each
becomes a strategy whose `active_service` returns one of two processors. What it explicitly
cannot do is what your `ACTION_PRIORITY` table does — collect competing actions from N layers and
resolve them by rank (`layers/pipeline.py:42`, via [[pipeline-composition]] and
[[processor-vocabulary]]). A switcher selects; it does not arbitrate. Write down that limit now,
because [[ch-12/read]] makes you derive the seam where arbitration has to live, and it will be a
better derivation if you arrive already knowing which stock class *almost* does it.

---

## 12. What to hold in your head

- The lego-block property has one cause: **one signature, one write verb, one two-valued
  direction enum, over one union type.** `process_frame` at `frame_processor.py:820`, `push_frame`
  at `:1004`, `FrameDirection` at `:60-69`.
- **Splicing is `list.insert`.** `link()` at `:671-679` is two pointer assignments and a log.
  `_link_processors` at `pipeline.py:197-202` is a fold with `link` as the operator. Nothing in
  `src/pipecat/pipeline/` validates a composition.
- **The algebra ships**: identity (`identity_filter.py:17`), near-zero (`null_filter.py:18`),
  associativity by type (`base_pipeline.py:19`, `pipeline.py:91`, relied on at `worker.py:537`),
  parallel (`parallel_pipeline.py:24`), conditional (`function_filter.py:21`), higher-order
  (`service_switcher.py:247`).
- **The control plane is not composable-away.** `SystemFrame` and `EndFrame` leak through every
  stock filter, by design. That is a guarantee, not an oversight.
- **Two silent taxes.** Forgetting to reach `FrameProcessor.process_frame` yields a black hole
  with an unreachable guard sitting next to it; a wrong arrangement yields a working, silent bot.
  Both are diagnosed by the position-N checklist, never by the compiler.
- **Numbers, re-measured at this commit:** 131 `process_frame` definitions in 117 files, of which
  **101 are the pipeline interface** in 87 files (100 overrides + 1 base) and **30 belong to
  seven unrelated hierarchies** in 30 files. **100/100** overrides reach the base — 99 via
  `super()`, 1 via an unbound grandparent call at `voicemail_detector.py:170`, **0** skipping it.
  **0** markdown files in the repo mention the rule.
- **boson-agent is call-and-return, not pipe-and-filter**, and violates both style invariants by
  construction. What that structure gives is one greppable control flow, two nameable
  cancellation sites, and stack traces. What the pipe structure gives is a seam at every
  position, distributed control flow, and no home for turn policy. No verdict here;
  [[ch-13/read]] is the only place anything is scored.

---

## 다음 챕터로

This chapter hands forward four things.

**To [[ch-02/read]] (the narrow waist).** §2.3 and §8 leave one question open and it is the
central one: the uniform interface only works because there is a single universal type, and §8
showed that the price of that type is paid a hundred times in `isinstance` branches. ch-02 opens
`Frame` as a sum type, counts what is actually in it, and asks what the waist has to be narrow
*enough* for — including the specific tension that `SystemFrame` / `DataFrame` / `ControlFrame`
is a three-way split doing two different jobs (priority and interruptibility) at once.

**To [[ch-04/read]] (the runtime).** §7.3 traced the two-queues-two-tasks structure only far
enough to explain one bug. The full picture — `FrameProcessorQueue(asyncio.PriorityQueue)` with
its `START_PRIORITY = 1` / `SYSTEM_PRIORITY = 10` / `DEFAULT_PRIORITY = 20` tiers at
`frame_processor.py:132-183`, `WorkerRunner`, and the asymmetric `CancelFrame` / `EndFrame`
shutdown — is ch-04's subject, re-founded on one Lina sales call so each mechanism answers a
question you already have.

**To [[ch-03/read]] (the baseline).** §10 gave you one counter-design deliberately: a
call-and-return agent loop. The second counter-design is the one you already shipped, and ch-03
characterises `realtime_voice` as a *closed* union where Pipecat has an open sum type. Every
later comparison in this course draws on that baseline, which is why it is established at the
start rather than produced at the end.

**To [[ch-12/read]] (rule layers).** §5.6 and §11 planted a specific limit: `ServiceSwitcher`
selects, it does not arbitrate, and `ACTION_PRIORITY`-style resolution has no stock home. ch-12
gives you three constraints and makes you derive the rule-processor seam yourself. Arrive with
the position-N checklist memorised — it is the tool that derivation uses.

**Practical carry-over.** Before ch-02, do the §10.4 exercise: name every piece of
`run_agent_loop` as either *mechanics* (free in Pipecat) or *policy* (no home). You do not need
Pipecat installed to do it, and the classification is the input to nearly every design decision
the rest of this course asks you to make.

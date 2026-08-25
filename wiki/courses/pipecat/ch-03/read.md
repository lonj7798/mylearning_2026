---
title: "What You Already Built: realtime_voice as the Baseline"
chapter: ch-03
phase: composition
course: pipecat
sources:
  - rtv-vs-pipecat-gap
  - rtv-pipeline-session
  - rtv-vad-chunking
  - rtv-webrtc-transport
  - theory-narrow-waist
deps:
  - ch-01
  - ch-02
figure: figures/rtv-baseline.html
---

# Chapter 3 — What You Already Built: realtime_voice as the Baseline

> **Scope, stated up front and enforced for the whole chapter.** This chapter is **mechanism and
> evidence only**. It casts no keep/replace/hybrid vote. It states no preference. It reaches no
> verdict, and it does not lean toward one. Every sentence below is either a measurement, a
> quotation, or a question. Scoring is [[ch-13/read]]'s job and deferring it is deliberate — you
> cannot score subsystems you have not yet seen the mechanics of, and eight of them are still ahead
> of you ([[ch-04/read]] through [[ch-12/read]]).
>
> **What this chapter hands forward is a fact sheet, not a recommendation.**

---

## 왜 이 챕터인가

[[ch-01/read]] gave you Pipes-and-Filters: a uniform interface at every boundary is what makes
splicing possible. [[ch-02/read]] gave you the narrow waist and its price: Pipecat's uniform
interface is **one open sum type**, `Frame`, and an open sum type makes new *functions* (processors)
cheap and new *cases* (frame types) expensive — Wadler's Expression Problem, playing out in a
production codebase at a measurable rate of **577 `isinstance(frame, ...)` call sites across 136
files**.

That is a lesson about a trade-off. Trade-offs are abstract until you have seen both sides
implemented by someone who had to ship. You have. `packages/realtime_voice/` on branch
`voice-chat-dev` is a working full-duplex Korean voice stack, and its unit of data is **the opposite
bet**: a *closed* union, declared on one line.

```python
# packages/realtime_voice/realtime_voice/types.py:201
# (boson-agent, private; excerpt-attested via [[rtv-pipeline-session]])
VoiceRuntimeEvent = VADEvent | ASREvent | AgentTextDelta | VoiceEvent
```

So this chapter closes the composition phase by making ch-02's lesson concrete on your own code. The
framing question — the one every later chapter will draw on — is not "does realtime_voice have
frames?" (it does not). It is:

> **What did each bet buy, and what did each bet cost, and for which population of author?**

That is a framework-extension question, which is the mode you are strongest in. The rest of the
chapter is the evidence you need to answer it, laid out subsystem by subsystem in both directions:
what realtime_voice implements that Pipecat does not, and what Pipecat implements that
realtime_voice does not. Neither list is a scorecard. Both lists are inputs.

---

## 0. How to read the evidence in this chapter

There are **two classes of claim** in this chapter and they have different verifiability. You must
be able to tell them apart, because [[ch-13/read]] will weigh them differently.

| Class | Source of truth | How you check it |
|---|---|---|
| **Pipecat claims** — file paths, line numbers, class names, counts, LOC, grep results | `wiki/raw-data/pipecat/pipecat-src` at commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` | Open the file. Every number below was re-measured against that tree on 2026-08-25 and the exact command is printed next to it. |
| **realtime_voice / boson-agent claims** — the 3,886 lines, the 561-line `VoiceSession`, the 60 tests, `types.py:201` | the `rtv-*` excerpts under `wiki/raw-data/pipecat/excerpts/`, read directly from the private repo `boson-agent`, branch `voice-chat-dev`, commit `034ce4ca09a2f109e6c248a43bc989f8d26a6abf` (2026-07-29) | You check it against your own repo. It is **not** checkable from this wiki, and nothing in this chapter pretends otherwise. |

Every boson-side code block below carries the boson path **and** the excerpt wikilink it came from.
Every Pipecat-side code block carries the repo-relative path and a line range you can `sed -n` on
right now. When the two disagree with each other, I say so rather than smoothing it over — there is
one such disagreement in §9 and I have flagged it.

**One honest caveat before the numbers start.** The two snapshots are six weeks apart: the boson
excerpt is from 2026-07-29, the Pipecat tree from 2026-08-25. If `realtime_voice` moved after
2026-07-29 — and it is your active repo, so assume it did — every boson number here is a floor, not
a current reading. Re-measure before [[ch-13/read]].

---

## 1. The scale strip, measured

Start with the crude number, because it frames everything and because it is the number that is
easiest to over-read.

```bash
# run in wiki/raw-data/pipecat/pipecat-src
$ find src   -name '*.py' | xargs wc -l | tail -1
  168847 total
$ find tests -name '*.py' | xargs wc -l | tail -1
   92538 total
$ grep -rno "def test_" tests | wc -l
    4236
$ find tests -name 'test*.py' | wc -l
     236
```

Against that, from [[rtv-vs-pipecat-gap]] (excerpt-attested):

| | realtime_voice | Pipecat |
|---|---|---|
| source lines | 3,886 | 168,847 |
| test lines | 1,504 | 92,538 |
| test functions / `def test_` matches | 60 | 4,236 |
| test files | 6 | 236 |
| test:src line ratio | 0.39 | 0.55 |

168,847 / 3,886 ≈ **43×**. That ratio is real and it is also nearly uninformative on its own,
because the two numbers are not measuring the same object. A very large fraction of Pipecat's
168,847 lines is **breadth** — 62 service directories, 11 transport packages, 9 serializers, a CLI,
a registry, templates — none of which realtime_voice attempted, because realtime_voice was written
against exactly one ASR vendor, one TTS vendor, one transport, and one language.

So do not read 43× as 43× the capability. Read it as: *these two artifacts were built to answer
different questions.* The rest of this chapter is about which questions, precisely.

The one place the raw number does carry weight is the last row of §9, where the *names* of the 60
tests — not the ratio — are the evidence.

---

## 2. The unit of data: an open sum type against a closed union

### 2.1 What Pipecat declares

Recall from [[ch-02/read]] and [[theory-narrow-waist]]: the waist carries no payload at all.

**`src/pipecat/frames/frames.py` L64–89**

```python
@dataclass
class Frame:
    """Base frame class for all frames in the Pipecat pipeline.

    All frames inherit from this base class and automatically receive
    unique identifiers, names, and metadata support.

    Parameters:
        id: Unique identifier for the frame instance.
        name: Human-readable name combining class name and instance count.
        pts: Presentation timestamp in nanoseconds.
        broadcast_sibling_id: ID of the paired frame when this frame was
            broadcast in both directions. Set automatically by
            ``broadcast_frame()`` and ``broadcast_frame_instance()``.
        metadata: Dictionary for arbitrary frame metadata.
        transport_source: Name of the transport source that created this frame.
        transport_destination: Name of the transport destination for this frame.
    """

    id: int = field(init=False)
    name: str = field(init=False)
    pts: int | None = field(init=False)
    broadcast_sibling_id: int | None = field(init=False)
    metadata: dict[str, Any] = field(init=False)
    transport_source: str | None = field(init=False)
    transport_destination: str | None = field(init=False)
```

And the three-way branch under it, whose docstrings are a **scheduling contract**, not a content
taxonomy:

**`src/pipecat/frames/frames.py` L104–138**

```python
@dataclass
class SystemFrame(Frame):
    """System frame class for immediate processing.

    A frame that takes higher priority than other frames. System frames are
    handled in order and are not affected by user interruptions.
    """

    pass


@dataclass
class DataFrame(Frame):
    """Data frame class for processing data in order.

    A frame that is processed in order and usually contains data such as LLM
    context, text, audio or images. Data frames are cancelled by user
    interruptions.
    """

    pass


@dataclass
class ControlFrame(Frame):
    """Control frame class for processing control information in order.

    A frame that, similar to data frames, is processed in order and usually
    contains control information such as update settings or to end the pipeline
    after everything is flushed. Control frames are cancelled by user
    interruptions.

    """

    pass
```

There is a fourth name in that neighbourhood and it is worth being precise about, because it is
easy to mis-file as a fourth branch:

**`src/pipecat/frames/frames.py` L147–157**

```python
class UninterruptibleFrame:
    """A marker for data or control frames that must not be interrupted.

    Frames with this mixin are still ordered normally, but unlike other frames,
    they are preserved during interruptions: they remain in internal queues and
    any task processing them will not be cancelled. This ensures the frame is
    always delivered and processed to completion.

    """

    pass
```

`UninterruptibleFrame` does **not** inherit from `Frame`. It is a mixin, applied as a second base —
`class EndFrame(ControlFrame, UninterruptibleFrame)` at L1899. So the branch structure is three
classes plus one orthogonal marker, not four peers.

### 2.2 How wide the "narrow" waist actually is

Counting frame classes is a place where honest people get different numbers depending on their
method, so here is mine, printed, with its method stated.

```python
# AST walk over src/pipecat/frames/frames.py at commit 0cbf9c5b
# ---------------------------------------------------------------
# total classes declared in the file ....................... 133
# classes whose name ends in "Frame" ....................... 131
# transitive descendants of `Frame` ........................ 123
#   subtree under SystemFrame (excl. the root) .............  48
#   subtree under DataFrame   (excl. the root) .............  33
#   subtree under ControlFrame(excl. the root) .............  39
#   ------------------------------------------------- sum ... 120
#   UNION of the three subtrees, deduplicated .............. 119
#   carrying the UninterruptibleFrame mixin ................  13
```

The three lines that do not add up are the interesting ones, and they reconcile exactly:

- the branch subtrees **sum** to 120 but their **union** is 119, because `InputTextRawFrame` is
  declared `class InputTextRawFrame(SystemFrame, TextFrame)` — it sits under `SystemFrame` *and*
  under `DataFrame` (via `TextFrame`) and is therefore counted twice;
- `LLMContextFrame` at L551 subclasses `Frame` **directly**, in no branch at all — the only such
  class in the file — so it is in the 123 and in neither the 119 nor the 120;
- 119 (union) + 1 (`LLMContextFrame`) + 3 (the branch roots themselves) = **123**.

[[theory-narrow-waist]] reports the same file at 123 descendants / 120 "concrete" and calls out the
same two anomalies. The course outline's figure spec for this chapter describes the ch-02 AST walk as
`129 → 122 → 119`; the `119` is the deduplicated branch union above and the `122` is that plus the
three roots, which reconciles two of its three numbers. **I could not reproduce `129` by any counting
rule I tried, so I do not assert it.** Reconcile it against [[ch-02/read]] when you get there; if it
cannot be reproduced there either, it is wrong and should be corrected rather than repeated.

The dispatch tax, which is the number that actually matters:

```bash
$ grep -rn "isinstance(frame," src | wc -l        # 577
$ grep -rln "isinstance(frame," src | wc -l       # 136
```

**577 sites, 136 files.** Every new `Frame` subclass is a new obligation on all of them.

### 2.3 Why an unhandled frame vanishes rather than raising

This is the mechanical consequence of the open sum type, and it is worth reading the actual method
rather than trusting a summary of it. `FrameProcessor.process_frame` — the base implementation that
every processor calls via `super()` — handles exactly five frame shapes and **has no `else`**:

**`src/pipecat/processors/frame_processor.py` L820–847**

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

Note what is not there: no fallthrough branch, no `await self.push_frame(frame, direction)` at the
bottom. Forwarding is a separate method the *subclass* must call:

**`src/pipecat/processors/frame_processor.py` L1004–1015**

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

So: you add a frame type, and every processor whose author did not anticipate it drops it —
silently, with no exception, no log line at error level, no type error. The frame reaches the
processor, matches none of the five `isinstance` arms, the method returns, and the frame is gone.
That is the failure mode the open sum type ships with.

### 2.4 What realtime_voice declares instead

**`packages/realtime_voice/realtime_voice/types.py:201`** — boson-agent, private, excerpt-attested
via [[rtv-pipeline-session]]:

```python
VoiceRuntimeEvent = VADEvent | ASREvent | AgentTextDelta | VoiceEvent
```

That single line is the entire frame taxonomy. Not a base class with 123 descendants — a **PEP 604
union of four frozen dataclasses**, closed by construction.

The payload types it unions over, per [[rtv-pipeline-session]], are all
`@dataclass(frozen=True, slots=True)`: `AudioFrame` (L48, PCM16 only — `__post_init__` raises
`InvalidAudioFrameError` for any other `AudioFormat` and for byte lengths not aligned to
`2 * channels`), `VADEvent` (L95), `ASREvent` (L113), `AgentRequest` (L126), `AgentTextDelta`
(L134), `TextChunk` (L142), `TTSRequest` (L149), `SynthesizedAudio` (L161), `VoiceEvent` (L188).

And one structural detail that has no Pipecat counterpart at all: correlation is **on every
payload**. `SessionId / TurnId / GenerationId / PhraseId` are `NewType("...", str)` at L16–19, and
every frame carries a `CorrelationIds(session_id, turn_id, generation_id)` (L27). In Pipecat, the
base `Frame` carries `id`, `name`, `pts`, `broadcast_sibling_id`, `metadata`, `transport_source`,
`transport_destination` — see the L64–101 quote above — and **no conversational correlation
fields**. Session/turn/generation identity, when Pipecat code needs it, rides in the open
`metadata: dict[str, Any]`, which [[theory-narrow-waist]] identifies as the ossification escape
hatch: the base field set is effectively frozen, so anything new goes in the untyped dict.

There is also a structural absence to name plainly. Pipecat's `SystemFrame` / `DataFrame` /
`ControlFrame` split encodes **scheduling priority in the type of the datum**. `VoiceRuntimeEvent`
has no such split. In realtime_voice, priority is not a property of the datum at all — it is a
property of which queue the datum was put on, and there are four separate queues with three
different overflow policies (§3.3). Same concern, different location.

### 2.5 Where the closed union surfaces at the boundary

The interesting thing about a closed union is not the declaration. It is what happens at the edge of
the system, when something arrives that the union does not name. In realtime_voice that edge is the
WebRTC transport's wire-mapping function, from [[rtv-webrtc-transport]]:

**`packages/realtime_voice/realtime_voice/transport/webrtc/transport.py` L118–156** — `_control_event()`,
the hand-written mapping from the closed union to wire types (excerpt-attested):

- `VoiceEvent` → `event.kind.value` (already dotted, e.g. `"assistant.audio_committed"`)
- `AgentTextDelta` → `"text_delta"`
- `ASREvent` → `"transcript.interim" | "transcript.final" | "asr.end_of_turn" | "asr.error"`
- `VADEvent` → `"vad.speech_started" | "vad.speech_stopped"`
- anything else → `TypeError(f"unsupported voice event: {type(event).__name__}")`

Put the two failure modes side by side, because they are the whole trade:

| | new datum that nobody handles |
|---|---|
| **Pipecat, open sum type** | reaches the processor, matches no `isinstance` arm in `process_frame`, method returns, **frame is silently dropped**; the pipeline keeps running and the symptom shows up somewhere downstream as missing audio or a stalled turn |
| **realtime_voice, closed union** | reaches `_control_event`, matches no arm, **`TypeError` raised at the wire boundary**; and before that, `mypy` had a shot at rejecting it at the union declaration |

### 2.6 What each bet buys — the actual question

Say it flatly, in both directions, with no thumb on the scale.

**The open sum type buys third-party extension without touching the core.** Anyone — a vendor
shipping an STT integration, you writing a Korean rules processor — can declare a `Frame` subclass
in their own module and it composes with the entire existing pipeline. Pipecat's own `flows/`
package does exactly this: it defines **two** frames, `FunctionActionFrame(ControlFrame)` and
`ActionFinishedFrame(ControlFrame)`, in `flows/actions.py` rather than in `frames.py`
([[theory-narrow-waist]], and see [[ch-10/read]]). The core file never changed. The price is the 577
sites and the silent drop: nothing anywhere can prove that every processor handles every frame,
because the set of frames is not knowable at compile time.

**The closed union buys exhaustiveness.** Because the set of cases is written on one line, a type
checker can walk a `match` over `VoiceRuntimeEvent` and tell you at check time that you forgot a
case. An unknown event is a loud `TypeError` at a named boundary rather than a frame vanishing at an
unknown hop. The price is that **the core is the only place extension can happen**: adding an event
type means editing `types.py:201` and then every `isinstance` chain that consumes the union, and a
third party cannot do it at all without a patch to your package.

Notice that these are not two answers to one question. They are answers to **two different
questions about who is going to write the next component**:

- Pipecat is optimising for a population of **external authors it will never meet** — 62 service
  directories' worth of vendors, arriving weekly. For that population, "you can extend without a
  PR to the core" is the requirement, and exhaustiveness is unattainable anyway because the case set
  is open by design.
- realtime_voice is optimising for a population of **one team that owns the whole file** — you. For
  that population, editing `types.py:201` is a two-minute cost, and what you get back is that the
  type checker becomes a second reviewer on every event-handling site in the package.

Both are defensible. Which one is right is a function of *who writes the next component*, and that
is a fact about your team and your roadmap, not a fact about either codebase. It is also exactly
what [[ch-13/read]] has to decide, which is why this chapter refuses to.

### 2.7 Use the figure here

→ **[Open the ch-03 baseline viewer](./figures/rtv-baseline.html)** and work the *headline panel*
before reading on: press **"add a frame type"** on the Pipecat side and watch which processors light
up as silent-drop sites, then press **"add an event type"** on the realtime_voice side and follow
the forced edit from `types.py:201` out through every `isinstance` chain to `_control_event` raising
`TypeError`. The checker strip under each column is the point — it shows what a type checker can
prove on the right and cannot prove on the left. Do both directions before you form an opinion; the
panel is deliberately symmetric and has no vote button.

### 2.8 Framework-extension probe #1

You are going to add `RuleViolation` — a verdict from boson's rule layer ([[ch-12/read]]) that a
downstream guard must see **before** TTS emits — to each system. Do not answer in prose; answer as a
diff.

1. In Pipecat: which file does the new class go in, which base does it subclass, and — given the
   `process_frame` quote in §2.3 — how many processors between the rule layer and `transport.output()`
   have to change for the guard to actually see it?
2. In realtime_voice: same question. Count the edits. `types.py:201` is one. What is the rest of the
   list, and which of those edits would a type checker have found for you?
3. Now change the population: a third party wants to ship a Korean honorific-consistency checker as
   a pip-installable package, against each system. Which of the two edits above is even *possible*
   for them?

---

## 3. The unit of work: a processor abstraction against a supervisor

### 3.1 What Pipecat composes

Pipecat's unit of work is `FrameProcessor`, and composition is a doubly-linked list built by one
method:

**`src/pipecat/processors/frame_processor.py` L671–679**

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

Two pointers, `_next` and `_prev`. Two pointers is why frames can travel in two directions:

**`src/pipecat/processors/frame_processor.py` L60–69**

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

and the push method dispatches on exactly that:

**`src/pipecat/processors/frame_processor.py` L1170–1183**

```python
            if direction == FrameDirection.DOWNSTREAM and self._next:
                logger.trace(f"Pushing {frame} downstream from {self} to {self._next}")

                if observer:
                    data = FramePushed(
                        source=self,
                        destination=self._next,
                        frame=frame,
                        direction=direction,
                        timestamp=timestamp,
                    )
                    await observer.on_push_frame(data)
                await self._next.queue_frame(frame, direction)
            elif direction == FrameDirection.UPSTREAM and self._prev:
```

`Pipeline` is then almost nothing — a list, wrapped in a source and a sink, linked pairwise:

**`src/pipecat/pipeline/pipeline.py` L113–121, L197–202**

```python
        super().__init__(enable_direct_mode=True)

        # Add a source and a sink queue so we can forward frames upstream and
        # downstream outside of the pipeline.
        self._source = source or PipelineSource(self.push_frame, name=f"{self}::Source")
        self._sink = sink or PipelineSink(self.push_frame, name=f"{self}::Sink")
        self._processors: list[FrameProcessor] = [self._source, *processors, self._sink]

        self._link_processors()
```

```python
    def _link_processors(self):
        """Link all processors in sequence and set their parent."""
        prev = self._processors[0]
        for curr in self._processors[1:]:
            prev.link(curr)
            prev = curr
```

That six-line loop is the whole topology mechanism. Insert an element into `processors`, and the
pipeline is different. That is the Pipes-and-Filters property [[ch-01/read]] identified: **topology
is data**.

What happens *inside* each processor at runtime — the two tasks, the two queues, the priority
scheduling — is [[ch-04/read]]'s subject and I am deliberately not spending it here.

### 3.2 What realtime_voice supervises

From [[rtv-pipeline-session]], `packages/realtime_voice/realtime_voice/pipeline/session.py`:

- `VoiceSession` is declared at **L89** and runs **561 lines**.
- Its docstring, L90: *"Supervise audio -> VAD -> ASR -> text agent -> TTS -> queued audio."*
- The constructor takes the four stages as keyword args (`vad`, `asr`, `agent`, `tts`) plus a
  `VoiceSessionConfig`.
- `_supervise()` at **L257** opens **one** `asyncio.TaskGroup` and creates exactly **two**
  long-lived tasks for the whole session: `"voice-input"` (`_input_loop`) and `"voice-asr-events"`
  (`_asr_loop`).
- Per-turn work spawns nested tasks: `f"asr-finalize:{gen}"`, `f"voice-generation:{gen}"`, and
  inside `_run_generation` an inner `TaskGroup` with `f"agent-text:{gen}"` + `f"tts:{gen}"`.

Two long-lived tasks for the entire session, plus a short-lived nest per turn. Pipecat's
per-processor runtime — how many tasks and queues each element of the chain owns — is [[ch-04/read]];
the figure's second panel draws the realtime_voice side against an explicit placeholder for it, so
you can see the shape difference before you have the mechanics.

Also absent, per [[rtv-pipeline-session]], and stated as absence rather than as deficiency: there is
no `Pipeline` equivalent, no `PipelineTask`, no `PipelineRunner`, no `ParallelPipeline`, no observer
plane, no frame-level metrics. `errors.py` (30 L) is the entire error surface for the core package:
`RealtimeVoiceError(Exception)` with `InvalidAudioFrameError(RealtimeVoiceError, ValueError)`,
`QueueOverflowError`, `QueueClosedError`, `ProviderError`,
`ProviderTimeoutError(ProviderError, TimeoutError)`, `SessionClosedError`.

### 3.3 Swappable slots, fixed topology

`protocols.py` is 82 lines and defines five `@runtime_checkable Protocol`s and nothing else
([[rtv-pipeline-session]]):

```python
# packages/realtime_voice/realtime_voice/protocols.py — shape as recorded in [[rtv-pipeline-session]]
class VAD(Protocol):    async def process(self, frame: AudioFrame, correlation: CorrelationIds) -> Sequence[VADEvent]
class StreamingASR(Protocol):  async def start / push_audio / finalize / close;  def events() -> AsyncIterator[ASREvent]
class StreamingTTS(Protocol):  async def start / cancel(generation_id) / close;  def synthesize(request) -> AsyncIterator[SynthesizedAudio]
class StreamingConversationAgent(Protocol):  def stream(request) -> AsyncIterator[AgentTextDelta];  async def cancel / close
class VoiceTransport(Protocol):  async def start / send_audio / send_event / close;  def incoming_audio() -> AsyncIterator[AudioFrame]
```

This is the sentence to hold onto: **the slots are replaceable, the topology is not.** You can hand
`VoiceSession` a different `StreamingTTS`. You cannot put a processor *between* the ASR and the
agent, because there is no "between" — the order VAD → ASR → agent → TTS is written into
`_supervise()` and its nested task structure. There is no `link()`, no `FrameDirection`, and no
upstream push at all: data moves one way.

That last item is not cosmetic. In Pipecat, closing a tool-call loop means pushing an
`LLMContextFrame` **upstream** ([[function-calling]], and [[ch-09/read]]). A one-directional
supervisor has no mechanism for that shape; realtime_voice instead delegates the whole tool loop out
of the package entirely, which is §7.6.

Interposition being impossible is precisely the property [[ch-01/read]] said Pipes-and-Filters buys.
realtime_voice forgoes it. What it gets for forgoing it is that there is exactly one file where the
order of operations lives, and reading that one file tells you the whole data path — which is a real
property, and the reason a 561-line class is comprehensible at all.

The bounded queues are where scheduling policy went, since it is not in the type of the datum.
`VoiceSessionConfig` at L41, defaults verbatim from [[rtv-pipeline-session]]:

```python
sample_rate=16_000; channels=1; language="ko"
ingress_queue_size=64; event_queue_size=256; phrase_queue_size=8; audio_queue_size=32
vad_prefix_frames=5; phrase_min_chars=12; phrase_max_chars=60
phrase_hard_max_chars=None; phrase_batch_max_chars=320; adaptive_sentence_batching=True
```

Three different overflow policies across four queues:

| queue | policy | mechanism |
|---|---|---|
| ingress (64) | **reject on overflow** | `push_audio` (L164) raises `QueueOverflowError("ingress queue full; frame rejected instead of adding latency")` |
| phrase (8) | backpressure | producer awaits |
| audio (32) | backpressure | producer awaits |
| event (256) | effectively unbounded | `_emit` awaits |

Class docstring L92–94: *"Ingress uses reject-on-overflow so a transport cannot silently extend user
turn latency."* That is a latency-preservation decision encoded as a queue policy — hold it for
[[ch-11/read]], where Pipecat's latency accounting gets measured.

### 3.4 Shutdown: a propagating frame against 24 hand-sequenced lines

Pipecat shuts a pipeline down by pushing a frame that travels the linked list:

**`src/pipecat/frames/frames.py` L1899–1910**

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

Ordering, flush semantics and interruption-survival are all expressed *in the type of the frame*:
`ControlFrame` gives it in-order delivery relative to data, the `UninterruptibleFrame` mixin makes it
survive a barge-in. Each processor's own `__cancel` / shutdown path runs as the frame passes.

realtime_voice has no `EndFrame`, so `close()` at L231 is an explicit sequence
([[rtv-pipeline-session]], 24 lines):

```
emit SESSION_CLOSING
  → cancel active generation with semantic_interrupt=False
      (comment L239-241: "Closing a transport is lifecycle cleanup, not a customer interruption")
  → _ingress.put(_STOP)
  → asr.close()
  → await supervisor
  → vad.close()
  → agent.close()
  → tts.close()
  → _audio.close()
  → emit SESSION_CLOSED
  → _events.put(_STOP)
```

And because there is no terminal frame, loop termination needs a sentinel object instead:

```python
# packages/realtime_voice/realtime_voice/pipeline/session.py L37 — via [[rtv-pipeline-session]]
_STOP = object()
```

`_STOP` is pushed onto the ingress, event and phrase queues to end their loops, so **every queue is
typed `asyncio.Queue[X | object]` and every consumer does a `cast(...)`**. That `| object` in the
type is the direct, visible cost of not having an `EndFrame` — the queue's element type had to be
widened to admit a thing that is not a `VoiceRuntimeEvent`, and the closed union's exhaustiveness
guarantee stops at the queue boundary as a result.

That is a genuinely interesting detail and worth sitting with: the closed union is exhaustive over
*events*, but the lifecycle signal is not an event, so it escaped the union and took the type safety
with it. Pipecat put the lifecycle signal *inside* the sum type (`EndFrame` is a `ControlFrame`) and
therefore did not have to widen anything — at the cost that `EndFrame` is one more of the 123
descendants that all 577 sites might have to reason about.

### 3.5 One more absence, reported because absence is evidence

[[rtv-pipeline-session]] records a dead-code finding: `clock.py` (16 L) declares
`MonotonicClock(Protocol)` and `SystemMonotonicClock`, and `grep -rn MonotonicClock` finds **neither
imported anywhere** in the package or the dental gateway. `VoiceSession` takes
`now_ns: Callable[[], int] = time.monotonic_ns` directly at L108; tests inject `FakeMonotonicClock`
from `testing/fakes.py` instead; `clock.py` is not re-exported from `__init__.py`.

Pipecat has a live equivalent — `FrameProcessor.get_clock()` returns `self.processor_setup.clock`
(`src/pipecat/processors/frame_processor.py` L681–690) and the clock is threaded through
`FrameProcessorSetup` to every processor. Stated as mechanism: one system wires the clock through
the setup object every processor receives; the other declared the protocol and then passed the
function directly instead. Confirm the dead code is still dead before [[ch-13/read]] — it is a
six-week-old reading.

### 3.6 Framework-extension probe #2

Lina TMR needs a **barge-in confirmation delay**: when VAD fires, do not cancel the assistant
immediately; wait 250 ms and cancel only if speech is still present, so a cough does not kill a
sentence.

1. Where does that component go in Pipecat? It is a `FrameProcessor` — but *between which two
   elements*, and which direction do the frames it cares about travel in? (You have `link()`,
   `FrameDirection`, and `_link_processors` above; you do not yet have [[ch-08/read]]'s cascade, so
   answer from topology alone.)
2. Where does it go in realtime_voice? Name the specific method in `session.py` that changes and say
   whether the change is an insertion or an edit.
3. Third question, and this is the one that matters: in which of the two can you **A/B test** the
   delay — ship both paths and flip between them per session — without editing the file that
   contains the rest of the turn logic?

---

## 4. VAD: two hysteresis machines that count different things

### 4.1 What Pipecat implements

**`src/pipecat/audio/vad/vad_analyzer.py` L25–44**

```python
VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2
VAD_MIN_VOLUME = 0.6


class VADState(Enum):
    """Voice Activity Detection states.

    Parameters:
        QUIET: No voice activity detected.
        STARTING: Voice activity beginning, transitioning from quiet.
        SPEAKING: Active voice detected and confirmed.
        STOPPING: Voice activity ending, transitioning to quiet.
    """

    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4
```

**`src/pipecat/audio/vad/vad_analyzer.py` L47–60**

```python
class VADParams(BaseModel):
    """Configuration parameters for Voice Activity Detection.

    Parameters:
        confidence: Minimum confidence threshold for voice detection.
        start_secs: Duration to wait before confirming voice start.
        stop_secs: Duration to wait before confirming voice stop.
        min_volume: Minimum audio volume threshold for voice detection.
    """

    confidence: float = VAD_CONFIDENCE
    start_secs: float = VAD_START_SECS
    stop_secs: float = VAD_STOP_SECS
    min_volume: float = VAD_MIN_VOLUME
```

The gate is a **conjunction**, and the state machine is four states:

**`src/pipecat/audio/vad/vad_analyzer.py` L202–246**

```python
        while len(self._vad_buffer) >= num_required_bytes:
            audio_frames = self._vad_buffer[:num_required_bytes]
            self._vad_buffer = self._vad_buffer[num_required_bytes:]

            confidence = self.voice_confidence(audio_frames)

            volume = self._get_smoothed_volume(audio_frames)
            self._prev_volume = volume

            speaking = confidence >= self._params.confidence and volume >= self._params.min_volume

            if speaking:
                match self._vad_state:
                    case VADState.QUIET:
                        self._vad_state = VADState.STARTING
                        self._vad_starting_count = 1
                    case VADState.STARTING:
                        self._vad_starting_count += 1
                    case VADState.STOPPING:
                        self._vad_state = VADState.SPEAKING
                        self._vad_stopping_count = 0
            else:
                match self._vad_state:
                    case VADState.STARTING:
                        self._vad_state = VADState.QUIET
                        self._vad_starting_count = 0
                    case VADState.SPEAKING:
                        self._vad_state = VADState.STOPPING
                        self._vad_stopping_count = 1
                    case VADState.STOPPING:
                        self._vad_stopping_count += 1

        if (
            self._vad_state == VADState.STARTING
            and self._vad_starting_count >= self._vad_start_frames
        ):
            self._vad_state = VADState.SPEAKING
            self._vad_starting_count = 0

        if (
            self._vad_state == VADState.STOPPING
            and self._vad_stopping_count >= self._vad_stop_frames
        ):
            self._vad_state = VADState.QUIET
            self._vad_stopping_count = 0
```

Read the `else` branch carefully. `STARTING → QUIET` is a **transition that emits nothing**. A blip
that entered `STARTING` and then went quiet before `_vad_start_frames` elapsed leaves no trace: no
`UserStartedSpeakingFrame`, no downstream cancellation, no turn. The state machine is the mechanism
by which a false start is discarded rather than published.

Chunk size is fixed by the analyzer, not by the transport:

**`src/pipecat/audio/vad/silero.py` L191–197**

```python
    def num_frames_required(self) -> int:
        """Get the number of audio frames required for VAD analysis.

        Returns:
            Number of frames required (512 for 16kHz, 256 for 8kHz).
        """
        return 512 if self.sample_rate == 16000 else 256
```

Two sample rates, both supported: **512 frames at 16 kHz, 256 at 8 kHz**. The 8 kHz branch is what
makes telephony audio analysable at all (§6.3).

Two more knobs that live above the analyzer:

```bash
$ grep -n "_MODEL_RESET_STATES_TIME" src/pipecat/audio/vad/silero.py
23:_MODEL_RESET_STATES_TIME = 5.0
218:            if diff_time >= _MODEL_RESET_STATES_TIME:

$ grep -n "audio_idle_timeout" src/pipecat/audio/vad/vad_controller.py | head -3
75:        audio_idle_timeout: float = 1.0,
83:            audio_idle_timeout: Timeout in seconds to force speech stop
103:        self._audio_idle_timeout = audio_idle_timeout
```

`VADController(audio_idle_timeout=1.0)` — *"Timeout in seconds to force speech stop"* — is the
watchdog for a microphone that stops delivering audio mid-utterance, e.g. a dropped WebRTC track.
Without it, a VAD sitting in `SPEAKING` never leaves, because nothing arrives to move it.

### 4.2 What realtime_voice implements

From [[rtv-vad-chunking]], two implementations sharing one state machine:

- `EnergyVADConfig` (`vad/energy.py` L15): `speech_rms: float = 500.0`, `min_speech_frames: int = 2`,
  `min_silence_frames: int = 4`. `EnergyVAD.rms()` (L104) is pure-Python
  `math.sqrt(sum(s*s)/n)` over an `array("h")` — no numpy, no model. Docstring: *"RMS hysteresis VAD
  intended for fallback and deterministic tests."*
- `SileroVADConfig` (`vad/silero.py` L21): `threshold: float = 0.5`, `min_speech_frames: int = 2`,
  `min_silence_frames: int = 6`. `SileroVAD.process` raises
  `ValueError("SileroVAD requires 16 kHz mono PCM")` at L58 for anything but 16 kHz mono.
  `from_pretrained()` (L44) lazily imports `silero_vad.load_silero_vad`; the model runs via
  `await asyncio.to_thread(call_model)` at L123.

Parameter for parameter ([[rtv-vad-chunking]], [[vad-silero]]) — this table describes, it does not
rank:

| | realtime_voice | Pipecat |
|---|---|---|
| confidence gate | `SileroVADConfig.threshold = 0.5` | `VADParams.confidence = 0.7` |
| volume gate | none | `VADParams.min_volume = 0.6`, **ANDed** with confidence |
| speech onset | `min_speech_frames = 2` (**frames**) | `start_secs = 0.2` (**seconds**) |
| speech offset | `min_silence_frames = 6` (**frames**) | `stop_secs = 0.2` (**seconds**) |
| states | 2 (`self._speaking: bool`) | 4 (`QUIET / STARTING / SPEAKING / STOPPING`) |
| analysis chunk | whatever the transport delivers | `num_frames_required()` — 512 @16 kHz, 256 @8 kHz |
| model state reset | `reset()` calls `model.reset_states()` if present | forced every `_MODEL_RESET_STATES_TIME = 5.0` s |
| idle-mic watchdog | none | `VADController(audio_idle_timeout=1.0)` |
| 8 kHz path | none — `ValueError` at `vad/silero.py` L58 | `256` frames branch |
| self-measured latency | `endpoint_latency_ms = self._silence_frames * frame.duration_seconds * 1000` (`energy.py` L79, `silero.py` L89) | reported through the observer plane (§6.5, [[ch-11/read]]) |

Both systems offload the model call from the event loop; the mechanism differs — Pipecat runs
`loop.run_in_executor(self._executor, self._run_analyzer, buffer)` against a
`ThreadPoolExecutor(max_workers=1)` (imported at `vad_analyzer.py` L16, constructed at L92, used at
L191), realtime_voice does `await asyncio.to_thread(call_model)`, which takes a thread from the
default executor per call.

And the seconds→frames conversion Pipecat performs, since §4.3 and §4.4 both depend on it:

**`src/pipecat/audio/vad/vad_analyzer.py` L159–165**

```python
        self._vad_frames = self.num_frames_required()
        self._vad_frames_num_bytes = self._vad_frames * self._num_channels * 2

        vad_frames_per_sec = self._vad_frames / self.sample_rate

        self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
        self._vad_stop_frames = round(self._params.stop_secs / vad_frames_per_sec)
```

(`vad_frames_per_sec` is a misleading name — it is seconds *per chunk*, `512 / 16000 = 0.032`.) At
16 kHz that gives `round(0.2 / 0.032) = round(6.25) = 6` chunks for both onset and offset.
[[vad-silero]] and [[rtv-vad-chunking]] both report **7**; the arithmetic in the code above gives
**6**, and I am reporting the code.

### 4.3 The unit mismatch, worked before the formula

The frames-versus-seconds row is the one worth doing concretely, because it is an operational
property that does not look like one.

`min_silence_frames = 6` means: after the VAD stops asserting speech, wait until six consecutive
frames have been silent, then declare the turn over. How long is six frames?

- The transport delivers audio in **20 ms** frames → 6 × 20 ms = **120 ms** of end-of-turn wait.
- The transport delivers audio in **100 ms** frames → 6 × 100 ms = **600 ms** of end-of-turn wait.

Same constant, same code, **480 ms of difference in time-to-first-response**, decided by a property
of the transport that the VAD never inspects. [[rtv-vad-chunking]] records that nothing in the code
asserts the frame duration. And in this stack, the frame duration is not a constant you choose once:
[[rtv-webrtc-transport]] records that `InboundAudioPump` emits **one `AudioFrame` per PyAV resampler
output**, so the size is whatever the resampler produced from whatever the browser sent.

Pipecat sidesteps the coupling by expressing the thresholds in seconds (`start_secs` / `stop_secs`)
and by fixing the analysis chunk itself at `num_frames_required()`, so the transport's framing does
not reach the state machine at all. That is a description of two mechanisms, not a ranking of them:
the frame-count formulation is simpler and exact when the frame size is pinned; the seconds
formulation costs a conversion and a fixed re-chunking buffer and is invariant to the transport.

Now the formula, which you did not need first:

```
endpoint_wait_seconds  =  min_silence_frames × frame_duration_seconds      # realtime_voice
endpoint_wait_seconds  =  stop_secs                                        # Pipecat
```

The right-hand side of the first line contains a term the VAD does not control. That is the whole
observation.

### 4.4 The two-frame blip, traced in both machines

Inject two frames of speech-like energy — a cough, a chair, a Korean backchannel "네" — into each
machine, mid-assistant-turn.

**Pipecat.** `speaking` is `confidence >= 0.7 and volume >= 0.6`; suppose both are met for two
chunks. `QUIET → STARTING`, `_vad_starting_count` reaches 2. `_vad_start_frames` is derived from
`start_secs = 0.2` through the L159–165 conversion above — 512-frame chunks at 16 kHz are 32 ms each,
so `_vad_start_frames = round(0.2 / 0.032) = 6` chunks are needed.
Frame 3 is silent → the `else` branch fires `case VADState.STARTING: self._vad_state = VADState.QUIET`.
Nothing is emitted. The assistant keeps talking.

**realtime_voice.** `min_speech_frames = 2` and the state is a bool. Two frames over
`threshold = 0.5` — with no `min_volume` conjunct — flips `self._speaking` to `True` and emits a
real `SPEECH_STARTED`. Per [[rtv-vad-chunking]], `VoiceSession._on_speech_started` (L284) then
**advances the generation and cancels the assistant**, and per [[rtv-pipeline-session]] it emits the
VAD event *before* awaiting the cancellation — comment L290–292: *"Publish media invalidation before
awaiting provider/Gateway cancellation."*

Both behaviours are the machines doing what they were written to do. The two-state machine has no
representation for "maybe speech", so it cannot hold a hypothesis; the four-state machine's
`STARTING` state *is* that representation. Whether holding the hypothesis is worth the added onset
latency is a product question about Korean tele-sales calls with real hold music and real
backchannels, and it is not settled by either code listing.

→ Use the figure's **third panel** here: it runs the same injected blip through both machines and
reports only what each does, frame by frame. Run it once with the blip at 2 frames and once at 8,
and note where the two behaviours converge.

### 4.5 One realtime_voice mechanism with no Pipecat counterpart, in this layer

`VoiceSessionConfig.vad_prefix_frames = 5` maintains a `deque(maxlen=5)` of pre-speech frames
(`session.py` L131) that is **replayed into ASR on `SPEECH_STARTED`** (L296–299), so the first ~100 ms
of the utterance — the part that was still in `QUIET` when the user actually started — is not lost
([[rtv-vad-chunking]]).

Pipecat's analogue exists but is not exposed at this layer: `SegmentedSTTService`'s docstring
(`src/pipecat/services/stt_service.py` L797–813) says it *"Maintains a small audio buffer to account
for the delay between actual speech start and VAD detection"* — a buffer inside the STT service,
not a tunable on the VAD. Same problem, different owner, and only one of the two makes the depth a
config field.

---

## 5. Speech-to-text: a streaming interface against a unary call

### 5.1 What Pipecat declares

Pipecat's STT layer has **two** base classes, and the difference between them is exactly the
mechanism at issue.

**`src/pipecat/services/stt_service.py` L51–70**

```python
class STTService(AIService):
    """Base class for speech-to-text services.

    Provides common functionality for STT services including audio passthrough,
    muting, settings management, and audio processing. Subclasses must implement
    the run_stt method to provide actual speech recognition.

    Includes an optional keepalive mechanism that sends silent audio when no real
    audio has been sent for a configurable timeout, preventing servers from closing
    idle connections (e.g. when behind a ServiceSwitcher). Subclasses that enable
    keepalive must override ``_send_keepalive()`` to deliver the silence in the
    appropriate service-specific protocol.

    A streaming STT reports latency through TTFB — speech end to final transcript —
    and not through processing metrics. Audio arrives continuously, so there is no
    discrete request whose duration a
    :meth:`~pipecat.processors.frame_processor.FrameProcessor.start_processing_metrics`
    window could measure; anchoring one to a speech or turn boundary measures how
    long the user talked. :class:`SegmentedSTTService` does issue a discrete
    request per utterance, so its subclasses time that call and report both.
```

**`src/pipecat/services/stt_service.py` L797–813**

```python
class SegmentedSTTService(STTService):
    """STT service that processes speech in segments using VAD events.

    Uses Voice Activity Detection (VAD) events to detect speech segments and runs
    speech-to-text only on those segments, rather than continuously.

    Requires VAD to be enabled in the pipeline to function properly. Maintains a
    small audio buffer to account for the delay between actual speech start and
    VAD detection.

    The buffered segment is passed to :meth:`run_stt` as a WAV container by
    default, which is what cloud providers want for their upload APIs. Local
    models that consume raw 16-bit PCM directly override
    :attr:`wants_wav_segments` to return ``False`` so they receive the
    unwrapped buffer instead. This is a subclass-level contract, not a
    user-configurable option: the format is dictated by what the model expects.
    """
```

Read those two docstrings against each other. Pipecat did not choose streaming *over* segmented — it
ships both, and the first docstring explains why the choice changes how latency is even *measurable*:
a streaming STT has no discrete request, so it reports TTFB from speech-end to final transcript; a
segmented one has a request, so it times the call. Two frame types carry the two outputs —
`TranscriptionFrame` at `frames.py` L450 and `InterimTranscriptionFrame` at L476.

### 5.2 What realtime_voice implements

Per [[rtv-vs-pipecat-gap]]: `OpenAICompatibleUnaryASR` **buffers the whole utterance into a WAV and
issues one `audio.transcriptions.create` call at `finalize()`** — `openai_compat.py` L194–242,
`timeout_seconds=1.5`. It is, in Pipecat's vocabulary, a `SegmentedSTTService` shape. There is no
streaming shape in the package.

The type system says the streaming shape was intended. `ASREventKind.INTERIM` and
`ASREventKind.END_OF_TURN` are **declared in `types.py`** and, per [[rtv-vs-pipecat-gap]], **never
emitted by any real provider** — only by a test fake (`FakeStreamingASR` in `testing/fakes.py`). The
`StreamingASR` Protocol in §3.3 has the streaming signature —
`def events() -> AsyncIterator[ASREvent]` — and the one shipped implementation yields exactly one
final event through it.

This is the closed union's other face, and it is worth naming precisely: the union *declares* the
interim case, `mypy` will insist every consumer handles it, the wire mapper `_control_event` maps it
to `"transcript.interim"`, and there is a test named
`test_interim_asr_is_observable_but_never_calls_agent` asserting the consumer behaviour. Everything
downstream of the event is built and verified. Only the producer is missing. An exhaustive type
system tells you the case exists; it cannot tell you nothing ever constructs it.

### 5.3 Where the call lands on the timeline

Mechanism, not judgement:

```
realtime_voice, per turn:
  [user speaks] ... [VAD: min_silence_frames × frame_dur] → finalize()
      → one HTTPS round trip, whole-utterance WAV upload, timeout_seconds=1.5
      → first AgentTextDelta possible
      → KoreanPhraseChunker holds until first sentence boundary
      → first TTS request → first audible sample

Pipecat with a streaming STTService, per turn:
  [user speaks] → partial transcripts arriving during speech
      → [VAD stop_secs] → final transcript (already mostly computed)
      → first LLM token
      → TTS
```

In the first shape the transcription round trip is **serial after** the endpoint decision. In the
second it overlaps the user's own speech. `CLAUDE.md`'s target for boson — quoted in
[[rtv-vs-pipecat-gap]] — is *"P50 at or below 1.0 seconds and P95 at or below 1.5 seconds,"* measured
*"from the last voiced user sample to the first audible assistant sample, including end-of-turn/VAD
time."* Both terms above are inside that measurement window. [[ch-06/read]] takes the turn-boundary
chain apart properly and [[ch-11/read]] does the budget arithmetic; this chapter's job is only to
put the two shapes on the same timeline so you can see which terms are serial in each.

---

## 6. Breadth, counted rather than characterised

Everything in this section is a `ls` or a `grep`. The commands are printed so you can re-run them.

### 6.1 Service providers

```bash
$ ls -d src/pipecat/services/*/ | wc -l
62
```

**62 service directories** in Pipecat. realtime_voice ships **2** providers — Boson ASR and Boson
TTS — both OpenAI-compatible, both, per [[rtv-vs-pipecat-gap]], in one 478-line file.
`BosonHiggsASR(OpenAICompatibleUnaryASR)` is in-house-model-specific by construction: the package was
written to talk to your own model, and it does.

### 6.2 Transports

```bash
$ ls -d src/pipecat/transports/*/
src/pipecat/transports/daily/
src/pipecat/transports/heygen/
src/pipecat/transports/lemonslice/
src/pipecat/transports/livekit/
src/pipecat/transports/local/
src/pipecat/transports/moq/
src/pipecat/transports/smallwebrtc/
src/pipecat/transports/tavus/
src/pipecat/transports/vonage/
src/pipecat/transports/websocket/
src/pipecat/transports/whatsapp/
```

**11 transport packages.** realtime_voice ships **1** — aiortc WebRTC — plus `FakeVoiceTransport` for
tests.

(Note for precision: [[rtv-vs-pipecat-gap]] says "12 transports" and lists eleven directory names.
The directory count at this commit is 11. I am reporting the count I ran.)

The two WebRTC implementations are close in size and in library choice — both are aiortc, both do
PyAV resampling, both drive a data channel:

```bash
$ wc -l src/pipecat/transports/smallwebrtc/*.py
     825 connection.py
     266 request_handler.py
    1085 transport.py
    2176 total
```

against realtime_voice's ~960 L across `manager.py` 248, `control.py` 226, `peer.py` 231,
`tracks.py` 216, `buffer.py` 123, `config.py` 64, `transport.py` 168 ([[rtv-webrtc-transport]]).

### 6.3 Telephony serializers

```bash
$ ls src/pipecat/serializers/
__init__.py  base_serializer.py  exotel.py  genesys.py  plivo.py
protobuf.py  telnyx.py  twilio.py  vonage.py
```

**6 telephony serializers** — `exotel`, `genesys`, `plivo`, `telnyx`, `twilio`, `vonage` — plus
`protobuf` (not telephony) and the base class. realtime_voice ships **0**, and cannot host one as
written: `SileroVAD.process` raises `ValueError("SileroVAD requires 16 kHz mono PCM")` and there is
no μ-law path ([[rtv-vad-chunking]], [[rtv-vs-pipecat-gap]]). [[rtv-vs-pipecat-gap]] records that
boson's `CLAUDE.md` names *"future SIP/RTP or telephony adapters"* as an intent. [[ch-05/read]] is
where the serializer-as-transport-adapter idea gets built out.

For a Korean insurance tele-sales agent this row is worth reading twice, and then not acting on
until [[ch-13/read]].

### 6.4 Connection recovery

Pipecat's WebRTC connection exposes renegotiation and ICE restart as first-class methods:

```bash
$ grep -n "def renegotiate\|def ask_to_renegotiate\|def pc_id" src/pipecat/transports/smallwebrtc/connection.py
302:    def pc_id(self) -> str:
443:    async def renegotiate(self, sdp: str, type: str, restart_pc: bool = False):
799:    def ask_to_renegotiate(self):
```

and the request handler's DTO carries the restart flag end to end:

**`src/pipecat/transports/smallwebrtc/request_handler.py` L25–41**

```python
@dataclass
class SmallWebRTCRequest:
    """Small WebRTC transport session arguments for the runner.

    Parameters:
        sdp: The SDP string (Session Description Protocol).
        type: The type of the SDP, either "offer" or "answer".
        pc_id: Optional identifier for the peer connection.
        restart_pc: Optional whether to restart the peer connection.
        request_data: Optional custom data sent by the customer.
    """

    sdp: str
    type: str
    pc_id: str | None = None
    restart_pc: bool | None = None
    request_data: Any | None = None
```

realtime_voice has neither: per [[rtv-webrtc-transport]] its only recovery path is a fresh
`accept_offer(reconnect=True)`. Also absent there: video (`RawVideoTrack`), screen share, and any
transport other than WebRTC — the last by explicit construction, since
`WebRTCTransportConfig.__post_init__` rejects `output_channels != 1` with *"the first WebRTC
transport supports mono assistant audio."*

### 6.5 Observability

Pipecat has a read-only plane that sees every frame at every hop without being in the chain:

**`src/pipecat/observers/base_observer.py` L90–97**

```python
class BaseObserver(BaseObject):
    """Base class for pipeline frame observers.

    Observers can view all frames that flow through the pipeline without
    needing to inject processors into the pipeline structure. This enables
    non-intrusive monitoring capabilities such as frame logging, debugging,
    performance analysis, and analytics collection.
    """
```

```bash
$ ls src/pipecat/observers/
__init__.py  base_observer.py  loggers  startup_timing_observer.py
turn_tracking_observer.py  user_bot_latency_observer.py
```

Both the processing hook (`observer.on_process_frame`, quoted at §2.3, `frame_processor.py` L835)
and the push hook (`observer.on_push_frame`, quoted at §3.1, L1181) are already wired into the
base class, so an observer sees the pipeline without a single `link()` change.

realtime_voice's instrumentation, per [[rtv-vs-pipecat-gap]] and [[rtv-webrtc-transport]], is:
`provider_latency_ms` / `endpoint_latency_ms` fields on events, a `VoiceEvent` stream with 14
`VoiceEventKind` values fanned out to the data channel, and `BoundedAudioOutput.discarded_frames` —
a counter that is exposed and, per the excerpt, **never read by anything**. No OTel, no spans, no
aggregation. Note the shape of that: the *event stream* is rich and typed and goes to the client;
what is missing is the aggregation layer above it. [[ch-11/read]] is where the observer plane and
the latency budget get treated together.

---

## 7. What realtime_voice implements that Pipecat has no equivalent of

Everything in this section was checked in the other direction too: for each item I grepped the
Pipecat tree for a counterpart and report what I found. "No equivalent" here means "I looked and did
not find one at commit `0cbf9c5b`," not "it cannot be built."

Each of these expands into its **algorithm** in the figure's fifth panel, not into a checkmark. Open
them there; a checkmark would let you skip the part that matters.

### 7.1 `KoreanPhraseChunker` — 283 lines

From [[rtv-vad-chunking]]:

```
__init__(*, min_chars=12, max_chars=60, hard_max_chars=None,
         batch_max_chars=320, adaptive_batching=True)
# when hard_max_chars is None it resolves to min(batch_max_chars, max_chars * 2)   (L56-60)
```

Docstring L28–34: *"Adaptive mode emits the first complete sentence immediately, batches the next two
complete sentences, then holds the remaining response as one final group until `flush`. `max_chars`
is a soft latency target rather than an immediate cut point."*

That is a **1 → 2 → bounded-tail schedule**, implemented as `_batch_phase` 0/1/2 in `_accept_adaptive`
(L115–149). Phase 0 optimises time-to-first-audio by shipping one sentence the instant it is
complete; phase 1 amortises TTS request overhead over pairs; phase 2 stops fragmenting the tail once
the customer is already listening.

`_BoundaryKind` is a `StrEnum` of `SENTENCE / SOFT / OVERLONG / FINAL_TAIL`, over
`_STRONG_END = frozenset(".!?。！？\n")`, `_SOFT_END = frozenset(",，;；:")`,
`_CLOSING_PUNCTUATION = frozenset("\"'”’)]}」』】")` — note that the strong-end set is CJK-aware and
the closing set includes `」』】`.

The guards are where the Korean-specific knowledge is:

- `_is_safe_period` (L255) refuses to split on a dot inside `1.5`, inside `...`, or between ASCII
  token characters. Comment L266–269: *"A dot between ASCII token characters belongs to a model name,
  hostname, abbreviation, or identifier rather than ending a Korean sentence."* That is the rule that
  keeps `gpt-4.1` and a domain name from becoming two TTS requests.
- `_is_numeric_separator` (L277) protects `1,000` from the `_SOFT_END` comma.
- `_INTERNAL_TAG = re.compile(r"\[(?:interruption|system|tool|objection|customer|assistant)[^\]]*\]")`
  strips gateway control tags out of the spoken text while `start_char` / `end_char` keep the
  **source** span intact — so the ledger in §7.2 can still map spoken characters back to the original
  string.

That last property is the one to notice: stripping and span-preservation are done together, because
the downstream consumer needs to answer "which characters of the *original* text were heard."

**Pipecat side, checked.** Pipecat's TTS services do their own sentence splitting with generic
heuristics; there is no 1→2→tail schedule, no Korean numeric/identifier guard, and no
tag-strip-with-span-preservation. See [[tts-service-interface]] and [[ch-07/read]].

### 7.2 `AudioTextPlayoutLedger` — 110 lines

From [[rtv-vad-chunking]]. Four dicts keyed by `GenerationId`: `_phrases` (a list of
`PhrasePlayout(request, sample_start, sample_end, complete)`), `_by_phrase`, `_next_sample`,
`_played_sample`. `begin(request)` / `append(request, sample_count) -> (start, end)` /
`finish(request)` build the text→sample map as TTS streams; `acknowledge(generation_id, played_sample)`
moves the client cursor with `max(current, played_sample)` — monotonic, so a late acknowledgement
cannot rewind the cursor.

`audible_text()` at L74 is the payoff. It walks phrases until the cursor and, for the phrase that was
only partly played, computes:

```
ratio = (cursor - sample_start) / (sample_end - sample_start)
heard = text[: int(len(text) * ratio)]
```

Two properties fall straight out of that expression and both are load-bearing:

1. It needs **nothing from the TTS but a sample count**. No word timestamps, no alignment metadata.
   Any TTS that emits PCM works.
2. It is a **linear character-per-sample approximation**. Within a word, and across a phrase whose
   speaking rate varies, the character index it returns is an estimate, not an alignment.

`playout_complete()` at L98 — all phrases `complete` **and** `played_sample >= queued_samples` — is
the predicate that lets `_cancel_generation` (`session.py` L502–507) set the `semantic_interrupt`
flag correctly, i.e. distinguish *"the customer cut me off"* from *"I finished my turn and they
replied."* Those two produce different conversation histories and different stage transitions in
gateway, so the distinction is not cosmetic.

**Pipecat side, checked.** Pipecat reaches a related guarantee structurally rather than
arithmetically: the assistant context aggregator sits **after** `transport.output()`, so it only ever
saw text that was actually released, and it is paced by word-timestamped `TTSTextFrame`s
([[interruption-cascade]], [[ch-08/read]]). No `[interrupted]` marker is written into the context.
Two different mechanisms with different input requirements — one needs a timestamp-emitting TTS, one
needs a sample counter. There is no ledger object in the Pipecat tree.

→ The figure's **fourth panel** is built for this: drag the mid-word slider to see where the linear
approximation drifts from a true alignment, then flip the "timestamp-less TTS" toggle to see which of
the two mechanisms still has an input.

### 7.3 `WebRTCSessionManager` — 248 lines

From [[rtv-webrtc-transport]]. `manager.py` L51, docstring: *"Create short-lived authorized sessions
and enforce one live peer each."*

- `create_session(customer_id, *, session_id=None, metadata=None) -> VoiceSessionTicket(session_id, token, expires_at, customer_id)`
- mints `secrets.token_urlsafe(32)` and stores **only** `hashlib.sha256(token).digest()`
- `_authorize` (L227) checks expiry, then `hmac.compare_digest`
- `session_token_ttl_seconds = 15 * 60`
- `accept_offer(..., reconnect: bool = False)` raises
  `SessionConflictError("this voice session already has a live peer")` unless `reconnect=True` —
  **explicit reconnect, no silent takeover**
- `accept_offer(sdp, type="offer")` raises `SignalingError("only an SDP offer may be accepted")` —
  answer-only; the server never initiates
- per-session concurrency is an `asyncio.Lock` on `_ManagedSession`, and **only `accept_offer` takes
  it** — `send_audio`, `activate_generation`, `send_control` read `session.peer` unlocked and return
  `False`/`0` when it is `None`, which is a deliberate "never block the media path on signaling"
  choice

**Pipecat side, checked, with the command:**

```bash
$ grep -rn "token_urlsafe\|compare_digest" src
src/pipecat/runner/run.py:324:    if not hmac.compare_digest(expected, sig):
src/pipecat/transports/whatsapp/client.py:181:        if not hmac.compare_digest(expected_signature, received_signature):
```

Both hits are webhook-signature checks, not voice-session authorization. `SmallWebRTCConnection.__init__`
takes `ice_servers` and `connection_timeout_secs` and nothing else:

**`src/pipecat/transports/smallwebrtc/connection.py` L245–248**

```python
    def __init__(
        self,
        ice_servers: list[str] | list[IceServer] | None = None,
        connection_timeout_secs: int = 60,
    ):
```

No token, no TTL, no customer binding, and `request_handler.py` is a bare offer/answer endpoint. This
is application-layer territory in Pipecat's design; realtime_voice put it in the transport package.

### 7.4 `ControlEvent` v1 — 226 lines in `control.py`

From [[rtv-webrtc-transport]]. `@dataclass(frozen=True, slots=True)` at L25 with fields
`session_id, type, sequence, payload, turn_id, generation_id, version=CONTROL_PROTOCOL_VERSION`,
where `CONTROL_PROTOCOL_VERSION = 1`.

Docstring L28–30: *"Audio bytes are intentionally prohibited. Microphone and assistant audio belong
on RTP tracks, never in JSON or base64."*

And the prohibition is enforced, not documented. `_reject_audio_payload` (L117) walks the payload
**recursively** and raises on:

- any key normalising into `{"audio","audio_base64","audio_data","base64_audio","pcm","pcm16","wav"}`
- any string starting with `data:audio/`
- any `bytes` / `bytearray` / `memoryview`

`from_json` (L64) rejects unknown top-level fields with
`SignalingError(f"unexpected control fields: ...")`, rejects non-object payloads, and rejects version
mismatches.

`OrderedControlChannel` (L136) refuses a partially-reliable channel **at construction**: `ordered=False`,
a non-`None` `maxRetransmits`, or a non-`None` `maxPacketLifeTime` each raise `SignalingError`
(L147–154), with the comment *"Control events must not silently disappear."* `receive()` enforces
strict in-order delivery —
`SignalingError(f"out-of-order control event: expected {self._next_inbound}, received {event.sequence}")`
— and `send()` refuses anything over `max_control_message_bytes = 64 * 1024`. The outbound sequence is
a private counter, so **the server owns ordering**, not the client.

**Pipecat side, checked.** The data channel is an untyped passthrough all the way to the application
event handler:

```bash
$ grep -n "_on_app_message\|on_app_message" src/pipecat/transports/smallwebrtc/transport.py
66:        on_app_message: Called when an application message is received.
71:    on_app_message: Callable[[Any, str], Awaitable[None]]
261:        async def on_app_message(connection: SmallWebRTCConnection, message: Any):
578:        await self._callbacks.on_app_message(message, sender)
989:            on_app_message=self._on_app_message,
1001:        self._register_event_handler("on_app_message")
1047:    async def _on_app_message(self, message: Any, sender: str):
1051:        await self._call_event_handler("on_app_message", message, sender)
```

`message: Any`. No schema, no sequence check, no size cap, no audio ban. Pipecat does have a typed
client protocol — RTVI ([[rtvi-observability]]) — but it rides a different layer and is not enforced
at the data channel itself.

Notice that §7.3 and §7.4 are the same shape of finding, and it is worth naming: both are **policy**,
not plumbing. The plumbing (aiortc, SDP, a data channel) is the same in both systems. What differs is
whether the policy was written down as enforced code inside the transport package or left to the
application.

### 7.5 `GenerationAudioQueue.discard_generation()`

From [[rtv-pipeline-session]]: `queues.py` L12, 66 lines, a hand-rolled `deque` plus an
`asyncio.Condition` — written that way for exactly one operation.

`discard_generation(generation_id) -> int` at L42 **rebuilds the deque under the condition lock as an
atomic filter**. `asyncio.Queue` has no such operation: you cannot remove selected items from an
`asyncio.Queue` without draining and re-putting, which is not atomic against a concurrent consumer.
`close()` clears and wakes all waiters; `get()` on a closed empty queue raises `QueueClosedError`.

Its transport-side twin is `BoundedAudioOutput` (`buffer.py` L21): an `asyncio.Queue(maxsize=64)`
whose `activate_generation(generation_id)` (L53) **synchronously drains everything older and returns
the drop count**; `put()` returns `False` for a stale generation and raises `AudioBufferFull` if the
live peer has not drained within `backpressure_ms = 250` ([[rtv-webrtc-transport]]).

**Pipecat side.** Pipecat solves the same problem with the priority queue plus task cancellation
rather than with a filtered rebuild — see §8.

**And one finding to carry to [[ch-13/read]] unmodified**, from [[rtv-webrtc-transport]]:
`WebRTCVoiceTransport` — the protocol-conformant adapter — **never calls `activate_generation()`**.
The dental `voice_server.py` wires the manager callbacks directly instead. So the adapter that
satisfies the `VoiceTransport` Protocol is the less capable of the two live paths, and the capable
path bypasses the abstraction. That is a fact about the abstraction boundary, and it will matter when
someone asks "what exactly is behind the `VoiceTransport` interface?"

### 7.6 `StreamingConversationAgent` — the slot itself

The Protocol is one line of intent (§3.3):

```python
class StreamingConversationAgent(Protocol):
    def stream(request) -> AsyncIterator[AgentTextDelta]
    async def cancel / close
```

It yields **only** `AgentTextDelta`. Not tool calls, not context objects, not messages — text deltas.
Per [[rtv-vs-pipecat-gap]], tools live in `packages/basement` + `packages/gateway`, reached via
`GatewayConversationAgent.stream()` → `bridge.dispatch_transcript()`
(`agents/dental-w-tool-gateway/voice_server.py` L163–205). boson's own `CLAUDE.md` states the
constraint the slot exists to satisfy: *"Keep Basement and the dental business logic text-native"*,
and *"Basement and Gateway must not import provider-specific audio code."*

So the voice package's contract with the brain is: **give me text, take my text, and never make me
know what a tool is.**

**Pipecat side.** Pipecat's LLM service owns the context and closes the tool loop in-pipeline:
`LLMContext(tools=[...])`, `register_function`, `params.result_callback(...)`, and the loop is closed
by pushing an `LLMContextFrame` **upstream** ([[function-calling]]). `LLMContextFrame` is the class at
`frames.py` L551 that subclasses `Frame` directly, in no branch.

There is no Pipecat analogue of a "text-in / text-out agent slot that is forbidden to know about
tools." Whether the two contracts can coexist, and what it would take, is [[ch-09/read]]'s entire
subject. Do not resolve it here.

---

## 8. Parity, named as parity: barge-in and cancellation

This section exists because the honest finding is *sameness*, and sameness is easy to under-report
when you are cataloguing differences.

Both systems reach the same guarantee — *when the customer speaks over the assistant, the
already-queued assistant audio must not play, and the assistant's in-flight work must stop* — by
different mechanisms.

**Pipecat: one signal, cascading.** `InterruptionFrame` is a `SystemFrame`:

**`src/pipecat/frames/frames.py` L1142–1150**

```python
class InterruptionFrame(SystemFrame):
    """Frame pushed to interrupt the pipeline.

    This frame is used to interrupt the pipeline. For example, when a user
    starts speaking to cancel any in-progress bot output. It can also be pushed
    by any processor.
    """

    pass
```

Being a `SystemFrame` is what makes it jump every queue. The priority queue assigns tiers by
`isinstance`:

**`src/pipecat/processors/frame_processor.py` L132–143**

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
```

and `put()` assigns them by `isinstance`, three lines of it:

**`src/pipecat/processors/frame_processor.py` L162–168**

```python
        frame, _, _ = item
        if isinstance(frame, StartFrame):
            priority = self.START_PRIORITY
        elif isinstance(frame, SystemFrame):
            priority = self.SYSTEM_PRIORITY
        else:
            priority = self.DEFAULT_PRIORITY
```

and the input task routes system frames to immediate execution rather than onto the process queue at
all:

**`src/pipecat/processors/frame_processor.py` L1304–1307**

```python
            if isinstance(frame, SystemFrame):
                await self.__process_frame(frame, direction, callback)
            elif self.__process_queue:
                await self.__process_queue.put((frame, direction, callback))
```

Then each processor cancels **its own** task when the frame arrives:

**`src/pipecat/processors/frame_processor.py` L1130–1150**

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

One frame, N processors, each responsible for its own cancellation. That is the composable
formulation: a processor you write next year participates in interruption correctly by inheriting
this method and doing nothing.

**realtime_voice: one integer, compared everywhere.** Per [[rtv-pipeline-session]],
`self._active_generation` is compared at **six** points, each dropping stale work with
`VoiceEventKind.GENERATION_DROPPED`:

| site | file:line |
|---|---|
| `next_audio` | `session.py` L180 |
| `_asr_loop` | L328 |
| `_produce_phrases` | L410 |
| `_consume_phrases` | L447 |
| `_consume_phrases` | L456 |
| `acknowledge_playout` | L199 |

plus `GenerationAudioQueue.discard_generation()` (§7.5) and, on the transport side,
`BoundedAudioOutput.activate_generation()` (§7.5) flushing the jitter buffer, and
`OutboundAudioTrack` throwing away its `av.AudioFifo` on generation change (`tracks.py` L142–146,
L152–154 — jitter-buffer flush as cancellation).

`_on_speech_started` (L284) advances the generation and publishes the invalidation **before** awaiting
provider cancellation — comment L290–292: *"Publish media invalidation before awaiting
provider/Gateway cancellation"* — which is why there is a test named
`test_barge_in_event_precedes_slow_provider_cancellation`.

Same guarantee, two shapes:

| | Pipecat | realtime_voice |
|---|---|---|
| the signal | a `SystemFrame` that traverses the chain | an integer field on the session |
| how staleness is decided | the task processing the stale work is cancelled | each site compares `generation_id` and drops |
| where it is enforced | in the base `FrameProcessor`, once, inherited by all | at 6 named sites plus 2 buffers, written by hand |
| a new component participates | automatically, by inheriting `_start_interruption` | only if its author adds the comparison |
| you can enumerate all enforcement points | no — they are wherever processors are | yes — the 6 sites are in one file |

Both columns are true statements about mechanism. The trade is the same trade as §2.6, wearing
different clothes: distributed and automatic against centralised and enumerable. [[ch-08/read]] takes
the Pipecat cascade apart end to end.

---

## 9. Test names as evidence

Test *count* is a weak signal. Test *names* are a strong one, because a name states what invariant the
author believed was worth pinning. From [[rtv-vs-pipecat-gap]]: 6 files, 1,504 lines, 60 test
functions, `asyncio_mode = "auto"`.

**`test_session_pipeline.py` (483 L, 15 tests)**

- `test_confirmed_barge_in_purges_server_audio_and_cancels_old_work`
- `test_barge_in_event_precedes_slow_provider_cancellation`
- `test_late_stale_asr_final_is_dropped`
- `test_playout_acknowledgement_returns_only_heard_prefix`
- `test_late_playout_ack_cannot_extend_interrupted_audible_prefix`
- `test_next_turn_after_completed_playout_is_not_an_interruption`
- `test_each_generation_has_exactly_one_terminal_event`
- `test_ingress_overflow_rejects_frame_instead_of_growing_latency`
- `test_two_sessions_do_not_share_generation_or_provider_state`
- `test_interim_asr_is_observable_but_never_calls_agent`

**`test_chunking_and_ledger.py` (270 L, 22 tests)**

- `test_safety_split_never_breaks_a_long_latin_token`
- `test_dots_inside_ascii_identifier_are_not_sentence_boundaries`
- `test_overlong_token_waits_across_streamed_deltas_for_whitespace`
- `test_playout_ledgers_are_generation_isolated`

**`test_webrtc_transport.py` (257 L, 6 tests)** includes a real
`test_aiortc_loopback_negotiates_ordered_control_channel` — an actual negotiation, not a mock — and
`test_outbound_track_underflow_is_explicit_zero_pcm_silence`, which pins the PyAV bug recorded in
`tracks.py` L201: *"PyAV does not guarantee zero-initialized AudioFrame storage. Sending a fresh
allocation as 'silence' can therefore produce full-scale random PCM."* That is a regression test for
a bug where **silence came out as full-scale noise into a customer's ear**, and it is pinned by name.

Read that list as a specification and notice what it is a specification *of*. Six of the fifteen
session tests are about the interaction between interruption, staleness and playout accounting.
Nothing in the list is a smoke test. Nothing asserts "the pipeline runs." Every name asserts an
invariant that, if violated, produces a specific bad experience on a phone call: the assistant keeps
talking over the customer, or a stale transcript triggers a reply to something already superseded, or
the conversation history claims the assistant said a sentence the customer never heard.

`realtime_voice/testing/fakes.py` (271 L) ships six deterministic doubles as **public API**, exported
from `realtime_voice.testing`: `ScriptedVAD`, `FakeStreamingASR`, `FakeStreamingConversationAgent`,
`FakeStreamingTTS`, `FakeVoiceTransport`, `FakeMonotonicClock`. Determinism was designed in, not
retrofitted — which is also, per §5.2, the reason `ASREventKind.INTERIM` has a producer at all.

**The disagreement I said I would flag.** [[rtv-vs-pipecat-gap]] reports Pipecat as *"226 test files /
92,538 L / 3,959 test functions (0.55 ratio)."* My re-measurement at commit `0cbf9c5b` gives **236
test files** and **4,236 `def test_` matches** (4,231 of which are at line-start after whitespace).
The line count agrees exactly at 92,538. Use 236 / 4,236 — the commands are in §1 and you can re-run
them — and treat the excerpt's file/function counts as taken from a slightly different tree state or
matcher.

---

## 10. The fact sheet

This is what the chapter hands forward. It is a table of measurements and absences. It contains no
recommendation, and the fact that a row is longer on one side is not a score.

| Layer | realtime_voice (excerpt-attested) | Pipecat (code-verified at `0cbf9c5b`) |
|---|---|---|
| unit of data | closed union, `VoiceRuntimeEvent`, `types.py:201`, 4 members over 9 frozen dataclasses; correlation IDs on every payload | open sum type, `Frame` (`frames.py` L65), 123 descendants in a 2,415-line file, 3 scheduling branches + `UninterruptibleFrame` mixin; no correlation fields on the base |
| extension cost of a new datum | edit 1 line + every `isinstance` chain; type checker finds the sites | declare a subclass anywhere; 577 `isinstance(frame, ...)` sites in 136 files may need updating; nothing finds them for you |
| unhandled datum | `TypeError` at `transport.py:_control_event` (L118–156) | silently dropped — `process_frame` (L820–847) has no fallthrough |
| unit of work | one `VoiceSession`, 561 L, `_supervise()` L257 → 1 `TaskGroup`, 2 long-lived tasks per session | `FrameProcessor` + `link()` (L671–679); per-processor runtime is [[ch-04/read]] |
| topology | fixed VAD→ASR→agent→TTS; 5 `Protocol` slots swappable; no interposition | `Pipeline._link_processors()` (L197–202) over a list; interposition is a list edit |
| direction | one-way; no upstream push | `FrameDirection.{DOWNSTREAM,UPSTREAM}` (L60–69), `_next` / `_prev` |
| shutdown | `close()` L231, 24 hand-sequenced lines, `_STOP = object()` sentinel widening every queue type | `EndFrame(ControlFrame, UninterruptibleFrame)` (L1899) propagates and survives interruption |
| VAD | 2-state bool; `threshold=0.5`; no volume gate; frame-count thresholds; no idle watchdog; 16 kHz mono only | 4-state `VADState`; `confidence=0.7` **and** `min_volume=0.6`; seconds-based; `audio_idle_timeout=1.0`; 512@16k / 256@8k |
| pre-roll into ASR | `vad_prefix_frames=5`, replayed at `SPEECH_STARTED` (L296–299) | buffer inside `SegmentedSTTService`, not a VAD tunable |
| STT | unary only — WAV + one `audio.transcriptions.create` at `finalize()`; `INTERIM`/`END_OF_TURN` declared, produced only by a fake | `STTService` (streaming) **and** `SegmentedSTTService` (per-utterance) as separate base classes |
| TTS | 1 streaming provider (`OpenAICompatibleStreamingTTS`, 24 kHz PCM) | one of 62 service directories; see [[ch-07/read]] |
| providers | 2 | 62 service directories |
| transports | 1 (aiortc WebRTC ~960 L) + a fake | 11 transport packages; smallwebrtc is 2,176 L |
| telephony | 0; `SileroVAD` rejects 8 kHz | 6 serializers (`exotel/genesys/plivo/telnyx/twilio/vonage`) |
| recovery | fresh `accept_offer(reconnect=True)` only | `renegotiate()` L443, `ask_to_renegotiate()` L799, `restart_pc` on the request DTO |
| session auth | `WebRTCSessionManager`: `token_urlsafe(32)`, SHA-256 digest, `compare_digest`, 15-min TTL, one-live-peer + explicit reconnect | none in the transport — `SmallWebRTCConnection.__init__` takes `ice_servers`, `connection_timeout_secs` |
| control channel | `ControlEvent` v1: dotted-type validation, strict in-order `sequence`, 64 KiB cap, recursive `_reject_audio_payload`, ordered-channel enforcement at construction | `on_app_message(message: Any, sender: str)` — untyped passthrough |
| text chunking | `KoreanPhraseChunker` 283 L: 1→2→tail schedule, `_is_safe_period`, `_is_numeric_separator`, `_INTERNAL_TAG` strip with source spans | no equivalent; TTS services split with generic heuristics |
| playout accounting | `AudioTextPlayoutLedger` 110 L: sample-ratio `audible_text()`, monotonic ack, `playout_complete()` → `semantic_interrupt` | no ledger; aggregator placement after `transport.output()` + word-timestamped `TTSTextFrame`s |
| agent slot | `StreamingConversationAgent` yields only `AgentTextDelta`; tools live outside the package by contract | LLM service owns context and closes the tool loop in-pipeline via upstream `LLMContextFrame` |
| barge-in | generation-ID equality at 6 sites + `discard_generation()` + `activate_generation()` + FIFO flush | `InterruptionFrame(SystemFrame)` + priority queue (L132–170, L1304) + per-processor `_start_interruption` (L1130–1150) |
| observability | `provider_latency_ms` / `endpoint_latency_ms`, 14 `VoiceEventKind` values to the data channel, `discarded_frames` counter that nothing reads | `BaseObserver` plane hooked into `process_frame` and `push_frame`; 3 shipped observers + loggers |
| dead code found | `clock.py` `MonotonicClock` / `SystemMonotonicClock` imported nowhere | — (not audited for this chapter) |
| scale | 3,886 L src / 1,504 L tests / 60 tests / 6 files | 168,847 L src / 92,538 L tests / 4,236 `def test_` / 236 files |

**Absences on the realtime_voice side, listed plainly, with no softening:** no streaming STT, no
telephony path, no 8 kHz, no renegotiation or ICE restart, no video, no observer plane, no metrics
aggregation, no processor interposition, no upstream direction, no second transport, no second
provider pair, and one dead module.

**Absences on the Pipecat side, listed with the same plainness:** no session token / TTL / one-live-peer
enforcement, no validated control-plane schema at the data channel, no Korean phrase chunker, no
playout ledger, no text-only agent slot that is forbidden to own tools, and no correlation IDs on the
base frame.

Neither list is longer in any way that means anything yet.

---

## 11. Open questions this chapter deliberately leaves open

Answer them in writing before [[ch-13/read]]; do not answer them now.

1. **Who writes the next component?** Over the next four quarters, how many voice-stack components
   will be written by someone who cannot edit `types.py`? If the answer is zero, what does the open
   sum type buy you? If it is not zero, what does the closed union cost you?
2. **What is the exhaustiveness actually worth in incidents?** Of the last N production voice
   incidents in boson, how many would have been caught by `mypy` at the union, and how many were
   caused by something the type system cannot see — like `ASREventKind.INTERIM` being declared,
   handled, wire-mapped, tested, and never produced (§5.2)?
3. **Is the abstraction boundary in §7.5 load-bearing?** `WebRTCVoiceTransport` satisfies the
   `VoiceTransport` Protocol and never calls `activate_generation()`; the production path bypasses it.
   Is the Protocol describing your system, or describing a system you no longer run?
4. **What is `min_silence_frames = 6` in milliseconds on your production traffic, right now?** Not in
   principle — measured, on real Korean calls, across the frame sizes the browser actually delivers.
   §4.3 gives you the arithmetic; you have the recordings.
5. **Does your TTS emit word timestamps?** The answer determines whether §7.2's linear approximation
   is the only mechanism available to you or one of two. It is a one-line check against the provider
   and it changes the shape of the [[ch-08/read]] discussion completely.
6. **Can a text-only agent slot and an in-pipeline tool loop coexist at all?** Do not answer.
   [[ch-09/read]] is built to answer it, and answering it early with an intuition is how you end up
   defending the intuition instead of the evidence.

---

## 다음 챕터로

This chapter hands forward three things and no fourth.

**A fact sheet** (§10). Every later chapter draws on it. When [[ch-05/read]] shows you six telephony
serializers, the baseline is "0, and `SileroVAD` raises on 8 kHz." When [[ch-08/read]] takes the
interruption cascade apart, the baseline is "generation-ID equality at six named sites plus two
buffer flushes." When [[ch-11/read]] builds the latency budget, the baseline is
`endpoint_latency_ms`, `provider_latency_ms`, and a `discarded_frames` counter nothing reads. The
sheet is what makes those chapters comparisons instead of tours.

**A framing question that is not yet answered.** Open sum type versus closed union is not a question
about two codebases; it is a question about **who writes the next component**, and you now have both
implementations in front of you at file-and-line resolution. Hold the question open through the
mechanics chapters. It will be asked again, with more evidence, in [[ch-13/read]].

**Two named collisions, deferred on purpose.** The agent slot (§7.6) — realtime_voice's
`StreamingConversationAgent` yields text and is contractually forbidden to know about tools, while
Pipecat's LLM service owns the context and closes the tool loop by pushing `LLMContextFrame`
upstream. And the topology question (§3.3) — slots are swappable, order is not. Neither is resolved
here. [[ch-09/read]] owns the first; [[ch-12/read]] owns the second when it places rule layers as
middleware.

Next is [[ch-04/read]], which opens the mechanics phase by answering the question this chapter kept
having to defer: **what actually runs when you call `worker.run()`?** Two tasks and two queues *per
processor*, a `FrameProcessorQueue` with three priority tiers, and the out-of-band path that lets an
`InterruptionFrame` overtake a queue full of audio. You have seen realtime_voice's answer — one
`TaskGroup`, two long-lived tasks, four bounded queues with three overflow policies. Now go see the
other one.

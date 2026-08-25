---
title: "The Narrow Waist: Frame as a Sum Type and What It Costs"
chapter: ch-02
phase: composition
course: pipecat
sources:
  - theory-narrow-waist
  - frame-taxonomy
  - flows-actions
  - pipecat-design-philosophy
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
---

# The Narrow Waist: Frame as a Sum Type and What It Costs

## 왜 이 챕터인가

[[ch-01/read]] showed you the **mechanism** of splicing: one method signature, one write
verb, a `link()` whose entire body is two pointer assignments and a log line, and a
`_link_processors` fold that validates nothing. That chapter answered *how* you can drop a
processor into position N.

It did not answer *why that is even possible*. A uniform method signature is worthless if
the thing flowing through it is not uniform. `async def process_frame(self, frame: Frame,
direction: FrameDirection)` type-checks in every position only because **exactly one
datatype crosses every boundary** — `Frame`. Lego blocks are the symptom. The hourglass is
the cause.

This chapter is about that one datatype: what it carries (nothing), how it is split (a
scheduling contract, not a content taxonomy), how many of them there are (119 concrete
classes in one file, 150 repo-wide, counted by AST walk and shown to you), what that costs
(577 `isinstance` sites across 136 files, and no automatic pass-through anywhere), and —
this is the part you will actually use — **the rule that decides which boson concepts are
allowed to become frames at all.**

That rule is not advice. Pipecat's own `flows/` package is a conversation state machine
with nodes, transitions and actions — structurally the same problem as boson's stage
machine ([[boson-stage-machine]]) plus its layered rule engine ([[boson-layers-rules]]).
`flows/` added **exactly two** frame classes. Not twenty. That precedent is the budget you
inherit, and section 12 turns it into a three-way test you can run on every boson concept
before you write a line of migration code.

One thing this chapter does **not** do: it does not tell you whether the migration is worth
doing. It tells you what the waist demands if you do it. The keep/replace argument is
[[ch-13/read]]'s and nowhere else's.

---

## 1. Count the top, count the bottom, count the middle

Before any theory, do the measurement. The hourglass claim is falsifiable: it says plurality
lives above and below and singularity lives in the middle. So count.

**The top of the glass — services.**

```
$ ls src/pipecat/services | wc -l
73
$ ls -d src/pipecat/services/*/ | wc -l
62
$ ls src/pipecat/services/*.py | wc -l
11
```

62 provider **directories** (one per vendor) plus 11 loose `.py` modules that are the
abstract service bases, not vendors:

```
src/pipecat/services/__init__.py        src/pipecat/services/stt_latency.py
src/pipecat/services/ai_service.py      src/pipecat/services/stt_service.py
src/pipecat/services/image_service.py   src/pipecat/services/tts_service.py
src/pipecat/services/llm_service.py     src/pipecat/services/vision_service.py
src/pipecat/services/mcp_service.py     src/pipecat/services/websocket_service.py
src/pipecat/services/settings.py
```

Note for accuracy: [[pipecat-design-philosophy]] reports "73 dirs under
`src/pipecat/services/`". 73 is the count of **entries**, not directories. The directory
count at this commit is 62. Where an excerpt and the tree disagree, the tree wins — that is
the standing rule for this whole course.

**The bottom of the glass — transports and serializers.**

```
$ ls src/pipecat/transports/
__init__.py  base_input.py  base_output.py  base_transport.py
daily  heygen  lemonslice  livekit  local  moq  smallwebrtc  tavus  vonage  websocket  whatsapp
```

11 provider packages, plus three abstract bases. And under `serializers/`:

```
$ ls src/pipecat/serializers/
__init__.py  base_serializer.py  exotel.py  genesys.py  plivo.py
protobuf.py  telnyx.py  twilio.py  vonage.py
```

7 concrete `FrameSerializer` implementations in that package —
`ExotelFrameSerializer`, `GenesysAudioHookSerializer`, `PlivoFrameSerializer`,
`ProtobufFrameSerializer`, `TelnyxFrameSerializer`, `TwilioFrameSerializer`,
`VonageFrameSerializer`. (Precision: there is an eighth `FrameSerializer` subclass in the
tree, `RTVIEvalSerializer` at `src/pipecat/evals/serializer.py:87`, but it is a test-harness
serializer and not part of the transport bottom. Seven is the number for the bottom of the
glass.)

**The middle of the glass.**

One class. `Frame`, at `src/pipecat/frames/frames.py:65`.

62 vendors above, 11 transports and 7 wire formats below, one datatype between them. That
is the hourglass, measured, at this commit. Everything else in this chapter is a consequence.

---

## 2. The literature, named and dated

You do not need this section to write code. You need it because the failure modes ahead are
*known* failure modes with names, and naming them is how you recognise them in boson before
they cost you a week.

**A worked example first, because "narrow waist" is an unfamiliar operational concept and
the diagram usually gets taught before the mechanism.**

Suppose you want to send a message from a laptop on Wi-Fi to a server on fibre, through
four intermediate carriers you have never heard of. There are, say, 40 physical link
technologies in the world and 400 applications. If every application had to speak every
link technology, you would need 16,000 pieces of adapter code, and adding one new link
technology would mean writing 400 new adapters. Instead everyone agrees on **one** format
in the middle — an IP packet — that says almost nothing: source address, destination
address, payload, a few flags. No reliability, no ordering, no security, no session. A new
link technology writes **one** adapter (carry IP packets), and a new application writes
**one** adapter (emit IP packets). 16,000 becomes 440.

The counter-intuitive part, and the whole thesis of the literature: **the middle format is
adoptable precisely because it promises so little.** If IP had guaranteed in-order delivery,
every link technology would have had to implement reordering to qualify, and most would not
have bothered. Generality at the waist is bought by *removing* capability, not adding it.

Now the citations.

- **Origin of the figure.** CSTB/NRC, *Realizing the Information Future: The Internet and
  Beyond*, National Academy Press, **1994** — many transmission technologies below, many
  applications above, one packet-transport "bearer service" at the waist.
- **The vocabulary.** Steve Deering, **"Watching the Waist of the Protocol Hourglass,"**
  keynote, **ICNP '98**, Austin TX, Oct 1998. This is the talk that made "narrow waist" and
  *IP over everything / everything over IP* standard terminology.
- **The theory.** Micah Beck, **"On the Hourglass Model," *CACM* 62(7):48–57, July 2019**.
  The thesis is the **Deployment Scalability Trade-off**: the *weaker* and more minimal the
  spanning layer, the more implementations support it and the more applications build on
  it. Beck credits David D. Clark for the term "spanning layer."
- **Evolvability, empirically.** Akhshabi & Dovrolis, *The Evolution of Layered Protocol
  Stacks Leads to an Hourglass-Shaped Architecture*, **SIGCOMM '11**. Their EvoArch model
  shows the hourglass emerges from competition, and that the waist **ossifies** — layers
  above and below innovate freely *precisely because* the waist cannot move.
- **The bill.** Philip Wadler, **"The Expression Problem,"** java-genericity mailing list,
  **12 Nov 1998**, verbatim: *"The goal is to define a datatype by cases, where one can add
  new cases to the datatype and new functions over the datatype, without recompiling
  existing code, and while retaining static type safety (e.g., no casts)."*

Hold Wadler for section 8. Hold Akhshabi & Dovrolis for section 10. Beck is immediately
testable, so test him now.

---

## 3. `Frame` carries nothing — Beck's minimality, literally

If Beck is right, the waist class should assert as close to nothing as it can while still
being useful. Open it.

**`src/pipecat/frames/frames.py:64-101`**

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

    def __post_init__(self):
        self.id: int = obj_id()
        self.name: str = f"{self.__class__.__name__}#{obj_count(self)}"
        self.pts: int | None = None
        self.broadcast_sibling_id: int | None = None
        self.metadata: dict[str, Any] = {}
        self.transport_source: str | None = None
        self.transport_destination: str | None = None

    def __str__(self):
        return self.name
```

Read what is *not* there. There is no payload field. No `data`, no `bytes`, no `content`,
no `kind` discriminator string, no timestamp you can trust (`pts` is set to `None` and
filled in later by whoever cares), no direction, no schema version.

Every single field is `field(init=False)`. That is a deliberate constructional fact, not a
style choice: **you cannot pass any of these to the constructor.** A `Frame` subclass's
`__init__` signature is entirely its own dataclass fields; the seven base fields are
assigned in `__post_init__` afterwards. The base class contributes zero constructor
arguments to the 119 classes below it.

Two of the seven fields are pure identity (`id`, `name`), two are routing hints that
transports fill in (`transport_source`, `transport_destination`), one is a correlation
pointer used by the both-directions broadcast (`broadcast_sibling_id`), one is a timestamp
slot (`pts`), and one is an untyped escape hatch (`metadata`). That is the whole spanning
layer.

This is exactly Beck's prediction. The waist asserts almost nothing, so almost anything can
implement it. Concretely: to become a Pipecat frame, a type must supply *nothing at all* —
`@dataclass class MyFrame(ControlFrame): pass` is a complete, legal, routable frame. The
cost of admission is zero. That is why there are 119 of them.

---

## 4. The three-way split is a scheduling contract, not a taxonomy

Directly under `Frame` sit three classes. The obvious guess is that they classify *content* —
system stuff, data stuff, control stuff. That guess is wrong, and the docstrings say so in
their own words. Read all three together.

**`src/pipecat/frames/frames.py:104-140`**

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

Every discriminating clause in those three docstrings is about **time**, not content:
"higher priority," "handled in order," "not affected by user interruptions," "cancelled by
user interruptions," "after everything is flushed." The one clause that mentions content —
"usually contains data such as LLM context, text, audio or images" — is hedged with
"usually" and is descriptive, not normative.

So the question a frame author answers by picking a base class is not *what is this?* but
**when must this run, and does it survive a barge-in?**

The code proves it in two places.

**Place one — the priority queue.** `src/pipecat/processors/frame_processor.py:132-171`

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

Three tiers, and the third tier is `else`. Look at the branch structure hard, because
section 5 lives inside it: the queue tests for `StartFrame`, then for `SystemFrame`, then
gives up and assigns 20. It never mentions `DataFrame` or `ControlFrame` by name.

**Place two — the input loop.** `src/pipecat/processors/frame_processor.py:1295-1312`

```python
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
```

A `SystemFrame` is executed **inline, on the input task**. Everything else is pushed into a
second, separately-cancellable queue. That is the entire out-of-band mechanism, and it is
six lines. The reason `InterruptionFrame` can reach a processor that is busy speaking is not
a scheduler and not a supervisor — it is `isinstance(frame, SystemFrame)` on line 1304.

The design consequence you should carry into boson: **priority in Pipecat is structural and
declared once, at class-definition time.** There is no per-send priority argument, no
`push_frame(frame, urgent=True)`. If you want a boson signal to jump the queue during a
barge-in, that is a decision you make when you write `class X(SystemFrame)`, and you cannot
change it per call site.

---

## 5. Honest finding #1 — at runtime the split is two-way

Now cash the observation from section 4. Grep the tree for dispatch on each branch.

```
$ grep -rn --include='*.py' "isinstance(frame, SystemFrame)" src/pipecat/ | wc -l
10
$ grep -rn --include='*.py' "isinstance(frame, DataFrame)" src/pipecat/ | wc -l
0
$ grep -rn --include='*.py' "isinstance(frame, ControlFrame)" src/pipecat/ | wc -l
0
```

`SystemFrame` is dispatched on ten times. `DataFrame` and `ControlFrame` are dispatched on
**zero** times, anywhere in `src/pipecat`. Widen the grep and the result holds: every
non-import occurrence of `DataFrame` outside `frames.py` is a `class X(DataFrame):`
declaration — `DailySIPTransferFrame`, `OpenClawSendFrame`, `LLMSearchResponseFrame` and so
on. `DataFrame` is used exclusively as a base to inherit from, never as a thing to test for.

So the runtime taxonomy is: **`StartFrame`, then `SystemFrame`, then everything else.**
Data-versus-Control is a documentation convention that helps humans read `frames.py`. It has
no runtime meaning in the core pipeline.

Which raises the obvious follow-up. The `DataFrame` docstring says "Data frames are
cancelled by user interruptions." If nothing tests for `DataFrame`, what actually cancels
them?

**A non-`Frame` mixin does.** `UninterruptibleFrame` at `frames.py:147`:

**`src/pipecat/frames/frames.py:146-158`**

```python
@dataclass
class UninterruptibleFrame:
    """A marker for data or control frames that must not be interrupted.

    Frames with this mixin are still ordered normally, but unlike other frames,
    they are preserved during interruptions: they remain in internal queues and
    any task processing them will not be cancelled. This ensures the frame is
    always delivered and processed to completion.

    """

    pass
```

It does not inherit from `Frame`. It is a bare marker dataclass, used as
`class EndFrame(ControlFrame, UninterruptibleFrame)`. And it is dispatched on — three times,
all in one file.

**`src/pipecat/utils/frame_queue.py:73-95`**

```python
    def _put(self, item: Any) -> None:
        if isinstance(self._frame_getter(item), UninterruptibleFrame):
            self._uninterruptible_count += 1
        super()._put(item)

    def _get(self) -> Any:
        item = super()._get()
        if isinstance(self._frame_getter(item), UninterruptibleFrame):
            self._uninterruptible_count -= 1
        return item

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

`reset()` is the interruption purge. It keeps `UninterruptibleFrame`s and throws away
everything else — regardless of whether the discarded frame is a `DataFrame`, a
`ControlFrame`, or (see section 6) in neither branch. And the caller is the interruption
path itself:

**`src/pipecat/processors/frame_processor.py:1130-1151`**

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

State it plainly, because this is the sentence the docstrings obscure:

> **A frame is cancelled on interruption by default. It survives only if it carries the
> `UninterruptibleFrame` mixin. Membership in `DataFrame` or `ControlFrame` has nothing to
> do with it.**

The `EndFrame` / `CancelFrame` pair makes the same point from the other side. `EndFrame`
(`frames.py:1899`) is `ControlFrame, UninterruptibleFrame` — graceful drain: it arrives
after everything queued ahead of it, and it cannot be purged. `CancelFrame`
(`frames.py:999`) is a `SystemFrame` — hard stop: it arrives *ahead* of them. Two different
shutdown semantics, expressed entirely through base-class choice, with no shutdown code
comparing them.

**Why you care.** When you write `class RuleViolationFrame(ControlFrame)` in section 12, you
are making a barge-in decision, and the base class alone does not fully make it. If the
verdict must still be delivered even though the customer interrupted — say, a DNC
registration that has legal consequences ([[boson-stage-machine]] routes DNC through
`register_dnc`) — `ControlFrame` alone is not enough. You need
`class X(ControlFrame, UninterruptibleFrame)`, and you need to know that because you read
`frame_queue.py`, not because any docstring told you.

---

## 6. Honest finding #2 — the taxonomy leaks

`LLMContextFrame` is arguably the most important frame in a voice pipeline: it is what tells
the LLM service "here is the conversation, generate." Look at what it inherits.

**`src/pipecat/frames/frames.py:550-561`**

```python
@dataclass
class LLMContextFrame(Frame):
    """Frame containing a universal LLM context.

    Used as a signal to LLM services to ingest the provided context and
    generate a response based on it.

    Parameters:
        context: The LLM context containing messages, tools, and configuration.
    """

    context: LLMContext
```

`class LLMContextFrame(Frame)`. Not `SystemFrame`, not `DataFrame`, not `ControlFrame` —
`Frame` directly. It is in **no branch**.

My AST walk over `frames.py` finds exactly one such class in the whole file:

```
in no branch: ['LLMContextFrame']
in two branches: ['InputTextRawFrame']
```

(`InputTextRawFrame` at `frames.py:1481` is the mirror-image anomaly:
`class InputTextRawFrame(SystemFrame, TextFrame)`, so it is counted under both `SystemFrame`
and `DataFrame`. The two anomalies cancel in the arithmetic — see section 7.)

What happens to a frame in no branch? Go back to `FrameProcessorQueue.put()`. It tests
`StartFrame`, then `SystemFrame`, then `else: priority = self.DEFAULT_PRIORITY`.
`LLMContextFrame` falls through to 20. It behaves exactly like a `DataFrame` — ordered,
purgeable on interruption — but **by fallthrough, not by declaration.**

Practically nothing breaks today, because default-priority is the behaviour you want for it.
The problem is epistemic, and it is the kind of thing that bites during a migration:
`LLMContextFrame`'s scheduling behaviour is not stated anywhere in its class definition. It
is an emergent property of an `else` branch in a different file. If someone later changed
that `else` — say, to raise on unclassified frames, or to route unclassified frames to
`SYSTEM_PRIORITY` — `LLMContextFrame` would silently change scheduling class and nothing in
`frames.py` would look different.

This is what a leaky sum type looks like in practice. The type system permits
`class X(Frame)`, so the three-way contract that the docstrings describe is a convention the
compiler does not enforce. Which is a preview of the whole cost structure. On to the count.

---

## 7. The cost, measured — and the counting trap

> **Figure.** Open [`figures/frame-waist.html`](figures/frame-waist.html) now and keep it
> open for sections 7 and 8. It draws this commit's hourglass from the counts you are about
> to verify, then turns into the cost calculator: click "add a column" to watch a new
> processor compose instantly, then click "add a row" to watch the same action light up all
> 577 `isinstance` sites and flag the processors that would silently drop your frame. Do
> those two clicks back to back — the asymmetry between them is the entire argument of
> section 8, and it lands faster as an animation than as prose.

`frames.py` is 2,415 lines. The question is how many frame types it defines, and this is
where every naive measurement of Pipecat goes wrong — including the one in
[[theory-narrow-waist]], which reports 120 concrete in-file and 151 repo-wide. The correct
numbers at this commit are **119** and **150**, and the discrepancy is instructive enough to
walk through, because the same trap will bite you when you audit your own migration.

**Do not grep.** `grep -c "^class .*Frame"` overcounts (mixins, non-frame helpers) and
undercounts (subclasses whose base is not literally named `*Frame`). Walk the AST and
compute a transitive closure.

```python
import ast
def basename(b):
    # normalize subscripted bases: Generic[X] -> Generic
    while isinstance(b, ast.Subscript): b = b.value
    if isinstance(b, ast.Name): return b.id
    if isinstance(b, ast.Attribute): return b.attr
    return None

t = ast.parse(open('src/pipecat/frames/frames.py').read())
tops = [n for n in t.body if isinstance(n, ast.ClassDef)]
bases = {c.name: [basename(b) for b in c.bases] for c in tops}

desc, changed = {'Frame'}, True
while changed:                       # fixed point
    changed = False
    for n, bs in bases.items():
        if n not in desc and any(b in desc for b in bs):
            desc.add(n); changed = True
```

Result:

```
top-level ClassDefs: 129
Frame descendants (incl Frame): 123
NON-descendants: 6 ['UninterruptibleFrame', 'AudioRawFrame', 'ImageRawFrame',
                    'FunctionCallResultProperties', 'DTMFFrame', 'FunctionCallFromLLM']
concrete (desc minus Frame + 3 branch): 119
SystemFrame 46
DataFrame 33
ControlFrame 40
```

**129 top-level classes → 122 proper `Frame` subclasses → 119 concrete frame types**, after
subtracting the three branch classes (`SystemFrame`, `DataFrame`, `ControlFrame`), which are
abstract markers nobody instantiates.

**The 7 top-level classes that are not `Frame` descendants**, named so you can see exactly
which ones a careless count sweeps in:

| Class | Line | What it actually is |
|---|---|---|
| `Frame` | 65 | the waist itself — not a *descendant* |
| `UninterruptibleFrame` | 147 | mixin (section 5) |
| `AudioRawFrame` | 161 | payload mixin: `audio`, `sample_rate`, `num_channels`, `num_frames` |
| `ImageRawFrame` | 181 | payload mixin: `image`, `size`, `format` |
| `DTMFFrame` | 843 | keypad-digit payload mixin |
| `FunctionCallResultProperties` | 748 | plain result-metadata dataclass |
| `FunctionCallFromLLM` | 1330 | plain tool-call descriptor dataclass |

Three of those seven end in `Frame` and are not frames. That is why grep fails.

**Now the trap.** Run the *same* walk but read `ast.ClassDef.bases` literally — take only
`ast.Name` bases, do not normalise subscripts:

```python
bases = {c.name: [b.id for b in c.bases if isinstance(b, ast.Name)] for c in tops}
```

```
naive descendants: 120
missing: ['AudioRawFrame', 'DTMFFrame', 'FunctionCallFromLLM',
          'FunctionCallResultProperties', 'ImageRawFrame', 'UninterruptibleFrame',
          'LLMUpdateSettingsFrame', 'STTUpdateSettingsFrame', 'TTSUpdateSettingsFrame']
```

The first six are correctly excluded. The last three are **real frames that vanished**, and
120 − 1 (`Frame`) − 3 (branches) = 116, three short. Here is why:

**`src/pipecat/frames/frames.py:2250-2251, 2282-2299`**

```python
@dataclass
class ServiceUpdateSettingsFrame(ControlFrame, UninterruptibleFrame, Generic[TSettings]):
```

```python
@dataclass
class LLMUpdateSettingsFrame(ServiceUpdateSettingsFrame[LLMSettings]):
    """Frame for updating LLM service settings."""

    pass


@dataclass
class TTSUpdateSettingsFrame(ServiceUpdateSettingsFrame[TTSSettings]):
    """Frame for updating TTS service settings."""

    pass


@dataclass
class STTUpdateSettingsFrame(ServiceUpdateSettingsFrame[STTSettings]):
    """Frame for updating STT service settings."""

    pass
```

`ServiceUpdateSettingsFrame` itself survives the naive walk — its first base, `ControlFrame`,
is a plain `ast.Name`. What breaks is its **three subclasses**, whose sole base is
`ServiceUpdateSettingsFrame[LLMSettings]`, an `ast.Subscript`. A walk that only collects
`ast.Name` bases sees them as having no bases at all and drops them out of the hierarchy.

That matters more than a rounding error, because those three are among the most
migration-relevant frames in the tree. `LLMUpdateSettingsFrame` is the frame you will use in
section 12 to swap boson's stage prompt, and `TTSUpdateSettingsFrame` is how you would swap
a Korean TTS voice mid-call. A frame audit that silently drops them would tell you the
capability does not exist.

**The branch split.**

```
SystemFrame  : 46
DataFrame    : 33
ControlFrame : 40
             -----
               119
```

46 + 33 + 40 = 119 exactly, which is arithmetic luck of a specific kind: `LLMContextFrame`
sits in no branch (subtracts one from the sum's coverage) and `InputTextRawFrame` sits in
two (adds one back). Two independent anomalies that happen to cancel. Do not treat the sum
matching as a validation of the branch counts.

Some representative members, so the numbers are not abstract:

- **SystemFrame (46):** `StartFrame` (924), `CancelFrame` (999), `ErrorFrame` (1016),
  `InterruptionFrame` (1142), `UserStartedSpeakingFrame` (1154),
  `UserStoppedSpeakingFrame` (1165), `VADUserStartedSpeakingFrame` (1226),
  `ProposedUserStartedSpeakingFrame` (1256), `BotStartedSpeakingFrame` (1282),
  `InputAudioRawFrame` (1449), `InputDTMFFrame` (1552).
- **DataFrame (33):** `OutputAudioRawFrame` (201), `TTSAudioRawFrame` (241),
  `TextFrame` (303), `LLMTextFrame` (343), `TTSTextFrame` (417),
  `TranscriptionFrame` (450), `InterimTranscriptionFrame` (476), `LLMRunFrame` (634),
  `LLMMessagesAppendFrame` (645), `LLMSetToolsFrame` (694), `TTSSpeakFrame` (795).
- **ControlFrame (40):** `EndFrame` (1899), `StopFrame` (1923), `HeartbeatFrame` (2007),
  `LLMFullResponseStartFrame` (2059), `TTSStartedFrame` (2210), `TTSStoppedFrame` (2231),
  `ServiceUpdateSettingsFrame` (2251), `LLMUpdateSettingsFrame` (2283),
  `VADParamsUpdateFrame` (2320).

**Repo-wide.** Run the same closure over every `.py` under `src/pipecat/`:

```
repo-wide concrete frame classes (top-level only): 150
outside frames.py: 31
  9  src/pipecat/processors/frameworks/rtvi/frames.py
  8  src/pipecat/transports/daily/transport.py
  6  src/pipecat/services/openclaw/frames.py
  3  src/pipecat/transports/livekit/transport.py
  2  src/pipecat/flows/actions.py
  1  src/pipecat/tests/utils.py
  1  src/pipecat/services/google/frames.py
  1  src/pipecat/pipeline/sync_parallel_pipeline.py
```

**150 = 119 + 31.** Note that pattern — it is section 11's whole argument in advance. The
31 out-of-tree frames are not scattered randomly; they cluster in three subsystem-local
`frames.py` files (RTVI 9, Daily 8, OpenClaw 6). Nobody added their subsystem's vocabulary
to the shared waist.

**One caveat on 150.** The walk counts top-level `ClassDef`s only, and `frames.py` hides
four more inside an indented block:

**`src/pipecat/frames/frames.py:1830-1848`**

```python
# The leaf aliases below intentionally subclass the deprecated TaskFrame /
# TaskSystemFrame bases; silence the subclassing DeprecationWarning that
# @deprecated would otherwise emit while this module is imported.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)

    @deprecated(
        "`EndTaskFrame` is deprecated since 1.4.0 and will be removed in 2.0.0. "
        "Use `EndWorkerFrame` instead."
    )
    @dataclass
    class EndTaskFrame(EndWorkerFrame, TaskFrame):
        """Deprecated alias for :class:`EndWorkerFrame`.

        .. deprecated:: 1.4.0
            Use :class:`EndWorkerFrame` instead. Will be removed in 2.0.0.
        """

        pass
```

Four such aliases exist — `EndTaskFrame` (1841), `StopTaskFrame` (1855),
`CancelTaskFrame` (1869), `InterruptionTaskFrame` (1883) — nested inside a
`with warnings.catch_warnings():` block, not inside a function. (I checked: an AST walk over
function bodies finds zero nested classes in this file; the container is the `with`.) Count
them and the repo total is 154. They are deprecated 1.4.0 aliases from the
`PipelineTask → PipelineWorker` rename, so 150 is the number I will use, with the caveat
stated rather than hidden.

**The dispatch tax.**

```
$ grep -rn --include='*.py' "isinstance(frame, " src/pipecat/ | wc -l
577
$ grep -rln --include='*.py' "isinstance(frame, " src/pipecat/ | wc -l
136
```

**577 sites, 136 files.** That is the number to hold in your head for the next section.

---

## 8. Wadler's bill: columns are cheap, rows are expensive

Wadler's framing, applied here:

- **Rows = cases of the datatype = frame types.** 119 of them in `frames.py`.
- **Columns = functions over the datatype = processors.** Every `process_frame` override.

A sum type (a class hierarchy with dispatch by `isinstance`) makes one axis cheap and the
other expensive, and Pipecat picked its cheap axis deliberately.

**Adding a column is nearly free.** Implement one method with one signature and you compose
with every existing frame:

```python
class MyProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
```

That is a complete, correct, universally-composable processor. It goes at any position N in
any `Pipeline([...])` and type-checks ([[ch-01/read]]). No existing file changes. No
recompilation of anything. **Zero cells in the matrix are touched except the new column.**

**Adding a row is free for the pipeline and a liability for every processor.** This is the
part people get wrong, so nail down the mechanism. Go back to the base implementation.

**`src/pipecat/processors/frame_processor.py:820-847`**

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

Read the last line and then read the whole method again. **There is no `else`. There is no
`await self.push_frame(frame, direction)` anywhere in this method.**

That is the single most consequential fact in this chapter. The base class handles exactly
six frame kinds — `StartFrame`, `InterruptionFrame`, `CancelFrame`, and the pause/resume
pairs — and for every other frame in existence it **does nothing and returns**. Forwarding
is not inherited. It is the subclass author's job, every time, via `push_frame` at
`frame_processor.py:1004`.

**Worked example — kill a frame in five lines.** Suppose you add a row:

```python
@dataclass
class RuleVerdictFrame(ControlFrame):
    verdict: str
    rule_name: str
```

You push it from your rule processor. It travels downstream through
`STTMuteFilter → LLMUserContextAggregator → BosonRuleProcessor → OpenAILLMService → …`.
Now suppose one processor in that chain was written like this — which is the normal shape of
a filtering processor:

```python
async def process_frame(self, frame: Frame, direction: FrameDirection):
    await super().process_frame(frame, direction)
    if isinstance(frame, TextFrame):
        await self.push_frame(self._transform(frame), direction)
    elif isinstance(frame, (StartFrame, EndFrame, InterruptionFrame)):
        await self.push_frame(frame, direction)
```

Your `RuleVerdictFrame` matches neither branch. `super().process_frame()` matched none of its
six either. So the method returns having done nothing. The frame is **gone** — not queued,
not errored, not logged. No exception, no warning, no metric. The downstream guard you wrote
the frame for never sees it, and the failure presents as "the rule sometimes doesn't fire,"
which is the worst possible bug shape in a tele-sales call you cannot reproduce.

That is the row cost, and it is why the number 577 matters: 577 `isinstance(frame, ...)`
sites across 136 files are 577 places where an author decided, once, which frames to
forward. Every one of them was written before your frame existed. Every one of them is a
potential silent drop.

Compare the two operations honestly:

| Operation | Files changed | Existing code touched | Failure mode if wrong |
|---|---|---|---|
| Add a processor (column) | 1 new file | none | loud — pipeline is silent, you notice in one test call |
| Add a frame (row) | 1 new file | none *required* | silent — the frame vanishes at an unknown hop |

Note the perverse symmetry: adding a row requires touching nothing, which is exactly why it
is dangerous. The compiler will not tell you which of the 136 files needed updating, because
in a sum type there is no exhaustiveness check to fail. (Hold that thought — [[ch-03/read]]
is about a design that made the opposite bet and gets exhaustiveness in exchange for
something else.)

---

## 9. Pipecat chose columns, and for a streaming framework that is the right axis

The choice is defensible on one measurable ground: **which axis actually grows.**

Frame vocabulary is *media* vocabulary — audio, text, image, LLM context, transcription,
DTMF. That set has been stable for a decade and will be stable for another. Service
vocabulary is *vendor* vocabulary, and vendors arrive weekly. 62 service directories at this
commit, and the README states the intent in one word:

**`README.md:23-29`**

```
## 🧠 Why Pipecat?

- **Voice-first**: Integrates speech recognition, text-to-speech, and conversation handling
- **Pluggable**: Supports many AI services and tools
- **Composable Pipelines**: Build complex behavior from modular components
- **Multi-Agent Ready**: Each pipeline is an agent. Compose them with handoff, parallel fan-out, sidecar workers, or distributed deployments
- **Real-Time**: Ultra-low latency interaction with different transports (e.g. WebSockets or WebRTC)
```

"Pluggable" and "Composable Pipelines" are both column-axis claims. There is no bullet about
extending the frame vocabulary.

The governance position makes it sharper. From [[pipecat-design-philosophy]], the only
explicit refusal in the entire repo is `COMMUNITY_INTEGRATIONS.md:9-11`: *"**What we don't
do:** The Pipecat team does not code review, test, or maintain community integrations."* The
core team optimises the column axis so hard that it hands the columns themselves to other
people. A framework that intends to accept unreviewed third-party columns *must* make columns
cheap and must not require them to understand the whole row set.

And the one direction rule `AGENTS.md` actually states (`:207`) is the column-side contract:
*"By default, all frames should be pushed in the direction they came."* Not "handle every
frame" — just "do not reverse anything you forward." The framework asks little of a column
author precisely because it expects a lot of column authors.

**Extension move, for boson.** Your growth vector is not vendors. Over the next year boson
will add rule checks, stages, script steps and compliance verdicts — 13 live `@check`s today
([[boson-layers-rules]]), nine registered stages ([[boson-stage-machine]]). Ask yourself
which Pipecat axis each of those maps onto. A rule check is *logic over a completed
utterance* — that is a column. A stage is *a bundle of prompt + tools + legal successors* —
that is not a frame at all, it is state (section 12). If you find yourself reaching for a new
frame class for the fourth time in a week, you are trying to grow the row axis of a
framework that priced rows expensively on purpose. That is the check.

---

## 10. Ossification, and the escape hatch that Pipecat itself barely uses

Akhshabi & Dovrolis's finding is that the waist ossifies, and that this is a *feature*: the
layers above and below can innovate freely precisely because the waist cannot move. Pipecat
shows both halves.

**The waist is frozen.** `Frame`'s seven fields (section 3) are effectively unchangeable now.
Every one of the 119 concrete classes inherits them; every `__post_init__` in the tree
depends on them; adding an eighth would touch the constructor semantics of everything. So
when a need arises for per-frame side-channel data, the answer cannot be a new base field.

**The declared escape hatch is `metadata: dict[str, Any]`** — untyped, free to every
processor, invisible to the type system. That is the ossification pressure valve the
literature predicts.

**Honest finding #3: Pipecat writes it once and reads it never.** I grepped:

```
$ grep -rn --include='*.py' "\.metadata\[" src/pipecat/ | wc -l
0
```

The only frame-metadata assignment anywhere in `src/pipecat/`:

**`src/pipecat/pipeline/worker.py:1226` and `:1457-1464`**

```python
        start_frame.metadata = self._create_start_metadata()
```

```python
    def _create_start_metadata(self) -> dict[str, Any]:
        """Build and return start metadata including user-provided values."""
        start_metadata = {}

        # Update with user provided metadata.
        start_metadata.update(self._params.start_metadata)

        return start_metadata
```

One write, on `StartFrame`, populated from `WorkerParams.start_metadata`. No read anywhere
in the core. So `frame.metadata` is a **provisioned but unexercised** convention: it is a
real dict on every frame and it will carry whatever you put in it, but nothing in Pipecat
propagates it, merges it across frame boundaries, or logs it. If you use it for correlation
IDs in boson, you own both ends — the write, the read, and the propagation across every
transformation that constructs a new frame from an old one. Sections 12's advice to use it
still stands; the caveat is that you are the first serious user in this pipeline.

**The waist is being actively narrowed, not widened.** This is the strongest evidence that
the Beck reading is not my imposition — the maintainers are removing capability from frames
right now. `StartFrame` carries seven configuration fields, and all seven are deprecated as
of 1.8.0:

**`src/pipecat/frames/frames.py:923-995` (abridged)**

```python
@dataclass
class StartFrame(SystemFrame):
    """Initial frame to start pipeline processing.

    This is the first frame that should be pushed down a pipeline to
    initialize all processors with their configuration parameters.

    Parameters:
        audio_in_sample_rate: Input audio sample rate in Hz.

            .. deprecated:: 1.8.0
                Read ``audio_in_sample_rate`` in ``FrameProcessorSetup.setup()`` instead.
                Will be removed in 2.0.0.
    ...
    """

    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    enable_metrics: bool = False
    enable_tracing: bool = False
    enable_usage_metrics: bool = False
    report_only_initial_ttfb: bool = False
    tracing_context: TracingContext | None = None

    def __getattribute__(self, name: str) -> Any:
        # Reads warn, writes don't: assignment goes through ``__setattr__``. The None
        # guard is for ``tracing_context``, the only field whose default is None and
        # so can't be told apart from unset.
        if name in _START_FRAME_DEPRECATED_FIELDS:
            value = object.__getattribute__(self, name)
            if value is not None:
                warn_deprecated_read(
                    f"`StartFrame.{name}` is deprecated since 1.8.0, "
                    f"read `{name}` in `FrameProcessorSetup.setup()` instead. "
                    "Will be removed in 2.0.0."
                )
            return value
        return object.__getattribute__(self, name)
```

The changelog fragment states the intent without hedging:

**`changelog/5316.deprecated.md`**

```
- Deprecated `StartFrame.audio_in_sample_rate`, `StartFrame.audio_out_sample_rate`,
  `StartFrame.enable_metrics`, `StartFrame.enable_tracing`, `StartFrame.enable_usage_metrics`,
  `StartFrame.report_only_initial_ttfb` and `StartFrame.tracing_context`, which will be
  removed in 2.0.0. Read the same values from `FrameProcessorSetup` in `setup()` instead.
  The fields still carry the pipeline's configuration, so a processor that reads one keeps
  working and emits a `DeprecationWarning`, once per call site.
```

Where does the configuration go instead? To a plain dataclass that is **not a frame**:

**`src/pipecat/processors/frame_processor.py:76-115` (abridged)**

```python
class FrameProcessorSetup:
    """Configuration parameters for frame processor initialization.
    ...
    """

    clock: BaseClock
    task_manager: BaseTaskManager
    pipeline_worker: PipelineWorker
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    enable_metrics: bool = False
    enable_tracing: bool = False
    enable_usage_metrics: bool = False
    observer: BaseObserver | None = None
    report_only_initial_ttfb: bool = False
    tracing_context: TracingContext | None = None
```

The rule being enforced: **frames carry data; setup carries configuration.** It rhymes with
`AGENTS.md:143`, which splits `@dataclass` (frames — "high-frequency, no validation needed")
from Pydantic `BaseModel` (service params, transport/VAD params, serializer params).

Write this down for section 12, because it is the most common migration mistake and boson is
positioned to make it: **if your port stuffs per-turn configuration into a frame payload, you
are pushing against the direction the maintainers are actively pulling.** Stage prompts,
tool lists and rule thresholds are configuration. They belong in a manager object or in
`FrameProcessorSetup`, and they reach services through the settings frames — not as
bespoke payload on a bespoke frame.

*(Side note, one line, for completeness: Pipecat runs a second, parallel hourglass for
cross-worker communication — `bus/messages.py` defines `BusMessage` splitting into
`BusDataMessage` / `BusSystemMessage`, the same priority split at a different waist, and
`BusFrameMessage(BusDataMessage)` is literally Frame-over-Bus, the IP-over-everything move
inside one codebase. See [[bus-and-extensions]]; it is not this chapter's subject.)*

---

## 11. The load-bearing precedent: `flows/` added exactly two frames

Everything above is diagnosis. Here is the prescription, and it is not my opinion — it is
what Pipecat's own team did when they faced boson's exact problem.

`src/pipecat/flows/` is a conversation state machine: nodes, transitions, actions, a manager
that drives the pipeline. Structurally that is boson's stage machine plus its action
vocabulary. If any subsystem in this repo had a licence to grow the row axis, it is this one.

**It defined two frames. Both `ControlFrame`. Neither in `frames.py`.**

**`src/pipecat/flows/actions.py:49-66`**

```python
@dataclass
class FunctionActionFrame(ControlFrame):
    """Frame containing a function action to be executed.

    Parameters:
        action: Action configuration dictionary.
        function: Function handler to execute.
    """

    action: dict
    function: FlowActionHandler


@dataclass
class ActionFinishedFrame(ControlFrame):
    """Frame indicating that an action has completed execution."""

    pass
```

Two classes, twelve lines of definition. And they exist for one specific reason: action
execution needs **in-band ordering against speech**. A `function` action must run at a known
point relative to the TTS output around it, and the only way to express "at a known point in
the stream" in Pipecat is to be a frame in the stream. `ActionFinishedFrame` exists so a
downstream observer can decrement the ongoing-action counter when the effect has actually
landed, rather than when the handler returned ([[flows-actions]]).

**Everything else rides existing frames.** `FlowManager` drives the pipeline by queueing
frames that already exist:

**`src/pipecat/flows/manager.py:764-770`**

```python
            frames = []

            # New path: role_message as LLM system instruction (persists until changed)
            if role_message:
                frames.append(
                    LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=role_message))
                )
```

**`src/pipecat/flows/manager.py:827-841`**

```python
            # Use an "update" (replace) frame for the RESET/RESET_WITH_SUMMARY
            # strategies; otherwise append. (Note that even the first node follows
            # the same rule: appending ensures any prior context contributions,
            # such as by tts_say pre-actions, is preserved rather than replaced).
            frame_type = (
                LLMMessagesUpdateFrame
                if update_config.strategy
                in [ContextStrategy.RESET, ContextStrategy.RESET_WITH_SUMMARY]
                else LLMMessagesAppendFrame
            )

            frames.append(frame_type(messages=messages))
            frames.append(LLMSetToolsFrame(tools=functions))

            await self._worker.queue_frames(frames)
```

Change the system prompt → `LLMUpdateSettingsFrame`. Change the tools →
`LLMSetToolsFrame`. Add or replace messages → `LLMMessagesAppendFrame` /
`LLMMessagesUpdateFrame`. Trigger inference → `LLMRunFrame()` (`manager.py:709`). Speak a
fixed line → `TTSSpeakFrame` (`actions.py`, the `tts_say` handler). Every one of those is an
existing frame that every existing LLM and TTS service already handles correctly.

**And the state stays out of the pipeline entirely.**

**`src/pipecat/flows/manager.py:147-149`**

```python
        self._state: dict[str, Any] = {}  # Internal state storage
        self._current_functions: set[str] = set()  # Track registered functions
        self._current_node: str | None = None
```

The current node is a `str`. The shared state is a plain `dict`. `NodeConfig`, `FlowResult`
and `ActionConfig` are `TypedDict`s in `flows/types.py` (lines 182, 41, 112). **None of them
is ever a frame.** State that only one component reads has no business in a universal waist.

The pattern holds across all 31 out-of-tree frames from section 7 — RTVI put its 9 in
`processors/frameworks/rtvi/frames.py`, Daily put its 8 in its own `transport.py`, OpenClaw
put its 6 in `services/openclaw/frames.py`. Nobody grew `frames.py`. `frames.py` is reserved
for the **shared media vocabulary**, and every subsystem keeps its private vocabulary
private.

*(How `flows/` actually works — `set_node`, the transition mechanics, the pre/post-action
ordering, and the finding that transitions do not have to be LLM function calls — is
[[ch-10/read]]'s subject. This section takes exactly one thing from it: the frame budget.)*

---

## 12. The rule for boson, and the three-way test

Here is the rule, stated once:

> **Subclass `ControlFrame`. Do not touch `DataFrame`. Put the class in `boson/frames.py`,
> never in `frames.py`. Budget two to four total.**

Now the reasoning, so you can defend a deviation when you find one.

**Why not `DataFrame`?** `DataFrame` is the conversation's *content* — the audio and text the
customer hears and says. A stage transition is not content. Putting boson signals in the
`DataFrame` branch buys nothing at runtime (nothing dispatches on it, section 5) and
mislabels the frame for every human who reads it afterwards.

**Why not `SystemFrame`?** Because `SystemFrame` means "jump the queue and survive
barge-in," and that is almost never what a boson signal wants. If a rule verdict about turn
N jumps ahead of turn N's audio, it arrives before the thing it is about. Worse, a
`SystemFrame` is executed **inline on the input task** (`frame_processor.py:1304`), so any
work it triggers blocks the processor's frame intake. Choose `SystemFrame` only when the
signal is genuinely a barge-in-class event.

**Why `ControlFrame`?** It gives you exactly what a rule verdict or a stage signal needs:
ordered relative to the speech around it, and purged when the customer interrupts — which is
correct, because a verdict about an utterance that got superseded should die with it. Add
`UninterruptibleFrame` (section 5) only for the specific signals that must be delivered even
through a barge-in. In boson that is a short list: DNC registration and consent recording are
the plausible candidates, because they have legal consequences and dropping one silently is
a compliance failure, not a UX glitch.

### The three-way test

Run every boson concept through this before it goes anywhere near a `Pipeline([...])`.

**(a) Is it STATE? → not a frame.**

The current stage, the accumulated rule results, the script cursor, the sentiment window,
the round counter. These live in a plain `BosonStageManager` object holding plain
dataclasses, mirroring `FlowManager._state` and `FlowManager._current_node`
(`manager.py:147-149`). The test: *does more than one component need to read this, in
order, as part of the stream?* If no — and for state, the answer is almost always no — it
is a field on a manager, not a frame.

Concretely, from [[boson-stage-machine]]: `session.active_stage` is a string. `StageMachine`
is explicitly stateless and shared. `MAX_ROUNDS` and `FALLBACK_TRANSITIONS` are dicts. All
of that ports as-is into a manager object and touches the waist zero times.

**(b) Is it an EFFECT? → an existing frame.**

This is where most of boson's vocabulary lands, and it lands cleanly. The mapping:

| boson concept | Pipecat frame that already exists | Line |
|---|---|---|
| stage transition changes the system prompt | `LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=...))` | `frames.py:2283` |
| stage transition changes the visible tools | `LLMSetToolsFrame(tools=[...])` | `frames.py:694` |
| `Respond(text)` — verbatim script line | `TTSSpeakFrame(text=..., append_to_context=...)` | `frames.py:795` |
| `Inject(content)` — rule verdict into context | `LLMMessagesAppendFrame(messages=[...])` | `frames.py:645` |
| context reset on stage entry | `LLMMessagesUpdateFrame(messages=[...])` | `frames.py:661` |
| trigger inference after a transition | `LLMRunFrame()` | `frames.py:634` |
| swap Korean TTS voice mid-call | `TTSUpdateSettingsFrame(delta=TTSSettings(...))` | `frames.py:2290` |

Seven boson behaviours, zero new frame classes, and every existing LLM/TTS service in the
tree already handles all seven correctly. That is what "full composability" cashes out to.

Note what this costs you, honestly, since I promised not to soften: the boson script engine
([[boson-script-engine]]) returns `Respond(step.text)` as an **LLM replacement** — the model
never sees the turn. Expressing that as `TTSSpeakFrame` gets the utterance spoken, but the
"suppress the LLM for this turn" semantics is not carried by the frame; it is a control
decision your processor has to make by not emitting `LLMRunFrame`. The frame maps; the
control inversion does not. That gap is real and is [[ch-12/read]]'s problem, not this
chapter's.

**(c) Is it a genuinely NEW in-band signal? → `boson/frames.py`, subclassing `ControlFrame`.**

The bar, stated precisely: **a signal must be ordered against speech, and its consumer must
be a processor that is not the component that produced it.** Both clauses. If the producer
and consumer are the same object, call a method — a frame that goes out and comes back to
the same processor is a very expensive function call.

Run your candidates through it:

| Candidate | Ordered against speech? | Different consumer? | Verdict |
|---|---|---|---|
| `RuleViolationFrame` — a downstream guard must see a verdict before TTS emits | yes | yes (guard ≠ rule processor) | **earns a class.** `ControlFrame`. |
| `StageEnteredFrame` — observability/analytics wants stage boundaries in the stream | yes | yes (observer ≠ manager) | **earns a class**, if and only if you actually build the observer. Otherwise it is state. |
| `ScriptStepFrame` — the script engine tells itself which step it is on | no | no (same object) | **rejected.** Manager field. |
| `StageTransitionFrame` — "please move to `purchase`" | — | no (the manager both decides and applies it) | **rejected.** A method call on the manager, whose *effects* are (b)'s frames. |
| `SentimentScoreFrame` — the 5-turn negative window | no | no | **rejected.** Manager field. |
| `ConsentCheckpointFrame` — legally-required consent acknowledgement | yes | yes | **earns a class**, and this is the one that needs `ControlFrame, UninterruptibleFrame`. |

That is two to three survivors out of six candidates, which lands inside the budget. The
budget itself: **2–4**, matching flows' 2 and OpenClaw's 6. Correlation IDs and stage tags
ride on `frame.metadata` (section 10's caveat applies — nothing in Pipecat reads it, so you
own the propagation).

**And the failure condition, stated as a tripwire rather than a warning:** if you find
yourself writing a tenth frame class, stop and read what you have built. Ten classes means
you have re-implemented boson's internal protocol — the 8-verb `ActionType` union from
`gateway/schemas/actions.py` plus stage and script signals — *inside* Pipecat's waist. At
that point every one of your ten types is subject to the row cost from section 8: 577
`isinstance` sites that do not know about them, no automatic pass-through, silent drops at
unknown hops. You would be paying the full sum-type tax and getting none of the
interchangeability you migrated for. The tenth frame is not a code smell; it is the signal
that the design went wrong three frames ago.

**A useful sanity check before you write any of them.** boson's protocol today is ten
strings: `VALID_CLIENT_TYPES = {"user_message", "partial_transcript", "interrupt",
"get_history"}` and `VALID_SERVER_TYPES = {"text_delta", "turn_end", "error", "interrupted",
"stage_changed", "history"}` ([[frame-taxonomy]]). Map them first, before inventing anything:
`"interrupt"` → `InterruptionFrame` (`frames.py:1142`); `"partial_transcript"` splits into
the real distinction `InterimTranscriptionFrame` (476) vs `TranscriptionFrame(finalized=...)`
(450); `"text_delta"` → `LLMTextFrame` (343); `"turn_end"` → `UserStoppedSpeakingFrame`
(1165) or `TTSStoppedFrame` (2231) depending on whose turn ended; `"error"` → `ErrorFrame`
(1016). Ten strings, and eight of them already have homes. Only `"stage_changed"` and
`"history"` are candidates for anything new — and `"history"` is state.

---

## 13. What this chapter does not settle

Three things are deliberately open, so you do not mistake them for closed.

1. **Whether boson's rule layers can express a veto at all inside a Pipecat pipeline.** A
   `FrameProcessor` acts the instant it calls `push_frame()`; boson's `LayerPipeline` stages
   every action and lets a later layer's `Filter` discard all of them, including the appended
   user message ([[boson-layers-rules]]). Sum types have nothing to say about two-phase
   commit. That is [[ch-12/read]]'s problem, and it turns out to be a latency problem.
2. **Whether the frames you choose in section 12 can be produced early enough to matter.**
   A rule verdict is only useful if it lands before inference starts, and where you can stand
   in the pipeline to guarantee that is a runtime question, not a type question — [[ch-04/read]]
   for the runtime, [[ch-11/read]] for the millisecond budget.
3. **Whether an open sum type is the right bet at all.** It is a bet, not a law. The
   alternative bet exists, is shipped, and you wrote it.

---

## 다음 챕터로

This chapter hands three things forward.

**To [[ch-03/read]], immediately:** the phrase **open sum type**, with its price tag
attached — anyone may add a `Frame` subclass without touching the core, and pays for it
across 577 `isinstance` sites with no exhaustiveness check and no automatic pass-through.
ch-03 characterises the design that made the exact opposite bet: `packages/realtime_voice/`
on branch `voice-chat-dev`, whose unit of data is a **closed union**,
`VoiceRuntimeEvent = VADEvent | ASREvent | AgentTextDelta | VoiceEvent` ([[rtv-pipeline-session]]).
Two designs, two answers to Wadler, optimising for different populations of author. ch-03
states what each one does. It casts no vote, and neither does this chapter.

**To [[ch-10/read]]:** the fact that `flows/` added exactly two `ControlFrame`s and kept
`_current_node` as a `str`. This chapter used that as a budget. ch-10 reads the package as
what it actually is — a state machine that deliberately lives *outside* the pipeline and
drives it through `queue_frames`.

**To [[ch-12/read]]:** the three-way test itself, plus the two-to-four budget and the
tripwire at ten. When you sit down to design boson's rule seam, the frame question will
already be answered; what remains is where the processor stands and what that position costs
in milliseconds.

One sentence to carry: **the waist is narrow because it promises nothing, and every promise
you add to it is a promise 136 files have not read.**

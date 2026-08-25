# The Narrow Waist: Why Frame Is Pipecat's IP (And What That Costs)
<!-- slug: theory-narrow-waist · type: theory · source: src/pipecat/frames/frames.py + Beck CACM 2019 + Wadler 1998 -->

**Core Insight.** "Lego blocks" is the symptom; the *hourglass* is the cause. Processors are
interchangeable only because exactly **one** datatype crosses every boundary: `Frame`. Plurality lives
above (services) and below (transports); the middle is singular. But a waist built as a **sum type**
pays the Expression Problem tax: nominally narrow (one base class), structurally wide (**120 concrete
frame classes in `frames.py`, 151 repo-wide**). Pipecat bought cheap processors by making frame-set
changes expensive — right for streaming, and the constraint that decides how boson migrates.

**Guideline.** Never add a `Frame` subclass to `frames.py` for a capability only your subsystem
understands. Follow Pipecat's own precedent: reuse existing frames to *drive* the pipeline, put
subsystem-local frames in `<subsystem>/frames.py`, keep state in a plain manager object. A new variant
is a new obligation on all **577 `isinstance(frame, ...)` sites across 136 files**.

## Technical Details

### 1. The literature, verified
- **Origin.** NRC/CSTB, *Realizing the Information Future: The Internet and Beyond* (National Academy Press, 1994) — the hourglass figure: many transmission technologies below, many applications above, one packet-transport "bearer service" at the waist. **Popularized** by Steve Deering (Cisco), **"Watching the Waist of the Protocol Hourglass,"** keynote, **ICNP '98**, Austin TX, Oct 1998 (re-given at IETF 51, TERENA) — the talk that made "narrow waist" and *IP over everything / everything over IP* standard vocabulary.
- **The theory.** Micah Beck, **"On the Hourglass Model," *CACM* 62(7):48–57, July 2019.** Thesis = the **Deployment Scalability Trade-off**: the *weaker* and more minimal the spanning layer, the more implementations support it and the more applications build on it. Generality at the waist is bought by *removing* capability, not adding it. Beck credits **David D. Clark** for "spanning layer."
- **Evolvability, empirically.** Akhshabi & Dovrolis, *The Evolution of Layered Protocol Stacks Leads to an Hourglass-Shaped Architecture*, **SIGCOMM '11**: their EvoArch model shows the hourglass emerges from competition and the waist **ossifies** — layers above and below innovate freely *precisely because* the waist cannot move.
- **The cost.** Philip Wadler, **"The Expression Problem,"** java-genericity mailing list, **12 Nov 1998**, verbatim: *"The goal is to define a datatype by cases, where one can add new cases to the datatype and new functions over the datatype, without recompiling existing code, and while retaining static type safety (e.g., no casts)."* Cases = rows, functions = columns; sum types make columns cheap and rows expensive, class hierarchies the reverse.

### 2. Frame *is* the waist — `src/pipecat/frames/frames.py` (2,415 lines)
```python
@dataclass
class Frame:                                       # L65 — verbatim
    id: int = field(init=False)
    name: str = field(init=False)
    pts: int | None = field(init=False)
    broadcast_sibling_id: int | None = field(init=False)
    metadata: dict[str, Any] = field(init=False)
    transport_source: str | None = field(init=False)
    transport_destination: str | None = field(init=False)
```
The base carries **no payload** — identity, timestamp, routing hints, an open `metadata` dict. That is
Beck's minimality: the waist asserts almost nothing, so almost anything can implement it. The
three-way split is **not a taxonomy of content — it is a scheduling contract**:

| Branch | Verbatim docstring | Category |
|---|---|---|
| `SystemFrame` L105 | *"higher priority... handled in order and are **not affected by user interruptions**"* | **out-of-band control plane** — jumps the queue, survives barge-in |
| `DataFrame` L116 | *"processed in order... **cancelled by user interruptions**"* | **in-band payload** — the conversation's content |
| `ControlFrame` L128 | *"similar to data frames, is processed in order... update settings or to end the pipeline after everything is flushed"* | **in-band configuration** — ordered *relative to* data |

The discriminator is time, not content, and the code proves it twice. `FrameProcessorQueue(
asyncio.PriorityQueue)` (`processors/frame_processor.py` L132) declares `START_PRIORITY = 1`,
`SYSTEM_PRIORITY = 10`, `DEFAULT_PRIORITY = 20`, and its `put()` (L163–168) assigns them by
`isinstance`; separately, L1304 routes `isinstance(frame, SystemFrame)` to immediate execution and
everything else to `__process_queue`. **Honest finding:** it is really **two-way at runtime** —
`isinstance(frame, DataFrame)` appears **zero times** in the whole `src/pipecat` tree, and the core
pipeline never dispatches on `ControlFrame` either, so Data-vs-Control is a documentation convention.
The taxonomy also leaks: `LLMContextFrame` (L551) subclasses `Frame` **directly**, in no branch,
landing in the default-priority bucket by fallthrough rather than by declaration.

### 3. The cost, measured
- `frames.py` = **2,415 lines**, **123** `Frame` descendants = **120 concrete frame types**: **48**
  under `SystemFrame`, **33** under `DataFrame`, **39** under `ControlFrame` (`InputTextRawFrame` is
  multiply-inherited and double-counted; `LLMContextFrame` uncounted). Repo-wide: **151 concrete frame
  classes**, **31 outside `frames.py`**. Dispatch tax: **577 `isinstance(frame, ...)` sites, 136 files.**

Wadler's table with the rows exploding. Adding a **column** (a processor) is nearly free — implement
`async def process_frame(self, frame: Frame, direction: FrameDirection)` and you compose with
everything. Adding a **row** (a frame) is free *for the pipeline*, a liability *for every processor*:
`FrameProcessor.process_frame` (L820–847) handles only `StartFrame`, `InterruptionFrame`,
`CancelFrame`, and the pause/resume pair — **there is no automatic pass-through**. Every subclass must
call `await self.push_frame(frame, direction)` itself (L1004), so an unknown frame is silently dropped
by any processor whose author didn't anticipate it.

**Pipecat chose columns, and that is right.** A streaming framework's growth vector is *new services* —
another STT vendor, TTS, transport ([[processor-vocabulary]], [[stt-service-interface]]). Vendors
arrive weekly; the media vocabulary (audio/text/image/context) is stable, so optimize the axis that
actually changes. Beck's ossification warning is the flip side: `Frame`'s field set is now effectively
unchangeable, which is why the escape hatch is the untyped `metadata: dict`, not a new base field.

### 4. Pipecat's own precedent — what `flows/` actually did
I checked, because `flows/` is exactly boson's case: a conversation **state machine** with nodes,
transitions, and actions ([[flows-state-machine]], [[flows-actions]]). **It defined 2 frames, not
20** — `FunctionActionFrame(ControlFrame)` (`flows/actions.py` L50) and
`ActionFinishedFrame(ControlFrame)` (L63) — **in `flows/actions.py`, not in `frames.py`**. Both exist
only because *action execution* needs in-band ordering against speech. Everything else rides existing
frames: `FlowManager` signals transitions with `LLMUpdateSettingsFrame(delta=LLMSettings(...))`
(`flows/manager.py` L768), `LLMSetToolsFrame(tools=functions)` (L839),
`LLMMessagesUpdateFrame`/`LLMMessagesAppendFrame` (L830–838), `LLMRunFrame()` (L709). Node state
itself — `NodeConfig`, `FlowResult`, `ContextStrategyConfig` (`flows/types.py`) — is plain `TypedDict`
held in `FlowManager._current_node`, and **never becomes a frame at all**. The pattern holds across all
31 out-of-tree frames: `rtvi/frames.py` (9), `transports/daily/transport.py` (8),
`services/openclaw/frames.py` (6) — `frames.py` is reserved for the shared media vocabulary.

Third data point: the bus is a **second, parallel hourglass**. `bus/messages.py` defines `BusMessage`
(L32) splitting into `BusDataMessage` (L60) / `BusSystemMessage` (L70) — same priority split, different
waist — stating the boundary verbatim: *"Bus messages are independent of pipeline `Frame`s — if a
worker needs to ship a frame between pipelines it wraps it in a `BusFrameMessage`."* And
`BusFrameMessage(BusDataMessage)` (L86) is literally *Frame-over-Bus*: the IP-over-everything move
inside one codebase ([[bus-and-extensions]]).

- **Migration angle:** boson's stage transitions ([[boson-stage-machine]]), rule verdicts
  ([[boson-layers-rules]]), and script steps ([[boson-script-engine]]) must **not** become `Frame`
  subclasses in `frames.py`. Apply the `flows/` precedent as a three-way test.
  **(a) State → not a frame.** Current stage, rule results, script cursor live in a `BosonStageManager`
  as plain dataclasses, mirroring `FlowManager._current_node`. State only one component reads has no
  business in a universal waist.
  **(b) Effects → existing frames.** Stage transition changing the system prompt =
  `LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=...))`; changing tools =
  `LLMSetToolsFrame(tools=...)`; scripted utterance = `TTSSpeakFrame`; injecting a rule verdict =
  `LLMMessagesAppendFrame`. Zero new frame types, full composability, every existing processor already
  handles them correctly.
  **(c) Genuinely new in-band signals → `boson/frames.py`.** Only when a signal must be *ordered
  against speech* by a processor that is not the stage manager — e.g. a `RuleViolationFrame` a
  downstream guard must see before TTS emits. Budget **2–4 frames total**, matching flows' 2 and
  openclaw's 6; a tenth means you re-implemented boson's internal protocol inside Pipecat's waist and
  lost the interchangeability you came for. Use `frame.metadata` for correlation IDs and stage tags
  (the ossification escape hatch, free to every processor); cross-worker signalling goes on the bus as
  a `BusMessage` subclass, not a frame ([[frame-taxonomy]], [[pipecat-design-philosophy]]).

## Citation
pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (read 2026-08-25), all under
`src/pipecat/`: `frames/frames.py` (2,415 lines; L65, L105, L116, L128, L551);
`processors/frame_processor.py` (L132, L163–168, L820–847, L1004, L1304); `flows/actions.py` (L50, L63);
`flows/manager.py` (L709, L768, L830–839); `flows/types.py`; `bus/messages.py` (L32, L60, L70, L86).
- CSTB/NRC, *Realizing the Information Future: The Internet and Beyond*, National Academy Press, 1994.
- Deering, "Watching the Waist of the Protocol Hourglass," ICNP '98, Oct 1998 — https://ant.isi.edu/csci551/images/3/32/Deering98a.pdf
- Beck, "On the Hourglass Model," *CACM* 62(7):48–57, 2019 — https://cacm.acm.org/research/on-the-hourglass-model/
- Akhshabi & Dovrolis, SIGCOMM '11 — https://doi.org/10.1145/2018436.2018460 · Wadler, "The Expression
  Problem," 12 Nov 1998 — https://homepages.inf.ed.ac.uk/wadler/papers/expression/expression.txt

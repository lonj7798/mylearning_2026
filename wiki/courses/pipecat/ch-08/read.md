---
title: "Barge-In and the Interruption Cascade"
chapter: ch-08
phase: read
course: pipecat
sources:
  - interruption-cascade
  - frame-taxonomy
  - frame-processor
  - theory-out-of-band-priority
  - boson-interrupt-subsystem
  - rtv-vad-chunking
  - rtv-vs-pipecat-gap
figure: figures/bargein-timeline.html
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
---

# Barge-In and the Interruption Cascade

## 왜 이 챕터인가

[[ch-06/read]] ended on one line of code:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1269-1270`**
```python
        if params.enable_interruptions:
            await self.broadcast_interruption()
```

Everything ch-06 taught — the VAD hysteresis machine, the `UserTurnStartStrategy` chain, the three layers at which a turn boundary can be proposed and ratified — exists to reach that `if`. This chapter starts on the next line. The decision that an interruption has occurred is *behind* us. Provider selection is behind us. The question here is narrower and, once you have a customer on a real phone line, more expensive:

> Between the instant `broadcast_interruption()` is called and the instant the last already-spoken byte stops coming out of the customer's handset — what happens, in what order, on which task, and what is left in the conversation history when it is over?

This chapter is placed at position eight rather than position five because the cascade is genuinely unteachable earlier. To follow it you need two things you now own:

- **[[ch-04/read]] §4** — the per-processor two-queue / two-task runtime. Priority tiers 1/10/20, `SystemFrame`s executed inline on the input task, everything else relayed to a second, *cancellable* process task, `FrameQueue.reset()` keeping the `UninterruptibleFrame`s. Without that, "the interruption cancels the processor" is a slogan.
- **[[ch-07/read]]** — the TTS output queue, the word-timestamp path, and the output transport's clock task. Without that, the sentence "truncation is emergent from pipeline position" is not checkable.

There is exactly **one idea** in this chapter, and it is section 7. Sections 1–6 are the machinery you need to read section 7 without hand-waving; sections 8–10 are what you do with it.

### Two corrections before we start

Both of these were in the material this course was built from, and both are wrong against the tree at commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`. Where an excerpt disagrees with the source, **the source wins**, and I say so in the text rather than quietly picking one.

**Correction 1 — the frames you were told about do not exist.** There is no `StartInterruptionFrame` and no `StopInterruptionFrame`. There is one field-less `InterruptionFrame`. §2 runs the grep.

**Correction 2, the larger one — there is no truncation routine.** Not a slow one, not a hidden one, not one with a different name. `LLMContext` has no interruption handling of any kind. What looks from the outside like "Pipecat truncates the assistant message to what was spoken" is a *side effect of where the assistant aggregator sits in the list* plus the fact that its input is paced by an audio clock. §7 traces it in real code, and then shows you the two places that emergent behaviour degrades — one of which will hit you specifically, because it depends on whether your Korean TTS vendor emits word timestamps.

### How to read the evidence

Every Pipecat line number below was opened at commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (2026-08-25; `CHANGELOG` head `[1.7.0] - 2026-08-01`). Every claim about `boson-agent`'s `gateway/interrupt/` and about `realtime_voice` comes from the excerpt library — [[boson-interrupt-subsystem]], [[rtv-vad-chunking]], [[rtv-vs-pipecat-gap]] — which was read from your private repos on 2026-08-25. Those repos are not opened here.

And the standing invariant for this course, restated because this chapter is where it is easiest to break: **§9 is a mechanism differential, not a scoring.** It states what each of the three systems *does*. It contains no "better", no "wins", no "should adopt". [[ch-13/read]] is the only place anything is scored.

---

## 1. Where the signal comes from — six ignition points, one call

`broadcast_interruption()` is a `FrameProcessor` method, which means *any* processor can fire it. The tree has ten call sites; one is the deprecated shim inside `frame_processor.py` itself, leaving nine real originators:

```
$ grep -rn "await self.broadcast_interruption()" src/
src/pipecat/extensions/voicemail/voicemail_detector.py:370
src/pipecat/processors/frame_processor.py:1036        ← deprecated shim
src/pipecat/processors/aggregators/llm_response_universal.py:1270
src/pipecat/processors/aggregators/dtmf_aggregator.py:106
src/pipecat/processors/frameworks/rtvi/processor.py:146
src/pipecat/turns/user_turn_processor.py:210
src/pipecat/services/ultravox/llm.py:650
src/pipecat/services/google/gemini_live/llm.py:1333
src/pipecat/services/aws/nova_sonic/llm.py:1528
src/pipecat/services/aws/nova_sonic/llm.py:1554
```

| Originator | Line | What decided |
|---|---|---|
| `LLMUserAggregator._on_user_turn_started` | `llm_response_universal.py:1270` | ch-06's turn-start strategy chain ratified a user turn |
| `UserTurnProcessor` | `turns/user_turn_processor.py:210` | the same three lines, in the standalone processor form |
| `DTMFAggregator` | `aggregators/dtmf_aggregator.py:106` | the customer pressed a key |
| `RTVIProcessor` | `frameworks/rtvi/processor.py:146` | a *client* sent an explicit interrupt over the RTVI protocol |
| `VoicemailDetector` | `extensions/voicemail/voicemail_detector.py:370` | we are talking to an answering machine |
| Gemini Live / Nova Sonic / Ultravox | `:1333` / `:1528`,`:1554` / `:650` | the **provider** detected barge-in server-side and told us |

The one you will use is the first. Here it is in full, because the surrounding four lines are the seam between ch-06 and this chapter:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1253-1272`**
```python
    async def _on_user_turn_started(
        self,
        controller: UserTurnController,
        strategy: BaseUserTurnStartStrategy,
        params: UserTurnStartedParams,
    ):
        logger.debug(f"{self}: User started speaking (strategy: {strategy})")

        self._user_turn_start_timestamp = time_now_iso8601()
        self._full_user_turn_aggregation = None

        if params.enable_user_speaking_frames:
            await self.broadcast_frame(UserStartedSpeakingFrame)

        await self._user_idle_controller.process_frame(UserStartedSpeakingFrame())

        if params.enable_interruptions:
            await self.broadcast_interruption()

        await self._call_event_handler("on_user_turn_started", strategy)
```

Three facts to carry forward.

**(a) `enable_interruptions` defaults to `True`.** `base_user_turn_start_strategy.py:56` — `enable_interruptions: bool = True`. Barge-in is on unless you turn it off, per strategy, and `BaseUserTurnStartStrategy` lets an individual strategy override it at trigger time (`:200-220`). That is the knob to reach for if you ever want "the customer may not interrupt during the compliance disclosure" — a real requirement in Korean insurance tele-sales, and one you would otherwise be tempted to hack in downstream.

**(b) `UserStartedSpeakingFrame` and `InterruptionFrame` are different frames doing different jobs.** Both are field-less `SystemFrame`s (`frames.py:1154` and `:1142`). `UserStartedSpeakingFrame` marks the *turn* — its own docstring says it "usually means that some transcriptions are already available". `InterruptionFrame` means *stop what you are doing*. They are broadcast one line apart here, but nothing structurally binds them: `enable_user_speaking_frames` and `enable_interruptions` are independent booleans, and five of the six originators above push an interruption without any turn frame at all. Do not build a processor that infers one from the other.

**(c) There is a seventh path that skips the aggregator entirely.** When the interruption arrives from outside the pipeline — over the worker bus, e.g. from a supervisor process or a REST endpoint that says "stop talking to this customer now" — the worker injects the frame directly:

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

Read the comment. It bypasses `_push_queue` — the engine loop you traced in [[ch-04/read]] §6.2 — precisely because that loop can be sitting inside `_wait_for_pipeline_end(frame)` and would not get around to the interruption. This is the out-of-band principle applied one level up: even the *worker's* single ordered queue is a queue, and a control signal cannot be allowed to sit in it. For Lina this is the mechanism behind "kill the bot's current utterance from the CRM console" — [[ch-11/read]] and [[ch-12/read]] will both want it.

From here on, the signal exists. Everything else in this chapter is consequence.

---

## 2. The frame that does not exist, and the frame that does

### 2.1 The grep

The material this course was assembled from — and most Pipecat blog content, and every LLM that has memorised a 2024 version of this library — refers to `StartInterruptionFrame` / `StopInterruptionFrame`. They are gone:

```
$ grep -rn "StartInterruptionFrame\|StopInterruptionFrame\|BotInterruptionFrame" src/
$ echo $?
1
```

Zero hits across the whole package. There is no start/stop pair, no bot-side variant. There is one frame:

**`src/pipecat/frames/frames.py:1141-1151`**
```python
@dataclass
class InterruptionFrame(SystemFrame):
    """Frame pushed to interrupt the pipeline.

    This frame is used to interrupt the pipeline. For example, when a user
    starts speaking to cancel any in-progress bot output. It can also be pushed
    by any processor.
    """

    pass
```

`pass`. **No fields.** Not a generation id, not a timestamp, not a reason, not a "how much was played" hint. Every field on the instance comes from the `Frame` root — `id`, `name`, `pts`, `broadcast_sibling_id`, `metadata`, `transport_source`, `transport_destination` (`frames.py:83-98`), all `field(init=False)`.

That emptiness is the whole design. A frame with a generation id would require every processor to *compare* it against something, which means every processor must hold that something, which means adding a processor means remembering to add the comparison. A field-less frame requires the receiver to know only one thing: *stop*. The cost is that no processor can ask "stop **which** turn?" — and §7.7 is where that bill comes due.

[[ch-03/read]] §8 already put this beside `realtime_voice`'s choice, where the signal is an integer (`self._active_generation`) compared at six named sites. Both are true statements about mechanism; neither is being scored here.

### 2.2 Two instances, not one

`broadcast_interruption` is four lines:

**`src/pipecat/processors/frame_processor.py:1017-1022`**
```python
    async def broadcast_interruption(self):
        """Broadcast an `InterruptionFrame` both upstream and downstream."""
        logger.debug(f"{self}: broadcasting interruption")
        self.__reset_process_task()
        await self.stop_all_metrics()
        await self.broadcast_frame(InterruptionFrame)
```

Note line 1020 before anything is pushed: `self.__reset_process_task()`. The originating processor tears down its *own* process task first, synchronously, before the frame goes anywhere. It does not wait to receive its own broadcast — it cannot, because `broadcast_frame` pushes to neighbours, not to self.

Then:

**`src/pipecat/processors/frame_processor.py:1038-1054`**
```python
    async def broadcast_frame(self, frame_cls: type[Frame], **kwargs):
        """Broadcasts a frame of the specified class upstream and downstream.

        This method creates two instances of the given frame class using the
        provided keyword arguments (without deep-copying them) and pushes them
        upstream and downstream.

        Args:
            frame_cls: The class of the frame to be broadcasted.
            **kwargs: Keyword arguments to be passed to the frame's constructor.
        """
        downstream_frame = frame_cls(**kwargs)
        upstream_frame = frame_cls(**kwargs)
        downstream_frame.broadcast_sibling_id = upstream_frame.id
        upstream_frame.broadcast_sibling_id = downstream_frame.id
        await self.push_frame(downstream_frame)
        await self.push_frame(upstream_frame, FrameDirection.UPSTREAM)
```

**Two distinct objects.** Different `id`s, different `name`s, cross-linked by `broadcast_sibling_id`, racing outward in opposite directions from wherever the interruption fired.

Map that onto the canonical seven-processor chain from [[ch-04/read]] §11, with the interruption originating at position 3:

```
  transport.input()    stt    user_agg    llm    tts    transport.output()    assistant_agg
        1               2        3         4      5            6                   7
        ◄───────────────◄────  [FIRE]  ────►─────►──────────────►───────────────────►
             upstream instance            downstream instance
```

- **Upstream** reaches `stt` and `transport.input()`. Nothing there is producing bot speech, so this half is mostly about resetting the input side's own metrics and process task. `STTService.process_frame` has an explicit branch (`services/stt_service.py:510-512`) that resets its TTFB state and forwards; `BaseInputTransport` has no `InterruptionFrame` branch at all — `grep -n InterruptionFrame src/pipecat/transports/base_input.py` returns nothing — so the input transport gets only the base-class behaviour of §4.
- **Downstream** reaches everything that is producing, buffering, or recording bot speech: the LLM, the TTS service, the output transport, and the assistant aggregator. That is where the whole rest of this chapter lives.

### 2.3 What reads `broadcast_sibling_id` — exactly one thing

The cross-link looks like coordination. It is not. Grep the whole tree:

```
$ grep -rn "broadcast_sibling_id" src/
src/pipecat/transports/base_output.py:716,717,768,769     ← writes (BotStarted/StoppedSpeaking)
src/pipecat/frames/frames.py:75,86,95                      ← declaration
src/pipecat/processors/frame_processor.py:1051,1052,1086,1087  ← writes
src/pipecat/processors/frameworks/rtvi/observer.py:429      ← the ONLY read
```

One reader, and it is an observer:

**`src/pipecat/processors/frameworks/rtvi/observer.py:427-430`**
```python
        # For broadcast frames (pushed in both directions), only process
        # the downstream copy to avoid sending duplicate RTVI messages.
        if frame.broadcast_sibling_id is not None and direction != FrameDirection.DOWNSTREAM:
            return
```

So the sibling id is a **de-duplication handle for the observability plane** ([[ch-11/read]]'s subject), not a coordination handle for the cascade. No processor in the interruption path reads it. If you were hoping to write a processor that says "I already saw this interruption from the other side, skip" — nothing in the framework does that, and you would be the first.

---

## 3. `EndFrame` is not a `SystemFrame`, and that is the proof

Here is the single cleanest piece of evidence that the data/signal split of [[ch-04/read]] §4 is deliberate design rather than an accident of implementation. Compare the two ways a Pipecat pipeline can stop.

**`src/pipecat/frames/frames.py:998-1009`** — the violent one:
```python
@dataclass
class CancelFrame(SystemFrame):
    """Frame indicating pipeline should stop immediately.

    Indicates that a pipeline needs to stop right away without
    processing remaining queued frames.

    Parameters:
        reason: Optional reason for pushing a cancel frame.
    """

    reason: Any | None = None
```

**`src/pipecat/frames/frames.py:1899-1912`** — the graceful one:
```python
@dataclass
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

`EndFrame` is deliberately **in-band**. It is a `ControlFrame`, so `FrameProcessorQueue.put` gives it `DEFAULT_PRIORITY = 20` and the input task relays it to the slow process queue like any data frame. It arrives *after* the audio it follows — which is exactly what you want, because "end the call" must not cut off the bot's closing sentence. If `EndFrame` rode the priority channel, `worker.stop_when_done()` would truncate every goodbye.

And because it is in-band it would be destroyed by the interruption flush — so it carries the `UninterruptibleFrame` mixin, and the docstring says why in as many words.

**`src/pipecat/frames/frames.py:146-157`**
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

Note it is **not a `Frame` subclass** — it is a bare `@dataclass` mixin, exactly as [[frame-taxonomy]] records, so it composes with any of the three real base classes without disturbing the priority tiering. Ten classes declare it directly:

```
$ grep -n "UninterruptibleFrame)" src/pipecat/frames/frames.py
770:  class FunctionCallResultFrame(DataFrame, UninterruptibleFrame)
1735: class EndWorkerFrame(WorkerFrame, UninterruptibleFrame)
1754: class StopWorkerFrame(WorkerFrame, UninterruptibleFrame)
1899: class EndFrame(ControlFrame, UninterruptibleFrame)
1923: class StopFrame(ControlFrame, UninterruptibleFrame)
1939: class PipelineFlushFrame(ControlFrame, UninterruptibleFrame)
2142: class LLMContextSummaryResultFrame(ControlFrame, UninterruptibleFrame)
2164: class FunctionCallInProgressFrame(ControlFrame, UninterruptibleFrame)
2363: class AudioBufferStartRecordingFrame(ControlFrame, UninterruptibleFrame)
2368: class AudioBufferStopRecordingFrame(ControlFrame, UninterruptibleFrame)
```

([[ch-03/read]]'s AST walk counted 13 *transitively*, i.e. including subclasses of these ten.)

Read that list as a policy statement. What must survive a barge-in is: pipeline termination (`EndFrame`, `StopFrame`, the two worker variants), the *settlement* of an in-flight tool call (`FunctionCallInProgressFrame`, `FunctionCallResultFrame`, `LLMContextSummaryResultFrame`), and recording boundaries. Everything else — every byte of bot audio, every word of bot text, every context frame — is expendable by default.

The selective flush that honours the mixin is nine lines:

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

Drain everything, re-enqueue only the survivors, in order. `_uninterruptible_count` is kept O(1) by overriding `_put`/`_get` (`frame_queue.py:73-82`) so the `has_uninterruptible` property that §5.5 depends on is a comparison, not a scan.

### 3.1 GStreamer drew this line in 2001, and Pipecat vendors it

Per [[theory-out-of-band-priority]]: GStreamer's own plugin-writer documentation splits downstream events into two kinds — *in-band*, "serialised with the buffer flow", and *out-of-band*, "travelling through the pipeline instantly, possibly not in the same thread as the streaming thread that is processing the buffers, **skipping ahead of buffers being processed or queued in the pipeline**". `SEGMENT`, `CAPS`, `TAG` and `EOS` are serialised. `FLUSH_START` is out-of-band, and it "unblocks the streaming thread by making all pads reject data."

The correspondence is exact:

| GStreamer | Pipecat |
|---|---|
| buffers | `DataFrame` |
| out-of-band events | `SystemFrame` |
| `FLUSH_START` (out-of-band) | `InterruptionFrame`, `CancelFrame` |
| `EOS` (serialised) | `EndFrame` is a `ControlFrame` |
| pads reject data after flush | `_cancelling` → `queue_frame` returns early (`frame_processor.py:714-715`) |

And this is not a resemblance someone noticed after the fact — Pipecat ships real `Gst` graphs:

**`src/pipecat/processors/gstreamer/pipeline_source.py:39`**
```python
class GStreamerPipelineSource(FrameProcessor):
```

Pipecat re-derived a 2001 media-framework design. If you find yourself arguing that `EndFrame` "should" be a `SystemFrame` for symmetry, you are arguing that `EOS` should be out-of-band, and twenty-five years of GStreamer says no.

---

## 4. The cascade has no coordinator

This is the structural claim, and it is worth stating as bluntly as possible: **nothing in Pipecat orchestrates an interruption.** There is no `InterruptionManager`, no ordered teardown sequence, no barrier, no acknowledgement. There is one frame that reaches N processors, and each of the N independently does the same small thing to itself.

The small thing is in the base class, which every processor inherits and — per [[ch-01/read]] §7.2 — every processor is obliged to call:

**`src/pipecat/processors/frame_processor.py:837-847`**
```python
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

and:

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

Two branches. Either the frame currently in flight is uninterruptible — in which case only the *queue* is flushed and the running coroutine is left alone — or the process task is killed outright and rebuilt.

Because `InterruptionFrame` is a `SystemFrame`, all of this runs **on the input task**, which is not the task being cancelled. That is the asymmetry [[ch-04/read]] §4.3 set up: the process task is disposable, the input task is not. A processor cancels its own data half while remaining live and listening for the *next* system frame. If both halves were one task, a processor would have to cancel the task that is executing the cancellation.

### 4.1 The price, cashed: adjacency is a lie

`await self.__cancel_process_task()` at `:1149` throws `CancelledError` into your coroutine **wherever it happens to be**, at whatever `await` it was suspended on, inside your `process_frame()`.

Consider the most ordinary processor anyone writes — accumulate on one frame, flush on another:

```python
class SentenceLogger(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)          # mandatory, ch-01 §7.2
        if isinstance(frame, LLMTextFrame):
            self._buffer += frame.text                          # frame A
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._sink.write(self._buffer)                # frame B
            self._buffer = ""
        await self.push_frame(frame, direction)
```

Frames A and B look adjacent. They are not: they are two separate trips through `__process_frame_task_handler`, and an interruption can land between them. When it does, `_buffer` is left half full, the process task is rebuilt, and the *next* turn's first `LLMTextFrame` appends onto last turn's fragment. Your CRM log now reads `"고객님 이 상품은 65세까지 갱네, 안녕하세요"`.

That is not a hypothetical. **Pipecat hits it in its own code and pays for it with an explicit line**:

**`src/pipecat/services/tts_service.py:1041`**
```python
        self._aggregated_frame_sequencer.clear()  # discard all pending slots on interruption
```

The comment exists because someone shipped the bug. Every accumulator in the framework's own TTS service is cleared by hand on `InterruptionFrame` — §5.3 lists them — for exactly this reason.

The rule that falls out, and it is the most important thing to remember when you write the rule-layer processor in [[ch-12/read]]:

> **Write every `process_frame()` as if it can be cancelled between any two frames it thinks are adjacent, because it can.**

§8 turns that into a checklist.

---

## 5. Hop by hop, in the canonical pipeline

Now walk the downstream instance through positions 4, 5, 6 and 7 of the canonical chain. Each hop is independent; each hop does something different; nothing waits for anything else.

### 5.1 The LLM — there is no abort API

Start with the negative result, because it is the one people assume away. `LLMService` does handle the frame:

**`src/pipecat/services/llm_service.py:688-689`**
```python
        if isinstance(frame, InterruptionFrame):
            await self._handle_interruptions(frame)
```

and the handler is three lines:

**`src/pipecat/services/llm_service.py:758-761`**
```python
    async def _handle_interruptions(self, _: InterruptionFrame):
        for function_name, entry in self._functions.items():
            if entry.cancel_on_interruption:
                await self._cancel_function_call(function_name)
```

That is **the entire** LLM-side interruption handling: cancel tool calls. There is no `client.abort()`, no `response.close()`, no cancellation token sent to OpenAI or Anthropic. Note also that the branch does **not** forward the frame — `LLMService.process_frame` falls through without pushing. The forwarding happens in the concrete provider subclass, in its `else`:

**`src/pipecat/services/openai/base_llm.py:590-615`**
```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for LLM completion requests.

        Handles LLMContextFrame to trigger LLM completions.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(frame.context)
            except httpx.TimeoutException as e:
                await self._call_event_handler("on_completion_timeout")
                await self.push_error(error_msg="LLM completion timeout", exception=e)
            except Exception as e:
                await self.push_error(error_msg=f"Error during completion: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)
```

So what actually kills the generation? **§4.** `_process_context` is executing inside `__process_frame_task`, suspended at an `await` inside the provider's streaming `async for`. `__cancel_process_task()` throws `CancelledError` in there; the async generator is closed; the HTTP/websocket connection to the provider is torn down by its own context manager.

> **Source correction.** [[interruption-cascade]] states that "`LLMContextFrame` is a `DataFrame`". It is not. **`src/pipecat/frames/frames.py:551`** is `class LLMContextFrame(Frame)` — it subclasses the root directly, in none of the three branches. ([[ch-03/read]] §7.6 already flagged this class as a taxonomy leak.) The *consequence* the excerpt draws is nonetheless correct, and now you can derive it yourself instead of taking it: `FrameProcessorQueue.put` (`frame_processor.py:162-168`) assigns `SYSTEM_PRIORITY` only on `isinstance(frame, SystemFrame)`, so a bare `Frame` lands in the `else` branch at `DEFAULT_PRIORITY = 20`; `__input_frame_task_handler` (`:1304-1307`) relays anything that is not a `SystemFrame` to `__process_queue`; and `FrameQueue.reset()` drops anything that is not an `UninterruptibleFrame`. Same fate, different reason. This is a live example of ch-02's warning that the taxonomy leaks — a frame outside the three branches gets *default* treatment silently, and here the default happens to be right.

**The consequence you must internalise: the `finally` block never runs to completion on an interrupted turn.** `CancelledError` propagates out of `_process_context`; Python does execute `finally` blocks during cancellation, but `await self.push_frame(LLMFullResponseEndFrame())` is itself an `await` inside a cancelled task, and the task is being torn down. In practice **`LLMFullResponseEndFrame` is not delivered downstream for an interrupted turn**.

Which raises the question the rest of this chapter answers: if the frame that normally closes an assistant turn never arrives, **who closes it?** Look at the assistant aggregator's dispatch table:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1552-1566`** (excerpt)
```python
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruptions(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._handle_end_or_cancel(frame)
            await self.push_frame(frame, direction)
        ...
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_end(frame)
```

Two different doors into the same room. On a clean turn, `_handle_llm_end` → `_trigger_assistant_turn_stopped()`. On an interrupted turn, `_handle_interruptions` → `_trigger_assistant_turn_stopped(interrupted=True)`. §7.4 opens that room.

### 5.2 Tool calls — and a second source correction

[[boson-interrupt-subsystem]] and [[theory-out-of-band-priority]] both assert that "`InterruptionFrame` truncates the bot turn but never synthesizes a `ToolResultBlock`", and [[ch-04/read]] parked that as an open question for [[ch-09/read]]. **That claim is wrong at this commit, and it is worth correcting now because it changes what you have to port.**

Trace it. `_cancel_function_call`:

**`src/pipecat/services/llm_service.py:2016-2020`**
```python
    async def _cancel_function_call(self, function_name: str | None):
        await self._cancel_function_call_tasks(
            lambda item: item.registry_item.function_name == function_name,
            reason="interruption",
        )
```

whose docstring names the three things it does:

**`src/pipecat/services/llm_service.py:1896-1901`**
```
        Cancelling a call delivers ``asyncio.CancelledError`` to its handler so
        cancel-aware handlers run their cleanup, broadcasts a
        ``FunctionCallCancelFrame`` so the rest of the pipeline can settle the
        call, and notifies application code via ``on_function_calls_cancelled``.
```

`FunctionCallCancelFrame` is itself a `SystemFrame` (`frames.py:1363`), carrying `function_name`, `tool_call_id`, `run_llm: bool = False` — and the field docstring is precise about the last one: *"an interruption must not trigger inference"*. The assistant aggregator receives it and settles the message:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1931-1950`**
```python
    async def _handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        logger.debug(
            f"{self} FunctionCallCancelFrame: [{frame.function_name}:{frame.tool_call_id}]"
        )
        function_call = self._function_calls_in_progress.get(frame.tool_call_id)
        if not function_call:
            return

        # Update context with the function call cancellation. Async calls are
        # settled with a developer message, the same channel their results
        # arrive on.
        if function_call.cancel_on_interruption:
            self._update_function_call_result(frame.function_name, frame.tool_call_id, "CANCELLED")
        else:
            self._context.add_message(
                async_tool_messages.build_cancelled_message(frame.tool_call_id)
            )

        group_id = function_call.group_id
        del self._function_calls_in_progress[frame.tool_call_id]
```

and `_update_function_call_result` overwrites the placeholder in place:

**`src/pipecat/processors/aggregators/llm_response_universal.py:2157-2165`**
```python
    def _update_function_call_result(self, function_name: str, tool_call_id: str, result: Any):
        for message in self._context.get_messages():
            if (
                not isinstance(message, LLMSpecificMessage)
                and message["role"] == "tool"
                and message["tool_call_id"]
                and message["tool_call_id"] == tool_call_id
            ):
                message["content"] = result
```

The placeholder it overwrites was written the moment the call started:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1769-1781`**
```python
        is_async = not frame.cancel_on_interruption
        if is_async:
            self._context.add_message(async_tool_messages.build_started_message(frame.tool_call_id))
        else:
            self._context.add_message(
                {
                    "role": "tool",
                    "content": "IN_PROGRESS",
                    "tool_call_id": frame.tool_call_id,
                }
            )

        self._function_calls_in_progress[frame.tool_call_id] = frame
```

So the corrected statement, precisely: **Pipecat maintains strict tool-message alternation through an interruption by writing an `IN_PROGRESS` placeholder up front and overwriting it with `"CANCELLED"`.** It is *placeholder-then-overwrite*, where boson's `handler.py` is *synthesize-on-demand*: per [[boson-interrupt-subsystem]], `InterruptHandler.handle_barge_in` walks `_collect_unanswered_tool_uses` at the next user turn and emits a `ToolResultBlock(tool_use_id, content=f"canceled: {tname}")` for every unanswered `tool_use`, so that Anthropic-style strict role alternation survives.

Two mechanisms, and the gap between them is real but much smaller than the excerpts implied:

| | Pipecat | boson `gateway/interrupt/handler.py` (per [[boson-interrupt-subsystem]]) |
|---|---|---|
| when the tool message exists | at call start, as `"IN_PROGRESS"` | only when a result or a repair is produced |
| what settles it on barge-in | `"CANCELLED"` written in place | `ToolResultBlock(content="canceled: {tname}")` synthesized at the next user turn |
| which calls are settled | only those reached by `_cancel_function_call`, i.e. registered `cancel_on_interruption=True` | every unanswered `tool_use` found by scan |
| async / long-running tools | `cancel_on_interruption=False` → not cancelled at all; settled through `async_tool_messages` | `_TOOL_CANCEL_HANDLERS` per-tool, else a default user+assistant pair |
| user-visible text | none | `"[tool call canceled, user interrupted: {tool_name}]"` |

The residue you would actually have to port is the *narrative* half — boson's tag strings — not the alternation repair. [[ch-09/read]] owns the rest of the tool-loop collision; what this chapter fixes is the premise it inherits.

### 5.3 TTS layer A — the service clears every accumulator it owns

`TTSService.process_frame` dispatches:

**`src/pipecat/services/tts_service.py:773-775`**
```python
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruption(frame, direction)
            await self.push_frame(frame, direction)
```

and the handler is the longest interruption routine in the framework. Read it as a catalogue of *everything a streaming TTS service accumulates*, because that is what it is:

**`src/pipecat/services/tts_service.py:1030-1056`**
```python
    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        self._processing_text = False
        self._sent_non_whitespace_in_context = False
        self._bot_speaking = False
        await self._text_aggregator.handle_interruption()
        for filter in self._text_filters:
            await filter.handle_interruption()

        self._llm_response_started = False
        self._streamed_text = ""
        self._text_aggregation_metrics_started = False
        self._aggregated_frame_sequencer.clear()  # discard all pending slots on interruption
        self._pending_llm_response_end_frames.clear()
        await self.reset_word_timestamps()

        await self._stop_audio_context_task()
        # Drops non-UninterruptibleFrame items while keeping uninterruptible ones
        # (e.g. FunctionCallResultFrame) that must not be lost mid-flight.
        self._serialization_queue.reset()
        audio_contexts = self.get_audio_contexts()
        if audio_contexts:
            for ctx_id in audio_contexts:
                await self.on_audio_context_interrupted(context_id=ctx_id)
        self.reset_active_audio_context()
        self._turn_context_id = None
        self._word_last_pts = 0
        self._create_audio_context_task()
```

Fourteen distinct pieces of state, cleared by hand. Nothing did it automatically. Six of them are worth naming for what they protect against:

| Cleared | If you did not |
|---|---|
| `self._text_aggregator.handle_interruption()` | the sentence splitter's partial sentence prefixes the next turn's first sentence |
| `self._aggregated_frame_sequencer.clear()` | pending word slots from the dead turn hold up the next turn's frames in the sequencer |
| `self._pending_llm_response_end_frames.clear()` | the held `LLMFullResponseEndFrame` of the dead turn is re-pushed later (§7.6's path) |
| `reset_word_timestamps()` | the next turn's word PTS values are computed against the old baseline — words scheduled in the past, or seconds in the future |
| `_serialization_queue.reset()` | the ordering queue still holds the dead turn's frames |
| `_word_last_pts = 0` | the clock-queue trick of §7.3 stamps the next turn's aggregation frame with a stale timestamp |

`_serialization_queue` is the one place the `UninterruptibleFrame` mixin earns its keep inside a service, and the constructor comment says so:

**`src/pipecat/services/tts_service.py:402-407`**
```python
        # Created once here so it survives interruptions: on interruption we call reset()
        # which drops non-UninterruptibleFrame items while keeping uninterruptible ones
        # (e.g. FunctionCallResultFrame) that must not be lost mid-flight.
        self._serialization_queue: FrameQueue = FrameQueue(
            frame_getter=lambda item: item if isinstance(item, Frame) else None
        )
```

`FunctionCallResultFrame` is `DataFrame, UninterruptibleFrame` (`frames.py:770`) — a *data* frame that must not be dropped. That combination is unusual enough to be worth pausing on: it means "keep your place in line, but do not evaporate." A tool result that arrives while the customer is interrupting still has to be written to the context, or §5.2's `IN_PROGRESS` placeholder never gets settled.

The routine ends with a nine-line comment about a deadlock, which is a good measure of how much sharp edge lives here:

**`src/pipecat/services/tts_service.py:1057-1066`**
```python
        # When pause_frame_processing=True, the process task may be blocked at
        # __process_event.wait() because pause_processing_frames() was called
        # after LLMFullResponseEndFrame and an UninterruptibleFrame was dequeued
        # before the interrupt arrived. _start_interruption() in the base class
        # handles the common case (non-uninterruptible frames) by cancelling and
        # recreating the process task. But when _start_interruption() detects an
        # UninterruptibleFrame it only resets the queue, leaving the process task
        # blocked. BotStoppedSpeakingFrame never arrives (no audio played), so we
        # must resume here to prevent a permanent deadlock.
        await self._maybe_resume_frame_processing()
```

That is the uninterruptible branch of `_start_interruption` (§4) interacting with the pause machinery of [[ch-04/read]] §4.2 to produce a hang. It took a fix and a paragraph of comment. Budget accordingly when you write processors that pause.

### 5.4 TTS layer B — the socket bounce, and who actually does it

Some TTS services go further and **throw the provider connection away**. The reason is stated plainly in the class docstring, and it is a fact about the vendor's server, not about Pipecat:

**`src/pipecat/services/tts_service.py:1969-1974`**
```python
class InterruptibleTTSService(WebsocketTTSService):
    """Websocket-based TTS service that handles interruptions without word timestamps.

    Designed for TTS services that don't support word timestamps. Handles interruptions
    by reconnecting the websocket when the bot is speaking and gets interrupted.
    """
```

**`src/pipecat/services/tts_service.py:1992-2002`**
```python
    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        # If the bot is not speaking we don't need to reconnect when the user
        # speaks. If the bot is speaking and the user interrupts we need to
        # reconnect. Captured before calling super(), which clears
        # _bot_speaking as part of its own interruption handling.
        should_reconnect = self._bot_speaking or self._tts_started
        self._tts_started = False
        await super()._handle_interruption(frame, direction)
        if should_reconnect:
            await self._disconnect()
            await self._connect()
```

Disconnect and reconnect, mid-call, because a half-spoken server-side synthesis context cannot be rewound over the wire. You cannot tell the vendor "forget the last 1.4 seconds"; the only verb available is *hang up*.

> **Source correction.** [[interruption-cascade]] says "Websocket TTS subclasses go further (L1992-2003)". The class hierarchy is more specific than that, and the specificity is the interesting part:
> ```
> $ grep -n "^class " src/pipecat/services/tts_service.py
> 109:  class TTSService(AIService)
> 1882: class WordTTSService(TTSService)
> 1899: class WebsocketTTSService(TTSService, WebsocketService)
> 1969: class InterruptibleTTSService(WebsocketTTSService)          ← the bounce lives here
> 2040: class WebsocketWordTTSService(WebsocketTTSService)
> 2062: class InterruptibleWordTTSService(InterruptibleTTSService)
> 2083: class AudioContextTTSService(WebsocketTTSService)
> 2121: class AudioContextWordTTSService(AudioContextTTSService)
> ```
> `WebsocketTTSService` does **not** bounce. Only `InterruptibleTTSService` and its subclass do, and the docstring says the family exists for providers **without word timestamps**. The providers that inherit it in this tree: `lmnt`, `rime`, `deepgram`, `smallest`, `neuphonic`, `nvidia/sagemaker`, `sarvam`, `fish`.

Hold that thought — it is the same axis §7.6 turns on. **The TTS providers that need a socket bounce are the providers whose text does not arrive word-by-word, which are the providers where Pipecat's emergent truncation degrades to all-or-nothing.** Those are two consequences of one vendor capability, and when you choose a Korean TTS vendor in [[ch-07/read]]'s terms, you are choosing both at once.

Practical cost: a websocket disconnect/reconnect to a Korean TTS vendor is a TLS handshake plus an auth round trip, and it lands squarely in the window where the customer has just started speaking and the bot must be ready to answer. [[ch-11/read]]'s latency budget has to carry a line for it.

### 5.5 The output transport — cancel the clock, allocate a new queue

The output transport is where the audio physically lives, and its interruption handling has an ordering detail worth reading carefully:

**`src/pipecat/transports/base_output.py:359-361`**
```python
        elif isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
            await self._handle_frame(frame)
```

**Push first, then handle.** The frame is forwarded to the assistant aggregator *before* this transport tears down its own queues. That is not decorative: it means the aggregator's commit (§7.4) is enqueued ahead of the discard, not behind it. It does not, however, buy you a synchronous ordering guarantee — `push_frame` enqueues on the neighbour's input queue and the neighbour's own input task runs it. §7.7 is about exactly what leaks through that gap.

Routing:

**`src/pipecat/transports/base_output.py:373-384`**
```python
    async def _handle_frame(self, frame: Frame):
        """Handle frames by routing them to appropriate media senders."""
        if frame.transport_destination not in self._media_senders:
            logger.warning(
                f"{self} destination [{frame.transport_destination}] not registered for frame {frame}"
            )
            return

        sender = self._media_senders[frame.transport_destination]

        if isinstance(frame, InterruptionFrame):
            await sender.handle_interruptions(frame)
```

and the sender's teardown:

**`src/pipecat/transports/base_output.py:566-593`**
```python
        async def handle_interruptions(self, _: InterruptionFrame):
            """Handle interruption events by restarting tasks and clearing buffers.

            Args:
                _: The start interruption frame (unused).
            """
            # Cancel tasks.
            await self._cancel_clock_task()
            await self._cancel_video_task()

            if self._audio_queue.has_uninterruptible or self._mixer:
                # Keep the audio task running but drain all interruptible frames
                # so the pending UninterruptibleFrames are still delivered. With
                # a mixer, cancelling the task would also stop mixer-only output
                # during the restart, causing an audible gap in the background
                # audio (made worse by telephony serializers that clear the
                # playout buffer on interruptions).
                self._audio_queue.reset()
            else:
                await self._cancel_audio_task()
                self._create_audio_task()

            # Create tasks.
            self._create_video_task()
            self._create_clock_task()

            # Let's send a bot stopped speaking if we have to.
            await self._bot_stopped_speaking()
```

Note the parameter name — `_` — and the docstring that still says "start interruption frame". A vestige of the frame that no longer exists (§2.1). The frame carries nothing, so the handler needs nothing.

Two teardowns matter for the rest of the chapter.

**(a) The clock queue is not flushed — it is replaced.**

**`src/pipecat/transports/base_output.py:1067-1077`**
```python
        def _create_clock_task(self):
            """Create the clock/timing processing task."""
            if not self._clock_task:
                self._clock_queue = asyncio.PriorityQueue()
                self._clock_task = self._transport.create_task(self._clock_task_handler())

        async def _cancel_clock_task(self):
            """Cancel and cleanup the clock processing task."""
            if self._clock_task:
                await self._transport.cancel_task(self._clock_task)
                self._clock_task = None
```

`_cancel_clock_task` sets `self._clock_task = None`; `_create_clock_task` then sees the falsy value and **binds a brand-new `asyncio.PriorityQueue()`**. The old queue — holding every timestamped frame that had not yet reached its presentation time — is simply dereferenced and collected. There is no selective flush here and no `UninterruptibleFrame` exemption: a timed frame is by definition a frame scheduled to be audible at a moment that is no longer going to happen.

Remember that when you get to §7.3. Every unspoken word of the assistant's sentence is in that queue.

**(b) The sub-chunk remainder is discarded, not flushed.**

**`src/pipecat/transports/base_output.py:746-756`**
```python
        async def _bot_stopped_speaking(self):
            """Handle bot stopped speaking event."""
            if not self._bot_speaking:
                return

            self._bot_speaking = False
            self._tts_audio_received = False

            # Any remaining leftover here (e.g. from an interruption) is
            # discarded rather than flushed, since it's no longer wanted.
            self._audio_buffer = bytearray()
```

`handle_audio_frame` only enqueues *complete* chunks, so up to one chunk's worth of PCM can be sitting in `_audio_buffer`. On a clean stop it is padded with silence and flushed (`handle_tts_stopped`, `:659-672`); on an interruption it is dropped. That is a deliberate, commented, sub-40-millisecond decision — a good indicator of how finely this path has been tuned.

### 5.6 Past your process — the carrier's playout buffer

And now the part that no amount of internal flushing can reach. Bytes you have already written to the wire are in *someone else's* buffer. Twilio has them. The customer's handset may have them. Flushing your own queue does nothing about audio that has already left.

So every telephony serializer emits a carrier-side flush command, and each carrier spells it differently:

| Serializer | Line | Emitted on `InterruptionFrame` |
|---|---|---|
| `serializers/twilio.py` | `:186-188` | `{"event": "clear", "streamSid": self._stream_sid}` |
| `serializers/plivo.py` | `:138-140` | `{"event": "clearAudio", "streamId": self._stream_id}` |
| `serializers/telnyx.py` | `:150-151` | `{"event": "clear"}` |
| `serializers/exotel.py` | `:98-99` | `{"event": "clear", "stream_sid": self._stream_sid}` |
| `serializers/vonage.py` | `:90-92` | `{"action": "clear"}` |
| `serializers/genesys.py` | `:602-603` | `json.dumps(self.create_barge_in_event())` |

> **Source correction, minor.** [[interruption-cascade]] and [[theory-out-of-band-priority]] cite `telnyx.py:150`, `vonage.py:90`, `exotel.py:98` as the *assignment* lines. At this commit those are the `elif isinstance(frame, InterruptionFrame):` lines; the assignments are one line later (`:151`, `:92`, `:99`). Neither excerpt mentions `plivo.py` or `genesys.py` at all, and both handle it.

The Twilio one, verbatim, because it is the shape you will actually ship against if Lina dials out through a US-fronted carrier:

**`src/pipecat/serializers/twilio.py:186-188`**
```python
        elif isinstance(frame, InterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)
```

Genesys is the outlier and worth seeing, because it names the concept in the protocol itself:

**`src/pipecat/serializers/genesys.py:470-486`**
```python
    def create_barge_in_event(self) -> dict[str, Any]:
        """Create a barge-in event message.

        This notifies Genesys Cloud that the user has interrupted the bot's
        audio output. Genesys will stop any queued audio playback.

        Returns:
            Dictionary of the barge-in event message.
        """
        msg = self._create_message(
            AudioHookMessageType.EVENT,
            parameters={"entities": [{"type": "barge_in", "data": {}}]},
        )

        logger.debug("🔇 Barge-in event sent to Genesys")

        return msg
```

Six serializers, six wire formats, one concept. And the concept is the boundary of your authority: **the cascade stops at the edge of your process, and past that edge you can only ask.** Whether the carrier honours the clear, and how fast, is not in this repo. For Lina on a Korean carrier or an SBC that Pipecat has no serializer for, this row of the table is *yours to write* — a `FrameSerializer` subclass whose `serialize()` returns your carrier's flush command on `InterruptionFrame`. [[ch-05/read]] gave you the interface; this is the one method of it that barge-in depends on.

---

## 6. Chunk size is an interrupt-granularity decision

[[ch-04/read]] §5.1 gave you the arithmetic. This is where you spend it, and where I correct one detail of how it was stated.

### 6.1 The constants

**`src/pipecat/transports/base_output.py:132-136`**
```python
        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks
```

**`src/pipecat/transports/base_transport.py:72`**
```python
    audio_out_10ms_chunks: int = 4
```

Four ten-millisecond chunks: **40 ms of PCM per written chunk.** The comment says the quiet part out loud — *"This will help with interruption handling."* This is not a buffering knob dressed up as an interruption knob; the source calls it what it is.

### 6.2 What actually drains the audio queue — a correction

Both [[theory-out-of-band-priority]] and the outline this chapter was specified from say the `_audio_queue` is "drained by `_clock_task_handler` (`base_output.py:1079`) at realtime." **It is not.** There are two independent queues in `MediaSender`, and they carry different things:

| Queue | Created | Drained by | Carries |
|---|---|---|---|
| `_audio_queue` | `base_output.py:690` (`FrameQueue()`) | `_audio_task_handler` (`:896`) via `_next_frame` (`:829`) | `OutputAudioRawFrame` chunks + sync frames |
| `_clock_queue` | `base_output.py:1070` (`asyncio.PriorityQueue()`) | `_clock_task_handler` (`:1079`) | frames with a `pts` — the word-level `TTSTextFrame`s of §7.3 |

The audio task pulls as fast as the queue supplies:

**`src/pipecat/transports/base_output.py:836-843`**
```python
            async def without_mixer(vad_stop_secs: float) -> AsyncGenerator[Frame, None]:
                while True:
                    try:
                        frame = await asyncio.wait_for(
                            self._audio_queue.get(), timeout=vad_stop_secs
                        )
                        yield frame
                        self._audio_queue.task_done()
                    except TimeoutError:
                        # Fallback: notify the bot stopped speaking upstream if necessary based on timeout.
                        await self._bot_stopped_speaking()
```

The realtime pacing is one layer lower, in the transport's `write_audio_frame`. Here is the websocket server transport:

**`src/pipecat/transports/websocket/server.py:474-480`**
```python
        if not await self._write_frame(frame):
            return False

        # Simulate audio playback with a sleep.
        await self._write_audio_sleep()

        return True
```

**`src/pipecat/transports/websocket/server.py:506-515`**
```python
    async def _write_audio_sleep(self):
        """Simulate audio device timing by sleeping between audio chunks."""
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval
```

with the interval set at setup:

**`src/pipecat/transports/websocket/server.py:379`**
```python
        self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2
```

That expression looks wrong on first read, so do the units. `audio_chunk_size` is in **bytes**; at 16-bit mono there are 2 bytes per sample, so `audio_chunk_size / sample_rate` is *twice* the chunk duration in seconds, and the `/ 2` corrects it. At the defaults: `audio_chunk_size = (24000/100) × 1 × 2 × 4 = 1920` bytes; `1920 / 24000 = 0.08`; `/2 = 0.04` s. **40 ms — exactly one chunk's duration.** The sleep paces writes at realtime, and *that* is where `r` lives.

The correction does not change the conclusion; it changes where you look for the residency. **N is spread across three places**, not one: `_audio_queue`, whatever the transport's write path has accepted but not yet sent, and the carrier's playout buffer. Only the first two are yours.

### 6.3 The arithmetic, on a Lina sentence

Lina is halfway through *"고객님, 이 상품은 65세까지 갱신 없이 보장이 되고요, 지금 가입하시면…"*. Call it 3 seconds of speech. Your Korean TTS vendor streamed the whole thing to you over its websocket in roughly 400 ms — vendors do that, because the audio is smaller than the network is wide.

- 3 s ÷ 40 ms = **~75 chunks** resident, plus whatever the carrier holds.
- The drain rate `r` is fixed by physics. It is the rate at which sound leaves a speaker. You do not get to tune it.
- An in-band stop signal appended behind those chunks is delivered at `N/r` ≈ **3 seconds late** — no matter how fast your handler is. A handler that is a single `return` is still three seconds late.

That is the entire argument for the priority channel, and it is arithmetic, not sloppiness. It is also why `audio_out_10ms_chunks` is a real product decision: halving it to `2` (20 ms chunks) doubles the number of write-loop iterations — more syscalls, more `asyncio.sleep` wakeups — in exchange for finer cancellation granularity on the last chunk in flight. [[ch-11/read]] spends that number; here you only need to know it is not a buffering preference.

→ **[bargein-timeline.html](figures/bargein-timeline.html)** — open it now and keep it beside you from here to §9. Start in the CASCADE panel: drag the interruption point along Lina's 3-second Korean sentence and watch the independent hops fire — `broadcast_frame` constructing two sibling-linked instances, each processor cancelling its own task, `FrameQueue.reset()` keeping the `UninterruptibleFrame` `EndFrame`, the TTS accumulator clear at `tts_service.py:1041`, the output transport binding a fresh `PriorityQueue`, and the Twilio `{"event": "clear"}` reaching past your process. Then run the accumulate-then-flush toy of §4.1 with the cancellation point dragged between frame A and frame B, first without a reset handler and then with one. Two notes on scope: the figure **begins at an interruption signal that already exists** — there is no VAD control anywhere in it, because that layer was built and made interactive in [[ch-06/read]]'s `turn-boundary.html` and a caption points there — and its in-band/out-of-band overlay reuses [[ch-04/read]]'s `N/r` calculator rather than re-deriving it.

---

## 7. The heart: there is no truncation routine

Everything so far has been about stopping sound. This section is about the conversation history, and it is the one idea the chapter exists for.

The behaviour people describe is: *"Pipecat truncates the assistant's message to what the user actually heard."* The behaviour is real. **The routine does not exist.** Not under another name, not in a helper module, not buried in the context class. What exists is a *position in a list* and an *audio clock*, and the truncation is what those two produce when you interrupt them.

### 7.1 The negative evidence, run

Start by proving the absence, because a claim of absence has to be checkable.

```
$ grep -n "interrupt" src/pipecat/processors/aggregators/llm_context.py
$ echo $?
1
$ wc -l src/pipecat/processors/aggregators/llm_context.py
510 src/pipecat/processors/aggregators/llm_context.py
```

**Zero hits in 510 lines.** `LLMContext` — the class that holds the messages, the tools, and the tool choice; the object every LLM service reads on every inference — does not know that interruptions exist. It is interruption-unaware **by design**, and that is a deliberate layering decision, not an oversight: the context is a data structure, and turn-boundary policy lives in the aggregator.

Now the near-miss, because `grep truncat` *does* hit this file and you should know why it does not count:

**`src/pipecat/processors/aggregators/llm_context.py:221-260`**
```python
    def get_messages(
        self,
        llm_specific_filter: str | None = None,
        *,
        truncate_large_values: bool = False,
    ) -> list[LLMContextMessage]:
        """Get the current messages list.

        Args:
            llm_specific_filter: Optional filter to return LLM-specific
                messages for the given LLM, in addition to the standard
                messages. If messages end up being filtered, an error will be
                logged; this is intended to catch accidental use of
                incompatible LLM-specific messages.
            truncate_large_values: If True, return deep copies of messages with
                large values shortened. For standard messages, known binary
                data (base64-encoded images, audio) is replaced with short
                placeholders. For LLM-specific messages, long string values
                are truncated.
        ...
        if truncate_large_values:
            messages = LLMContext._truncate_large_values_from_messages(messages)

        return messages
```

`_truncate_long_strings(value, *, max_length: int = 100)` at `:315`. Every caller in the tree passes `truncate_large_values=True` from exactly one kind of place:

**`src/pipecat/adapters/services/open_ai_adapter.py:230-236`**
```python
        Returns:
            List of messages in a format ready for logging about OpenAI.
        """
        return cast(
            list[dict[str, Any]],
            self.get_messages(context, truncate_large_values=True),
        )
```

*"in a format ready for logging."* The only "truncate" in `LLMContext` shortens base64 blobs so your log lines are readable. It has nothing to do with barge-in.

So: no truncation routine, and no interruption awareness in the context. Where does the behaviour come from?

### 7.2 Fact one — the aggregator sits after the output transport

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

[[ch-04/read]] §11 already told you this list is the house pattern, and that position 7 is fixed by *what evidence exists only at that point*. Now cash it: **the only text the assistant aggregator can ever see is text the output transport has already released.** Not text the LLM generated. Not text the TTS was asked to speak. Text that came out the far side of the thing that owns the speaker.

Move `assistant_aggregator` to position 5 — before `transport.output()` — and it will faithfully record every word the LLM produced, including the twelve words the customer cut off. Nothing errors. Nothing warns. The pipeline is still type-correct, because [[ch-01/read]] §8's point stands: *"any processor anywhere" is a type-level truth and a semantic lie.* This is the most expensive instance of that lie in the whole framework, and it is invisible at the call site.

Truncation is a property of **position in a list**.

### 7.3 Fact two — assistant text is paced by an audio clock

Position alone is not enough. If words arrived at position 7 as fast as the TTS produced them, the aggregator would hold the whole sentence long before it was audible, and an interruption would commit all of it. So the second half of the mechanism is that word-level text is **timestamped and released by the same clock that releases audio**.

Follow one Korean word through.

**Step 1 — the TTS service stamps each word with a presentation timestamp.**

**`src/pipecat/services/tts_service.py:1472-1490`**
```python
        for word, timestamp in word_times:
            ts_ns = seconds_to_nanoseconds(timestamp)
            if self._initial_word_timestamp == -1:
                # Cache until we have audio and can compute PTS.
                self._initial_word_times.append(
                    (word, timestamp, context_id, includes_inter_frame_spaces)
                )
            else:
                pts = self._initial_word_timestamp + ts_ns
                # Build TTSTextFrame(s) for this word token, advancing the active
                # slot's tracker and flushing any skipped frames now unblocked.
                for f in self._aggregated_frame_sequencer.process_word(
                    word, pts, context_id, includes_inter_frame_spaces
                ):
                    if isinstance(f, TTSTextFrame):
                        # The sequencer stamps every word frame it builds.
                        assert f.pts is not None
                        self._word_last_pts = f.pts
                    await self.push_frame(f)
```

`_initial_word_timestamp` is the baseline established on the **first audio chunk** — `start_word_timestamps()` is called from `_handle_audio_context` at `:1737-1740`, guarded by `if not timestamps_started`. So the word clock is anchored to the audio, not to wall time.

`TTSTextFrame` is `AggregatedTextFrame(TextFrame(DataFrame))` at `frames.py:417` — a plain data frame, priority 20, interruptible, carrying a `pts`.

**Step 2 — the output transport routes anything with a `pts` to the clock queue.**

**`src/pipecat/transports/base_output.py:395-400`**
```python
        elif isinstance(frame, TTSStoppedFrame):
            await sender.handle_tts_stopped(frame)
        elif frame.pts:
            await sender.handle_timed_frame(frame)
        else:
            await sender.handle_sync_frame(frame)
```

**`src/pipecat/transports/base_output.py:643-649`**
```python
        async def handle_timed_frame(self, frame: Frame):
            """Handle frames with presentation timestamps.

            Args:
                frame: The frame with timing information to handle.
            """
            await self._clock_queue.put((frame.pts, next(self._clock_queue_counter), frame))
```

**Step 3 — the clock task sleeps until the word is audible, then pushes it.**

**`src/pipecat/transports/base_output.py:1079-1100`**
```python
        async def _clock_task_handler(self):
            """Main clock/timing task handler for timed frame delivery."""
            running = True
            while running:
                timestamp, _, frame = await self._clock_queue.get()

                # If we hit an EndFrame, we can finish right away.
                running = not isinstance(frame, EndFrame)

                # If we have a frame we check it's presentation timestamp. If it
                # has already passed we process it, otherwise we wait until it's
                # time to process it.
                if running:
                    current_time = self._transport.get_clock().get_time()
                    if timestamp > current_time:
                        wait_time = nanoseconds_to_seconds(timestamp - current_time)
                        await asyncio.sleep(wait_time)

                    # Push frame downstream.
                    await self._transport.push_frame(frame)

                self._clock_queue.task_done()
```

`await asyncio.sleep(wait_time)` — the word is held until the moment it is heard, then and only then pushed to the next processor, which is the assistant aggregator.

**That is the whole trick.** A word reaches the thing that writes history at the instant it becomes audible. And §5.5(a) told you what an interruption does to that queue: it does not flush it, it *replaces* it. Every word scheduled for a moment that will never arrive is dereferenced along with the queue object.

**Step 4 — and the framework goes out of its way to keep the flush frame behind the words.**

This is the piece that proves the design is deliberate rather than lucky. When a TTS context ends, the service pushes a frame telling the aggregator to commit. That frame has no natural `pts`, so it would take the *audio* queue path and could overtake the final word frames — committing before the last words arrive. So the service stamps it:

**`src/pipecat/services/tts_service.py:920-933`**
```python
        if isinstance(frame, TTSStoppedFrame) and frame.context_id:
            if frame.context_id in self._tts_contexts:
                if self._tts_contexts[frame.context_id].push_assistant_aggregation:
                    aggregation_frame = LLMAssistantPushAggregationFrame()
                    # When word-level TTSTextFrames are routed through the
                    # transport's clock queue (PTS-based), the aggregation frame
                    # would otherwise take the audio (sync) queue path and
                    # could overtake the final word frames. Stamping it with a
                    # PTS just past the last word forces it through the clock
                    # queue too, so the assistant aggregator sees every word
                    # before flushing.
                    if self._word_last_pts:
                        aggregation_frame.pts = self._word_last_pts + 1
                    await self.push_frame(aggregation_frame)
```

`self._word_last_pts + 1`. One nanosecond after the last word. Someone found this bug and fixed it with a `+ 1`, and left a seven-line comment explaining the queue race. The same trick is applied to the segment-announcement frame twelve lines below (`:950-964`).

Nobody writes that comment by accident. The audio-paced aggregation is load-bearing, known, and defended.

### 7.4 What the aggregator actually does on interruption

Now the four methods, in the order they run.

**Dispatch** — and note where in the file this lands: `InterruptionFrame` is a `SystemFrame`, so per [[ch-04/read]] §4.2 this executes **on the input task**, inline, ahead of every priority-20 frame queued at this processor.

**`src/pipecat/processors/aggregators/llm_response_universal.py:1545-1554`**
```python
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self._start(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruptions(frame)
            await self.push_frame(frame, direction)
```

`super().process_frame` at 1545 runs `_start_interruption()` first — cancelling this aggregator's own process task, the one that might have been mid-`_handle_text`. Then line 1553.

**The handler** — two lines:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1723-1725`**
```python
    async def _handle_interruptions(self, frame: InterruptionFrame):
        await self._trigger_assistant_turn_stopped(interrupted=True)
        await self.reset()
```

**The turn-stop** — and read the ordering carefully, it matters in §7.8:

**`src/pipecat/processors/aggregators/llm_response_universal.py:2175-2197`**
```python
    async def _trigger_assistant_turn_stopped(self, *, interrupted: bool = False):
        if not self._assistant_turn_start_timestamp:
            return

        aggregation = await self.push_aggregation()
        if aggregation:
            # Strip turn completion markers from the transcript
            aggregation = self._maybe_strip_turn_completion_markers(aggregation)

        message = AssistantTurnStoppedMessage(
            content=aggregation,
            interrupted=interrupted,
            timestamp=self._assistant_turn_start_timestamp,
        )
        await self._call_event_handler("on_assistant_turn_stopped", message)
        if aggregation:
            await self.broadcast_frame(
                LLMContextAssistantTurnFrame,
                text=aggregation,
                timestamp=self._assistant_turn_start_timestamp,
            )

        self._assistant_turn_start_timestamp = ""
```

**The commit** — the line that writes history:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1677-1694`**
```python
    async def push_aggregation(self) -> str:
        """Push the current assistant aggregation with timestamp."""
        if not self._aggregation:
            return ""

        aggregation = self.aggregation_string()
        await self.reset()

        self._context.add_message({"role": "assistant", "content": aggregation})

        # Push context frame
        await self.push_context_frame()

        # Push timestamp frame with current time
        timestamp_frame = LLMContextAssistantTimestampFrame(timestamp=time_now_iso8601())
        await self.push_frame(timestamp_frame)

        return aggregation
```

And what filled `self._aggregation` in the first place — every released `TextFrame`, appended one at a time:

**`src/pipecat/processors/aggregators/llm_response_universal.py:2051-2072`**
```python
    async def _handle_text(self, frame: TextFrame):
        # Skip TextFrame types not intended to build the assistant context
        if isinstance(frame, (TranscriptionFrame, TranslationFrame, InterimTranscriptionFrame)):
            return

        if not frame.append_to_context:
            return

        # Make sure we really have text (spaces count, too!)
        if len(frame.text) == 0:
            return

        text = (
            frame.raw_text
            if isinstance(frame, AggregatedTextFrame) and frame.raw_text
            else frame.text
        )
        self._aggregation.append(
            TextPartForConcatenation(
                text, includes_inter_part_spaces=frame.includes_inter_frame_spaces
            )
        )
```

Read `push_aggregation` line 1685 one more time:

```python
        self._context.add_message({"role": "assistant", "content": aggregation})
```

**That is it.** An ordinary assistant message. No `[interrupted]` tag. No ellipsis. No `interrupted` field. No metadata. The *interrupted* turn and a *complete* turn produce byte-identical message shapes; the only difference is that one of them is shorter.

And if nothing was aggregated — the customer interrupted before the first word became audible — line 1679 returns `""` at the top and **`add_message` is never called at all**. The assistant turn does not appear in the history in any form. From the model's point of view on the next inference, Lina said nothing.

### 7.5 Run the test — the literal message list

You do not have to take any of that from me. Pipecat asserts it, twice.

**`tests/test_context_aggregators_universal.py:1319-1348`**
```python
        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            SleepFrame(),
            InterruptionFrame(),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            LLMTextFrame("there!"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
            InterruptionFrame,
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertEqual(should_start, 2)
        self.assertEqual(should_stop, 2)
        self.assertTrue(stop_messages[0].interrupted)
        self.assertEqual(stop_messages[0].content, "Hello")
        self.assertFalse(stop_messages[1].interrupted)
        self.assertEqual(stop_messages[1].content, "Hello there!")
```

Two turns, two `on_assistant_turn_stopped` events, and the first carries `interrupted=True` with `content="Hello"` (trailing space eaten by concatenation). Look at `expected_down_frames`: a full `LLMContextFrame` is pushed for the interrupted turn — the interrupted partial goes into the context and straight back to the LLM, same as a complete turn.

The second test is even more direct, because it asserts the **literal message list**:

**`tests/test_context_aggregators_universal.py:2434-2451`**
```python
        frames_to_send = [
            TranscriptionFrame(text="Hi!", user_id="", timestamp="now"),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            SleepFrame(),
            InterruptionFrame(),
        ]
        await run_test(
            Pipeline([user, assistant]),
            frames_to_send=frames_to_send,
        )

        roles_contents = [(m["role"], m["content"]) for m in context.get_messages()]
        # User message written when assistant started; assistant message
        # written immediately on interruption with interrupted=True.
        self.assertEqual(roles_contents, [("user", "Hi!"), ("assistant", "Hello")])
        self.assertEqual(len(assistant_messages), 1)
        self.assertTrue(assistant_messages[0].interrupted)
```

```
[("user", "Hi!"), ("assistant", "Hello")]
```

That is the context the model sees on the next turn. Not `"Hello…"`. Not `"Hello [interrupted]"`. `"Hello"`.

### 7.5.1 Where `interrupted=True` actually lives, and who can see it

The flag is real. It is just not on anything the LLM reads.

**`src/pipecat/processors/aggregators/llm_response_universal.py:327-345`**
```python
@dataclass
class AssistantTurnStoppedMessage:
    """An assistant turn stopped message containing an assistant transcript update.

    A message in a conversation transcript containing the assistant
    content. This is the aggregated transcript that is then used in the context.

    Parameters:
        content: The message content/text. May be empty if the LLM
            returned zero tokens (e.g. turn was interrupted before any tokens
            were received or pushed)
        interrupted: Whether the assistant turn was interrupted.
        timestamp: When the assistant turn started.

    """

    content: str
    interrupted: bool
    timestamp: str
```

A transient event payload, handed to `on_assistant_turn_stopped` handlers. Transcript observers, RTVI clients, your CRM logger — they all get it. The `LLMContext` does not.

Nor does the frame that carries the turn text to other processors:

**`src/pipecat/frames/frames.py:531-546`**
```python
@dataclass
class LLMContextAssistantTurnFrame(DataFrame):
    """The aggregated text of a completed assistant turn.

    Broadcast by the LLM assistant aggregator when a turn ends, carrying the
    same text that is stored in the LLM context. Processors upstream and
    downstream (e.g. STT services) can handle this frame to react to each
    completed bot reply without needing a separate observer.

    Parameters:
        text: The assistant's aggregated spoken text for this turn.
        timestamp: ISO-8601 timestamp of when the assistant turn started.
    """

    text: str
    timestamp: str
```

`text` and `timestamp`. **No `interrupted` field.** Its one consumer in the tree is the STT service:

**`src/pipecat/services/stt_service.py:513-515`**
```python
        elif isinstance(frame, LLMContextAssistantTurnFrame):
            await self._process_assistant_turn(frame.text)
            await self.push_frame(frame, direction)
```

So the STT service, which uses the bot's last utterance to condition itself, cannot tell whether that utterance was completed or cut off either.

Summary of where the fact "this turn was interrupted" is representable:

| Carrier | Has `interrupted`? | Who reads it |
|---|---|---|
| `AssistantTurnStoppedMessage` (`:328`) | **yes** | `on_assistant_turn_stopped` handlers — transcript observers |
| the context message (`:1685`) | no | **the LLM, on every subsequent turn** |
| `LLMContextAssistantTurnFrame` (`frames.py:533`) | no | `STTService._process_assistant_turn` |
| `InterruptionFrame` (`frames.py:1142`) | no fields at all | every processor, transiently |

The one consumer that most needs the fact is the one consumer that cannot get it.

### 7.6 The timestamp-less branch — all-or-nothing per sentence

Everything in §7.3 assumed word timestamps. Half the TTS ecosystem does not emit them, and Pipecat takes a completely different path for those providers. This is the branch that will decide your Korean vendor choice, so read it closely.

**`src/pipecat/services/tts_service.py:1322-1341`**
```python
        if self._push_text_frames and not self._is_streaming_tokens:
            # In TTS services that support word timestamps, the TTSTextFrames
            # are pushed as words are spoken. However, in the case where the TTS service
            # does not support word timestamps (i.e. _push_text_frames is True), we send
            # the original (non-transformed) text after the TTS generation has completed.
            # This way, if we are interrupted, the text is not added to the assistant
            # context and the context that IS added does not include TTS-specific tags
            # or transformations.
            #
            # In streaming (TOKEN) mode this is handled instead by the sequencer's
            # per-sentence promotion (see AggregatedFrameSequencer._promote): a call
            # here represents a single token, not the sentence-level unit this frame
            # should carry.
            frame = TTSTextFrame(text, aggregated_by=type)
            frame.will_be_spoken = True
            frame.includes_inter_frame_spaces = includes_inter_frame_spaces
            frame.context_id = context_id
            frame.append_to_context = append_tts_text_to_context
            # Appending to the context, so it preserves the ordering.
            await self.append_to_audio_context(context_id, frame)
```

Read the comment at lines 1326-1329 twice: *"we send the original (non-transformed) text **after** the TTS generation has completed. This way, **if we are interrupted, the text is not added to the assistant context**."*

The framework is telling you, in its own words, that the mechanism here is different. Without word timestamps there is nothing to pace against, so the whole sentence's `TTSTextFrame` is queued behind the sentence's audio in the audio-context serialization queue and released when synthesis finishes. Interrupt at any point before that and `_handle_interruption`'s `_stop_audio_context_task()` + `_serialization_queue.reset()` (§5.3) throw the frame away.

**The granularity of truncation is therefore the sentence, and the rule is all-or-nothing.**

Put the two paths side by side. Same barge-in, same moment, different vendor:

| | Word-timestamp provider | Timestamp-less provider (`InterruptibleTTSService` family) |
|---|---|---|
| what the aggregator receives | one `TTSTextFrame` per word, released at its `pts` by the clock task | one `TTSTextFrame` per **sentence**, released after synthesis completes |
| interrupted 60 % through sentence 3 | sentences 1–2 in full + the words of 3 that played | sentences 1–2 in full; **sentence 3 contributes nothing** |
| interrupted 95 % through sentence 3 | sentences 1–2 + nearly all of 3 | sentences 1–2; **sentence 3 still contributes nothing** |
| interrupted during sentence 1 | the words of 1 that played | **no assistant message at all** (`push_aggregation` returns `""`) |
| socket handling on barge-in | none required | full `_disconnect()` / `_connect()` bounce (§5.4) |

The two rows in bold are the ones to take to a vendor meeting. With a timestamp-less TTS, Lina reading a three-sentence 65세 갱신 explanation and being cut off in the middle of the third sentence records *"…65세까지 갱신 없이 보장이 됩니다."* as if she never started the third sentence — which, for a compliance transcript, is a materially different record from what the customer heard.

This is also the concrete form of the question [[ch-07/read]] left you with. "Does your Korean TTS vendor emit word timestamps?" is not a nice-to-have; it selects which of two truncation mechanisms you get, and whether every barge-in costs you a websocket reconnect.

### 7.7 The one-hop race the mechanism cannot close

One more property, derived from the machinery rather than asserted, and it is the sharpest limit on the guarantee.

`_start_interruption` (§4) resets `__process_queue`. It does **not** touch `__input_queue`. Trace what that means for a word that had just become audible:

1. `_clock_task_handler` reaches the word's presentation time and calls `self._transport.push_frame(frame)` (`base_output.py:1098`). The customer hears the word.
2. `push_frame` → `__internal_push_frame` → `self._next.queue_frame(frame, DOWNSTREAM)` — the aggregator's **input queue**, at `DEFAULT_PRIORITY = 20`.
3. The aggregator's input task pops it and relays it to `__process_queue`; the process task then runs `_handle_text` and appends it to `self._aggregation`.

Now interrupt between step 2 and step 3. The `InterruptionFrame` enters the same input queue at `SYSTEM_PRIORITY = 10` and is popped **first**. `_start_interruption` rebuilds the process queue; `_handle_interruptions` commits `self._aggregation` as it stands — *without that word*. Then the input task pops the word, relays it to the **new** process queue, and `_handle_text` appends it to a freshly `reset()` aggregation belonging to no committed turn.

Two consequences, both real:

- **The committed prefix is a lower bound on what was heard, not an exact match.** The window is one hop wide — sub-millisecond in the common case, wider whenever the aggregator's input task was suspended — so the typical loss is zero or one word. But it is one-directional: the mechanism can never commit *more* than was audible, only less.
- **The stray word survives into the next turn's aggregation.** `_assistant_turn_start_timestamp` is cleared at `:2197`, so the next `_trigger_assistant_turn_stopped` short-circuits — until `_trigger_assistant_turn_started` fires on the next `LLMFullResponseStartFrame`. At that point the leftover fragment is already sitting in `_aggregation` and becomes the *prefix* of the next assistant message. This is §4.1's half-full accumulator, occurring inside the framework's own aggregator.

I am flagging this as **derived, not measured.** The observation that would settle it: register an `on_assistant_turn_started` handler that logs `len(aggregator._aggregation)`, run a hundred barge-ins on real Korean audio, and count the non-zero readings. If it is never non-zero, the input-task hop is always faster than the interruption's arrival in practice and the concern is theoretical. If it is occasionally non-zero, you have found the mechanism behind an intermittent "the bot's reply starts with a word from the previous sentence" bug that would otherwise be unattributable.

### 7.8 Why the missing marker is a real behavioural gap

State the problem in one sentence: **a bare partial is indistinguishable, to the model, from a complete short reply.**

Concretely, on a Lina call. Lina begins:

> "고객님, 이 상품은 65세까지 갱신 없이 보장이 되고요, 지금 가입하시면 첫 달 보험료가…"

The customer cuts in at "지금". The context now contains:

```json
{"role": "assistant", "content": "고객님, 이 상품은 65세까지 갱신 없이 보장이 되고요, 지금"}
```

On the next inference the model reads a complete assistant turn that trails off on a dangling adverb. It has no way to know it was cut off. Common downstream failures, in rough order of how often you will see them: the model does not resume the interrupted point because it believes it already made that point; the model repeats the whole pitch from the top because the fragment reads as a false start; the model treats the dangling clause as a stylistic choice and imitates it.

For a scripted sales flow this compounds, because [[ch-10/read]]'s stage machine will be asking "has the 갱신 조건 been explained?" and the transcript says yes.

What a fix costs, concretely. `push_aggregation` is a plain `async def` on `LLMAssistantAggregator`, so the smallest change is a subclass:

```python
class TaggedAssistantAggregator(LLMAssistantAggregator):
    """Append an explicit interruption marker to a truncated assistant turn."""

    def __init__(self, *args, tag: str = "[고객 끼어듦]", **kwargs):
        super().__init__(*args, **kwargs)
        self._tag = tag
        self._interrupting = False

    async def _handle_interruptions(self, frame: InterruptionFrame):
        self._interrupting = True
        try:
            await super()._handle_interruptions(frame)
        finally:
            self._interrupting = False

    async def push_aggregation(self) -> str:
        if self._interrupting and self._aggregation:
            self._aggregation.append(
                TextPartForConcatenation(self._tag, includes_inter_part_spaces=True)
            )
        return await super().push_aggregation()
```

Two things to get right, both of which fall out of code quoted above:

- **Do not try to do this from an `on_assistant_turn_stopped` handler by appending a message.** `_trigger_assistant_turn_stopped` calls `push_aggregation()` at `:2179` and fires the event handler at `:2189` — the message is *already in the context* by the time your handler runs. An `on_assistant_turn_stopped` approach must **rewrite the last message**, not append one.
- **The empty-aggregation case still vanishes.** `push_aggregation` returns at `:1679` before your hook can do anything if `self._aggregation` is empty. If you want "Lina started to speak and was cut off before a word landed" to be representable, you need to write the message yourself — the tag alone, or a marker-only message — and the natural place is `_handle_interruptions`, not `push_aggregation`.

Per [[interruption-cascade]], boson already has the tag vocabulary you would use here: `_TAGS = {"interrupted": "[interrupted-by-user]", "tool_canceled": "[tool call canceled, user interrupted: {tool_name}]", "barge_in_prefix": "[barge-in] "}`, overridable via `set_interrupt_tags`, with the docstring's own example in Korean (`"[고객 끼어듦]"`). That vocabulary is business logic and it ports; the machinery around it does not need to.

---

## 8. The author's obligations

Everything above turns into six rules for anyone writing a `FrameProcessor` in this system — which, from [[ch-12/read]] onward, is you.

**1. Reset every accumulator you own on `InterruptionFrame`. Nothing does it for you.**
This is §4.1 and the fourteen hand-written lines of `tts_service.py:1030-1056`. If your processor holds partial state across frames — a buffer, a dict keyed by turn, a counter, a half-built rule-evaluation record — it will be left half-built when the process task is cancelled at an arbitrary `await`, and it will contaminate the next turn.

**2. Keep `SystemFrame` handling fast.**
Per [[frame-processor]] and [[ch-04/read]] §4.2: system frames run **on the input task**. A slow branch there stalls every subsequent system frame *at that processor* — including the next interruption, the `CancelFrame`, and the error path. Do not `await` a network call in your `InterruptionFrame` branch. (`TTSService` does — `_disconnect()`/`_connect()` in §5.4 — and it is a deliberate, expensive, documented exception.)

**3. Always `await super().process_frame(frame, direction)`, first line.**
[[ch-01/read]] §7.2 taught this; here is the interruption-specific consequence. `_start_interruption` is only reached through the base implementation (`frame_processor.py:839-841`). Skip the `super()` call and your processor **never cancels its own process task on barge-in** — it keeps chewing through the dead turn's frames while every other processor has moved on. There is no error and no warning.

**4. Mark must-not-drop frames with the `UninterruptibleFrame` mixin.**
`frames.py:147`. The mixin is not a `Frame` subclass, so you compose it: `class LinaRuleCommitFrame(ControlFrame, UninterruptibleFrame)`. Use §3's ten-class list as the test of whether yours belongs: does it terminate something, or settle something that is already half-done? Then yes. Does it carry bot output? Then no.

**5. Use `has_queued_frame(frame_type)` to ask what is pending.**
`frame_processor.py:1244`. One caveat inherited from [[ch-04/read]] §4.3: the docstring's O(1) claim is true of `has_uninterruptible` (a counter) and false of `has_frame`, whose body is a linear scan of the deque (`frame_queue.py:63-66`). Fine at voice-pipeline depths; do not put it in a hot loop.

**6. Remember `_cancelling` makes your frames vanish silently.**
`frame_processor.py:253` sets it `False`; `__cancel()` sets it `True` at `:1105`; and then:

**`src/pipecat/processors/frame_processor.py:713-715`**
```python
        # If we are cancelling we don't want to process any other frame.
        if self._cancelling:
            return
```

After a `CancelFrame`, `queue_frame` returns without enqueueing and without logging. Anything you push during shutdown — a final CRM write, a "call ended" event, a metrics flush — is dropped on the floor. If it must happen, it happens on the `EndFrame` path (which is in-band and uninterruptible, §3), not the `CancelFrame` path.

### 8.1 The rule in one worked example

Take §4.1's `SentenceLogger` and make it correct. The diff is four lines:

```python
class SentenceLogger(FrameProcessor):
    def __init__(self, sink, **kwargs):
        super().__init__(**kwargs)
        self._sink = sink
        self._buffer = ""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)     # obligation 3 — also runs _start_interruption

        if isinstance(frame, InterruptionFrame):
            self._buffer = ""                             # obligation 1 — fast, no await (obligation 2)
        elif isinstance(frame, LLMTextFrame):
            self._buffer += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._sink.write(self._buffer)
            self._buffer = ""

        await self.push_frame(frame, direction)
```

Three things this version gets right that the first one did not: the reset exists, it does no I/O on the input task, and the `super()` call is unconditional and first. The figure's accumulate-then-flush panel runs both versions against the same drag point — run it once before you write your first processor.

Note what is *not* required: registering anywhere, implementing an interface, or telling anyone that you handle interruptions. The cascade is emergent (§4), which means a correct processor participates by doing four ordinary lines and an incorrect one participates by silently not.

---

## 9. Three mechanisms, side by side

This section states what each of the three systems **does** on a barge-in. It contains no comparative adjective and reaches no conclusion. [[ch-13/read]] is the only place any of them is scored; the point of writing them down now is that you cannot score what you have not first described precisely.

Sources: Pipecat from the code quoted throughout this chapter; `boson-agent`'s `gateway/interrupt/` from [[boson-interrupt-subsystem]]; `realtime_voice` from [[rtv-vad-chunking]] and [[rtv-vs-pipecat-gap]], with [[ch-03/read]] §7.2 and §8 as the fuller treatment.

### 9.1 The differential

| Axis | **Pipecat** | **boson `gateway/interrupt/`** | **`realtime_voice`** |
|---|---|---|---|
| **What can create the signal** | any processor calling `broadcast_interruption()`; the shipped path is VAD → turn-start strategy (`llm_response_universal.py:1270`) | text only. Every decision point takes `text: str` — `PartialDetector.is_partial`, `WordFilterPolicy.evaluate`, `fillers.is_filler`, `InterruptionGate.allows`. No audio path, no energy threshold, no VAD | VAD frame-count hysteresis (`SileroVADConfig(threshold=0.5, min_speech_frames=2, min_silence_frames=6)`), 2-state, 16 kHz mono only |
| **Earliest the signal can exist** | after VAD start-of-speech | after an STT partial exists, plus `DurationPolicy(min_ms=500)`; the production gate is a 2000 ms silence timer in `server/websocket.py` | after `min_speech_frames` audio frames |
| **What the signal carries** | nothing — `InterruptionFrame` has no fields (`frames.py:1142`) | the interrupting `text`, the session id, and `elapsed_ms` | an integer: the new `_active_generation` |
| **How work is stopped** | preemptive: each processor cancels its own process task mid-`await` (`_start_interruption`, `:1130-1150`); no coordinator | cooperative: `CancellationFlag.set()` / `.check()` raising `CancellationError`; docstring is explicit — *"Cooperative — tool runs to completion, then flag is checked"* | generation-ID equality compared at six named sites, each dropping stale work with `GENERATION_DROPPED` |
| **Enumerable enforcement points** | no — wherever processors are | yes — the flag's check sites | yes — six sites in one file, plus two buffers |
| **A new component participates** | automatically, by inheriting `_start_interruption` (and only if it calls `super().process_frame`) | only if its author checks the flag | only if its author adds the generation comparison |
| **Queued output audio** | `MediaSender.handle_interruptions` (`base_output.py:566-593`): cancel clock + video tasks, `_audio_queue.reset()` or cancel-and-recreate, **new** `asyncio.PriorityQueue` for the clock (`:1067-1071`) | not in scope — boson's gateway is text-native by contract | `GenerationAudioQueue.discard_generation()` rebuilds the deque under the condition lock as an atomic filter; `BoundedAudioOutput.activate_generation()` drains everything older and returns the drop count |
| **Reaching past the process** | six serializers emit a carrier flush (§5.6) | n/a | `OutboundAudioTrack` throws away its `av.AudioFifo` on generation change |
| **TTS provider handling** | `InterruptibleTTSService` (`tts_service.py:1969`) bounces the websocket for timestamp-less providers | n/a | n/a |
| **How the spoken prefix is derived** | **pipeline position** — the aggregator sits after `transport.output()` — plus `TTSTextFrame.pts` released by `_clock_task_handler` at presentation time | not derived. The partial is whatever the agent had streamed out, `partial_text` | **sample-ratio reconstruction**: `AudioTextPlayoutLedger.audible_text()` walks phrases to the acknowledged cursor and, for the partial phrase, `ratio = (cursor - sample_start) / (sample_end - sample_start)` then `text[:int(len(text) * ratio)]` |
| **What the TTS must provide** | word timestamps, for the word-level path; without them the unit is the whole sentence (§7.6) | nothing | a sample count. Works with a TTS that emits no timestamps at all |
| **Accuracy characteristic** | exact at word boundaries where timestamps exist; sentence-granular where they do not; one-hop lower bound (§7.7) | not a spoken-prefix measure — it is a generated-prefix measure | linear character-per-sample approximation; drifts within a word and across a phrase whose speaking rate varies |
| **What is written to history** | `{"role": "assistant", "content": "<spoken-so-far>"}`, untagged. Nothing at all if zero words played (`:1679-1680`) | `Message(role="assistant", content=f"{partial_text}[interrupted-by-user]")` — `cancellation.py:128-132` | the ledger's audible prefix, recorded explicitly, with a `semantic_interrupt` flag distinguishing "cut off" from "finished and they replied" (`playout_complete()`, `session.py:502-507`) |
| **Is "this was interrupted" representable to the LLM** | no — `interrupted=True` exists only on the transient `AssistantTurnStoppedMessage` | yes — the literal tag, overridable, Korean by example | yes — the `semantic_interrupt` flag |
| **Tool-call repair** | `IN_PROGRESS` placeholder overwritten with `"CANCELLED"` (§5.2), for tools registered `cancel_on_interruption=True` | `_collect_unanswered_tool_uses` synthesizes a `ToolResultBlock(content=f"canceled: {tname}")` per unanswered `tool_use` at the next user turn, preserving strict role alternation | not in scope — tools live in gateway |

### 9.2 Three failure modes, named

Each mechanism fails differently, and the failures do not overlap.

**Pipecat fails on representation.** The prefix it commits is a good measure of what was heard, and then it discards the one bit that says the turn was cut off. The model reads a truncated sentence as a finished one (§7.8). Secondarily, the accuracy of the prefix is a function of the *vendor*: with a timestamp-less TTS, truncation collapses to sentence granularity (§7.6).

**boson fails on origination.** The subsystem is 581 lines of careful policy sitting downstream of a signal that cannot exist until a transcript does. Per [[boson-interrupt-subsystem]] and [[theory-out-of-band-priority]], that is *signal-origination* latency, not queue-depth latency, and it sits **upstream** of every mechanism this chapter described. Moving the signal onto the priority channel without first moving its origination onto audio buys nothing: `InterruptionFrame` overtakes buffered TTS, but the interrupt cannot be constructed until an ASR partial has arrived. Two other facts belong in the same row, per the same excerpt: `PartialDetector` is constructed once at `bootstrap.py:316` and **its field is never read anywhere** — dead code, with the real path being the `_partial_transcripts` dict plus the 2000 ms silence timer in `server/websocket.py:288-317, 616-735`; and `WordFilterPolicy(ignore_words=["hmm","uh","um","ah"], max_chars=3)` counts **characters**, so "네" and "아니요" are silently ignored on a Korean line.

**realtime_voice fails on approximation.** `audible_text()` is a linear char-per-sample estimate. It is exact at phrase boundaries and wrong inside a word, and the error grows with speaking-rate variation across the phrase. In exchange it needs nothing from the TTS but a sample count, which is the input requirement that survives a vendor change.

### 9.3 What observation would distinguish them on real Korean traffic

Three different mechanisms with three different dependencies is a conclusion you can only cash by measuring. Here is a protocol that would actually separate them, written so it can be run rather than admired.

**Setup.** Fifty recorded Korean barge-ins from real Lina calls, each with: the assistant audio as rendered, the customer's audio, and a human-annotated ground-truth character index — *the last character the customer could have heard before their own speech masked it*. Annotating that index by ear is the expensive part and there is no way around it; it is the ground truth all three mechanisms are estimating.

**Measurements, per barge-in:**

1. **Prefix error, in characters.** `len(committed_prefix) - ground_truth_index`, signed. Pipecat's should be ≤ 0 by construction (§7.7); realtime_voice's should straddle zero with variance concentrated inside words; boson's is not measuring the same quantity at all and should be reported separately as *generated* minus *heard*.
2. **Sentence-boundary collapse rate.** Fraction of barge-ins where the committed prefix ends exactly at a sentence boundary. Near 1.0 confirms the timestamp-less path of §7.6 is active on your vendor; near the base rate confirms the word path.
3. **Stray-fragment rate.** The §7.7 test: log `len(_aggregation)` in an `on_assistant_turn_started` handler. Any non-zero reading is a leaked fragment from the previous turn.
4. **Signal-origination latency.** Milliseconds from the first voiced customer sample to the moment the interrupt signal exists. This is the axis that separates boson from the other two, and it should be reported before any of the truncation numbers, because a 700 ms origination delay makes prefix accuracy academic.
5. **Downstream behavioural consequence.** For each recorded barge-in, replay the resulting context to the model and classify the next turn: *resumes correctly*, *repeats from the top*, or *treats the fragment as complete*. This is the measurement that turns §7.8 from an argument into a number.

Measurements 1 and 2 distinguish Pipecat's word path from its sentence path. Measurement 1 versus 5 distinguishes "the prefix is accurate" from "the model behaves correctly" — which are not the same property, and conflating them is how you end up optimising the wrong one. Measurement 4 is the one that has to be taken first.

Cast nothing yet. Write the numbers down.

---

## 10. Three framework-extension moves for Lina

Not summary — things to build, each one an application of a mechanism above to a problem the chapter did not pose.

**Move 1 — an uninterruptible compliance segment.**
Korean insurance tele-sales has statutory disclosures that must be read in full. §1(a) gave you the switch: `enable_interruptions` is per-strategy and defaults `True` (`base_user_turn_start_strategy.py:56`), and `BaseUserTurnStartStrategy` supports a per-trigger override (`:200-220`). But flipping it off is not sufficient, because the customer's audio is still arriving and the turn boundary still fires — you would suppress the interrupt while leaving the STT to accumulate an utterance that will be attributed to the wrong turn. The complete design has three pieces: (a) a `FrameProcessor` that gates `enable_interruptions` on a flag set by [[ch-10/read]]'s flow node; (b) a custom `ControlFrame, UninterruptibleFrame` marking the disclosure segment so it survives any interruption that does slip through (§3); (c) a decision about what to do with the customer speech captured during the segment — buffer and replay it after, or drop it. Note that (c) is a *product* question the framework will not answer, and that Pipecat's own answer for muting (`UserMuteStartedFrame`, `frames.py:1176`) is "suppress and discard."

**Move 2 — the interruption marker, plus the thing the marker cannot fix.**
§7.8 gave you the subclass. The extension is realising it is not enough on its own: with a timestamp-less TTS the marker is attached to a *sentence-granular* prefix (§7.6), so `"...보장이 됩니다.[고객 끼어듦]"` claims the customer heard a sentence they may have heard 5 % of. The honest version writes the marker **and** records the granularity it was computed at — e.g. a `metadata` entry on the message, or a second field on your `AssistantTurnStoppedMessage` handler's CRM write — so that later analysis can tell a word-accurate truncation from a sentence-accurate one. That distinction is exactly measurement 2 of §9.3, and building it in from the start is what makes the measurement cheap instead of a retrofit.

**Move 3 — port `AudioTextPlayoutLedger` as a `FrameProcessor`, but only under a condition.**
[[rtv-vad-chunking]]'s migration note says the ledger "would become redundant *only if* the chosen TTS emits word timestamps." §7.6 sharpens the condition into something you can check: run measurement 2 from §9.3 for one day on your candidate vendor. If the sentence-boundary collapse rate is near 1.0, the word path is not active, and the ledger — which needs only a sample count — is computing something Pipecat's mechanism is not. The port shape is a `FrameProcessor` placed **beside** `transport.output()` (position 6.5 in the canonical list), counting `TTSAudioRawFrame` samples on the way past and holding the phrase→sample map, with a `reset()` on `InterruptionFrame` per obligation 1. What makes this a genuine extension rather than a copy is that the ledger's `acknowledge(generation_id, played_sample)` cursor has no Pipecat input — Pipecat has no client playout acknowledgement — so you would have to source the cursor from the transport's own write position instead, which changes what the number means: *bytes we handed to the carrier*, not *bytes the customer's device played*. Deciding whether that substitution is acceptable is the real work, and it is a [[ch-13/read]] question.

---

## 다음 챕터로

What this chapter hands forward, named so later chapters cite it instead of re-deriving it:

- **The cascade is emergent and has no coordinator.** One field-less `InterruptionFrame` (`frames.py:1142`), broadcast as two sibling-linked instances by `broadcast_frame` (`frame_processor.py:1038-1054`), reaching N processors that each independently run `_start_interruption()` (`:1130-1150`) on themselves. Nothing registers, nothing acknowledges, nothing waits. [[ch-12/read]] inherits this directly: your rule-layer processor participates correctly by doing four ordinary lines (§8.1) and incorrectly by silently not.
- **The in-band / out-of-band split is design, and `EndFrame` proves it.** `class EndFrame(ControlFrame, UninterruptibleFrame)` at `frames.py:1899` versus `class CancelFrame(SystemFrame)` at `:999` — graceful end is deliberately ordered so the bot finishes its sentence; only the violent path rides the priority channel. Ten classes carry the mixin, and the list reads as a policy: terminate, settle, record.
- **Truncation is a property of pipeline position plus an audio clock, and Pipecat writes no marker.** The assistant aggregator at position 7, fed word-level `TTSTextFrame`s released by `_clock_task_handler` at their `pts`, commits `{"role": "assistant", "content": <spoken-so-far>}` with no tag, no ellipsis, no `interrupted` field — and nothing at all if no word played. `grep -n interrupt src/pipecat/processors/aggregators/llm_context.py` returns zero hits in 510 lines. Asserted verbatim by `tests/test_context_aggregators_universal.py:2449`: `[("user", "Hi!"), ("assistant", "Hello")]`.
- **The truncation unit is a function of your TTS vendor.** Word timestamps → per-word granularity. No word timestamps → the whole sentence, all-or-nothing (`tts_service.py:1322-1341`), *plus* a full websocket bounce on every barge-in (`InterruptibleTTSService`, `:1969`). [[ch-07/read]]'s vendor question has this as its consequence.
- **Chunk size is an interrupt-granularity decision, and the drain path is two queues not one.** `audio_out_10ms_chunks: int = 4` → 40 ms per written chunk, with `_audio_queue` drained by `_audio_task_handler` and paced at realtime by the transport's `_write_audio_sleep` (`websocket/server.py:379,506-515`), and `_clock_queue` drained separately by `_clock_task_handler`. [[ch-11/read]] spends both numbers in the latency budget.
- **The cascade stops at the edge of your process.** Six telephony serializers ask the carrier to flush (§5.6); what the carrier does is not in this repo. A Korean carrier with no shipped serializer means you write that method.
- **Two corrections to carry.** `LLMContextFrame(Frame)` at `frames.py:551` subclasses the root directly, not `DataFrame` — same fate at runtime, different reason, and a live instance of [[ch-02/read]]'s taxonomy leak ([[frame-taxonomy]] catalogues the branches this class sits outside of). And Pipecat **does** repair tool-message alternation on interruption, via `IN_PROGRESS` → `"CANCELLED"` (§5.2) — [[ch-04/read]] parked the opposite as an open question for [[ch-09/read]]; the premise was wrong and the gap is narrower than it looked.

[[ch-09/read]] takes the next collision. This chapter kept saying "the LLM's process task is cancelled and the generation dies" as if the LLM were one processor in a list — which, in Pipecat's design, it is. boson's design says the opposite: `StreamingConversationAgent` yields only `AgentTextDelta`, tools live in `basement` and `gateway`, and `CLAUDE.md` requires that "Basement and Gateway must not import provider-specific audio code." Two agent loops, two owners of the context, two owners of the tool loop, two definitions of when a turn ends. ch-09 is where that has to be resolved rather than described.

Open questions parked here so they are not lost:

- **The §7.7 stray-fragment race.** Derived from the machinery, not measured. The one-line experiment is in §7.7; run it before [[ch-11/read]] builds the observer plane, because the observer plane is where you would watch for it.
- **What the marker should say, and in which language.** [[interruption-cascade]] records boson's overridable tags with a Korean example (`"[고객 끼어듦]"`). Whether a Korean-language marker in an otherwise Korean context helps or confuses a multilingual model is an evaluation question, not an architecture one, and it belongs with [[ch-12/read]]'s rule layers.
- **Whether the ledger port is worth it.** Move 3's condition is a measurement, and the measurement needs a chosen TTS vendor. [[ch-13/read]] closes it.

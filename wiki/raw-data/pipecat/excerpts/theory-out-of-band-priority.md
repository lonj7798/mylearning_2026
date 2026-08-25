# Out-of-Band Priority — why a pure pipe physically cannot barge in

<!-- slug: theory-out-of-band-priority · type: theory · source: src/pipecat/processors/frame_processor.py, src/pipecat/frames/frames.py, src/pipecat/utils/frame_queue.py, src/pipecat/transports/base_output.py, src/pipecat/processors/gstreamer/ -->

**Core Insight.** The "lego block" feeling is a real, named architectural style — Garlan & Shaw's *pipe and filter* (1993), whose reuse property they state as *"any two filters can be hooked together, provided they agree on the data that is being transmitted between them."* But the same paper names its fatal limit: *"pipe and filter systems are typically not good at handling interactive applications."* Pipecat buys sub-second barge-in by **deliberately breaking a pipe-and-filter invariant**: `SystemFrame`s do not queue behind the data already in flight. Latency of a control signal in a pipeline is a function of **queue depth**, not of code speed — so no amount of fast code can rescue an in-band "stop talking" message. It must travel on a different channel. That is the whole trick, and it costs you the ability to reason about the pipeline as an ordered sequence.

**Guideline.** Classify every message you add to a pipeline as *data* (must stay ordered, may be discarded on interrupt) or *signal* (must overtake, must survive interrupt) **before** you write the processor. Then write every `process_frame()` as if it can be cancelled between any two frames it thinks are adjacent, because it can.

## Technical Details

### 1. The argument from first principles: latency = queue depth, not code speed

- A strictly ordered pipe has one channel. A "stop" message enqueued at time *t* is delivered only after every item enqueued before *t* is drained. If the pipe holds *N* items and the sink drains at realtime rate *r*, control latency is `N/r` — **independent of how fast your interrupt handler runs**.
- Real numbers from the repo. `base_transport.py:72` `audio_out_10ms_chunks: int = 4`, and `base_output.py:135-136` computes `audio_bytes_10ms = int(self._sample_rate / 100) * channels * 2; self._audio_chunk_size = audio_bytes_10ms * audio_out_10ms_chunks` → **40 ms of PCM per written chunk**. The comment above it (`base_output.py:132-134`) says the quiet part out loud: *"We will write 10ms\*CHUNKS of audio at a time… If we receive long audio frames we will chunk them. **This will help with interruption handling.**"* Chunk size *is* an interrupt-granularity decision.
- Now the depth. `_audio_queue` (`base_output.py:690` `self._audio_queue = FrameQueue()`) is drained by a **clock-paced** task (`_clock_task_handler`, `base_output.py:1079`; frames enqueued with their `pts` at `:649`) — i.e. at realtime. But it is **filled** by a TTS vendor websocket at whatever rate the vendor streams. A 3-second Korean sentence arriving in ~400 ms leaves ~3 s of audio (≈75 chunks) resident in that queue plus whatever the far end already holds. An in-band stop signal appended behind it arrives **~3 seconds late**. That is the entire failure, and it is arithmetic, not sloppiness.

### 2. The mechanism in real code — a priority tier plus a task split

`FrameProcessor` is **not** one queue. `frame_processor.py:132`:

```python
class FrameProcessorQueue(asyncio.PriorityQueue):
    START_PRIORITY = 1
    SYSTEM_PRIORITY = 10
    DEFAULT_PRIORITY = 20
```

`put()` (`:152-171`) assigns `START_PRIORITY` to `StartFrame`, `SYSTEM_PRIORITY` to any `SystemFrame`, `DEFAULT_PRIORITY` otherwise, and appends a monotonic `self.__counter` so same-tier frames keep arrival order (and so the queue never compares `Frame` objects). Everything enters through `queue_frame()` (`:700-721`).

The branch the thesis rests on is `__input_frame_task_handler` (`:1287-1313`):

```python
if isinstance(frame, SystemFrame):
    await self.__process_frame(frame, direction, callback)
elif self.__process_queue:
    await self.__process_queue.put((frame, direction, callback))
```

So there are **two tasks over two queues**: the input task executes system frames *inline*, and shovels data/control frames into `__process_queue`, drained by the separate, **cancellable** `__process_frame_task_handler` (`:1315-1333`). A system frame therefore overtakes in-flight data twice — by priority on the way in, and by never entering the slow queue at all. (Third path: `_enable_direct_mode` at `:717-719` skips both queues.)

**What is really a `SystemFrame`** (`frames.py:105`, docstring: *"A frame that takes higher priority than other frames. System frames are handled in order and are not affected by user interruptions."*). 33 subclasses; the load-bearing ones are `StartFrame` (`:924`), `CancelFrame` (`:999`), `ErrorFrame` (`:1016`), `InterruptionFrame` (`:1142`), `UserStartedSpeakingFrame`/`UserStoppedSpeakingFrame` (`:1154`/`:1165`), `BotStartedSpeakingFrame`/`BotStoppedSpeakingFrame` (`:1282`/`:1293`), `FunctionCallCancelFrame` (`:1363`), `FrameProcessorPauseUrgentFrame`/`ResumeUrgentFrame` (`:1106`/`:1122`), `MetricsFrame` (`:1317`), `UserImageRequestFrame` (`:1420`).

**Two expectations that do not hold — reporting rather than inventing:**
- **`StartInterruptionFrame` does not exist** in this commit. There is a single `InterruptionFrame`, no start/stop pair.
- **`EndFrame` is NOT a SystemFrame.** `frames.py:1899` `class EndFrame(ControlFrame, UninterruptibleFrame)` and `:1923` `class StopFrame(ControlFrame, UninterruptibleFrame)`. Graceful end is deliberately *in-band* — it must arrive after the audio it follows, so the bot finishes its sentence. Only the violent path (`CancelFrame`) is out-of-band. That contrast is the cleanest evidence the data/signal split is intentional design, not convenience.

### 3. The second out-of-band axis: direction

`frame_processor.py:60-69`:

```python
class FrameDirection(Enum):
    DOWNSTREAM = 1
    UPSTREAM = 2
```

`__internal_push_frame` (`:1160-1194`) routes to `self._next.queue_frame(...)` or `self._prev.queue_frame(...)` — this is where the style breaks a second Garlan–Shaw invariant, *"filters do not know the identity of their upstream and downstream filters."* Pipecat processors hold both pointers (`_prev`, `_next` at `:239-240`).

Real upstream traffic, verified:
- **Errors.** `push_error` ends in `await self.push_frame(error, FrameDirection.UPSTREAM)` (`:1002`) — a failure in TTS must reach the transport/worker that is *earlier* in the chain.
- **Context, and this one is structural.** The assistant aggregator sits **after** `transport.output()`, so to hand the updated conversation back to the LLM it must push *backwards*: `llm_response_universal.py:1620, 1706, 1889, 1985` `await self.push_context_frame(FrameDirection.UPSTREAM)`, plus `:2235` for `LLMContextSummaryRequestFrame` (docstring: *"Push the request frame UPSTREAM to the LLM service for processing"*).
- **Both at once.** `broadcast_interruption()` (`:1017-1022`) calls `broadcast_frame(InterruptionFrame)` (`:1038-1054`), which constructs **two** instances, links them via `broadcast_sibling_id`, and pushes one each way — so every processor on both sides of the interrupt origin is hit.

### 4. Back-pressure: Pipecat opts out, and that is the correct call

Back-pressure is a consumer telling a producer *"slow down"*. The canonical statement is the Reactive Streams spec: *"The purpose of Reactive Streams is to provide a standard for asynchronous stream processing with non-blocking backpressure,"* where *"backpressure is an integral part of this model in order to allow the queues which mediate between threads to be bounded"* (v1.0.4, 2022-05-26).

**Pipecat's queues are unbounded.** `grep -n maxsize` over `frame_processor.py`, `frame_queue.py`, `base_input.py`, `base_output.py` returns **zero hits**; `FrameProcessorQueue.__init__` and `FrameQueue.__init__` both call bare `super().__init__()`, `base_input.py:265` is `self._audio_in_queue = asyncio.Queue()`. There is no demand signal anywhere.

That is deliberate, and it follows from physics: **you cannot back-pressure a live microphone.** Blocking the producer does not pause the speaker; it just relocates the buffer and grows latency monotonically. A realtime media pipeline's only real options are *drop* or *flush*. Pipecat chooses **flush, triggered by the out-of-band signal**:
- `_start_interruption` (`:1130-1150`) either cancels and recreates the process task, or calls `__reset_process_queue()`.
- `FrameQueue.reset()` (`frame_queue.py:84-95`) drains every non-`UninterruptibleFrame` item and re-enqueues the uninterruptible ones — a *selective* flush, with `_uninterruptible_count` kept O(1) via `_put`/`_get` overrides.
- `MediaSender.handle_interruptions` (`base_output.py:566-593`) cancels the clock and video tasks, then either `self._audio_queue.reset()` (mixer/uninterruptible present) or cancel-and-recreate the audio task.
- Flushing your own queue is not sufficient: `serializers/twilio.py:187` emits `{"event": "clear", "streamSid": self._stream_sid}` on `InterruptionFrame`, because bytes already on the wire live in Twilio's playout buffer, beyond your reach.

**GStreamer got here first, and Pipecat vendors it.** `src/pipecat/processors/gstreamer/pipeline_source.py:39` `class GStreamerPipelineSource(FrameProcessor)` builds real `Gst` graphs (`Gst.ElementFactory.make("queue", None)` at `:181`/`:216`, appsinks at `:189`/`:225`). GStreamer's own design docs draw exactly Pipecat's line: downstream events are either *"in-band (serialised with the buffer flow)"* or *"out-of-band (travelling through the pipeline instantly, possibly not in the same thread as the streaming thread that is processing the buffers, **skipping ahead of buffers being processed or queued in the pipeline**)"*. `SEGMENT`, `CAPS`, `TAG`, `EOS` are serialized; `FLUSH_START` is out-of-band and *"unblocks the streaming thread by making all pads reject data."* Buffers:events :: DataFrame:SystemFrame, and `EOS`-is-serialized :: `EndFrame`-is-a-`ControlFrame`. Pipecat re-derived a 2001 design.

### 5. The price: adjacency is a lie

Out-of-band priority destroys sequential reasoning. Concretely, `_start_interruption` calls `await self.__cancel_process_task()` (`:1149`) — the task is cancelled **while it is inside your `process_frame()`**, at an arbitrary `await`. A processor that does `self._buffer += frame.text` on frame *A* and flushes on frame *B* can be killed between them, leaving `_buffer` half-full; the next turn then emits last turn's fragment. Pipecat hits this in its own code and pays for it explicitly: `services/tts_service.py:1041` `self._aggregated_frame_sequencer.clear()  # discard all pending slots on interruption`.

Defensive obligations for a processor author:
1. Handle `InterruptionFrame` and **reset every accumulator you own**. Nothing does it for you.
2. Keep `SystemFrame` handling fast — it runs on the input task, so a slow branch stalls *every* later system frame at that processor.
3. Always `await super().process_frame(frame, direction)`, or `StartFrame` never triggers `__create_process_task()` (`:1091-1097`) and data frames are silently never processed.
4. Mark must-not-be-dropped frames with the `UninterruptibleFrame` mixin (`frames.py:147`); check pending work with `has_queued_frame(frame_type)` (`:1244`).
5. Remember `_cancelling` (`:253`): after `CancelFrame`, `queue_frame` returns early (`:714-715`) and your frames vanish.

- **Migration angle:** boson's barge-in is **in-band by construction**, and no tuning fixes that. Every decision point takes text — `PartialDetector.is_partial(text, previous)`, `WordFilterPolicy.evaluate(text, *, elapsed_ms)`, `fillers.is_filler(content, status)`, `InterruptionGate.allows(session_id, content)` — so the interrupt cannot be *created* until an STT partial exists. The floor is the client's ASR partial-emission interval, and the production path adds `silence_timeout_ms=2000` plus `DurationPolicy(min_ms=500)` before a barge-in is even admitted. That is not queue-depth latency, it is *signal-origination* latency, and it sits **upstream** of everything Pipecat optimizes. Migration is therefore two independent moves, and doing only the second buys nothing: **(a)** move interrupt origination off text and onto VAD (`VADUserTurnStartStrategy`) so the signal exists ~200 ms after speech onset instead of after a transcript; **(b)** let that signal ride `broadcast_interruption()` → `InterruptionFrame` so it overtakes buffered TTS instead of queueing behind it. Second, boson's cooperative `CancellationFlag` (*"tool runs to completion, then flag is checked"*) is the **opposite** discipline from Pipecat's task cancellation — porting it as-is reintroduces exactly the in-band delay you removed, but porting it away loses `handler.py`'s tool_use/tool_result repair, which Pipecat has no equivalent for (`InterruptionFrame` truncates the turn but never synthesizes a `ToolResultBlock`). Budget that repair as hand-written processor state that you must reset on `InterruptionFrame`, per obligation (1) above. See `[[boson-interrupt-subsystem]]`, `[[interruption-cascade]]`, `[[frame-processor]]`, `[[latency-budget-voice]]`.

## Citation

- Pipecat: `pipecat-ai/pipecat`, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (2026-08-25), read 2026-08-25 at `src/pipecat/processors/frame_processor.py`, `src/pipecat/frames/frames.py`, `src/pipecat/utils/frame_queue.py`, `src/pipecat/transports/base_output.py`, `src/pipecat/transports/base_transport.py`, `src/pipecat/serializers/twilio.py`, `src/pipecat/services/tts_service.py`, `src/pipecat/processors/aggregators/llm_response_universal.py`, `src/pipecat/processors/gstreamer/pipeline_source.py`.
- D. Garlan and M. Shaw, "An Introduction to Software Architecture," *Advances in Software Engineering and Knowledge Engineering*, vol. 1, World Scientific, 1993, §3.1 "Pipes and Filters" (pp. 6–8). Quotations verified verbatim from https://www.cse.msu.edu/~cse870/Materials/Design/intro_softarch-Garlan-Shaw.pdf
- Reactive Streams Specification for the JVM, v1.0.4 (released 2022-05-26). https://github.com/reactive-streams/reactive-streams-jvm and https://www.reactive-streams.org/
- GStreamer Plugin Writer's Guide, "Events: Seeking, Navigation and More." https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/events.html

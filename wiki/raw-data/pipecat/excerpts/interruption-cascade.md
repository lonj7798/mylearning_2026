# The Interruption Cascade — one `InterruptionFrame`, four different unwindings
<!-- slug: interruption-cascade · type: source · source: src/pipecat/processors/frame_processor.py, services/tts_service.py, services/llm_service.py, transports/base_output.py, processors/aggregators/llm_response_universal.py -->

**Core Insight.** Pipecat does not "truncate the context" on barge-in. It never had the untruncated text in the context to begin with. The assistant aggregator is placed **after** `transport.output()`, so the only assistant text it ever sees is text the output transport already released at its audio presentation timestamp. Interruption throws away the output transport's pending clock queue, and the aggregator then commits whatever it happened to have accumulated as a normal `{"role": "assistant", "content": ...}` message. Truncation is an emergent property of pipeline *position*, not a truncation routine.

**Guideline.** If you want "the assistant remembers only what was heard", put the assistant aggregator downstream of the output transport and let word-timestamped `TTSTextFrame`s pace it. If you want an explicit `[interrupted]` marker in history, you must add it yourself — Pipecat writes no marker, no ellipsis, no `interrupted` flag into the message.

## Technical Details

### 0. The frames the outline assumed do not exist

`StartInterruptionFrame` and `StopInterruptionFrame` are **gone** — `grep -rn "StartInterruptionFrame\|StopInterruptionFrame" src/` returns **0 hits**. There is one frame, no start/stop pair:

```python
@dataclass
class InterruptionFrame(SystemFrame):   # frames.py L1142
    """Frame pushed to interrupt the pipeline. ..."""
    pass                                 # no fields at all
```
`UserStartedSpeakingFrame` (L1154) and `UserStoppedSpeakingFrame` (L1165) still exist, also field-less `SystemFrame`s, but they mark the **turn** ("usually means that some transcriptions are already available"), not the interruption. They are pushed by the turn-start/stop strategies, independently of the interruption.

### 1. Ignition — VAD → strategy → broadcast

`VADController._handle_vad` fires `on_speech_started` → `LLMUserAggregator._on_vad_speech_started` broadcasts `VADUserStartedSpeakingFrame` (`llm_response_universal.py` L1238) → `VADUserTurnStartStrategy` matches it and calls `trigger_user_turn_started()` → `LLMUserAggregator._on_user_turn_started` (L1252-1272):

```python
if params.enable_user_speaking_frames:
    await self.broadcast_frame(UserStartedSpeakingFrame)
await self._user_idle_controller.process_frame(UserStartedSpeakingFrame())
if params.enable_interruptions:
    await self.broadcast_interruption()
```
`enable_interruptions` defaults `True` (`turns/user_start/base_user_turn_start_strategy.py` L57); same three lines in `turns/user_turn_processor.py` L195-211. `FrameProcessor.broadcast_interruption` (`frame_processor.py` L1017-1022) does `self.__reset_process_task()` → `await self.stop_all_metrics()` → `await self.broadcast_frame(InterruptionFrame)`; `broadcast_frame` builds **two** instances, one pushed upstream and one downstream, so the frame reaches transport input *and* the LLM/TTS/output/aggregator chain. Other ignition points: `DTMFAggregator` (L106), `RTVIProcessor` (L146), `VoicemailDetector` (L370), realtime LLMs detecting server-side barge-in (Gemini Live L1333, Nova Sonic L1528/L1554, Ultravox L650), and `PipelineWorker` converting an `InterruptionWorkerFrame` from the bus (`pipeline/worker.py` L1280-1286).

### 2. Every processor: cancel the in-flight coroutine

`FrameProcessor.process_frame` (L839-841) runs in **every** processor, because `InterruptionFrame` is a `SystemFrame` and rides the priority path: `elif isinstance(frame, InterruptionFrame): await self._start_interruption(); await self.stop_all_metrics()`. `_start_interruption` (L1130-1144): if the frame currently being processed is an `UninterruptibleFrame`, only `__reset_process_queue()`; otherwise `await self.__cancel_process_task()` then `__create_process_task()`. Net effect: the per-processor data-frame queue is emptied (uninterruptible frames survive) and whatever coroutine was mid-`await` is thrown `CancelledError`.

### 3. What aborts the in-flight LLM generation

There is **no explicit LLM abort call**. `LLMService._handle_interruptions` (`services/llm_service.py` L758-761) only cancels tool calls:
```python
async def _handle_interruptions(self, _: InterruptionFrame):
    for function_name, entry in self._functions.items():
        if entry.cancel_on_interruption:
            await self._cancel_function_call(function_name)
```
The generation itself dies from step 2. `LLMContextFrame` is a `DataFrame`, so `OpenAILLMService.process_frame` (`services/openai/base_llm.py` L590-614) — which does `push_frame(LLMFullResponseStartFrame())` → `await self._process_context(frame.context)` → `finally: push_frame(LLMFullResponseEndFrame())` — is executing *inside* `__process_frame_task`. Cancelling that task raises `CancelledError` inside the streaming `async for`, which closes the provider connection. **The `finally` block's `LLMFullResponseEndFrame` is therefore never pushed on an interruption**; the assistant turn is closed by the `InterruptionFrame` path instead (see §5).

### 4. What stops the TTS mid-utterance — two layers

**Layer A, the service** — `TTSService._handle_interruption` (`services/tts_service.py` L1030-1066) clears `_processing_text` / `_bot_speaking` / `_llm_response_started` / `_streamed_text`, awaits `self._text_aggregator.handle_interruption()`, runs `self._aggregated_frame_sequencer.clear()  # discard all pending slots on interruption`, `self._pending_llm_response_end_frames.clear()`, `reset_word_timestamps()`, `_stop_audio_context_task()`, `self._serialization_queue.reset()` ("Drops non-UninterruptibleFrame items while keeping uninterruptible ones (e.g. FunctionCallResultFrame)"), then `on_audio_context_interrupted(context_id=...)` per open context. Websocket TTS subclasses go further (L1992-2003): `should_reconnect = self._bot_speaking or self._tts_started` → `await self._disconnect(); await self._connect()` — a full socket bounce so the provider stops streaming.

**Layer B, the output transport** — `BaseOutputTransport.process_frame` (L359-361) pushes the frame on *then* calls `_handle_frame`, routing to `sender.handle_interruptions(frame)` (L383-384). `MediaSender.handle_interruptions` (L566-591): `_cancel_clock_task()`, `_cancel_video_task()`, then either `self._audio_queue.reset()` (when uninterruptible frames or a mixer are present) or cancel+recreate the audio task; then `_create_video_task()`, `_create_clock_task()`, `await self._bot_stopped_speaking()`. **`_create_clock_task` (L1067-1071) allocates a brand-new `asyncio.PriorityQueue()`** — every queued, not-yet-due timed frame is discarded. Telephony serializers additionally flush the carrier's playout buffer: `serializers/twilio.py` L186-188 returns `{"event": "clear", "streamSid": self._stream_sid}`; same shape in `telnyx.py` L150, `vonage.py` L90, `exotel.py` L98.

### 5. HOW THE CONTEXT IS TRUNCATED — the actual mechanism

**It is not a truncation function.** Three facts compose:

**(a) The aggregator sits after the output transport.** `examples/getting-started/06-voice-agent.py` L80-89:
```python
pipeline = Pipeline([
    transport.input(), stt, user_aggregator, llm, tts,
    transport.output(),      # Transport bot output
    assistant_aggregator,    # Assistant spoken responses
])
```

**(b) Assistant text arrives paced by audio.** `TTSTextFrame` (a `TextFrame` subclass) carries a `pts` derived from provider word timestamps (`tts_service.py` L964, L1689, L932). In the output transport, `elif frame.pts: await sender.handle_timed_frame(frame)` (`base_output.py` L397-398) puts it in `_clock_queue`, and `_clock_task_handler` (L1079-1100) sleeps until `timestamp > current_time` is satisfied before `await self._transport.push_frame(frame)`. So a word only reaches the aggregator at the moment it is audible. `LLMAssistantAggregator._handle_text` (`llm_response_universal.py` L2051-2072) appends each such `TextFrame` to `self._aggregation` (skipping transcription frames and `append_to_context=False` frames).

**(c) On interruption the aggregator commits what it has, verbatim.**
```python
async def _handle_interruptions(self, frame: InterruptionFrame):   # L1723
    await self._trigger_assistant_turn_stopped(interrupted=True)
    await self.reset()
```
```python
async def _trigger_assistant_turn_stopped(self, *, interrupted: bool = False):   # L2175
    if not self._assistant_turn_start_timestamp:
        return
    aggregation = await self.push_aggregation()
    if aggregation:
        aggregation = self._maybe_strip_turn_completion_markers(aggregation)
    message = AssistantTurnStoppedMessage(
        content=aggregation, interrupted=interrupted,
        timestamp=self._assistant_turn_start_timestamp)
    await self._call_event_handler("on_assistant_turn_stopped", message)   # ...

async def push_aggregation(self) -> str:   # L1677
    if not self._aggregation:
        return ""
    aggregation = self.aggregation_string()
    await self.reset()
    self._context.add_message({"role": "assistant", "content": aggregation})
    await self.push_context_frame()
    timestamp_frame = LLMContextAssistantTimestampFrame(timestamp=time_now_iso8601())
    await self.push_frame(timestamp_frame)
    return aggregation
```

**What the assistant message is replaced with: nothing.** The message written is an ordinary `{"role": "assistant", "content": <spoken-so-far>}`. There is no `[interrupted]` tag, no ellipsis, no `interrupted` field on the context message — `interrupted=True` lives only on the transient `AssistantTurnStoppedMessage` event payload (dataclass at L327-345), which is for transcript observers, not for the LLM. If zero text was aggregated (interrupted before the first word played), `push_aggregation` returns `""` at the first line and **no assistant message is added at all** — the turn vanishes from history entirely. `grep -n "interrupt" src/pipecat/processors/aggregators/llm_context.py` returns nothing; `LLMContext` is interruption-unaware by design.

Confirmed by `tests/test_context_aggregators_universal.py::test_interruption` (L1298-1348): send `LLMFullResponseStartFrame, LLMTextFrame("Hello "), SleepFrame(), InterruptionFrame(), ...` and assert `self.assertTrue(stop_messages[0].interrupted)` / `self.assertEqual(stop_messages[0].content, "Hello")` — trailing space trimmed by concatenation, second turn's full `"Hello there!"` a separate message. Also `LLMFullResponseAggregator` (`aggregators/llm_response.py` L62-65) fires `on_completion(self._aggregation, False)` — the `False` is `completed` — then clears; same "partial is what you get" contract.

- **Migration angle:** this is where boson-agent and Pipecat *disagree on philosophy*, not just on API. boson writes the partial with an explicit marker — `packages/gateway/gateway/interrupt/cancellation.py` L128-132:
  ```python
  def cancel_during_streaming(partial_text: str) -> CancelResult:
      tag = _TAGS["interrupted"]
      entry = Message(role="assistant", content=f"{partial_text}{tag}")
      return CancelResult(discard_pending=False, history_entries=[entry])
  ```
  with defaults `_TAGS = {"interrupted": "[interrupted-by-user]", "tool_canceled": "[tool call canceled, user interrupted: {tool_name}]", "barge_in_prefix": "[barge-in] "}` (L88-93), and `InterruptHandler.handle_barge_in` (`handler.py` L78-151) then appends the user turn — including a `ToolResultBlock(content=f"canceled: {tname}")` for every unanswered `tool_use` so Anthropic-style strict role alternation survives. Moving to Pipecat, `basement/loop/interrupt.py` + `gateway/interrupt/cancellation.py` + `gateway/server/interruption.py` are **replaced** by `broadcast_interruption()` + the aggregator's `_handle_interruptions`. Two things do **not** come for free and must be ported as a subclass of `LLMAssistantAggregator` (override `push_aggregation`) or an `on_assistant_turn_stopped` handler that rewrites the last message: (1) the `[interrupted-by-user]` / Korean `[고객 끼어듦]` tag — Pipecat writes a bare partial that reads to the LLM as a *complete* short reply; (2) `boson`'s partial-text barge-in *source*, which becomes irrelevant since Pipecat's truncation point is audio-paced rather than transcript-paced. Conversely, boson's tool-cancel bookkeeping **does** have a Pipecat counterpart — `FunctionCallParams.cancel_on_interruption` drives `LLMService._handle_interruptions`, and the assistant aggregator writes `{"role": "tool", "content": "IN_PROGRESS", ...}` placeholders (L1774-1780) — so `_collect_unanswered_tool_uses` is the one piece of `handler.py` that is genuinely redundant after migration.

## Citation
pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (2026-08-25; CHANGELOG head `[1.7.0] - 2026-08-01`). Files: `src/pipecat/frames/frames.py`, `src/pipecat/processors/frame_processor.py`, `src/pipecat/processors/aggregators/llm_response_universal.py`, `src/pipecat/processors/aggregators/llm_response.py`, `src/pipecat/services/llm_service.py`, `src/pipecat/services/openai/base_llm.py`, `src/pipecat/services/tts_service.py`, `src/pipecat/transports/base_output.py`, `src/pipecat/serializers/twilio.py`, `tests/test_context_aggregators_universal.py`, `examples/getting-started/06-voice-agent.py`. boson-agent read-only at `/Users/jaewon/mywork_2026/Lina_2026/boson-agent-dev/boson-agent/packages/gateway/gateway/interrupt/`. Read 2026-08-25.

# The canonical voice bot — seven processors, and why the order is the design
<!-- slug: canonical-voice-bot · type: source · source: examples/getting-started/06-voice-agent.py -->

**Core Insight.** The reference voice bot is seven processors long, and every position encodes a
causal dependency on *evidence that only exists at that point in the chain*. The context
aggregator appears twice not because there are two contexts, but because there is one shared
`LLMContext` with two write points: the user's turn is only final after STT, and the assistant's
turn is only final after the audio has actually been played out by the transport.

**Guideline.** Read a Pipecat pipeline as a data-dependency chain, not a call sequence. When you
insert a processor, ask what frames it needs and which processor *first produces* them — that
answer is its position. Put the assistant aggregator after `transport.output()`, always.

## Technical Details
- File: `examples/getting-started/06-voice-agent.py` (133 L). The same seven-processor list
  appears verbatim in `examples/getting-started/06a-voice-agent-local.py` (L69–73),
  `07-function-calling.py` (L110–114) and across `examples/voice/voice-*.py`
  (e.g. `voice-cartesia.py` L86–96) — it is the house pattern, not one example's choice.
- The real list, L81–91:
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
  Built from `stt = DeepgramSTTService(...)` (L59), `tts = CartesiaTTSService(...)` (L61),
  `llm = OpenAILLMService(...)` (L68), and the pair (L75–79):
  ```python
  context = LLMContext()
  user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
      context,
      user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
  )
  ```
  Tuple-unpacking works via `LLMContextAggregatorPair.__iter__`
  (`processors/aggregators/llm_response_universal.py` L2331–2341); the same `context` object is
  handed to both halves at L2296 and L2309.
- **Position 1 — `transport.input()`** (`BaseTransport.input()`, `transports/base_transport.py`
  L122). Sole producer of inbound audio frames and of user-speaking frames. Nothing upstream of it
  exists, so everything downstream is derived from it.
- **Position 2 — `stt`.** Needs raw audio; produces transcription frames. Must sit between the
  input transport (audio) and anything that reasons over text.
- **Position 3 — `user_aggregator`** (`LLMUserAggregator`, L563). Docstring: it "collects and
  aggregates speech-to-text transcriptions that occur while a user turn is active and pushes the
  final aggregation when the user turn is finished." Turn boundaries come from the strategies
  configured via `LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())`. On turn end it writes
  the user message into the shared context and calls `push_context_frame()` whose default direction
  is `FrameDirection.DOWNSTREAM` (L489–496) — that downstream push *is* the LLM trigger. So the
  aggregator must be immediately upstream of the LLM: it is the thing that starts inference.
- **Position 4 — `llm`.** Consumes the context frame, emits `LLMFullResponseStartFrame`, streaming
  text frames, `LLMFullResponseEndFrame`, and function-call frames.
- **Position 5 — `tts`.** Consumes LLM text; emits `TTSStartedFrame`, `TTSAudioRawFrame`,
  `TTSTextFrame`, `TTSStoppedFrame`. After the LLM because it needs text; before the output
  transport because the transport plays audio, not text.
- **Position 6 — `transport.output()`** (`BaseTransport.output()`, L131). The only component that
  knows what was *actually played*. In `transports/base_output.py`, `_bot_started_speaking`
  (L699–720) and `_bot_stopped_speaking` (L746–772) each construct **two** sibling frames and push
  both directions — `await self._transport.push_frame(downstream_frame)` then
  `await self._transport.push_frame(upstream_frame, FrameDirection.UPSTREAM)`, linked by
  `broadcast_sibling_id`. The downstream copy exists precisely so something *after* the transport
  can observe playback.
- **Position 7 — `assistant_aggregator`** (`LLMAssistantAggregator`, L1384) — the answer to "why a
  second aggregator". Placed last, it receives the downstream `BotStartedSpeakingFrame` /
  `BotStoppedSpeakingFrame` produced at real playback boundaries and tracks `self._bot_speaking`
  (L305–313). Two behaviours depend on it: (a) a `FunctionCallResultFrame` arriving while the bot
  is speaking sets `_push_context_on_bot_stopped_speaking = True` (L1886) and the context frame is
  pushed once, on stop, "preventing the LLM from running multiple times and producing duplicated
  responses"; (b) `_handle_interruptions` on an `InterruptionFrame` calls
  `_trigger_assistant_turn_stopped(interrupted=True)` then `reset()`, so a barge-in truncates the
  assistant turn at what was reached rather than recording text the user never heard. Put it before
  `transport.output()` and both mechanisms lose their evidence — the aggregator would commit text
  that TTS produced but the caller never heard.
- **The chain is longer at runtime than in the source.** `PipelineWorker` defaults
  `enable_rtvi=True` and prepends an `RTVIProcessor` (`pipeline/worker.py` L548), then re-wraps in
  its own source/sink `Pipeline` (L549). The seven-item list is the *inner* chain.
- **Entry point.** Nothing runs until a frame is queued: `worker.queue_frames([LLMRunFrame()])`
  from `on_client_connected` (L107–114), after `context.add_message({"role": "developer", "content":
  "Please introduce yourself to the user."})`. Teardown is `await runner.cancel()` on
  `on_client_disconnected` (L116–119). Worker config at L93–101 uses
  `PipelineParams(enable_metrics=True, enable_usage_metrics=True)`,
  `idle_timeout_secs=runner_args.pipeline_idle_timeout_secs` (default 300, `runner/types.py` L172)
  and `processor_unusable_policy=ProcessorUnusablePolicy.END`.
- **Migration angle:** positions 1, 2, 5 and 6 are *net-new* for boson-agent, which has no
  server-side audio — there is nothing in `gateway/` to replace. Position 3 partially replaces
  `gateway/interrupt/detector.py`'s partial-transcript reasoning and `gateway/server/websocket.py`'s
  silence timer (`_on_silence` L627): user-turn detection moves from boson's text-partial heuristics
  to Pipecat turn strategies over VAD. Position 7 **collides hard** with boson's context ownership:
  `gateway/session/` + `gateway/compact/` currently own the message list and summarization, and
  `basement/context` owns the agent-side view, whereas `LLMContext` here is a single object mutated
  by two processors — a migration must pick one owner, and `LLMAssistantAggregatorParams` already
  ships `enable_auto_context_summarization` / `auto_context_summarization_config` (L243–244) that
  overlap with `gateway/compact/`. Position 4 leaves `basement/loop` and `basement/tools`
  structurally untouched: they live *inside* what becomes the LLM service processor.

## Citation
pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (v1.7.0 line, changelog head
dated 2026-08-01), read 2026-08-25. Repo paths: `examples/getting-started/06-voice-agent.py`,
`src/pipecat/processors/aggregators/llm_response_universal.py`,
`src/pipecat/transports/base_output.py`.

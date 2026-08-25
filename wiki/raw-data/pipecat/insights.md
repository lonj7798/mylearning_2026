# insights — `pipecat`

One row per excerpt that exists on disk: the single idea it contributes and the
chapters it feeds. 44 excerpts, all carrying a **Migration angle** line.
Pipecat read at commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (2026-08-25).

## Cluster A — Pipecat core (ch-01, ch-02)

| slug | type | core insight | feeds |
|------|------|--------------|-------|
| [[frame-taxonomy]] | source | The concurrency model is encoded in *which base class a frame inherits*, not in a scheduler; `UninterruptibleFrame` is a second, non-`Frame` axis the outline missed. | ch-01, ch-06 |
| [[frame-processor]] | source | Every processor runs **two** tasks over **two** queues; system frames execute inline on the input task, everything else on a separately *cancellable* task — that split *is* barge-in. | ch-01, ch-06, ch-08 |
| [[pipeline-composition]] | source | A `Pipeline` is a doubly-linked list in a source/sink envelope — no scheduler, no routing, no fan-out; composition *is* ordering. | ch-01, ch-02, ch-08 |
| [[pipeline-task-runner]] | source | `PipelineTask`/`PipelineRunner` are deprecated 1.3.0 shims; the real units are `PipelineWorker` (lifecycle) + `WorkerRunner` (process concerns). | ch-02, ch-09, ch-10 |
| [[parallel-pipeline]] | source | Fans the *same frame object* into every branch, merges by first-arrival `frame.id` dedup; only Start/End/Cancel are synchronized — no arbitration, no voting. | ch-02, ch-08 |
| [[canonical-voice-bot]] | source | Seven processors where every position encodes a data dependency; the aggregator appears twice because one `LLMContext` has two write points. | ch-02, ch-05, ch-06 |
| [[pipecat-design-philosophy]] | doc | No architecture doc exists; the enforced philosophy lives in `AGENTS.md` + a machine-checked deprecation registry (391 live, all `removed_in: 2.0.0`). | ch-01, ch-10 |
| [[processor-vocabulary]] | source | Six stock processor shapes (PASS/BLOCK, TRANSFORM, OBSERVE, TAP, GATE, BUFFER); write custom only to coordinate two *non-adjacent* pipeline points. | ch-02, ch-08 |

## Cluster B — Voice I/O (ch-03, ch-04, ch-05)

| slug | type | core insight | feeds |
|------|------|--------------|-------|
| [[transport-daily-webrtc]] | source | A transport is just a pair of `FrameProcessor`s (`input()`/`output()`); `TransportParams` no longer carries `vad_analyzer` at all. | ch-03 |
| [[transport-websocket]] | source | The socket owns the audio clock and delegates every wire byte to a pluggable `FrameSerializer`; that one field is the whole telephony seam. | ch-03 |
| [[transport-telephony]] | source | **There is no telephony transport** — a phone call is `FastAPIWebsocketTransport` + a provider serializer; 5 of 6 providers are 8 kHz μ-law. | ch-03, ch-04, ch-09 |
| [[stt-service-interface]] | source | `STTService` is a *latency contract*: it redefines TTFB as speech-end→final-transcript and broadcasts its P99 so turn-stop timers can size themselves. | ch-04 |
| [[stt-korean-providers]] | source | Korean support is verified / passthrough / excluded — only the first is evidence; **no accuracy or 8 kHz number exists anywhere in the repo**. | ch-04, ch-10 |
| [[endpointing-turn-boundary]] | module | "User stopped talking" is a negotiated verdict from a chain of vetoing strategies in `src/pipecat/turns/`; the *default* is an ML analyzer, not VAD. | ch-04, ch-06, ch-09 |
| [[tts-service-interface]] | source | TTS is a reordering buffer optimising one number — time to first *audible* sample; `TTFA = TTFB + leading_silence`. | ch-05, ch-06 |
| [[tts-korean-providers]] | source | 12 services map `Language.KO`; only **six** also emit word timestamps, and unmapped languages fail *silently* at runtime, not at config time. | ch-05, ch-10 |

## Cluster C — The collision surface (ch-06, ch-07, ch-08)

| slug | type | core insight | feeds |
|------|------|--------------|-------|
| [[vad-silero]] | source | Split three ways — model, 4-state hysteresis machine, edge-triggered controller; `start_secs`/`stop_secs` are the only real barge-in latency knobs. | ch-04, ch-06 |
| [[interruption-cascade]] | source | Pipecat **never truncates the context** — the aggregator sits after `transport.output()` and only ever saw audible text. No `[interrupted]` marker is written. | ch-06 |
| [[llm-service-context]] | source | One plain `LLMContext` owns the messages; the LLM service is stateless and two aggregators mutate the *live* list in place. | ch-07 |
| [[function-calling]] | source | There is **no tool loop and no turn cap** inside the service — think-act-observe is a topology closed by an upstream `LLMContextFrame`. | ch-07 |
| [[custom-processor-guide]] | source | A custom processor is four lines of ceremony around one `isinstance` chain; this is what lets boson's gateway survive intact. | ch-08 |
| [[rtvi-observability]] | source | Instrumentation is a second read-only plane over the frame graph — you instrument by *subscribing*, never by editing processors. | ch-09 |
| [[bus-and-extensions]] | source | The bus is push-only worker-to-worker pub/sub with **no read API**; for non-adjacent coordination Pipecat's own answer is a shared `EventNotifier`. | ch-08, ch-09 |
| [[flows-state-machine]] | module | Pipecat **does** ship a state machine (`FlowManager`, 898 L) — and it is not a `FrameProcessor`; it drives the pipeline from outside via `queue_frames`. | ch-08 |
| [[flows-node-types]] | source | A "node" is a `NodeConfig` TypedDict (`total=False`), an "edge" is a tuple element; `FlowConfig` does not exist and there is no transition validation. | ch-08 |
| [[flows-actions]] | source | Exactly three action verbs (`tts_say`, `end_conversation`, `function`); actions cannot choose a node, veto, or return a value. | ch-08 |
| [[flows-insurance-example]] | source | Nodes are *factory functions* — collected data is baked into the next node's prompt by f-string at transition time; there is no separate memory component. | ch-08, ch-10 |

## Cluster D — boson-agent, read-only

| slug | type | core insight | feeds |
|------|------|--------------|-------|
| [[boson-agent-loop]] | boson | One 561-line async generator owns turn bounding, context mutation, streaming, tool dispatch, hooks and cancel-repair — the exact cluster Pipecat splits apart. | ch-07 |
| [[boson-tool-router]] | boson | Tools are hidden from the model behind `use_tool`/`use_skill`; exposure, permission and stage-allowlist are **three separate gates**. | ch-07 |
| [[boson-gateway-server]] | boson | `gateway/server/` is not a transport but a turn-arbitration engine; ~700 of its 1,404 L have no Pipecat counterpart. | ch-03 |
| [[boson-interrupt-subsystem]] | boson | Every barge-in decision takes `text: str` — there is no audio path at all, so floor-yield latency is bounded by the client's ASR interval. | ch-06 |
| [[boson-stage-machine]] | boson | **Nine** stages (not seven); transitions are performed by deterministic rule code against an edge whitelist — the LLM never chooses. | ch-08 |
| [[boson-layers-rules]] | boson | A two-phase-commit vote over one *already-complete* utterance; a `Filter` from any later layer rolls back everything including the user message. | ch-08 |
| [[boson-script-engine]] | boson | A pure function returning `(state, Action)` that emits legally-fixed Korean text verbatim — the LLM is the *fallback*, not the primary. | ch-08 |
| [[boson-compact-session]] | boson | Compaction ports almost cleanly (deferred vs immediate apply); `SessionState` is the painful part — Pipecat has no equivalent object. | ch-07, ch-10 |

## Cluster E — Production (ch-09, ch-10)

| slug | type | core insight | feeds |
|------|------|--------------|-------|
| [[latency-budget-voice]] | source | The two biggest terms are *not* the LLM — they are the endpointing wait (`stop_secs` + STT P99) and TTS time-to-first-**audible**-sample. | ch-09, ch-10 |
| [[deployment-scaling]] | source | Pipecat ships a *development* runner, not an orchestrator; every session is an asyncio task on one loop and `[scaling] min_agents = 1` is the whole config surface. | ch-09, ch-10 |

## Cluster F — The learner's own `realtime_voice`

| slug | type | core insight | feeds |
|------|------|--------------|-------|
| [[rtv-pipeline-session]] | boson | The learner already built the equivalent — with the opposite bet: **no** Frame class, **no** processor abstraction, one 561-L `VoiceSession` with five swappable `Protocol` slots and a fixed topology. | ch-01, ch-02, ch-10 |
| [[rtv-vad-chunking]] | boson | VAD is frame-counting where Pipecat's is seconds-based (a latency bug); but `KoreanPhraseChunker` and `AudioTextPlayoutLedger` have **no Pipecat equivalent**. | ch-04, ch-05, ch-06 |
| [[rtv-webrtc-transport]] | boson | ~960 L of aiortc landing where `SmallWebRTCTransport` lands — plus HMAC session auth and an audio-forbidden ordered control protocol Pipecat ships neither of. | ch-03, ch-10 |
| [[rtv-vs-pipecat-gap]] | boson | The capstone is a **build-vs-buy decision on an artifact that already passes 60 contract tests**, decided per layer: replace VAD/ASR, keep chunker/ledger/auth. | ch-06, ch-10 |

## Cluster G — Architectural theory (why the design is the way it is)

| slug | type | core insight | feeds |
|------|------|--------------|-------|
| [[theory-pipes-and-filters]] | theory | "Lego block" is the named **Pipes and Filters** style (Garlan & Shaw 1994); the uniform interface is what makes any two processors connectable — and it moves ordering constraints out of the type system into your head. | ch-01, ch-02, ch-08 |
| [[theory-out-of-band-priority]] | theory | Control latency in a pipe is **queue depth, not code speed**, so an in-band "stop" can never barge in; Pipecat buys sub-second interruption by deliberately breaking a pipe-and-filter invariant. | ch-01, ch-06 |
| [[theory-narrow-waist]] | theory | `Frame` is an **hourglass waist** (Beck, *CACM* 2019): cheap to add processors (columns), expensive to add frames (rows) — the Expression Problem tax, and the rule that decides how boson's stage logic may enter the pipeline. | ch-01, ch-08, ch-10 |

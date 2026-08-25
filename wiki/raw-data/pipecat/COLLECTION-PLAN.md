# COLLECTION-PLAN — `pipecat`

Target coverage for the source library. Status: `TODO` until the researcher
agent writes the excerpt; flip to `DONE` when `excerpts/<slug>.md` exists.

## Cluster A — Pipecat core (feeds ch-01, ch-02)

| slug | artifact | status |
|------|----------|--------|
| `frame-taxonomy` | `pipecat/frames/frames.py` — SystemFrame / DataFrame / ControlFrame and the concrete frame types | TODO |
| `frame-processor` | `pipecat/processors/frame_processor.py` — `FrameProcessor`, `push_frame`, direction, per-processor task and ordering guarantees | TODO |
| `pipeline-composition` | `pipecat/pipeline/pipeline.py` — `Pipeline`, source/sink, linking | TODO |
| `pipeline-task-runner` | `PipelineTask` + `PipelineRunner` — lifecycle, params, cancellation, shutdown | TODO |
| `parallel-pipeline` | `ParallelPipeline` and processor filters — branch semantics, frame duplication | TODO |
| `canonical-voice-bot` | The reference voice-bot example pipeline, end to end | TODO |

## Cluster B — Voice I/O (feeds ch-03, ch-04, ch-05)

| slug | artifact | status |
|------|----------|--------|
| `transport-daily-webrtc` | Daily/WebRTC transport — media handling, room lifecycle | TODO |
| `transport-websocket` | WebSocket transport + serializers — the closest analogue to boson's current server | TODO |
| `transport-telephony` | Twilio / Telnyx / SIP transports — telephony codecs, 8 kHz, call lifecycle | TODO |
| `stt-service-interface` | The STT service base class — streaming, interim vs final transcript frames | TODO |
| `stt-korean-providers` | Korean-capable STT providers, measured accuracy and latency, telephony-audio caveats | TODO |
| `endpointing-turn-boundary` | Endpointing / turn detection — smart-turn vs threshold, and what defines a turn | TODO |
| `tts-service-interface` | The TTS service base class — streaming, word/sentence boundary frames | TODO |
| `tts-korean-providers` | Korean-capable TTS providers, time-to-first-byte, prosody | TODO |

## Cluster C — The collision surface (feeds ch-06, ch-07, ch-08)

| slug | artifact | status |
|------|----------|--------|
| `vad-silero` | VAD analyzer, `VADParams`, speech start/stop detection | TODO |
| `interruption-cascade` | `StartInterruptionFrame` handling — TTS stop, LLM abort, and context truncation | TODO |
| `llm-service-context` | LLM service + context aggregator pair — who owns the message list | TODO |
| `function-calling` | Pipecat function calling — registration, invocation, result frames | TODO |
| `custom-processor-guide` | Writing a `FrameProcessor` — gating, injecting, filtering, transforming | TODO |
| `rtvi-observability` | RTVI / observers / metrics — what is instrumentable | TODO |

## Cluster D — boson-agent, read-only (feeds every chapter, especially ch-06..ch-10)

| slug | artifact | status |
|------|----------|--------|
| `boson-agent-loop` | `basement/loop/agent_loop.py` (561 L) + `loop/interrupt.py` — the think-act-observe cycle | TODO |
| `boson-tool-router` | `basement/tools`, `metatool` — `@tool`, ToolRouter, `use_tool` / `use_skill` dispatch | TODO |
| `boson-gateway-server` | `gateway/server/` (1,404 L) — websocket, protocol, access, history, interruption | TODO |
| `boson-interrupt-subsystem` | `gateway/interrupt/` (581 L) — detector, policy, handler, cancellation, fillers | TODO |
| `boson-stage-machine` | `gateway/schemas/stage.py` + stage transitions — the seven Lina stages | TODO |
| `boson-layers-rules` | `gateway/layers/` + `gateway/rules/` (about 900 L) — layer pipeline, rule engine, actions | TODO |
| `boson-script-engine` | `gateway/script/` (517 L) — the scripted purchase flow | TODO |
| `boson-compact-session` | `gateway/compact/` + `gateway/session/` — summarization and session state | TODO |

## Cluster E — Production (feeds ch-09, ch-10)

| slug | artifact | status |
|------|----------|--------|
| `latency-budget-voice` | Published voice-to-voice latency breakdowns and perceptual thresholds | TODO |
| `deployment-scaling` | Running Pipecat in production — process model, concurrency, cold start | TODO |
| `migration-patterns` | Incremental-migration patterns for replacing a transport/orchestration layer | TODO |

## Gap log

- boson-agent has **no audio stack today**; chapters ch-04 / ch-05 / ch-06
  describe capabilities being *added*, not ported. Say so explicitly rather than
  implying a one-to-one mapping exists.
- Korean STT accuracy on **8 kHz telephony audio** is the migration's largest
  unknown. If no measured number can be found, record that fact rather than
  guessing.
- Pipecat moves fast. Every excerpt must record the version or commit it was
  read at, and prefer the source tree over prose docs where they disagree.
- boson's `d_model`-level model details are irrelevant here, but its **LLM
  provider set** (Anthropic / OpenAI / Google) matters for ch-07, since Pipecat's
  function-calling interface differs per provider.

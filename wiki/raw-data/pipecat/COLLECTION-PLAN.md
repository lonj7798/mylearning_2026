# COLLECTION-PLAN — `pipecat`

Target coverage for the source library. Status: `TODO` until the researcher
agent writes the excerpt; flip to `DONE` when `excerpts/<slug>.md` exists.

Status verified against the filesystem on 2026-08-25: **44 excerpts exist**.
Pipecat read at commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`; boson-agent
at `0a5e8ced2fd9e631dcbbe8c5f4adb68b89a4fafb`; `realtime_voice` at
`034ce4ca09a2f109e6c248a43bc989f8d26a6abf` (branch `voice-chat-dev`).

## Cluster A — Pipecat core (feeds ch-01, ch-02)

| slug | artifact | status |
|------|----------|--------|
| `frame-taxonomy` | `pipecat/frames/frames.py` — SystemFrame / DataFrame / ControlFrame **+ the `UninterruptibleFrame` mixin (second axis)** and the concrete frame types | DONE |
| `frame-processor` | `pipecat/processors/frame_processor.py` — `FrameProcessor`, `push_frame`, direction, the **two-queue / two-task** split and ordering guarantees | DONE |
| `pipeline-composition` | `pipecat/pipeline/pipeline.py` — `Pipeline`, source/sink, linking | DONE |
| `pipeline-task-runner` | **`PipelineWorker` + `WorkerRunner`** (`pipeline/worker.py`, `workers/runner.py`) — lifecycle, params, cancellation, shutdown. `PipelineTask`/`PipelineRunner` are deprecated 1.3.0 aliases | DONE |
| `parallel-pipeline` | `ParallelPipeline` and processor filters — branch semantics, fan-out **by reference (no copy)**, first-arrival dedup | DONE |
| `canonical-voice-bot` | The reference voice-bot example pipeline, end to end (`examples/getting-started/06-voice-agent.py`) | DONE |
| `pipecat-design-philosophy` | `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, the deprecation registry — the framework's stated and *enforced* design intent | DONE |
| `processor-vocabulary` | `processors/filters/`, `producer`/`consumer`, `text_transformer`, `logger`, `aggregators/{gated,sentence}` — the stock brick set before writing custom | DONE |

## Cluster B — Voice I/O (feeds ch-03, ch-04, ch-05)

| slug | artifact | status |
|------|----------|--------|
| `transport-daily-webrtc` | Daily/SmallWebRTC transport — media handling, room lifecycle | DONE |
| `transport-websocket` | WebSocket transport + serializer slot — the closest analogue to boson's current server | DONE |
| `transport-telephony` | **Corrected:** telephony is `FastAPIWebsocketTransport` + a provider `FrameSerializer` (`serializers/{twilio,telnyx,plivo,exotel,genesys,vonage}.py`). There is no Twilio/Telnyx/SIP *transport*. 8 kHz μ-law, call lifecycle, DTMF | DONE |
| `stt-service-interface` | The STT service base class — the three base shapes, streaming vs segmented, interim vs final transcript frames, redefined TTFB | DONE |
| `stt-korean-providers` | Korean-capable STT providers, measured **latency only** (no accuracy data exists), telephony-audio caveats | DONE |
| `endpointing-turn-boundary` | **Corrected:** `src/pipecat/turns/` (~4,900 L) — a start/stop *strategy chain*, not a config knob. Default stop is the ML smart-turn analyzer, not VAD | DONE |
| `tts-service-interface` | The TTS service base class — sentence aggregation, audio contexts, word timestamps on `TTSTextFrame.pts`, **TTFA not TTFB** | DONE |
| `tts-korean-providers` | Korean-capable TTS providers, **time-to-first-audible-sample (TTFA)**, word-timestamp capability, prosody | DONE |

## Cluster C — The collision surface (feeds ch-06, ch-07, ch-08)

| slug | artifact | status |
|------|----------|--------|
| `vad-silero` | VAD analyzer, `VADParams`, the 4-state speech machine, `VADController` | DONE |
| `interruption-cascade` | **Corrected:** there is one field-less `InterruptionFrame`; no `StartInterruptionFrame`/`StopInterruptionFrame`. TTS stop, LLM abort by task-cancel, and *positional* context truncation | DONE |
| `llm-service-context` | `LLMContext` + `LLMContextAggregatorPair` — who owns the message list | DONE |
| `function-calling` | Pipecat function calling — registration, invocation, result frames, per-provider adapters | DONE |
| `custom-processor-guide` | Writing a `FrameProcessor` — gating, injecting, filtering, transforming | DONE |
| `rtvi-observability` | RTVI / observers / metrics / OpenTelemetry — what is instrumentable | DONE |
| `bus-and-extensions` | `src/pipecat/bus/`, `registry/`, `extensions/{ivr,voicemail}/` — worker-to-worker pub/sub and the two shipped telephony extensions | DONE |
| `flows-state-machine` | `src/pipecat/flows/manager.py` — `FlowManager`, `_set_node`, the two transition paths | DONE |
| `flows-node-types` | `src/pipecat/flows/types.py` — `NodeConfig`, `ConsolidatedFunctionResult`, `ContextStrategy` | DONE |
| `flows-actions` | `src/pipecat/flows/actions.py` — the 3-verb side-effect vocabulary and the pre/post ordering table | DONE |
| `flows-insurance-example` | `examples/flows/{insurance_quote,warm_transfer,patient_intake,restaurant_reservation}.py` — node factories, data threading, human handoff | DONE |

## Cluster D — boson-agent, read-only (feeds every chapter, especially ch-06..ch-10)

Path correction: this is a **two-package monorepo**. Every path below needs a
`packages/<pkg>/` prefix — `packages/basement/basement/…` and
`packages/gateway/gateway/…`, with per-tenant configs under `agents/`.

| slug | artifact | status |
|------|----------|--------|
| `boson-agent-loop` | `packages/basement/basement/loop/agent_loop.py` (561 L) — the think-act-observe cycle. (`loop/interrupt.py` is a CLI SIGINT handler, **not** barge-in) | DONE |
| `boson-tool-router` | `packages/basement/basement/{tools,metatool,skills,llm}/` — `@tool`, ToolRouter, `use_tool` / `use_skill` dispatch, 6 LLM providers | DONE |
| `boson-gateway-server` | `packages/gateway/gateway/server/` (1,404 L) — websocket, protocol, access, history, interruption | DONE |
| `boson-interrupt-subsystem` | `packages/gateway/gateway/interrupt/` (581 L) — detector, policy, handler, cancellation, fillers | DONE |
| `boson-stage-machine` | `packages/gateway/gateway/stage/` + `schemas/stage.py` + `agents/*/stage_config.py` — the **nine** Lina stages | DONE |
| `boson-layers-rules` | `packages/gateway/gateway/{layers,rules}/` (**1,206 L**, not ~900) — layer pipeline, rule engine, actions | DONE |
| `boson-script-engine` | `packages/gateway/gateway/script/` (517 L) — the scripted purchase flow | DONE |
| `boson-compact-session` | `packages/gateway/gateway/{compact,session}/` (393 + 289 L) — summarization and session state | DONE |

## Cluster E — Production (feeds ch-09, ch-10)

| slug | artifact | status |
|------|----------|--------|
| `latency-budget-voice` | Voice-to-voice latency breakdowns as Pipecat measures them (`stt_latency.py`, `UserBotLatencyObserver`, TTFA/TTFAT) | DONE |
| `deployment-scaling` | Running Pipecat in production — process model, concurrency, cold start, the bus | DONE |
| `migration-patterns` | Incremental-migration patterns for replacing a transport/orchestration layer | TODO |

## Cluster F — The learner's own `realtime_voice` (feeds ch-03, ch-06, ch-10)

Discovered during the harvest and **not anticipated by the original plan**. The
learner maintains a separate repo at
`/Users/jaewon/mywork_2026/Lina_2026/boson-agent-voice-chat-dev/boson-agent`
(branch `voice-chat-dev`) whose `packages/realtime_voice/` is 3,886 L of source
+ 1,504 L of tests already doing server-side voice: VAD, Korean phrase chunking,
a WebRTC transport, and an audio/text playout ledger. This turns the course from
a port into a **comparison**, and the capstone into a keep-or-replace decision.

| slug | artifact | status |
|------|----------|--------|
| `rtv-pipeline-session` | `packages/realtime_voice/realtime_voice/{pipeline/session.py,protocols.py,types.py,queues.py,clock.py}` — the monolithic `VoiceSession` supervisor and its closed event union | DONE |
| `rtv-vad-chunking` | `realtime_voice/{vad/energy.py,vad/silero.py,chunking.py,ledger.py}` — frame-count VAD, `KoreanPhraseChunker`, `AudioTextPlayoutLedger` | DONE |
| `rtv-webrtc-transport` | `realtime_voice/transport/webrtc/` (~960 L) — aiortc peer, HMAC `WebRTCSessionManager`, the audio-forbidden `ControlEvent` protocol | DONE |
| `rtv-vs-pipecat-gap` | Feature-by-feature comparison + test-coverage evidence — the capstone's decision table | DONE |

## Cluster G — Architectural theory (feeds ch-01, ch-02, ch-06, ch-08)

Also unplanned. These answer *why* the design is shaped this way, grounding the
frame pipeline in named prior art (Garlan & Shaw, Beck, Wadler, GStreamer) rather
than treating it as Pipecat's invention. They are the conceptual spine of ch-01.

| slug | artifact | status |
|------|----------|--------|
| `theory-pipes-and-filters` | The Pipes-and-Filters style — uniform interface, the identity/zero/associativity algebra witnessed by real classes, and the position-N checklist the type system cannot run for you | DONE |
| `theory-out-of-band-priority` | Why control latency is queue depth not code speed; the priority tier + task split; back-pressure opted out of on purpose; GStreamer's in-band/out-of-band precedent | DONE |
| `theory-narrow-waist` | `Frame` as an hourglass waist and the Expression Problem tax (120 concrete frame types, 577 `isinstance` sites); `flows/` defined only 2 frames — the precedent boson must follow | DONE |

## Gap log

**Korean STT on telephony audio — still the migration's largest unknown.**

- **No measured Korean accuracy number exists anywhere in the Pipecat tree.**
  Verified by grep across `src/`: `WER` / "word error rate" / accuracy claims →
  **zero hits**, for any service, any language. Record this rather than guessing.
- **No Korean-on-8 kHz number either.** `stt_latency.py` is the repo's only
  benchmark table and it is silent on the language *and* the sample rate of the
  benchmark audio; it only states `VADParams.stop_secs=0.2`. The only `8000`
  values in the tree are telephony serializer defaults.
- **The physics make the transfer invalid anyway.** The PSTN wire is 8 kHz
  μ-law: a 4 kHz Nyquist ceiling plus 8-bit companding. Korean fricatives
  (ㅅ/ㅆ/ㅊ) and much of the 받침 discrimination cue sit at or above 4 kHz and are
  simply absent from the signal. Upsampling to 16 kHz satisfies the model's input
  contract, not the information loss. **Any Korean-STT number measured on 16 kHz
  studio audio is not transferable.** Blocking action: run
  <https://github.com/pipecat-ai/stt-benchmark> on real Lina TMR μ-law audio.
- **Deepgram is the lowest-latency option (0.35 s) and the *least* verified.** It
  has no `LANGUAGE_MAP` and no `language_to_service_language` override — Korean
  is a bare passthrough (`kwargs["language"] = str(s.language)`).
- **AssemblyAI does not support Korean** despite being an obvious candidate; its
  `LANGUAGE_MAP` and `language_code` docstring both enumerate the set and Korean
  is absent from both. **Sarvam** looks like a hit when grepping `KO` but maps
  `Language.KOK_IN` (Konkani). **CartesiaTurnsSTTService** is English-only.
- **`realtime_voice`'s `SileroVAD` hard-rejects 8 kHz** (`ValueError: requires
  16 kHz mono PCM`), so the learner's existing stack has *no* telephony path at
  all. Pipecat's `SileroVADAnalyzer` supports 8 kHz natively (256-sample frames)
  and the derived frame counts are identical, so tuning does not change.

**Assumed to exist but does not.**

- **`StartInterruptionFrame` / `StopInterruptionFrame` / `BotInterruptionFrame`**
  — zero hits across `src/`. There is one field-less `InterruptionFrame`
  (`frames.py` L1142), broadcast in *both* directions.
- **No telephony transport.** No `twilio/`, `telnyx/`, `plivo/`, `exotel/` or
  `sip/` directory under `src/pipecat/transports/`. Telephony = WebSocket
  transport + serializer.
- **`TransportParams.vad_analyzer` was removed.** VAD now mounts on
  `LLMUserAggregatorParams(vad_analyzer=…)` or a standalone `VADProcessor`.
  Any text saying "attach the analyzer to transport params" describes a dead API.
- **`create_context_aggregator()` does not exist** (removed with
  `OpenAILLMContext`, PR #4215). Replacement is `LLMContext` +
  `LLMContextAggregatorPair`.
- **No `endpointing.py`, no `EndpointingConfig`, no single turn-detector class.**
  The real subsystem is the strategy chain in `src/pipecat/turns/`.
- **No `max_turns` / turn cap anywhere in Pipecat** — zero hits. boson's
  `max_turns=50` guard must be rebuilt as a counting `FrameProcessor` or dropped.
- **No dedicated word-boundary or timestamp frame class.** Word info rides on
  `TTSTextFrame.pts`. Every class with "Word" in its name is deprecated (0.0.105).
- **`FlowConfig` does not exist**, and neither does static-vs-dynamic flows —
  zero hits. Flows determines structure at runtime only, with **no** transition
  legality check and no node registry.
- **No maintained self-hosted Korean TTS.** The only local service mapping
  Korean is `XTTSService`, deprecated 1.7.0 with "No replacement". Neither Kokoro
  nor Piper maps Korean. **Rime** is word-timestamp capable but supports only
  five languages, none of them Korean — the most tempting wrong pick.
- **No deployment docs in the repo.** `docs/` is Sphinx scaffolding; guidance is
  off-repo. Scaling surface is one `[scaling] min_agents = 1` line.
- **No perceptual latency targets stated by Pipecat.** The ~200 ms / ~800 ms
  figures are external voice-UX knowledge and must not be attributed to the repo.
  The learner's own `CLAUDE.md` *does* state one: P50 ≤ 1.0 s, P95 ≤ 1.5 s, from
  last voiced user sample to first audible assistant sample.

**boson-side surprises.**

- **`PartialDetector` is dead code in production.** Constructed once at
  `bootstrap.py:316`, stored, and **never read**. Real partial handling is the
  `_partial_transcripts` dict + 2 s silence timer in `server/websocket.py`.
- **Nine stages, not seven**; a tenth (`close`) is referenced in three modules
  and has a prompt file but is never registered — it can never be entered.
- **`VALID_SERVER_TYPES` declares `interrupted` and `stage_changed`; neither is
  ever emitted.** Porting them is *implementing*, not porting.
- **`realtime_voice` has no streaming STT.** `OpenAICompatibleUnaryASR` buffers
  the whole utterance to WAV and makes one blocking call at `finalize()`.
  `ASREventKind.INTERIM` / `END_OF_TURN` are emitted only by a test fake. This
  is the single largest functional gap against the P50 ≤ 1.0 s target.
- **boson has two capabilities Pipecat entirely lacks**: `WebRTCSessionManager`
  HMAC session auth with TTL and one-live-peer, and the versioned/ordered/
  audio-forbidden `ControlEvent` data-channel schema. Pipecat's `on_app_message`
  is an untyped passthrough with no auth layer.

**Process notes.**

- Pipecat moves fast: seven minors in under four months, 391 live deprecations
  all dated for removal in 2.0.0. Every excerpt records the commit it was read
  at; prefer the source tree over prose docs where they disagree.
- boson's **LLM provider set is six**, not three (anthropic, openai, google,
  boson, xai, openrouter) — three are OpenAI-compatible subclasses, so the
  effective wire formats are Anthropic / OpenAI / Gemini.
- Remaining open item: `migration-patterns` (Cluster E) has no excerpt. It is the
  only planned slug still missing, and it is the one with no in-repo source —
  it would have to be crawled.
- **Schema deviation, accepted:** the three Cluster G excerpts declare
  `type: theory`, which is not one of the `doc|source|paper|module|boson` values
  in `README.md`'s header schema. The type is apt (they are grounded in named
  literature rather than in one artifact) and they otherwise satisfy the schema
  in full, including the Migration angle line. Either widen the README's enum to
  include `theory` or re-label them `paper`; do not silently leave the two
  documents disagreeing.
- **Cosmetic, not a violation:** 19 excerpts put the `<!-- slug: … -->` comment on
  line 3 with a blank line after the title rather than on line 2 as the README
  sketches. Every one carries the correct slug, type and source. Harmless, but if
  a linter is ever written against the schema it should tolerate both.

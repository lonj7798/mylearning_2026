# boson-agent — Outline Draft

**Kind:** draft
**Course slug:** boson-agent
**Generated:** 2026-04-17T00:00:00Z
**Chapters:** 12

A chapter-by-chapter teaching plan for the Basement Agent Framework and its Gateway orchestration layer. Each chapter focuses on one concept and depends only on chapters listed above it. Tactic-blind: the shape is voice-invariant.

Revision note (2026-04-17): Learner feedback requested (1) removal of the capstone/integration chapter so each subsystem is studied independently, and (2) deeper coverage of the Gateway network layer. The old ch-07 has been split into three: conversation ownership (ch-07), WebSocket server and wire protocol (ch-08), and concurrency/interrupts/barge-in over WebSocket (ch-09). Rule engine, actions, and stages/compaction retain their scope but shift ids. WebRTC is not in the codebase and is therefore not covered; networking material is anchored exclusively to real files under `packages/gateway/gateway/server/`, `packages/gateway/gateway/session/`, `docs/plan/v0_3`, `docs/plan/v0_4`, `docs/plan/v0_6`, and the `agents/*/client.py` examples.

---

## ch-01 — Framework Orientation and Two-Package Architecture

**Depends on:** (none)

Purpose: give the learner the map before anything else. What is Basement? What is Gateway? Where does an agent live on disk?

Concepts:
- Basement (core) vs Gateway (orchestrator) split and what each owns
- Agent folder convention: `BOSON.md`, `config.yaml`, `tools/`, `hooks/`, `skills/`
- Zero-registration auto-discovery: why dropping files in standard folders works
- LLM-Oriented Design (LOD) principles: small files, one responsibility, deterministic vs flexible layers
- Installation, Python 3.11+ requirement, and the `python -m basement` / `python -m gateway` entry points

Primary sources: `boson-agent/README.md`, `packages/basement/README.md` (intro + architecture sections).

---

## ch-02 — The Think-Act-Observe Agent Loop

**Depends on:** ch-01

Purpose: make the core runtime concrete. Before we talk about tools, hooks, or the Gateway, the learner must understand what a turn actually *is*.

Concepts:
- `run_agent_loop()` as the core think → act → observe orchestrator
- Turn lifecycle: add user message, stream LLM, execute tool_uses, repeat until text-only response
- Tool chaining via inner loop and the `max_turns` safety bound
- StreamEvent types: `TextDelta`, `ToolUseStart`, `InputJsonDelta`, `ToolUseEnd`, `MessageEnd`
- `AgentRuntime` dataclass as the single bundle of runtime components (AD3)

Primary sources: `packages/basement/basement/loop/agent_loop.py`, `packages/basement/README.md` (Agent Loop section).

---

## ch-03 — LLM Providers and Streaming Abstraction

**Depends on:** ch-01

Purpose: round out the runtime by teaching what the loop streams *from*. Sits parallel to ch-02 in the DAG because later chapters need both.

Concepts:
- Provider protocol: Anthropic, OpenAI, Google behind a uniform streaming interface
- `LLMConfig` fields: provider, model, temperature, max_tokens, api_key
- The provider registry (`get_provider`) and how new providers are registered
- Streaming of text deltas vs `tool_use` blocks and how the agent loop consumes them
- API key resolution via environment variables and the `.env` discovery walk-up

Primary sources: `packages/basement/README.md` (Providers section), `agents/demo/config.yaml`.

---

## ch-04 — Defining Tools with the @tool Decorator

**Depends on:** ch-02

Purpose: first-class user-facing feature. Now that the loop is known, show how to give it hands.

Concepts:
- `@tool` decorator: required docstring (becomes description) and required type hints
- Automatic JSON Schema generation from Python type hints (`str`, `int`, `float`, `bool`, `list[X]`, `Optional[X]`)
- `ToolSpec` structure: `name`, `description`, `input_schema`, `handler`
- Sync vs async tool handlers and how return values become strings for the LLM
- `ToolRegistry` auto-discovery from the agent's `tools/` folder

Primary sources: `packages/basement/basement/tools/decorator.py`, `packages/basement/README.md` (Tools section).

---

## ch-05 — Observing and Intervening with Hooks

**Depends on:** ch-02

Purpose: the second first-class user extension point. Hooks are meaningful only once the loop is understood.

Concepts:
- `@hook` decorator and the `HookEvent` enum (`ON_TURN_START`, `PRE/POST_LLM_CALL`, `PRE/POST_TOOL_CALL`, `ON_ERROR`, `ON_COMPACT`, `ON_TURN_END`, `ON_SKILL_INVOKE`)
- `HookContext` fields and priority ordering (lower priority runs first)
- `ConversationAPI`: `inject_assistant_tool_use`, `inject_tool_result`, `inject_system_reminder`
- AD1 mutation timing: append operations are immediate, destructive operations (remove/replace) are deferred to turn boundary via `flush_pending`
- Supervisor hooks as syntactic sugar for `ON_ERROR` with early priority for retry/recovery

Primary sources: `packages/basement/README.md` (Hooks section), `agent_loop.py` hook call sites.

---

## ch-06 — Skills, Permissions, ToolRouter, and MCP

**Depends on:** ch-04

Purpose: the v0.2 plugin axis — optional but important extensions that all sit on top of the tool system.

Concepts:
- Skills as `.md` prompt injections activated via the `use_skill` meta-tool
- `PermissionChecker` semantics: `allowed_tools` whitelist, `denied_tools` blacklist, deny-overrides-allow rule
- `ToolRouter`: hiding native tools and exposing only `use_tool` + `use_skill` meta-tools to the LLM
- `router.dispatch()` unifying native and MCP tool execution behind one interface
- `MCPClient` / `MCPManager` for connecting to external Model Context Protocol servers

Primary sources: `packages/basement/README.md` (Skills, Permissions, ToolRouter, MCP sections).

---

## ch-07 — Gateway: Conversation Ownership

**Depends on:** ch-02

Purpose: enter package #2. The learner now crosses the boundary from single-turn agent to full-conversation orchestrator. This chapter is intentionally narrow: it covers conversation state and the per-turn flow, but does **not** cover the wire protocol (that is ch-08) or concurrency/interrupts (that is ch-09).

Concepts:
- Why the Gateway exists: it owns the conversation while the agent owns a single turn
- `GatewayCore.handle_message()` per-turn flow: session lookup, rules, executor, agent loop
- `SessionStore` and `SessionState` (`session_id`, `messages`, `active_stage`, `active_skill`, `pending_compact`)
- `SharedHistory` adapter: direct reference (not copy) so Gateway and agent mutations are mutually visible
- How turn-scoped (agent) and conversation-scoped (gateway) state stay in sync without copying

Primary sources: `packages/gateway/gateway/core.py`, `packages/gateway/gateway/session/store.py`, `packages/gateway/README.md` (architecture + shared-history sections).

---

## ch-08 — Gateway Networking: WebSocket Server and Protocol

**Depends on:** ch-07

Purpose: teach how a real client actually talks to the Gateway. ch-07 gave the learner the in-process conversation model; this chapter exposes the wire. No WebRTC, no speculative transports — only the WebSocket server and JSON protocol that exist in the codebase today.

Concepts:
- WebSocket server startup and endpoint wiring (`packages/gateway/gateway/server/websocket.py`)
- Connection lifecycle: how an incoming socket is bound to a `session_id` via `SessionStore` (`packages/gateway/gateway/session/store.py`)
- Inbound wire message: `user_message` schema as defined in `packages/gateway/gateway/server/protocol.py`
- Outbound wire messages: `text_delta` (streamed tokens), `turn_end` (completion marker), `error` (fail_open semantics)
- Per-turn translation: converting `GatewayCore` streaming output into protocol frames pushed to the client
- Client usage pattern demonstrated by `agents/demo-gateway/client.py` and `agents/test-lina-gateway/client.py`
- Protocol design history: `docs/plan/v0_3/06-phase5-websocket.md` (initial) and `docs/plan/v0_4/07-phase6-websocket-e2e.md` (E2E integration)

Primary sources: `packages/gateway/gateway/server/websocket.py`, `packages/gateway/gateway/server/protocol.py`, `packages/gateway/gateway/session/store.py`, `agents/demo-gateway/client.py`, `agents/test-lina-gateway/client.py`, `docs/plan/v0_3/06-phase5-websocket.md`, `docs/plan/v0_4/07-phase6-websocket-e2e.md`.

---

## ch-09 — Concurrency, Interrupts, and Barge-in Over WebSocket

**Depends on:** ch-08

Purpose: once the learner understands how frames flow, teach what happens when they overlap. Barge-in arrives as a WebSocket message mid-turn, so interrupt handling is fundamentally a networking story, not a capstone afterthought.

Concepts:
- Per-session locks: preventing overlapping turns on a single connection (`docs/plan/v0_6/04-phase4-websocket-concurrency.md`)
- Concurrent turn guards: how a new `user_message` mid-stream is detected and queued or rejected
- Barge-in as a wire event: `user_message` arriving while a turn is still emitting `text_delta` frames
- Cancel handlers and interrupt tags: cooperative cancellation of in-flight tool calls on the agent side
- Partial transcripts: preserving the assistant's emitted text up to the interrupt point in `SharedHistory`
- Reconnect behavior: session rebind after a dropped WebSocket, state restoration from `SessionStore`
- Interrupt-related schemas and policies referenced in `docs/plan/v0_4/01-phase0-interrupt-schemas.md` and `docs/plan/v0_4/02-phase1-bargein-policies.md`

Primary sources: `docs/plan/v0_6/04-phase4-websocket-concurrency.md`, `docs/plan/v0_4/01-phase0-interrupt-schemas.md`, `docs/plan/v0_4/02-phase1-bargein-policies.md`, `docs/plan/v0_4/03-phase2-partial-detection.md`, `packages/gateway/gateway/server/websocket.py`, `packages/gateway/gateway/session/store.py`.

---

## ch-10 — The Rule Engine: @check, Modes, and Priorities

**Depends on:** ch-07

Purpose: the first Gateway extension point. Explain *when* rules run and *how* they compose before showing *what* they can do. Depends on ch-07 (conversation ownership) rather than the network chapters because rules are evaluated inside `GatewayCore.handle_message`, independent of transport.

Concepts:
- `@check` decorator parameters: `name`, `mode` (sequential/parallel), `priority`, `check_type` (deterministic/llm)
- Sequential phase: priority-ordered, short-circuits on the first non-CONTINUE action
- Parallel phase: `asyncio.gather` concurrent execution, all non-CONTINUE results collected
- Check signature `(messages, user_message, session)` and stateful rules via custom session attributes
- Fail-open behavior: logged warning + `Continue()` on exception (toggled by `fail_open` flag)

Primary sources: `packages/gateway/gateway/rules/engine.py`, `packages/gateway/README.md` (Rule Engine section).

---

## ch-11 — Actions: Continue, Respond, Inject, PreTool, Compact, StageTransition

**Depends on:** ch-10

Purpose: now that the engine evaluates rules, show every instrument a rule can return.

Concepts:
- `Continue` / `Pass` as the no-op default that proceeds to the agent
- `Respond`: return a fixed string and skip the agent loop entirely
- `Inject`: append a system-reminder to history, then continue to the agent
- `PreTool`: execute a named tool before the agent turn to reduce latency or seed context
- `StageTransition` and `Compact` as orchestration-level actions coordinated with the stage machine and compact pipeline

Primary sources: `packages/gateway/README.md` (Rule Actions section), `agents/demo-gateway/layers/01-guard/rules/spam_filter.py`, `agents/demo-gateway/layers/03-orchestrator/rules/stage_manager.py`.

---

## ch-12 — Stage Machine, Layers, and Async Compaction

**Depends on:** ch-11

Purpose: the orchestration-layer capabilities that actions like `StageTransition` and `Compact` actually drive. Grouped because all three are conversation-scoped state systems. This is the final chapter — the course ends on stages/compaction per learner request; no capstone.

Concepts:
- Stage definitions in `stage_config.py`: tools, skills, transitions allowlist, preloads, `initial_stage`
- Per-stage tool and meta-tool filtering via `ToolRouter.set_allowed_tools()` at turn start
- Stage preloads: synthesized `tool_use`/`tool_result` pairs and skill prompt injection on stage entry
- Layer pipeline (e.g. `01-guard`, `02-analyzer`, `03-orchestrator`) with `ctx.data` used to pass state between layers
- `AsyncCompactPipeline`: `threshold_messages` trigger, background summarization task, `pending_compact` swap on next turn, `keep_recent` window

Primary sources: `agents/test-lina-gateway/stage_config.py`, `packages/gateway/README.md` (Compaction + Shared History sections), `agents/demo-gateway/README.md` (layers table).

---

## Dependency Graph (DAG verification)

```
ch-01  (root)
  ├── ch-02  (loop)
  │     ├── ch-04 (tools)
  │     │     └── ch-06 (skills/perm/router/mcp)
  │     ├── ch-05 (hooks)
  │     └── ch-07 (gateway: conversation ownership)
  │           ├── ch-08 (networking: WebSocket server + protocol)
  │           │     └── ch-09 (concurrency / interrupts / barge-in)
  │           └── ch-10 (rule engine)
  │                 └── ch-11 (actions)
  │                       └── ch-12 (stages / layers / async compact)
  └── ch-03  (providers)
```

Every `deps` entry references a chapter id that appears earlier in the `chapters` array. No cycles. Chapter count = 12 (within 3–12 range). No capstone chapter — course ends on ch-12 as requested by the learner.

### Changes from previous draft

- Removed old ch-11 (capstone "Lina Case Study").
- Split old ch-07 into three chapters: ch-07 (conversation ownership only), ch-08 (WebSocket server + wire protocol), ch-09 (concurrency, interrupts, barge-in).
- Renumbered: old ch-08 (rules) → ch-10, old ch-09 (actions) → ch-11, old ch-10 (stages/compact) → ch-12.
- Updated ch-07 concepts to remove all WebSocket-protocol content (now in ch-08).
- Interrupt / barge-in / partial-transcript / cancel-handler material migrated from the removed capstone to ch-09, anchored to `docs/plan/v0_4` and `docs/plan/v0_6/04`.
- Tool-filler / skill-filler / Lina case-study content from the old capstone is intentionally dropped per learner instruction (learner wants subsystem-by-subsystem, not a synthesized worked example).

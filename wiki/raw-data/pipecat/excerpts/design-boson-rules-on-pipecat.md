# Design — Carrying boson's Layered Rule System onto Pipecat
<!-- slug: design-boson-rules-on-pipecat · type: design · source: packages/gateway/gateway/{layers,rules,script,schemas}/ + agents/test-lina-gateway/layers/ vs pipecat src/pipecat/{flows,processors,turns} -->

**Core Insight.** The port has exactly one load-bearing decision, and it is a latency decision, not a structural one. Every boson rule is a pure Python function over a *finished* utterance, so the port is mechanical **up to the moment a rule wants to veto or steer**. Vetoing requires sitting between `LLMUserAggregator` and the LLM service — the only pipeline position where the turn is complete and inference has not started — and that position puts boson's two `check_type="llm"` rules (~250-400 ms of Qwen3.6-27B) directly onto the pre-LLM critical path, every turn. You can pay it, or you can demote in-turn transitions to next-turn transitions. There is no third option in Pipecat.

**Guideline.** Collapse all boson layers into **one** `FrameProcessor` placed between the user aggregator and the LLM, split its checks into a free deterministic tier and a paid LLM tier, and let the Flows node — not your processor — be the sole inference trigger on transition turns.

## Technical Details

### 1. boson's model, in its own vocabulary
- A **layer** is a directory. `discovery.py:24` `LAYER_PATTERN = re.compile(r"^(\d+)-(.+)$")` scans `<agent>/layers/NN-name/rules/*.py`, sorts by numeric prefix, and builds one `RuleEngine` per layer. Lina has 6 dirs but 4 live: `01-filler-filter`, `02-analyzer`, `03-orchestrator`, `04-committer` (`04-tmr/rules/` and `05-committer/rules/` are empty).
- A **rule** is a function stamped by `check(name, *, mode: Literal["sequential","parallel"]="sequential", priority: int=100, check_type: Literal["deterministic","llm"]="deterministic")` (`rules/check.py:16-42`), signature `(messages, user_message, session) -> Action | list[Action]`. **There is no condition DSL** — a condition is arbitrary Python. Lina has **13 live `@check`s**; exactly 2 are `check_type="llm"` (`intent_rules` prio 30, `sentiment_tracker` prio 10, both `mode="parallel"`).
- Inside a layer, `RuleEngine.evaluate` (`rules/engine.py:41-134`): sequential checks run in priority order and the first non-`continue`/`pass` **short-circuits** (L68-69); parallel checks all run under one `asyncio.gather` (L74-80) and every non-continue result is kept. Cross-phase arbitration (L100-127): at most one `stage_transition` survives; lowest `@check` priority wins; ties go to sequential; losers lose *only* their transition.
- **Actions are 8 verbs**, `schemas/actions.py:17-20`: `continue respond inject compact pre_tool stage_transition filter pass`. Constructors are functions, not classes.
- **Does a rule need the COMPLETE user turn? Yes, structurally.** `LayerPipeline.process(self, session_id: str, content: str, session: SessionState, *, user_message_appended: bool = False)` (`layers/pipeline.py:87-94`) — `content` is a `str`, one whole utterance. Partial ASR never reaches a rule.
- **Does a rule mutate the prompt, block a response, force a transition? All three — and one rule can do all three in a single return.** `Inject` folds `<system-reminder>…</system-reminder>` into the most recent user message (`_merge_system_reminder`, L341-372). `Respond(text)` appends an assistant message and the pipeline `return`s before `self._inner_handler` (L320-324) — the LLM never runs. `Filter` discards every staged action and deletes the appended user message **by object identity** (L215-235). `StageTransition` reaches `core._apply_stage_transition` (`core.py:591`, wired at `bootstrap.py:482`) → `StageMachine.transition()` → `TransitionResult(success=False, error="Transition 'a' -> 'b' not allowed")` (`stage/machine.py:57-60`).
- **One real rule** (`engine/intent_rules.py:104-130`), emitting all three kinds at once:
  ```python
  IntentRule(
      descs=["Agent (telemarketer) has explicitly asked the customer for permission to continue…",
             "Customer yields the floor for the agent to continue… (a) direct consent ('네','동의합니다'…)"],
      type="AND",                      # ALL descs must have matched, persisted in session.checklist_state
      actions=[StageTransition("product_focused"),
               Inject("[Active product for this customer: product_id='saedam_355_m35_plan_default'. …]"),
               PreTool("check_product_summary", {"product_id": "saedam_355_m35_plan_default"},
                       preamble=["감사합니다!"])],
  )
  ```
- **When in a turn each layer runs.** All layers run in one contiguous block *between* the arrival of the finished utterance and the first LLM token. `LayerPipeline.process` captures `pre_turn_status`, flips the tracker to GENERATING (L113-115), appends the plain user message **first** (L154-156, keeping an identity reference), then Phase 1 iterates layers in prefix order staging actions (L178-248), Phase 2 replays them in the same layer/array order (L254-316), and only then does `self._inner_handler(session_id, content)` start the agent loop (L334). The full live order for Lina, with `@check` priorities:

  | layer | checks (mode, priority) | job |
  |---|---|---|
  | `01-filler-filter` | `korean_filler_filter` (seq, 10) | drop backchannels **only** while `pre_turn_status in ("generating","tool_processing")` |
  | `02-analyzer` | `end_signal` (seq, 5), `response_classifier` (seq, 5), `hesitation_hook` (seq, 10) | classify the utterance; write findings onto the session for later layers |
  | `03-orchestrator` | `turn_counter` (seq, 1), `stage_round_tracker` (seq, 2), `preload_on_question` (seq, 4), `script_flow` (seq, 10), `sentiment_tracker` (par, 10, **llm**), `tool_gate` (par, 12), `intent_rules` (par, 30, **llm**) | all stage/script/tool decisions |
  | `04-committer` | `help_responder` (seq, 10), `auto_compact` (seq, 99) | terminal responses and history maintenance |

  Note the deliberate seq/par split in layer 03: `script_flow` is sequential at priority 10 so it can short-circuit; `tool_gate` is parallel at 12 precisely so its advisory `Inject` *cannot* short-circuit consent collection (`tool_gate.py:23-31`, verbatim: *"Running in the parallel phase means this check never short-circuits the sequential pipeline, so consent collection still completes (no deadlock)"*).
- **What state a rule sees.** `SharedLayerContext` (`layers/context.py:17`) carries `session, messages, user_message, signal_queue, get_agent_status, layer_name, data`, and `__getattr__` (L40) / `__setattr__` (L54) proxy every other name straight through to the live `SessionState` — so `session.active_stage`, `session.checklist_state`, `session.fired_rules`, `session.script_state` are read/write and persist. Plus `session.pre_turn_status`, the `AgentStatus` captured *before* this turn opened its own GENERATING window (`pipeline.py:113-114`); the filler filter must read it (`korean_fillers.py:66`) or it self-filters.

### 2. The mapping table

| boson mechanism | Pipecat home | why | what is lost |
|---|---|---|---|
| layer discovery (`layers/NN-name/`) | **NOTHING** | no layer or node registry exists; `_validate_node_config` (`manager.py:867-898`) checks 2 things only | ordering-by-filename → an explicit Python list |
| `@check` sequential + short-circuit | custom `FrameProcessor` internals | Pipecat ships no rule scheduler | nothing — pure Python, port verbatim |
| `@check(mode="parallel")` gather | same processor, same `asyncio.gather` | — | nothing |
| Phase-1/Phase-2 staged commit | **one** custom `FrameProcessor` | `push_frame` is irreversible; the only rollback surface is `LLMContext.set_messages()` (`llm_context.py:377`) | cross-*processor* veto — all layers must collapse into one object |
| `Filter(reason)` | swallow the `LLMContextFrame`, then `context.set_messages(snapshot)` | dropping = not calling `push_frame`; `FunctionFilter` (`function_filter.py:21`) cannot do the context rollback, so don't use it | cannot un-broadcast an interruption already sent by VAD |
| `Respond(text)` | push `TTSSpeakFrame`, swallow the context frame | Flows' `tts_say` only fires at node-set time, too coarse | nothing |
| `Inject(content)` | `context.add_message(...)` before pushing | the option-β *merge into the last user message* has no frame equivalent | the `\n---\n` separator + reminder-stacking convention is yours to reimplement |
| `PreTool(name, args, preamble)` | Flows `function` action in `pre_actions` | `actions.py:285` — `"function"` actions **always** wait, matching boson's synchronous-before-generation semantics | the preamble-as-first-stream-chunk; becomes a separate `tts_say` action ordered before it |
| `StageTransition(target)` | `await flow_manager.set_node_from_config(node)` (`manager.py:588`) | Path B, proven in-tree at `warm_transfer.py:658` | inline synchrony — Flows queues at the pipeline **head** (`manager.py:841`) |
| `StageMachine.transition()` legality | **NOTHING** — keep boson's `StageMachine` as a pure pre-check | Flows has no from→to check anywhere in the codebase | nothing if you keep the class; all legality if you drop it |
| `StageDefinition.prompt/tools` | a Flows `NodeConfig` (`task_messages` + `functions`) | `stage_config.py` already *is* the node graph | `skills` — no Pipecat concept at all |
| `_GLOBAL_TOOLS` | `FlowManager(global_functions=[...])` (`manager.py:100`, mixed in at L654) | exact match | nothing |
| `Compact()` | Flows `function` pre-action calling boson's compactor | `LLMContextSummaryRequestFrame` exists (`llm_service.py:709`) but Flows' summary path is `asyncio.wait_for(..., timeout=5.0)` and **silently degrades to APPEND** (`manager.py:815-822`) | background-async compaction; a pre-action blocks |
| `SignalQueue.get_recent(seconds)` | plain object owned by the processor | the bus is push-only — `BusSubscriber` is `name` + `on_bus_message`, no read API | nothing, in-process |
| `AgentStatusTracker` / `pre_turn_status` | processor state fed by `BotStartedSpeakingFrame`/`BotStoppedSpeakingFrame` | Pipecat has no agent-status enum | `TOOL_PROCESSING` has no frame that means it; the 500 ms `settling_ms` decay must be re-derived |
| `SharedLayerContext` proxy | keep a real `SessionState` object on the processor and pass it as the `session` arg | `flow_manager.state` is a `dict`; rewriting 13 rules' `getattr(session, …)` buys nothing | nothing — this is the zero-edit choice |
| `ScriptEngine.process_turn(state, msg, registry)` | runs unchanged inside the processor | already stateless dict-in/`Action`-out | nothing — cleanest port in the system |

### 3. The hard case: a rule that needs the whole turn, in a token-stream pipeline
The rule is `evaluate_intent_rules` (`transition_detector.py:82`). It calls `_llm_match_descs` (`intent_matcher.py:205-271`) — **one batched call**, prompt anchored on `"Most recent turn (PRIMARY SIGNAL — evaluate against THESE)"`, model `("boson", "Qwen3.6-27B-FP8")` at `temperature=0.1` (`llm_config.py:20,34`), output a comma-separated index list or `"none"`. `sentiment_tracker` fires concurrently under the same `gather`, so wall clock ≈ max, not sum.

**Option A — gate the context downstream of the user aggregator.** `LLMUserAggregator.push_aggregation()` (`llm_response_universal.py:856-873`) does `self._context.add_message({...})` (L863) and *then* `await self.push_context_frame()` (L866). `LLMContextFrame` is a bare `Frame` (`frames.py:551`), and `base_llm.py:601` is what starts the completion. So a processor at that seam holds the complete turn with inference not yet begun, and rollback is real: snapshot then `context.set_messages(snapshot)`. Every boson action semantic survives exactly.

**Option B — run the rule on the aggregated turn via an event handler.** Rejected, and the reason is decisive rather than stylistic: `on_user_turn_message_added` is fired at L871, *after* `push_context_frame()` at L866. The event is a notification, not a gate — an event handler can never veto. Option B is usable only for the observe-only rules (`end_signal`, `turn_counter`, `stage_round_tracker`).

**Option C — let generation start, then cancel.** `broadcast_interruption()` (`frame_processor.py:1017-1022`) resets the process task and broadcasts `InterruptionFrame`. Zero added latency on the pass path — but `Inject` and `StageTransition` exist to steer *the generation they precede*, so a cancel-and-redo pays the first generation's TTFT twice and may have already spoken audio.

**Decision: Option A, split into two tiers.** Tier 1 = the 11 deterministic checks (filler filter, `end_signal`, `response_classifier`, `hesitation`, `script_flow`, `preload`, `turn_counter`, `stage_round_tracker`, `tool_gate`, `help_responder`, `auto_compact`) — all pure Python, sub-millisecond, blocking, free. Tier 2 = the 2 LLM checks — **blocking, and this is the bill: ~250-400 ms of added pre-LLM latency on every turn** (one Qwen3.6-27B TTFT plus ~5 output tokens). For scale, Pipecat's own STT TTFS P99 reference is 0.45 s for Deepgram (`stt_latency.py`; Korean 8 kHz telephony has **no entry** and must be benchmarked), so Tier 2 roughly *doubles* the pre-LLM half of the budget. That cost buys in-turn veto and in-turn transitions. The only way to not pay it is to run Tier 2 concurrently with the LLM's first tokens and call `set_node_from_config()` on completion — which makes every stage change land **one turn late**. State the trade honestly to the product owner: **veto and in-turn steering cost 250-400 ms; next-turn transitions cost 0 ms.** boson pays the 250-400 ms today.

### 4. Pipeline position — the proposed list
```python
pipeline = Pipeline([
    transport.input(),
    stt,                       # Korean 8 kHz telephony STT
    BosonFillerGate(),         # boson layer 01
    user_aggregator,           # LLMContextAggregatorPair(context).user()
    BosonRuleProcessor(...),   # boson layers 02/03/04, Tier 1 + Tier 2
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
])
# FlowManager(llm=llm, context_aggregator=pair, worker=worker, global_functions=[...])
# is NOT in this list — it drives from outside via worker.queue_frames (manager.py:841).
```
- **`BosonFillerGate` between `stt` and `user_aggregator`.** It needs `TranscriptionFrame` text, which only `stt` produces, and must drop it before the aggregator commits it to `LLMContext`. One position earlier (before `stt`) it sees only audio and `_is_filler_text()` has no input — the rule is dead. One position later (after the aggregator) the "네" is already `add_message`d and the context frame already pushed, so a 1-line string check has become a rollback. It must also drop `InterimTranscriptionFrame` unconditionally, or every keyword rule fires on prefixes.
- **`BosonRuleProcessor` between `user_aggregator` and `llm`.** This is the only position where the turn is complete *and* inference has not begun — a data dependency (`push_aggregation` L856 writes then pushes; `base_llm.py:601` consumes), not a convention. One earlier: it sees `TranscriptionFrame`s, so `kw in user_message.lower()` in `end_signal.py` fires on fragments and the "PRIMARY SIGNAL" prompt gets a partial utterance. One later (after `llm`): the model has already generated, `Inject` can no longer steer what it was written to steer, and `Filter`/`Respond` degrade into Option C.
- **`assistant_aggregator` after `transport.output()`** — the house pattern, and boson has the same invariant for a different reason: a `Respond()` interrupted mid-TTS must not be recorded as spoken. Move it before `transport.output()` and history drifts exactly the way boson's identity-based rollback exists to prevent.
- **On a transition turn, `BosonRuleProcessor` swallows the context frame and lets the Flows node be the sole inference trigger** (`respond_immediately=True`, `LLMRunFrame` at `manager.py:707-709`). This is deliberate and reverses the naive advice: ordering `set_node_from_config()` before `push_frame()` does **not** fix the race, because the node's frames enter at the head and must traverse `stt` → gate → aggregator before reaching `llm`, while your pushed frame reaches `llm` immediately. Swallowing removes the race by construction. On non-transition turns, push normally.

### 5. Three open risks
1. **Filler-filter vs energy-based barge-in.** boson filters `"네"` by *content* and `pre_turn_status`; Pipecat interrupts on VAD energy upstream of STT, so the bot is already interrupted before the gate sees text. **Measure:** timestamp-diff interruption broadcast vs `TranscriptionFrame` arrival over a corpus of lone Korean backchannels on 8 kHz telephony audio. If the gap is positive (it will be, ~always), a custom `BaseUserTurnStartStrategy` (`base_user_turn_stop_strategy.py:38` for the sibling contract) that withholds turn-start until a transcript exists is mandatory — and it costs the unmeasured Korean STT TTFS.
2. **Transition frame race.** **Prototype:** rule processor calls `set_node_from_config()` and swallows the context frame; assert with a `FrameLogger` at the `llm` input that exactly **one** inference-triggering frame arrives per turn and that `LLMSetToolsFrame` precedes it. Failure modes to watch for explicitly: zero generations (both paths swallowed) and two (node ran *and* context pushed).
3. **Two-phase-commit blast radius.** boson rolls back by object identity over `session.messages`; `LLMContext` offers only `set_messages(list)` with no identity handle, and the aggregator has already written. **Prototype:** snapshot/restore around the whole rule round and replay the Lina e2e suite (`agents/test-lina-gateway/tests/`, `e2e_runner.py`), counting divergences — specifically turns where a `PreTool` appended synthetic tool-call history *before* a later layer filtered.

- **Migration angle:** the port is one processor, not a framework port. `BosonRuleProcessor` holds all 13 checks, the `RuleEngine`, the `SignalQueue`, the `StageMachine` pre-check, and a real `SessionState` — so the rule files themselves need **zero edits**. What genuinely has no Pipecat home and must stay boson code: cross-layer veto, transition legality, `skills`, per-session attribute namespaces, and the `TOOL_PROCESSING` status. What Pipecat gives back: `Flows` nodes for stages, `global_functions` for `_GLOBAL_TOOLS`, and `function` pre-actions for `PreTool`.

## Citation
boson-agent (private, `Lina_2026/boson-agent-dev/boson-agent`), branch `lina-new-dental-dev`, read 2026-08-25. Paths: `packages/gateway/gateway/{layers,rules,script,schemas,stage}/`, `packages/gateway/gateway/{core,bootstrap}.py`, `agents/test-lina-gateway/{stage_config,llm_config}.py`, `agents/test-lina-gateway/layers/`.
pipecat-ai/pipecat commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`, read 2026-08-25. Paths: `src/pipecat/flows/{manager,actions,types}.py`, `src/pipecat/processors/aggregators/{llm_response_universal,llm_context,gated_llm_context}.py`, `src/pipecat/processors/{frame_processor,filters/function_filter}.py`, `src/pipecat/services/{llm_service,openai/base_llm,stt_latency}.py`, `src/pipecat/turns/user_stop/`, `src/pipecat/frames/frames.py`, `examples/getting-started/06-voice-agent.py`, `examples/flows/warm_transfer.py`.

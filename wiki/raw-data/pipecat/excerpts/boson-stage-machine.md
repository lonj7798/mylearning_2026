# boson-agent stage machine — NINE Lina stages, and the LLM never drives them

<!-- slug: boson-stage-machine · type: boson · source: packages/gateway/gateway/stage/{machine.py,context.py} + schemas/stage.py + agents/test-lina-gateway/stage_config.py -->

**Core Insight.** A boson "stage" is a **context package** — prompt + visible tools + visible skills + a whitelist of legal successors — and the transition is performed by *deterministic rule code*, not by the model. Rules emit a `StageTransition(target)` action; `StageMachine.transition()` rejects it if the target is not in the current stage's `transitions` list; only then does `session.active_stage` move and the new stage prompt get injected as a `<system-reminder>`. The LLM is told which stage it is in and never gets to choose the next one.

**Guideline.** When porting to a flow framework, port the *edge whitelist* first, not the prompts. Every regression comment in `stage_config.py` (`v0.7.5 (#12)`) is a bug where a rule emitted a transition the whitelist silently rejected — a failure mode that is invisible in logs unless you check `TransitionResult.error`.

## Technical Details

- **Correction to the brief: there are NINE registered stages, not seven.** `agents/test-lina-gateway/stage_config.py` (87 L) declares `initial_stage = "introduction"` and a `stages: dict` with exactly these keys: `introduction`, `product_focused`, `escalate_to_human`, `consultation`, `purchase`, `reschedule`, `dnc_processing`, `informed_consent`, `end`. `agents/dental-gateway/stage_config.py` is byte-identical. A tenth name, `close`, appears in `session_tracker.MAX_ROUNDS`, `FALLBACK_TRANSITIONS`, and the guard `transition_detector.py:90` `if not stage or stage in ("end", "close")`, and has a prompt file `stages/close.md` — but **`close` is not registered as a stage**; it is a leftover. Also `agents/dental-gateway/stages/` is missing `informed_consent.md` and `purchase.md`, so in that agent those two stages load with `prompt=""` (`load_stage_prompts` only globs existing `*.md`).
- **Schema** (`schemas/stage.py`, 30 L) — two frozen-ish dataclasses, no Pydantic:
  ```python
  @dataclass
  class StageDefinition:
      name: str; prompt: str = ""; tools: list[str] = []; skills: list[str] = []; transitions: list[str] = []
  @dataclass
  class TransitionResult:
      success: bool; new_stage: StageDefinition | None = None; error: str | None = None
  ```
- **The machine is stateless and shared** (`stage/machine.py:17`): *"Shared across all sessions. Holds stage definitions and validates transitions. Does NOT track current stage — that's per-session on `session.active_stage`."* `transition(from_stage, to_stage) -> TransitionResult` (L45) fails with `f"Transition '{from_stage}' -> '{to_stage}' not allowed"` when `to_stage not in current.transitions`. `load_stages(config, prompts)` (L82) builds it from the agent's dict; `load_stage_prompts(stages_dir)` (L71) reads `stages/*.md` keyed by `md_file.stem`.
- **The real edge whitelist** (`stage_config.py`), verbatim targets per stage:
  | stage | allowed successors | stage tools |
  |---|---|---|
  | `introduction` (initial) | product_focused, purchase, dnc_processing, reschedule, escalate_to_human, end | — |
  | `product_focused` | consultation, purchase, reschedule, dnc_processing, escalate_to_human, end | check_product_detail, check_product_summary, lookup_faq |
  | `consultation` | purchase, informed_consent, reschedule, dnc_processing, escalate_to_human, end | + check_available_products; skill `product_manager` |
  | `informed_consent` | consultation, end, reschedule, dnc_processing, escalate_to_human | record_consent, get_consent_status |
  | `purchase` | end, escalate_to_human | agreement_record, agreement_status, check_product_detail, verify_personal_info, save_payment_info, save_address; skill `payment_manager` |
  | `reschedule` | consultation, end, escalate_to_human | reschedule |
  | `dnc_processing` | end, escalate_to_human | register_dnc, check_dnc_status |
  | `escalate_to_human` | end | escalate_to_human |
  | `end` | *(none — terminal)* | — |
- **Who performs a transition — the full chain, four possible triggers.**
  1. *LLM-scored intent rules.* `layers/03-orchestrator/rules/transition_detector.py:82` `@check("intent_rules", mode="parallel", priority=30, check_type="llm")` `async def evaluate_intent_rules(messages, user_message, session)`. It reads `session.active_stage`, flattens that stage's `INTENT_RULES` into descriptions, runs **one LLM call per turn** via `match_intents(...)`, evaluates pools, picks winners by *lowest priority number*, and returns the winner's `actions`. Example (`engine/intent_rules.py:120`): `actions=[StageTransition("product_focused"), Inject("[Active product … product_id='saedam_355_m35_plan_default' …]"), PreTool("check_product_summary", {...}, preamble=["감사합니다!"])]`, `priority=50`. Multi-turn triggers use `chain_consecutive(descs, actions, n=2)` — e.g. DNC requires **two consecutive** matches before `StageTransition("dnc_processing")` + `PreTool("register_dnc", {"reason": "Customer request Do Not Call"})` fires at `priority=20`.
  2. *Sentiment safety override.* `sentiment_tracker.py:92` — with `NEGATIVE_WINDOW_SIZE = 5`, `NEGATIVE_END_THRESHOLD = 4`, `_SCORE_DELTA = {"positive": 1, "neutral": 0, "negative": -1}` (`signals/sentiment_classify.py:76-84`): 4 negatives in the last 5 turns → `StageTransition("reschedule")` if `temperature_score >= -4`, else `StageTransition("end")`.
  3. *Round-budget auto-advance.* `session_tracker.py:27` `MAX_ROUNDS = {introduction: 6, product_focused: 5, consultation: 100, purchase: 100, reschedule: 10, dnc_processing: 3, escalate_to_human: 1, informed_consent: 10, end: 1}` with `FALLBACK_TRANSITIONS = {introduction: "end", product_focused: "consultation", consultation: "reschedule", purchase: "end", reschedule: "end", dnc_processing: "end", escalate_to_human: "end", informed_consent: "consultation"}`; on overflow it returns `StageTransition(fallback)` (L158).
  4. *Script engine.* `script/engine.py:221,232,281` returns `(state, StageTransition(next_ch or "end"))`.
- **Execution.** `StageTransition(target_stage)` is just `Action(type="stage_transition", payload={"target_stage": ...})` (`schemas/actions.py:78`). `LayerPipeline` (`layers/pipeline.py:62`) commits actions in array order and *"call[s] on_stage_transition (which sets `session._pending_stage_injection`); merge that pending injection as `<system-reminder>…</system-reminder>` into the most recent user msg."* `on_stage_transition` is wired at `bootstrap.py:482` to `core._apply_stage_transition` (`core.py:591`):
  ```python
  result = self._stage_machine.transition(from_stage=session.active_stage, to_stage=target)
  if not result.success:
      return                      # ← silent no-op on a rejected edge
  session.active_stage = target
  self._inject_stage(session, result.new_stage)
  ```
  The injected text is `build_stage_injection` (`stage/context.py:73`) = `f"[Stage: {stage.name}]"` + the stage prompt — deliberately **not** the tool/skill lists ("redundant noise the LLM doesn't need to re-read every turn"). The initial stage is bootstrapped once by `_apply_initial_stage` (`bootstrap.py:257`), optionally prefixed with `## Customer Info`.
- **Tool visibility** is enforced twice: `StageContext.from_stage()` → `filter_tools(all_tools)` / `is_skill_available(...)`, gated by `should_filter_tools(tool_router_enabled)` (`context.py:44`) — with the ToolRouter **on**, all tools stay loaded and only the prompt guides usage; with it **off**, the registry is filtered per stage. Also re-asserted per turn as `<system-reminder>Active stage: {session.active_stage}</system-reminder>` (`session/history.py:153-157`).
- **Migration angle:** this is the one boson subsystem with a genuine upstream counterpart — `pipecat.flows` (vendored into core at `src/pipecat/flows/`, formerly the standalone `pipecat-flows`). `StageDefinition` ↔ `NodeConfig` (`flows/types.py:182`: `role_messages`, `task_messages`, `functions`, `context_strategy`, `respond_immediately`); `StageMachine` ↔ `FlowManager` (`flows/manager.py:80`) with `initialize(initial_node)`, `set_node_from_config(node_config)`, `current_node`, `state`. But the transition *mechanism* collides hard: Pipecat Flows transitions are driven by **LLM function calls** — a `FlowsFunctionSchema` handler returns `ConsolidatedFunctionResult = tuple[Any, NodeConfig | None | _NoResponse]` and the returned `NodeConfig` *is* the transition (`_create_transition_func`, `manager.py:443`). boson's model never chooses a stage; nine deterministic rule modules do, one of which is itself an LLM classifier scoring pre-written intent descriptions. Porting naively would hand stage control to the sales LLM and delete the whitelist. The migration that preserves behavior: keep `StageMachine` + `stage_config.py` + the four trigger sources intact, and use `FlowManager.set_node_from_config()` as a *sink* driven by `_apply_stage_transition`, not as the decision-maker — i.e. Flows replaces `_inject_stage` and per-stage tool filtering (`ContextStrategy.APPEND/RESET`, `flows/types.py:134`), and replaces **nothing** in `transition_detector.py`, `sentiment_tracker.py`, or `session_tracker.py`. Untouched by any voice migration: the entire stage layer is text-domain and does not care whether input arrived as audio.

## Citation

boson-agent (private, Lina TMR), `packages/gateway` version `0.3.0`, commit `0a5e8ced2fd9e631dcbbe8c5f4adb68b89a4fafb` (2026-08-20), read 2026-08-25 at `packages/gateway/gateway/stage/`, `packages/gateway/gateway/schemas/stage.py`, `agents/test-lina-gateway/stage_config.py`, `agents/test-lina-gateway/layers/03-orchestrator/rules/`. Pipecat Flows API verified at `pipecat-ai/pipecat` commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (2026-08-25).

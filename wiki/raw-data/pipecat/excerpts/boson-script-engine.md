# boson-agent ScriptEngine — Verbatim Compliance Script That Suppresses the LLM
<!-- slug: boson-script-engine · type: boson · source: packages/gateway/gateway/script/ (engine.py 284 + schema.py 233 = 517 LOC) -->

**Core Insight.** The scripted purchase flow is a **stateless pure function over a state dict** that returns `(new_state, Action)`. Its whole purpose is to emit *word-for-word* YAML text as `Respond(step.text)`, which the layer pipeline treats as an LLM replacement — the model never sees the turn. Free-form generation is the *fallback*, reached only by returning `Continue()`. That inversion (script is primary, LLM is the exception handler) is the opposite of every LLM-first flow framework, and it exists because Korean insurance-consent script text is legally fixed.

**Guideline.** Anything that must be uttered verbatim for compliance cannot be a prompt. Keep it as data that bypasses the model, and give the model a documented handoff point (here: `objection_active`) plus a documented resume point (`interrupted_at`).

## Technical Details

Exactly 517 LOC as the outline claims: `engine.py` 284 + `schema.py` 233. `script/__init__.py` is 0 bytes.

### Step representation — `script/schema.py` L55-72
```python
@dataclass
class ScriptStep:
    id: str
    text: str
    action_type: str
    expected_answer: str
    required_recording: bool = False
    condition: Optional[str] = None
    next_step: Optional[str] = None
    branch_options: Optional[dict] = None
    item_code: Optional[str] = None

@dataclass
class ScriptChapter:
    stage_name: str
    steps: list
    next_stage: Optional[str] = None
```
`load_script_chapter(path: Path) -> ScriptChapter` (L91) parses YAML, requires `stage_name` and per-step `("id","text","action_type","expected_answer")`, and raises `basement.errors.ConfigError` on duplicate step ids. `ScriptRegistry.load_all(directory)` (L174) builds `_chapters`/`_steps` indices *into shadow dicts first*, then validates every `next_stage` and `next_step` reference resolves (L206-222) before committing — a dangling `next_step` is a load-time `ConfigError`, not a runtime `KeyError`. Lookups: `get_step(step_id)`, `get_chapter(stage_name)`.

Real data, `agents/test-lina-gateway/scripts/data/purchase_pre_consent.yaml` (10 chapter files total):
```yaml
stage_name: purchase_pre_consent
next_stage: purchase_brochure
steps:
- id: 1-1
  text: '"가입설계 및 맞춤형 보험상담을 위해 일반개인정보, 건강정보 고지사항과 라이나생명 보유 고객님의 보험계약정보,
    지급정보, 질병·상해정보를 동의일로부터 3개월까지 수집·이용 하고자 합니다. …이에 동의하십니까?"'
  action_type: consent
  expected_answer: 'yes'
  required_recording: true
```

### Advancing — `script/engine.py`
`class ScriptEngine` is **all `@staticmethod`**, no instance state. `init_state(start_chapter) -> dict` (L136) seeds 8 keys: `current_step_index=0`, `chapter`, `stage` (*"kept for backward compatibility"*), `awaiting_response=False`, `branch_decisions={}`, `agreements={}`, `interrupted_at=None`, `objection_active=False`.

`process_turn(script_state: dict, user_message: str, registry: ScriptRegistry) -> tuple[dict, Action]` (L179) copies the dict and both nested dicts first — *"never mutates input state"*. Its dispatch order (L189-284):
1. `objection_active` → `return (state, Continue())` — **defer to LLM**.
2. `interrupted_at is not None` → clear it, `Respond(chapter.steps[interrupted_at].text)` — replay the interrupted step verbatim.
3. `awaiting_response` → record: `agreements[step.item_code] = user_message` when `item_code` set, and `branch_decisions[step.id] = user_message` **always** (L206-208 comment: *"Record EVERY awaited answer so condition-source steps without `branch_options` can drive `should_skip_step` (issue #24)"*), then `awaiting_response=False`, index += 1.
4. Walk steps, accumulating `texts` for `expected_answer == "none"` and flushing them as `Respond("\n".join(texts))` before any awaiting step. Chapter exhaustion follows `chapter.next_stage`, carrying `texts` across chapter boundaries; `next_stage` of `None`/`"end"` yields `StageTransition("end")`.

`_AWAITABLE_ANSWERS = ("yes", "yes_no", "descriptive")` (L71) is the only set that sets `awaiting_response=True`. `advance(state)` bumps the index by 1; `pause_for_interrupt(state)` sets `interrupted_at = current_step_index`; `resume_from_interrupt(state)` clears `objection_active` but **preserves `interrupted_at`**.

Branching: `should_skip_step(state, step)` (L155) splits `step.condition` on `":"` into `step_id, value` and compares `_normalize_yes_no(branch_decisions[step_id]) == _normalize_yes_no(value)`. `_normalize_yes_no` (L86) is **token-exact, never substring** — `set(re.split(r"\W+", text.lower()))` intersected against `_AFFIRMATIVE_TOKENS` (24 tokens: `예/네/넵/응/맞아요/동의합니다/yes/ok/…`) and `_NEGATIVE_TOKENS` (11: `아니오/아니요/아뇨/싫어요/없어요/no/nope/…`); both or neither matched → return input unchanged, degrading to exact match. Comment L74: *"'아니 그게 아니라' or '글쎄요' stay ambiguous."*

Fail-loud data validation: `_condition_dead_reason(step, registry)` (L106) returns why a condition can never fire — self-referential, source missing, or *"source step '%s' has expected_answer '%s' and never awaits an answer"*. Named real defects in the docstring: `purchase_disclosures` step **6-207** (source **6-206** has `expected_answer: none`) and step **6-212** (condition references itself). The engine logs a `WARNING` and continues; it *"cannot repair agent data"* (L243-245).

### Interaction with free-form LLM generation
The engine emits four action kinds and the layer pipeline interprets them:
| Action | Effect in `layers/pipeline.py` |
|---|---|
| `Respond(text)` | appended as `assistant` message, yielded, `respond_emitted=True`; L320-324 *"Skip inner_handler — Respond replaced the LLM."* |
| `Continue()` | falls through to `inner_handler` = `core.handle_message` → real LLM turn |
| `Inject(content)` | merged as `<system-reminder>…</system-reminder>` into the most recent user message, then the LLM runs |
| `StageTransition("end")` | `core._apply_stage_transition` |

Wiring lives in the agent, not the framework: `agents/test-lina-gateway/layers/03-orchestrator/rules/script_flow.py` L133 `@check("script_flow", mode="sequential", priority=10)`, active only when `session.active_stage == "purchase"`. It unwraps the proxy with `_real = getattr(session, "session", session)` (L148) so `_real.script_state` persists, reads `session.script_response` produced by layer `02-analyzer/response_classifier.py`, and on `response_class == "unclear"` returns an `Inject` in Korean naming the chapter label and agreement summary rather than calling the engine — the documented LLM-clarification handoff.

- **Migration angle:** Pipecat *does* ship a flow engine — `src/pipecat/flows/` (2019 LOC: `manager.py` 898, `types.py` 518, `actions.py` 400) with `FlowManager`, `NodeConfig(task_messages, role_message, functions, context_strategy, respond_immediately)`, `ContextStrategyConfig`, and `FlowsFunctionSchema`. **It is not a drop-in replacement**, and the reason is structural: a Flows `NodeConfig` supplies `task_messages` that *steer* the LLM, and transitions fire from LLM **function calls**. boson's steps are not steering — `Respond(step.text)` is the literal string spoken, LLM excluded. Porting `purchase_pre_consent` onto `NodeConfig.task_messages` would let the model paraphrase a regulated consent disclosure. Practical split: keep `ScriptEngine` + `ScriptRegistry` verbatim (they are pure functions with zero gateway coupling — only `gateway.script.schema` and `gateway.schemas.actions`), wrap them in one custom `FrameProcessor`, and map `Respond` → push a `TTSSpeakFrame` while suppressing the downstream `LLMContextFrame`. Use `FlowManager` only for the non-scripted stages (introduction/consultation), and note that `FlowManager.state` is the natural home for `script_state` since Pipecat has no `SessionState`. `pause_for_interrupt`/`resume_from_interrupt` must be re-driven from `InterruptionFrame` rather than from boson's `AgentStatusTracker`.

## Citation
boson-agent (private), branch `lina-new-dental-dev`, commit `0a5e8ced2fd9e631dcbbe8c5f4adb68b89a4fafb`, 2026-08-20. Paths: `packages/gateway/gateway/script/{engine.py,schema.py}`, `agents/test-lina-gateway/scripts/data/*.yaml`, `agents/test-lina-gateway/layers/03-orchestrator/rules/script_flow.py`. Pipecat Flows read against `pipecat-ai/pipecat` commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`, read 2026-08-25.

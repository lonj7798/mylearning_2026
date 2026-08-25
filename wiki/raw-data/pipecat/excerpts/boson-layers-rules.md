# boson-agent Layer Pipeline & Rule Engine — Transactional Voting Over a Finished Turn
<!-- slug: boson-layers-rules · type: boson · source: packages/gateway/gateway/layers/ + packages/gateway/gateway/rules/ -->

**Core Insight.** boson-agent's product logic is a **two-phase-commit vote over one already-complete user utterance**. N layers each run a `RuleEngine`, every layer stages `Action`s without side effects, and a single `Filter` from *any* later layer discards *every* staged effect — including the appended user message itself. That transaction only closes because the input is a finished `str`: a rule can substring-match the whole utterance and a rollback has a bounded blast radius. Pipecat has neither the staging nor the guaranteed-complete input, so this is the module that migration hurts most.

**Guideline.** When porting, first decide the commit boundary. A Pipecat `FrameProcessor` acts the instant it calls `push_frame()` (which is *enqueue-on-neighbor*, see `[[frame-processor]]`) — there is no rollback. Either replicate the stage/veto buffer inside one processor, or accept that a veto can only be expressed as "drop the frame before it is pushed", which loses cross-layer veto entirely.

## Technical Details

Real sizes (read 2026-08-25): `gateway/layers/` = **905 LOC** (`pipeline.py` 396, `discovery.py` 122, `status.py` 111, `rules/reconciliation.py` 78, `signals.py` 71, `context.py` 68, `actions.py` 29, `engine.py` 17, `__init__.py` 13); `gateway/rules/` = **301 LOC** (`engine.py` 184, `registry.py` 63, `check.py` 42, `__init__.py` 12). Total **1206**, not ~900 — the outline undercounts.

### Action vocabulary — 8 types, one dataclass
`gateway/schemas/actions.py` L17-81. `Action` is `@dataclass` with `type: ActionType` + `payload: dict`. Constructors are **functions, not classes**: `Continue()`, `Pass()`, `Filter(reason="")`, `Respond(text)`, `Inject(content)`, `Compact()`, `PreTool(tool_name, arguments=None, preamble="")`, `StageTransition(target_stage)`. Header comment L10-11: *"v0.6: Single action set for ALL rules (layers and inner). No separate LayerAction type."*

Flow-control precedence, `layers/pipeline.py` L42-51 verbatim:
```python
ACTION_PRIORITY = {
    "filter": 0, "respond": 1, "inject": 2,
    "stage_transition": 3, "compact": 3, "pre_tool": 3,
    "pass": 4, "continue": 4,
}
```

### Condition vocabulary — there is none
There is **no declarative condition DSL**. A "condition" is arbitrary Python inside a `@check`-decorated function. `rules/check.py` L16-42: `check(name, *, mode: Literal["sequential","parallel"]="sequential", priority: int=100, check_type: Literal["deterministic","llm"]="deterministic")` stamps `__check_name__/__check_mode__/__check_priority__/__check_type__` onto `fn`. The contract (L32-33): *"must accept `(messages, user_message, session)` and return a `list[Action]` or a single `Action`."* Lower priority = runs first.

### Discovery is filesystem convention
`layers/discovery.py` L24 `LAYER_PATTERN = re.compile(r"^(\d+)-(.+)$")`; scans `<agent>/layers/NN-name/rules/*.py`. Live Lina agent has four: `01-filler-filter`, `02-analyzer`, `03-orchestrator`, `04-committer`. If `layers/` is absent but `rules/` exists, `_auto_wrap_rules` (L98) creates a virtual layer `"99-auto-wrapped"`.

### Two-phase commit
`LayerPipeline._process_active` (L128-335). **Phase 1** (L178-248) calls `await engine.evaluate(session.messages, content, ctx)` per layer and only *stages* results. On `decision == "filter"` (L205-239) it walks `session.messages` backwards by **object identity** and deletes the exact `pipeline_user_message` (or just the `pipeline_user_block` `TextBlock` it appended into an existing tool-result turn), clears `_pending_stage_injection`, and returns. **Phase 2** (L254-316) replays staged actions in original layer/array order. `RuleEngine.evaluate` (L41-134) also arbitrates cross-phase: at most one `stage_transition` survives; lowest `@check` priority wins, ties go to the sequential phase, and losers lose *only* their transition action.

Layer engines are **fail-closed**: `__main__.py` L96-102, `LayerRuleEngine(ld.checks, fail_open=False)` — comment: *"deliberately fail-closed regardless of `config.fail_open` — safety/analyzer layer rules must never silently degrade to CONTINUE on exceptions."* The non-layered engine uses `config.fail_open` (default `True`).

`SharedLayerContext` (`layers/context.py`) is a per-turn proxy: `__getattr__`/`__setattr__` forward unknown names to the real `SessionState`, so `session.script_response = x` inside a rule persists across turns. `_OWN_FIELDS` (L35) is the non-proxied set.

### Real rule, verbatim (`agents/test-lina-gateway/layers/01-filler-filter/rules/korean_fillers.py` L53-75)
```python
@check("korean_filler_filter", mode="sequential", priority=10)
def filter_fillers(messages, user_message, session):
    """Filter filler words ONLY during agent streaming."""
    status = getattr(session, "pre_turn_status", None) or session.get_agent_status()
    if status not in ("generating", "tool_processing"):
        return Pass()
    if _is_filler_text(user_message):
        return Filter(
            reason=f"filler_word:{_normalize(user_message) or user_message.strip()}"
            f" | agent_status:{status}"
        )
    return Pass()
```
`KOREAN_FILLERS` (L16-19) = `["네","예","아","음","어","그","응","아아","음음","네네","예예","아하","흠","그래요"]`. `_SPECIALS_RE = re.compile(r"[^0-9a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ]+")` so `"흠..."`/`"네~"` normalize to bare fillers. Every `Filter` also writes a `Signal(timestamp, source_layer, reason, content, action_type)` to the append-only `SignalQueue` (`layers/signals.py`) that any later layer can read via `get_recent(seconds, source_layer=None)`.

### **Does a rule inspect a COMPLETE user turn? Yes — provably.**
1. `LayerPipeline.process(self, session_id: str, content: str, session: SessionState, *, user_message_appended: bool = False)` — `content` is a `str`, never an iterator or a stream.
2. `server/websocket.py` L288-292, verbatim: *"Incremental ASR: keep only the latest hypothesis. A partial may stop an in-flight response promptly, but only after the same filler/policy gate used by explicit final frames has authorized the interruption. **Rules/LLMs/tools still do not see incomplete text.**"*
3. Every branch of the `msg.type == "partial_transcript"` handler (L293-317) ends in `continue`. Partials land in `self._partial_transcripts[session_id]` and never reach `_replace_active_task(...)` (L374) — the sole path to `_message_handler`.
4. `msg.type == "user_message"` (L327) is *"the client's explicit final ASR frame"*; `content = msg.content or buffered or ""` (L343) is a whole utterance, with an empty frame acting purely as an end-of-utterance marker.
5. Rules exploit this: `end_signal.py` does `kw in lower` over `user_message.lower().strip()` for 5 categories × ~10 Korean/English keywords, gated by `STAGE_SIGNALS` (L54-62) per stage. `_detect_natural_close` reaches back over `messages[-4:]`. Both are whole-string operations that would fire on a prefix mid-stream.

- **Migration angle:** This module has **no Pipecat equivalent and must survive as custom `FrameProcessor`s**. (a) Placement is forced: the processor must sit *after* `LLMUserAggregator` (see `[[llm-service-context]]`) so it receives an aggregated turn, and must ignore `InterimTranscriptionFrame` — otherwise every keyword rule fires on a prefix. (b) Filler-filtering collides head-on with Pipecat's VAD/turn strategy (`[[vad-silero]]`, `[[endpointing-turn-boundary]]`): boson filters `"네"` *by content and by `pre_turn_status`*, while `VADUserTurnStartStrategy` interrupts *by energy*, content-blind — so `korean_filler_filter` must be re-expressed as a processor upstream of interruption broadcast, or barge-in regresses. (c) The Phase-1/Phase-2 transaction cannot be spread across processors; `push_frame` is irreversible, so all layers must collapse into **one** processor holding the staged list. (d) `Respond(text)` maps to pushing a `TTSSpeakFrame` and swallowing the frame instead of forwarding to the LLM service. (e) `SharedLayerContext` proxying maps to nothing — Pipecat has no per-session mutable namespace; it becomes processor instance state or `FlowManager.state`.

## Citation
boson-agent (private, `Lina_2026/boson-agent-dev/boson-agent`), branch `lina-new-dental-dev`, commit `0a5e8ced2fd9e631dcbbe8c5f4adb68b89a4fafb`, 2026-08-20. Paths: `packages/gateway/gateway/{layers,rules,schemas}/`, `packages/gateway/gateway/server/websocket.py`, `agents/test-lina-gateway/layers/`. Pipecat comparison read against `pipecat-ai/pipecat` commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`, read 2026-08-25.

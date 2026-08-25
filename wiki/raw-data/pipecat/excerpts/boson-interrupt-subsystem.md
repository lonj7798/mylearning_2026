# boson-agent `gateway/interrupt/` — barge-in decided from text, never from audio (581 LOC)

<!-- slug: boson-interrupt-subsystem · type: boson · source: packages/gateway/gateway/interrupt/ (cancellation.py 187, handler.py 151, policy.py 116, detector.py 82, fillers.py 40, __init__.py 5) -->

**Core Insight.** boson cannot barge in until a *transcript string exists*. Every interruption decision in the codebase — `PartialDetector.is_partial`, `WordFilterPolicy.evaluate`, `fillers.is_filler`, `InterruptionGate.allows` — takes `text: str` as its primary argument. There is no audio path, no energy threshold, no VAD. Floor-yield latency is therefore bounded below by the client's ASR partial-emission interval, not by speech onset. This is the single largest behavioral gap between boson today and any Pipecat voice bot.

**Guideline.** Treat the subsystem as two separable halves. The *timing* half (`DurationPolicy(min_ms=500)`, `silence_timeout_ms=2000`, `PartialDetector.timing_threshold_ms`) is what Pipecat VAD replaces outright. The *semantic* half (`WordFilterPolicy`, the agent-registered `fillers` callback, `_TOOL_CANCEL_HANDLERS`, the tool_use/tool_result repair) is Korean-language and API-contract business logic Pipecat has no equivalent for — port it, do not delete it.

## Technical Details

### The detector reasons over TEXT — verified, quoted

- `detector.py:22` `class PartialDetector` — docstring: *"Detects partial transcript updates using content + timing."* Constructor (L34): `overlap_chars: int = 10, timing_threshold_ms: float = 1000, silence_timeout_ms: float = 2000`.
- The core predicate, verbatim (L44-51):
  ```python
  def is_partial(self, text: str, previous: str | None) -> bool:
      """Check if text is a partial update of previous via content overlap."""
      if not previous:
          return False
      compare_len = min(self.overlap_chars, len(previous))
      if compare_len == 0:
          return False
      return text[:compare_len] == previous[:compare_len]
  ```
- `detect(text, previous, elapsed_ms)` (L57-68) returns `DetectResult.PARTIAL` if the first ≤10 **characters** match, else `PARTIAL` if `elapsed_ms < 1000`, else `NEW_MESSAGE`. `should_finalize(elapsed_since_last_ms)` is `>= 2000`.
- Nothing in this file touches audio. Its only import is `basement.schemas.message_schema.Message`. The calling spec header reads `CALLED BY: interrupt/handler (on incoming text while partial_buffer active)`.

### Expected-but-absent: the detector is dead code in production

- `PartialDetector` is constructed exactly once — `bootstrap.py:316` `core.set_partial_detector(PartialDetector())` — stored by `core.py:175` `def set_partial_detector(self, d): self._partial_detector = d`, and **`self._partial_detector` is never read anywhere**. The only other hits are `core.py:70` (init to `None`) and a bootstrap calling-spec comment. Every remaining reference is in `packages/gateway/tests/`.
- The sibling `PartialBuffer` dataclass (`schemas/session.py:21`) is likewise unused; `tests/test_cancellation.py:90` calls it *"dead overlap-dedup path; the PartialBuffer class itself is retained for…"*.
- The **real** partial handling is the `_partial_transcripts` dict + silence timer in `server/websocket.py:288-317, 616-735`: keep the latest hypothesis (`self._partial_transcripts[msg.session_id] = msg.content`) and let the 2 s timer finalize it. No overlap dedup runs at all.

### What actually decides barge-in today

`server/interruption.py:36` `InterruptionGate.allows(session_id, content) -> bool`, in order:

1. No live task (`_active_tasks.get(sid) is None or .done()`) → **allow** unconditionally.
2. Read `session.status_tracker.get_status().value`, falling back to `"idle"` on any exception.
3. `if fillers.is_filler(content, status): return False` — logged as `"Filler %r ignored (status=%s, session=%s)"`.
4. If no `should_interrupt` callback → allow.
5. `elapsed_ms = max(0.0, (time.monotonic() - started_at) * 1000)`, then `should_interrupt(session_id, content, elapsed_ms)`. **Any exception here allows the interruption** — `"Barge-in decision failed for session %s; allowing interruption"`.
6. On allow, set `session._pending_bargein_approved = True`.

`preapprove_explicit(session_id)` (L80) sets the same flag with no policy check — the `interrupt` client frame bypasses filler and policy gates entirely (`websocket.py:319-326`).

### Policy rules (`policy.py`)

- `BargeInResult.ALLOW | IGNORE` with `.is_bargein`. Four evaluators, all `evaluate(text, *, elapsed_ms=0)`.
- `AlwaysPolicy` → always ALLOW.
- `DurationPolicy(min_ms=500)` → ALLOW iff `elapsed_ms >= min_ms`.
- `WordFilterPolicy(ignore_words=["hmm","uh","um","ah"], max_chars=3)` → IGNORE on exact lowercase match **or** `len(text.strip().lower()) <= 3`.
- `CompositePolicy(policies, mode="all"|"any")` → returns `IGNORE` for an empty policy list.
- `default_bargein_policy()` (L108) = `CompositePolicy([DurationPolicy(min_ms=500), WordFilterPolicy(...)], mode="all")`, wired at `bootstrap.py:315`, reached via `core.py:165 should_interrupt()` → `InterruptHandler.check_barge_in` → `policy.evaluate(...).is_bargein`.
- Note `max_chars=3` counts **characters**, applied to Korean: "네", "아니요" (3 chars) are silently ignored.

### Fillers are agent-supplied, not gateway logic

- `fillers.py` (40 L) is a module-global registry: `FillerCheck = Callable[[str, str], bool]` over `(text, agent_status)`, with `set_filler_check` / `get_filler_check` / `is_filler` / `clear_filler_check`.
- Docstring: *"The gateway has zero language knowledge — it only calls the registered callback."* `is_filler` returns `False` when unset.
- Lina's implementation lives outside the package at `agents/test-lina-gateway/layers/01-filler-filter/rules/korean_fillers.py`.

### Cancellation path — cooperative flag, not asyncio

- `CancellationFlag` (`cancellation.py:21`): `set()` / `reset()` / `check()` (raises `CancellationError`), plus v0.7.4 `pending_tool_cancel: tuple[str, str] | None` with `stash_tool_cancel(tool_use_id, tool_name)` / `consume_tool_cancel()`. `reset()` deliberately does **not** clear `pending_tool_cancel`.
- Three cancel points: `cancel_before_llm()` → `CancelResult(discard_pending=True, history_entries=[])`; `cancel_during_streaming(partial_text)` → one assistant `Message` of `f"{partial_text}[interrupted-by-user]"`; `cancel_during_tool(tool_name, arguments)` → a per-tool handler from `_TOOL_CANCEL_HANDLERS`, else the default pair `user:"[tool call canceled, user interrupted: {tool_name}]"` + `assistant:"[interrupted-by-user]"`.
- Docstring is explicit: *"Cooperative — tool runs to completion, then flag is checked."*
- Tags are overridable via `set_interrupt_tags(interrupted, tool_canceled, barge_in_prefix)`; default `barge_in_prefix` is `"[barge-in] "`, and the docstring's own example uses Korean (`"[고객 끼어듦]"`).

### History reconciliation (`handler.py`)

- `InterruptHandler.handle_barge_in(session, content, partial_agent_output, tagging=True, tool_cancel=None)` (L78) pauses an active script (`ScriptEngine.pause_for_interrupt`, sets `script_state["objection_active"] = True`), sets the flag, prefixes the user text.
- When a tool was cancelled mid-flight it emits the next user turn as `[ToolResultBlock(tool_use_id, content=f"canceled: {tname}", is_error=False), TextBlock(text=user_text)]` for **every** unanswered `tool_use` found by `_collect_unanswered_tool_uses` (L21) — preserving strict assistant→user alternation *"(some models reject consecutive same-role messages)"*.

- **Migration angle:** Pipecat replaces the timing half and the plumbing, and collides head-on with `handler.py`. **Replaced:** `DurationPolicy.min_ms=500` and the server's `silence_timeout_ms=2000` → `VADParams.start_secs`/`stop_secs` (see `[[vad-silero]]`); the `_partial_transcripts` + silence-timer path → `src/pipecat/turns/user_start/` strategies. The exact analogue of boson's text-only detector already exists upstream — `TranscriptionUserTurnStartStrategy` (`turns/user_start/transcription_user_turn_start_strategy.py:14`, `__init__(self, *, use_interim: bool = True)`) — so boson's current behavior is a *supported Pipecat configuration*: land text-triggered first, add `VADUserTurnStartStrategy` as phase two. `WordFilterPolicy`'s "ignore ≤3 chars" maps onto `MinWordsUserTurnStartStrategy(min_words=…)` (`min_words_user_turn_start_strategy.py:22`) but **not faithfully** — Pipecat counts words, boson counts characters, and Korean backchannels are 1-word/2-char. **Not replaced:** the `fillers` registry (no Pipecat hook for an agent-registered language-specific ignore callback — subclass `BaseUserTurnStartStrategy`), `_TOOL_CANCEL_HANDLERS`, and `handle_barge_in`'s tool_use/tool_result repair — `InterruptionFrame` (`frames/frames.py:1142`) truncates the bot turn but never synthesizes `ToolResultBlock`s to keep the Anthropic message contract valid. **Delete outright:** `detector.py` and `PartialBuffer`, already dead. Finally, `_pending_bargein_approved` / `_pending_partial` are boson-side session flags with no frame equivalent; they become processor state or are dropped.

## Citation

boson-agent (private, Lina TMR), `packages/gateway` version `0.3.0`, commit `0a5e8ced2fd9e631dcbbe8c5f4adb68b89a4fafb` (2026-08-20), read 2026-08-25 at `/Users/jaewon/mywork_2026/Lina_2026/boson-agent-dev/boson-agent/packages/gateway/gateway/interrupt/`. Pipecat identifiers verified at `pipecat-ai/pipecat` commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (2026-08-25).

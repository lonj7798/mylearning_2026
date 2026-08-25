# boson-agent Compaction & SessionState vs Pipecat's LLMContextSummarizer
<!-- slug: boson-compact-session · type: boson · source: packages/gateway/gateway/compact/ (393 LOC) + gateway/session/ (289 LOC) + gateway/schemas/session.py -->

**Core Insight.** Both systems solved the same problem — summarize old history, keep a recent tail, splice `[system?] + [summary] + [tail]` — and arrived at nearly identical designs. The divergence is *where the answer lands*. boson runs summarization as a detached `asyncio.create_task` that writes `session.pending_compact` and **applies it at the top of the next turn**, so no turn ever waits on it. Pipecat's `LLMContextSummarizer` applies the result the moment a `LLMContextSummaryResultFrame` arrives, mid-pipeline. That makes compaction the *least* painful part of the migration — and the `SessionState` object it hangs off is the *most* painful, because Pipecat has no equivalent at all.

**Guideline.** Port the compaction policy to `LLMAutoContextSummarizationConfig` and delete boson's `AsyncCompactPipeline`. Do **not** try to port `SessionState` — inventory every field on it and re-home each one individually, because ~15 of them are boson-specific turn scratch that Pipecat expresses as frames.

## Technical Details

Real sizes: `gateway/compact/` = **393 LOC** (`pipeline.py` 211, `hooks.py` 85, `strategy.py` 84, `__init__.py` 13); `gateway/session/` = **289 LOC** (`history.py` 166, `store.py` 110, `__init__.py` 13). The outline gave no number for these; both are smaller than `layers/`.

### boson: policy and lifecycle
`CompactConfig` (`schemas/config.py` L18-41), pydantic `extra="forbid"`: `enabled=True`, `threshold_messages=30 (ge=5)`, `provider="openai"`, `model="gpt-5.4-mini"`, `temperature=0.3`, `keep_recent=10 (ge=2)`, plus a `validate_compaction_window` validator rejecting `keep_recent > threshold_messages`. Trigger is **message-count only — there is no token-based trigger anywhere in boson.**

`AsyncCompactPipeline` (`compact/pipeline.py`):
- `_compaction_window(session) -> tuple[list, int]` (L85) snapshots `prefix_len = max(0, len(session.messages) - keep_recent)`.
- `should_compact(session)` (L107) requires `len(messages) > threshold_messages` **and** `_contains_uncompacted_message(prefix)` — i.e. a prefix consisting only of a `"[Compact Summary]\n"`-prefixed marker does not re-summarize itself.
- `trigger(session) -> bool` (L114) guards on `session.compact_in_progress`, then `asyncio.create_task(self._compact_task(session))` and returns.
- `_compact_task` (L129) re-snapshots the window (*"the shared history may shrink after `trigger()` has scheduled this task"*), runs the pre-hook, `await self.strategy.summarize(messages_to_compact, session.system_prompt)`, runs the post-hook, and stores `session.pending_compact = {"summary", "keep_recent", "compacted_prefix_len"}`. Every failure path sets `pending_compact = None` and the `finally` clears `compact_in_progress`.
- `apply_pending(session, shared_history) -> bool` (L186) is called by `bootstrap.build_layered_handler` **before the layers run** (`bootstrap.py` L455-458: *"Apply pending compact BEFORE layers"*), and discards a result whose effective prefix is 0.

`LLMCompactStrategy.summarize(messages, system_prompt) -> str` (`compact/strategy.py` L60) flattens history to `f"{msg.role}: {msg.content}"` lines in a single `user` message and streams `TextDelta`s from `basement.llm.registry.get_provider(config)`. Provider is built once and cached (`issue #36`).

Hooks (`compact/hooks.py`): module-level `set_pre_compact_hook` / `set_post_compact_hook` / `get_*` / `clear_compact_hooks`, typed `PreCompactHook = Callable[[SessionState, list[Message]], Union[list[Message], None, Awaitable[...]]]` and `PostCompactHook = Callable[[SessionState, str], Union[str, None, Awaitable[...]]]`. A hook raising is logged and swallowed; compact proceeds un-mutated.

`SharedHistory.swap_compact(compact_summary, keep_recent=10, compacted_prefix_len=None)` (`session/history.py` L82) builds `Message(role="user", content=f"[Compact Summary]\n{compact_summary}")` + tail, appends `<system-reminder>Active skill: …</system-reminder>` and `<system-reminder>Active stage: …</system-reminder>` when set, then does `session.messages.clear(); session.messages.extend(new_messages)` — **in-place, because `ContextManager._messages` is the same list object** (L67-69: *"Intentional direct assignment: share the same list object"*). Two named bugs are fixed in comments: **H2** (anchor the cut at the T0 snapshot so messages appended during in-flight summarization survive) and **H3** (`_safe_window_start`, L29, advances the window past leading orphaned `tool_result` messages whose `tool_use` was summarized away).

### boson: the session state model
`SessionState` (`schemas/session.py` L29-68) is a plain `@dataclass`, keyed by `session_id`, holding `messages`, `system_prompt`, `active_skill`, `active_stage`, `pending_compact`, `compact_in_progress`, `context_manager`, `conversation_api`, `cancellation_flag: CancellationFlag`, `status_tracker: AgentStatusTracker`, `history_lock: asyncio.Lock`, `signal_queue`, plus 8 underscore-prefixed per-turn scratch fields (`_pending_partial`, `_pending_bargein_approved`, `_bargein_user_appended`, `_pipeline_appended`, `_pending_stage_injection`, `_customer_context`, `_compact_ctx`, `_pending_partial_started_at`). L57-59 is explicit: *"Agent-rule state (`sentiment_history`, `script_response`, `end_signals`, …) is intentionally NOT declared here and stays dynamically attached."*

`SessionStore` (`session/store.py`) is a dict + idle TTL: `DEFAULT_SESSION_TTL_SECONDS = 1800.0`; `create()` opportunistically calls `sweep_expired()`; `get()` clears any disconnect mark (*"Access counts as liveness"*); `mark_disconnected()` starts the clock. Retention exists so the debug UI can auto-reconnect with the same `session_id`.

### Pipecat comparison — `processors/aggregators/llm_context_summarizer.py` (468 LOC) + `utils/context/llm_context_summarization.py` (641 LOC)

| Concern | boson | Pipecat |
|---|---|---|
| Trigger | messages only, `threshold_messages=30` | `_should_summarize()` L200: **either** `max_context_tokens=8000` **or** `max_unsummarized_messages=20` |
| Token estimate | none | `CHARS_PER_TOKEN = 4`, `TOKEN_OVERHEAD_PER_MESSAGE = 10`, `IMAGE_TOKEN_ESTIMATE = 500` |
| Tail kept | `keep_recent=10` | `min_messages_after_summary=4`, `target_context_tokens=6000` |
| Result shape | `[Compact Summary]\n…` as a **user** message + reminders | `summary_message_template = "Conversation summary: {summary}"`, also a **user** message — `_apply_summary` L436-439 comment: *"the summary is context provided *to* the assistant, not something the assistant said"* |
| System prompt | `session.system_prompt` passed to the summarizer, not spliced | `_apply_summary` L425-431 preserves `messages[0]` **only if** `role == "system"`; other system messages are treated as mid-conversation injections |
| Tool-pair safety | `_safe_window_start` drops leading orphan `tool_result`s **forward** | `_get_earliest_function_call_not_resolved_in_range` pulls `summary_end` **back** before an unresolved call (`get_messages_to_summarize` L552-571) |
| Concurrency guard | `session.compact_in_progress` bool | `_summarization_in_progress` + `_pending_summary_request_id: str` (uuid4) matched against `LLMContextSummaryResultFrame.request_id` |
| Application | deferred to next turn via `pending_compact` | immediate, in `_handle_summary_result` → `_apply_summary` → `context.set_messages(new_messages)` |
| Interruption | none (task runs to completion) | `_handle_interruption()` L184 clears `_summarization_in_progress` but **keeps** `_pending_summary_request_id`, because the result frame is uninterruptible |
| Dedicated cheap model | `CompactConfig.provider/model` | `LLMContextSummaryConfig.llm: Optional[LLMService]` → `_generate_summary_with_dedicated_llm` under `asyncio.wait_for(timeout=DEFAULT_SUMMARIZATION_TIMEOUT=120.0)` |
| Hooks | `set_pre_compact_hook` / `set_post_compact_hook` | **no equivalent** — the closest is the `on_summary_applied` event with `SummaryAppliedEvent(original_message_count, new_message_count, summarized_message_count, preserved_message_count)`, which is observability-only and cannot mutate input or output |

Note: `LLMContextSummarizer` extends `BaseObject`, **not** `FrameProcessor` — it has `process_frame(self, frame: Frame)` (single arg, no `direction`) and is driven by an aggregator, not linked into the pipeline itself.

- **Migration angle:** `gateway/compact/` is **replaced wholesale** by `LLMContextSummarizer` + `LLMAutoContextSummarizationConfig`; port `threshold_messages=30 → max_unsummarized_messages=30` and `keep_recent=10 → min_messages_after_summary=10`, and add the token trigger boson never had. Two real losses to plan for: (1) **pre/post compact hooks have no Pipecat equivalent** — any agent that strips noisy tool blocks pre-summary or appends extracted structured data post-summary must instead subclass `LLMContextSummarizer` or use a dedicated summarization `LLMService`; (2) the `<system-reminder>Active stage: …</system-reminder>` re-injection in `swap_compact` is boson stage-machine coupling that Pipecat expresses via `NodeConfig.role_message` / `ContextStrategy` in `[[boson-script-engine]]`'s Flows counterpart — it must be re-attached explicitly or stage identity is silently lost at every compaction. `gateway/session/` splits three ways: `session.messages` → `LLMContext` (`[[llm-service-context]]`), `SessionStore` TTL → the runner's per-connection worker lifecycle (`[[deployment-scaling]]`), and `status_tracker`/`cancellation_flag`/`_pending_bargein_*` → frames (`[[interruption-cascade]]`), not fields. `SharedHistory`'s shared-list aliasing (`ctx._messages = session.messages`) is exactly the pattern Pipecat forbids — `LLMContext.set_messages()` rebinds a new list, so any boson code holding a stale reference breaks.

## Citation
boson-agent (private), branch `lina-new-dental-dev`, commit `0a5e8ced2fd9e631dcbbe8c5f4adb68b89a4fafb`, 2026-08-20. Paths: `packages/gateway/gateway/compact/{pipeline,strategy,hooks}.py`, `packages/gateway/gateway/session/{history,store}.py`, `packages/gateway/gateway/schemas/{session,config}.py`. Pipecat: `src/pipecat/processors/aggregators/llm_context_summarizer.py`, `src/pipecat/utils/context/llm_context_summarization.py` @ `pipecat-ai/pipecat` commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`, read 2026-08-25.

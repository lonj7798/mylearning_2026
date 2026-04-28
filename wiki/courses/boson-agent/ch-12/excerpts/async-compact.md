---
calling_spec:
  purpose: "Full walkthrough of AsyncCompactPipeline and CompactStrategy"
  parent: "[[../read.md]]"
  phase: read
  chapter: ch-12
  course: boson-agent
---

# Async Compact Pipeline — Full Walkthrough

> Sub-page of [[../read.md]]. Covers `gateway/compact/pipeline.py` and
> `gateway/compact/strategy.py`.

---

## The Problem Compaction Solves

A long-running conversation session accumulates messages without bound. LLMs
have finite context windows; sending thousands of turns of history on every
request is expensive and eventually impossible. Compaction replaces old message
history with an LLM-generated summary, keeping the window manageable while
preserving key facts.

The design challenge: summarization is itself an LLM call that takes hundreds
of milliseconds. Blocking the current turn on compaction would make the agent
feel slow. The solution is to run compaction **asynchronously in the background**
and apply its result on the *next* turn — the current turn proceeds without delay.

---

## AsyncCompactPipeline

```python
# boson-agent/packages/gateway/gateway/compact/pipeline.py, lines 23-102

class AsyncCompactPipeline:
    """Manages async background compaction of session history.

    Trigger compaction when message count exceeds the configured threshold.
    The summarization runs as a background asyncio task so it does not block
    the main request path. The result is stored in session.pending_compact
    and applied on the next turn via apply_pending().
    """

    def __init__(self, config: CompactConfig) -> None:
        self._config = config
        self._strategy: CompactStrategy | None = None

    @property
    def strategy(self) -> CompactStrategy:
        """Lazy-initialise the LLM strategy from config."""
        if self._strategy is None:
            self._strategy = LLMCompactStrategy(
                provider=self._config.provider,
                model=self._config.model,
                temperature=self._config.temperature,
                system_prompt=self._config.system_prompt,
            )
        return self._strategy

    def should_compact(self, session: SessionState) -> bool:
        """Return True when the session message count exceeds the threshold."""
        return len(session.messages) > self._config.threshold_messages

    async def trigger(self, session: SessionState) -> bool:
        """Start a background compact task if conditions are met.

        Returns True when a new task was started, False otherwise (already
        in-progress or threshold not reached).
        """
        if session.compact_in_progress:
            return False
        if not self.should_compact(session):
            return False

        session.compact_in_progress = True
        asyncio.create_task(self._compact_task(session))
        return True
```

**Line-by-line:**

- **`strategy` property (lines 36-45):** Lazy initialization — the
  `LLMCompactStrategy` is not created until the first `trigger()` call that
  actually needs it. This avoids paying the initialization cost if compaction
  never fires (short conversations).

- **`should_compact()` (lines 47-49):** A single condition: `len(session.messages)
  > threshold_messages`. No age-based or token-based heuristics — pure message
  count. Simple to reason about, simple to configure.

- **`trigger()` (lines 51-65):** Two guards before launching: `compact_in_progress`
  (prevents a second compaction task from starting while one is already running)
  and `should_compact()` (prevents premature compaction). When both pass,
  `session.compact_in_progress = True` is set *before* `asyncio.create_task()` —
  this prevents a race where a second turn fires between the flag check and the
  task creation on the asyncio event loop.

**Notice:** `asyncio.create_task()` is fire-and-forget from the perspective of
`trigger()`. The task runs concurrently with whatever comes next in the event
loop. No `await`, no callback. The result is communicated back via
`session.pending_compact` — a shared mutable field that `apply_pending()` checks
at the start of the next turn.

---

## The Background Task

```python
# boson-agent/packages/gateway/gateway/compact/pipeline.py, lines 67-87

    async def _compact_task(self, session: SessionState) -> None:
        """Background task: summarise old messages and queue the result."""
        try:
            # Snapshot messages to compact (all except keep_recent)
            keep = self._config.keep_recent
            messages_to_compact = (
                list(session.messages[:-keep]) if keep
                else list(session.messages)
            )

            summary = await self.strategy.summarize(
                messages_to_compact,
                session.system_prompt,
            )

            session.pending_compact = {
                "summary": summary,
                "keep_recent": keep,
            }
        except Exception as exc:
            logger.error("compact_task failed: %s", exc)
            session.pending_compact = None
        finally:
            session.compact_in_progress = False
```

**Line-by-line:**

- **Lines 72-74** `messages_to_compact = list(session.messages[:-keep])` —
  takes a *snapshot* of the messages to summarize at the moment the task runs.
  The `list()` call makes a shallow copy, so subsequent turns appending to
  `session.messages` do not affect the snapshot. The last `keep_recent` messages
  are excluded from summarization — they will be preserved verbatim in the final
  swap.

- **Lines 76-79** Calls `strategy.summarize()`. This is the slow LLM call.
  The event loop is free to process other sessions or respond to other I/O while
  this awaits. From the session's perspective, this turn's agent response is
  streaming to the client at the same time the background task is running.

- **Lines 81-84** On success, sets `session.pending_compact` to a dict with the
  summary and the `keep_recent` count. This is the signal that `apply_pending()`
  checks.

- **Lines 85-87** On failure, sets `session.pending_compact = None` (a no-op
  for `apply_pending`) and releases the lock via `finally`. If compaction fails,
  the conversation continues normally — the only consequence is that the history
  stays long.

**Notice:** The `finally: session.compact_in_progress = False` is in a `finally`
block, not an `except` block. Whether the summarization succeeds or fails, the
lock is always released. Without this, a single failed compaction would
permanently prevent future compaction attempts for that session.

---

## Applying the Pending Compact

```python
# boson-agent/packages/gateway/gateway/compact/pipeline.py, lines 89-102

    def apply_pending(
        self, session: SessionState, shared_history: SharedHistory
    ) -> bool:
        """Apply a queued compact result if one is available.

        Returns True when the swap was performed, False when nothing was pending.
        """
        if not session.pending_compact:
            return False

        summary = session.pending_compact["summary"]
        keep_recent = session.pending_compact.get(
            "keep_recent", self._config.keep_recent
        )
        shared_history.swap_compact(summary, keep_recent=keep_recent)
        return True
```

This is called at the **very start** of `handle_message()`, before any rules
run, before the user message is appended:

```python
# boson-agent/packages/gateway/gateway/core.py, lines 119-121

# 3. Apply any pending compact from previous turn
if self._compact_pipeline is not None:
    self._compact_pipeline.apply_pending(session, shared_history)
```

The ordering matters: apply compact first, then process the new message.
This means the new message is added to an already-compacted history, keeping
the window small from the very start of the turn.

`shared_history.swap_compact(summary, keep_recent=keep_recent)` replaces
`session.messages[:-keep_recent]` with a single synthesized `[user]` message
containing the summary. The last `keep_recent` messages are preserved intact.
After the swap, `session.pending_compact` is implicitly stale — a new compaction
cycle can start if the threshold is exceeded again.

**Notice:** `apply_pending()` does NOT clear `session.pending_compact` after
applying it. This is safe because `trigger()` checks `compact_in_progress`
(which is already `False` at this point), not `pending_compact`. After `swap_compact`
shrinks the history, `should_compact()` will return `False` until messages
accumulate again — no double-apply is possible in practice. But a defensive
implementation might explicitly set `session.pending_compact = None` here.

---

## LLMCompactStrategy

```python
# boson-agent/packages/gateway/gateway/compact/strategy.py, lines 36-84

class LLMCompactStrategy(CompactStrategy):
    """Summarizes conversation history via an LLM provider."""

    def __init__(self, provider: str, model: str, temperature: float = 0.3,
                 system_prompt: str = "Summarize this conversation concisely, "
                 "preserving key facts, decisions, and context.") -> None:
        self._provider_name = provider
        self._model = model
        self._temperature = temperature
        self._system_prompt = system_prompt

    def _build_summary_messages(self, messages: list[Message]) -> list[Message]:
        """Format messages as 'role: content' lines for the summarization LLM."""
        lines: list[str] = []
        for msg in messages:
            if isinstance(msg.content, str):
                lines.append(f"{msg.role}: {msg.content}")
            elif isinstance(msg.content, list):
                text_parts = [
                    block.text
                    for block in msg.content
                    if isinstance(block, TextBlock)
                ]
                if text_parts:
                    lines.append(f"{msg.role}: {' '.join(text_parts)}")

        conversation_text = "\n".join(lines)
        return [Message(role="user", content=conversation_text)]

    async def summarize(self, messages: list[Message], system_prompt: str) -> str:
        """Call the LLM to produce a compact summary of messages."""
        config = LLMConfig(
            provider=self._provider_name, model=self._model,
            temperature=self._temperature,
        )
        provider = get_provider(config)
        summary_messages = self._build_summary_messages(messages)
        chunks: list[str] = []

        async for event in await provider.stream(
            messages=summary_messages,
            system=self._system_prompt,
            tools=None,
        ):
            if isinstance(event, TextDelta):
                chunks.append(event.text)

        return "".join(chunks)
```

**Key design points:**

- **`_build_summary_messages()` (lines 47-63):** Converts the full message
  history into a flat text transcript (`"user: …\nassistant: …"`), then wraps
  it in a single `[user]` message. The summarization LLM receives the entire
  history as one big user message. This is a pragmatic simplification — it
  avoids the complexity of reconstructing a multi-turn conversation for a
  provider that may have different API constraints.

- **`temperature=0.3` (line 40):** Lower than typical generation temperatures.
  Summaries should be factual and consistent, not creative. The same history
  should produce roughly the same summary on repeated calls.

- **`tools=None` (line 77):** The summarization LLM has no tools. It only reads
  the transcript and writes a summary. This is intentional — you do not want the
  summarization process to trigger tool calls.

**Notice:** `LLMCompactStrategy` calls `get_provider(config)` inside `summarize()`,
not in `__init__`. This means a different LLM provider and model can be used for
compaction than for the main agent. A common production pattern: use a fast,
cheap model (e.g., `claude-haiku`) for compaction and a more capable model for
the main agent. The compaction model does not need to be smart — just concise.

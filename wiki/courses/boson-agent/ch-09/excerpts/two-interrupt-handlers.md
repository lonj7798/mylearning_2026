---
chapter: ch-09
course: boson-agent
phase: read
excerpt_for: two-interrupt-handlers
created_at: "2026-04-19"
sources_cited:
  - "boson-agent/packages/basement/basement/loop/interrupt.py"
  - "boson-agent/packages/gateway/gateway/interrupt/handler.py"
  - "boson-agent/packages/gateway/gateway/interrupt/cancellation.py"
  - "boson-agent/packages/gateway/gateway/core.py"
---

# Excerpt: Two InterruptHandlers — Basement SIGINT vs Gateway Barge-in

This sub-page carries the full line-by-line contrast of the two classes
named `InterruptHandler` in the codebase. They share a name and the
conceptual role of "stop the in-flight agent response when the user speaks"
but they solve that problem at completely different layers and by completely
different mechanisms.

---

## Basement: `loop/interrupt.py` — SIGINT via OS Signal

```python
# boson-agent/packages/basement/basement/loop/interrupt.py, lines 16-52

class InterruptHandler:
    """Handle user interrupts during streaming.

    setup() registers SIGINT handler.
    check is_interrupted to see if user pressed Ctrl+C.
    reset() clears state for next turn.
    """

    def __init__(self):
        self._interrupted = asyncio.Event()
        self._original_handler = None

    def setup(self) -> None:
        """Register SIGINT handler. Call once at startup."""
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, signum, frame):
        self._interrupted.set()

    @property
    def is_interrupted(self) -> bool:
        """Check if user triggered an interrupt."""
        return self._interrupted.is_set()

    def get_context_message(self) -> str:
        """Return message to add to conversation after interrupt."""
        return "[Response interrupted by user]"

    def reset(self) -> None:
        """Clear interrupt state for next turn."""
        self._interrupted.clear()

    def teardown(self) -> None:
        """Restore original SIGINT handler."""
        if self._original_handler:
            signal.signal(signal.SIGINT, self._original_handler)
```

**What this does mechanically:** `setup()` replaces the process-level SIGINT
handler with `_on_sigint`, which sets an `asyncio.Event`. The caller (the CLI
`__main__`) periodically polls `is_interrupted` between streaming chunks to
decide whether to abort the loop. `reset()` clears the Event for the next
turn. `teardown()` restores the original handler so the OS returns to normal
behaviour when the agent exits.

**Notice — asyncio.Event, not a plain boolean:** The `asyncio.Event` allows
coroutines to `await self._interrupted.wait()` if a blocking-wait design were
ever needed. In current usage it is only polled (`is_set()`), so the Event is
slightly over-engineered, but it makes the implementation safe if a future
caller wants to sleep until interrupted rather than spin-poll.

**Wiring status:** The calling spec says `CALLED BY: __main__`. Looking at
`agent_loop.py` — there is zero import of this class. It is used only by the
CLI demo `__main__`. The agent loop itself has no knowledge of
`InterruptHandler`. The Basement interrupt is a CLI concern.

**Substrate forcing function:** A terminal CLI session has exactly one user
input channel — the keyboard. Ctrl+C is a POSIX signal. There is no other
wire, no session_id, no JSON frame. The mechanism is therefore forced to be
OS-level.

---

## Gateway: `interrupt/handler.py` — Flag via Barge-in Detection

```python
# boson-agent/packages/gateway/gateway/interrupt/handler.py, lines 22-83

class InterruptHandler:
    """Static methods for interrupt handling. Extracted from core.py."""

    @staticmethod
    def reset_cancellation(session: SessionState) -> None:
        """Reset cancellation flag at start of new turn."""
        session.cancellation_flag.reset()

    @staticmethod
    def detect_and_handle_partial(
        session: SessionState,
        content: str,
        timestamp: float,
        detector: PartialDetector,
    ) -> bool:
        """Detect if message is partial update. Returns True if handled."""
        if session.partial_buffer is None:
            return False
        elapsed_ms = (timestamp - session.partial_buffer.timestamp) * 1000
        result = detector.detect(
            text=content,
            previous=session.partial_buffer.text,
            elapsed_ms=elapsed_ms,
        )
        if result == DetectResult.PARTIAL:
            replace_partial_in_history(
                session.messages,
                new_text=content,
                index=session.partial_buffer.message_index,
            )
            session.partial_buffer = PartialBuffer(
                text=content,
                timestamp=timestamp,
                message_index=session.partial_buffer.message_index,
            )
            return True
        return False

    @staticmethod
    def check_barge_in(
        content: str,
        policy: BargeInPolicy,
        elapsed_ms: float,
    ) -> bool:
        """Check if incoming message is barge-in. Returns True if so."""
        return policy.evaluate(content, elapsed_ms=elapsed_ms).is_bargein

    @staticmethod
    def handle_barge_in(
        session: SessionState,
        content: str,
        partial_agent_output: str,
        tagging: bool = True,
    ) -> None:
        """Handle barge-in: set flag, save partial output, add user message."""
        session.cancellation_flag.set()
        if partial_agent_output:
            result = cancel_during_streaming(partial_agent_output)
            session.messages.extend(result.history_entries)
        prefix = get_barge_in_prefix() if tagging else ""
        user_content = f"{prefix}{content}"
        session.messages.append(Message(role="user", content=user_content))
```

**What this does mechanically:** All four methods are static — no instance
state. `reset_cancellation` clears the per-session `CancellationFlag` at the
start of each turn. `detect_and_handle_partial` runs the `PartialDetector`
heuristic and replaces the in-progress partial message in `session.messages`
in-place without triggering a new agent turn. `check_barge_in` delegates to
whichever `BargeInPolicy` the operator configured. `handle_barge_in` does the
actual interruption: sets the flag, appends the partial assistant text with
`[interrupted-by-user]` tag, then appends the user's barge-in message with
the `[barge-in]` prefix.

**Notice — wiring status (critical carryforward from ch-01):** Grepping
`core.py` and `agent_loop.py` reveals that:

1. `InterruptHandler.reset_cancellation` IS called at `core.py` line 113 at
   the top of `handle_message`.
2. `InterruptHandler.detect_and_handle_partial` IS called at `core.py`
   lines 123–127.
3. **`InterruptHandler.check_barge_in` and `handle_barge_in` are NOT called
   anywhere in `core.py` or `websocket.py`.** The `_bargein_policy` and
   `_bargein_tagging` fields are set on `GatewayCore` via
   `set_bargein_policy()`, but `core.handle_message` never invokes
   `check_barge_in`. There is no code path that calls `handle_barge_in`
   during an active stream.

This is not a bug in the schemas — the data structures are correct. It is an
integration gap: the barge-in detection machinery (`check_barge_in` /
`handle_barge_in`) was designed and implemented but has not yet been wired
into the hot path inside `_process_message` or `handle_message`. The docs
describe the intended design; the code implements the building blocks; the
glue that calls those blocks during an active stream is absent. The ch-01
session that flagged "defined but not integrated" was correct and the gap
persists in the current code.

**Substrate forcing function:** A WebSocket server has many concurrent
sessions, each with its own message stream arriving as JSON frames. A
process signal is meaningless here — you cannot Ctrl+C session "abc" without
killing all sessions. The mechanism must therefore be per-session state
(`CancellationFlag` on `SessionState`) and must be set programmatically from
within the message-routing code, not from an OS signal handler.

---

## Side-by-Side Comparison

| Dimension | Basement `InterruptHandler` | Gateway `InterruptHandler` |
|---|---|---|
| Trigger source | OS SIGINT (Ctrl+C) | Incoming WebSocket `user_message` frame |
| State carrier | `asyncio.Event` on the instance | `CancellationFlag` on `SessionState` |
| Scope | Process-wide (all sessions) | Per-session |
| Caller | CLI `__main__` | `core.handle_message` (partially) |
| Polling point | CLI loop after each stream chunk | `core.py` line 171 before LLM call |
| Agent loop awareness | Not wired into `agent_loop.py` | Not wired into `agent_loop.py` |
| Reset point | Explicit `reset()` call | `reset_cancellation()` at turn start |
| Partial-text preservation | `get_context_message()` fixed string | `cancel_during_streaming()` with actual partial |
| Integration completeness | Complete for CLI use case | `reset` + `detect_partial` wired; `check_barge_in` / `handle_barge_in` not yet wired |

The invariant across both: neither handler reaches inside `run_agent_loop`.
Both rely on the loop being abandoned at a checkpoint outside the loop body —
either because the CLI main loop stops iterating, or because `core.py` returns
early before calling `run_agent_loop`. The agent loop itself is interrupt-blind.

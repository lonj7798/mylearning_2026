---
chapter: ch-09
course: boson-agent
phase: read
excerpt_for: cancel-propagation
created_at: "2026-04-19"
sources_cited:
  - "boson-agent/packages/gateway/gateway/interrupt/cancellation.py"
  - "boson-agent/packages/gateway/gateway/core.py"
  - "boson-agent/packages/basement/basement/loop/agent_loop.py"
---

# Excerpt: Cancel Propagation — CancellationFlag and the Three Interruption Points

---

## CancellationFlag: the shared mutable cell

```python
# boson-agent/packages/gateway/gateway/interrupt/cancellation.py, lines 21-46

class CancellationFlag:
    """Cooperative cancellation flag.

    Gateway sets the flag. Agent loop checks between iterations.
    Flag is reset at the start of each new turn.
    """

    def __init__(self) -> None:
        self._is_set: bool = False

    @property
    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        """Set the cancellation flag. Called by Gateway on barge-in."""
        self._is_set = True

    def reset(self) -> None:
        """Reset the flag. Called at start of new turn."""
        self._is_set = False

    def check(self) -> None:
        """Check flag. Raises CancellationError if set."""
        if self._is_set:
            raise CancellationError("Cancelled by user interruption")
```

This is a plain boolean wrapped in a class. The design is intentional: the
class name makes the intent explicit at every call site, and the `check()`
method raises `CancellationError` (a typed exception) rather than returning
`False`, which would require every caller to write an `if` branch. The
`asyncio.CancelledError` pattern was not used here because the cancellation is
cooperative and controlled by the gateway layer, not by asyncio's task
cancellation mechanism.

**Notice — `check()` is never called in `agent_loop.py`:** Grepping the
entire `agent_loop.py` for `cancellation_flag`, `check()`, `CancellationError`,
or any import from `gateway.interrupt` yields zero matches. The agent loop is
interrupt-blind. The only checkpoint in the current code is in `core.py` at
line 171, before `run_agent_loop` is called.

---

## The single checkpoint in core.py

```python
# boson-agent/packages/gateway/gateway/core.py, lines 108-176 (key section)

async def handle_message(self, session_id: str, content: str) -> AsyncIterator[str]:
    session = self._get_or_create_session(session_id)

    # v0.4: Reset cancellation flag at turn start
    InterruptHandler.reset_cancellation(session)

    # ... rule engine, action executor ...

    # v0.4: Check cancellation before LLM
    if session.cancellation_flag.is_set:
        return

    # 8 & 9. Build runtime and run agent loop
    runtime = self._build_agent_runtime(session, shared_history)
    async for event in run_agent_loop(runtime, content):
        # ... yield chunks to WebSocket ...
```

The flag is checked exactly once: after the rule engine has run but before
`run_agent_loop` is entered. If barge-in sets the flag during the rule engine
phase, the function returns and yields nothing. If barge-in sets the flag
during streaming (i.e., `run_agent_loop` is already executing), there is no
checkpoint inside the loop that reads the flag. The asyncio task cancellation
from `_cancel_active_task` in `websocket.py` is what actually terminates an
in-flight streaming turn.

---

## The three CancelResult factories

```python
# boson-agent/packages/gateway/gateway/interrupt/cancellation.py, lines 92-156

def cancel_before_llm() -> CancelResult:
    """Cancel before LLM call: discard pending, no trace."""
    return CancelResult(discard_pending=True, history_entries=[])


def cancel_during_streaming(partial_text: str) -> CancelResult:
    """Cancel during LLM streaming: save partial with tag."""
    tag = _TAGS["interrupted"]
    entry = Message(role="assistant", content=f"{partial_text}{tag}")
    return CancelResult(discard_pending=False, history_entries=[entry])


def cancel_during_tool(tool_name: str, arguments: dict) -> CancelResult:
    """Cancel during tool execution.

    Checks per-tool handler first, falls back to default behavior.
    NOTE: Cooperative — tool runs to completion, then flag is checked.
    """
    handler = _TOOL_CANCEL_HANDLERS.get(tool_name)
    if handler:
        return handler(tool_name, arguments)

    tool_msg = _TAGS["tool_canceled"].format(tool_name=tool_name)
    tool_cancel = Message(role="user", content=tool_msg)
    interrupted = Message(role="assistant", content=_TAGS["interrupted"])
    return CancelResult(
        discard_pending=False,
        history_entries=[tool_cancel, interrupted],
    )
```

Each factory returns a `CancelResult` with two fields: `discard_pending`
(whether to throw away the pending LLM request) and `history_entries` (what
messages to commit to `session.messages` to preserve the cancellation record).

- **Before LLM:** Nothing has been emitted yet, so discard everything silently.
- **During streaming:** The partial text is worth preserving — the LLM began
  answering. Append it with the `[interrupted-by-user]` tag so the next turn's
  LLM call knows an interruption occurred mid-sentence.
- **During tool:** The tool ran to completion (cooperative cancellation — the
  tool is not killed mid-execution). Append a record of both the cancellation
  and the interruption tag.

**Notice — `cancel_during_tool` is cooperative, not preemptive:** The comment
"Cooperative — tool runs to completion, then flag is checked" is load-bearing.
A tool call that takes 5 seconds will run all 5 seconds before the flag is
ever consulted. There is no mechanism to abort a running tool mid-execution.
This is a deliberate choice: tools may hold resources (open files, network
connections) and killing them mid-execution risks corruption. The tradeoff is
that long-running tools delay barge-in acknowledgment.

---

## Per-tool cancel handlers

```python
# boson-agent/packages/gateway/gateway/interrupt/cancellation.py, lines 104-132

_TOOL_CANCEL_HANDLERS: dict[str, ToolCancelHandler] = {}


def set_tool_cancel_handler(tool_name: str, handler: ToolCancelHandler) -> None:
    """Register a custom cancel handler for a specific tool."""
    _TOOL_CANCEL_HANDLERS[tool_name] = handler


def cancel_during_tool(tool_name: str, arguments: dict) -> CancelResult:
    handler = _TOOL_CANCEL_HANDLERS.get(tool_name)
    if handler:
        return handler(tool_name, arguments)
    # ... default behavior ...
```

The per-tool handler registry lets operators override the default "append two
messages" behavior for specific tools. The docstring example shows two
patterns: a retry handler (keep the result, add a user message requesting
retry) and a silent-ignore handler (`CancelResult(discard_pending=True,
history_entries=[])`). This extensibility point is important for tools where
the cancellation semantics differ from the default — for example, a tool that
fetches a disclosure form mid-call should not silently discard its result even
if the user interrupted; it should queue a retry.

**Notice — the tag system is process-global mutable state:** `_TAGS` is a
module-level dict. `set_interrupt_tags()` modifies it globally. If two agents
with different locale requirements were running in the same process (unlikely
with the current architecture but possible), they would overwrite each other's
tags. The current design assumes one agent configuration per process.

Connection to universal pattern: the three-point cancellation model (before
LLM / during streaming / during tool) maps directly onto the three observable
states of a turn in the agent loop. Any system that streams from an LLM and
executes tools must handle interruption differently at each state because the
amount of work done — and the amount of state worth preserving — differs at
each point.

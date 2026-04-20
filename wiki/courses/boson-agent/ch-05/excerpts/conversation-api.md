---
chapter: ch-05
course: boson-agent
phase: excerpt
title: "ConversationAPI — mutation facade with AD1 timing contract"
source_file: boson-agent/packages/basement/basement/context/conversation_api.py
created_at: 2026-04-19
---

# ConversationAPI — conversation_api.py

**Source:** `boson-agent/packages/basement/basement/context/conversation_api.py`
**Role in system:** The only surface through which hooks may read or mutate conversation state. Enforces the AD1 mutation timing contract: append-ops are immediate, destructive ops are deferred to turn boundary.

---

## PendingMutation dataclass

```python
# boson-agent/packages/basement/basement/context/conversation_api.py, lines 30-36

@dataclass
class PendingMutation:
    """A queued destructive mutation to apply at turn boundary."""

    op: Literal["remove", "replace", "compact"]
    index: int | None = None
    new_content: Any = None
    preserve_rules: dict = field(default_factory=dict)
```

`PendingMutation` is the unit of deferred work. The `op` field is a `Literal` type (not another enum) — a deliberate minimalism choice. Three variants, expressed inline. The `preserve_rules` field only matters for `compact` ops; it is ignored for `remove` and `replace`.

---

## Append-ops (immediate)

```python
# boson-agent/packages/basement/basement/context/conversation_api.py, lines 52-79

    async def inject_assistant_tool_use(
        self, tool_name: str, tool_input: dict
    ) -> str:
        """Inject a tool_use message. Returns generated tool_use_id."""
        tool_use_id = f"toolu_{uuid4().hex[:12]}"
        self._manager.add_message(
            "assistant",
            [ToolUseBlock(id=tool_use_id, name=tool_name, input=tool_input)],
        )
        return tool_use_id

    async def inject_tool_result(
        self, tool_use_id: str, result: str, is_error: bool = False
    ) -> None:
        """Inject a tool_result message."""
        self._manager.add_message(
            "user",
            [
                ToolResultBlock(
                    tool_use_id=tool_use_id, content=result, is_error=is_error
                )
            ],
        )

    async def inject_system_reminder(self, content: str) -> None:
        """Queue content for injection into next user message."""
        self._manager.add_pending_reminder(content)
```

### Walkthrough

**`inject_assistant_tool_use`:** Calls `self._manager.add_message("assistant", [...])` directly — no queuing. The message is live in the conversation history immediately after this line returns. The `toolu_` prefix with 12 hex chars matches Anthropic's tool-use ID format, so injected tool uses look native to the LLM. Returns the generated ID so the caller can chain an `inject_tool_result` with the same ID.

**`inject_tool_result`:** Also immediate. Adds a `user`-role message containing a `ToolResultBlock`. The Anthropic API requires tool results to be in `user` messages, hence the role. Pairs with `inject_assistant_tool_use` to synthesize a complete tool exchange in the history.

**`inject_system_reminder`:** Calls `self._manager.add_pending_reminder(content)` — the reminder is stored in a pending list on the `ContextManager`, not added to messages directly. The agent loop consumes pending reminders by wrapping them in `<system-reminder>` tags and prepending them to the next user message text (see `agent_loop.py` lines 70-77). This is why the method is named "inject" but the effect is deferred one step — it's immediate into the pending queue, but the queue drains at the next user message boundary, not at the next LLM call.

**Notice:** All three methods are declared `async` even though none of them perform I/O. This is forward-compatibility design — if a future implementation needs to write to a persistent store or emit an event, the callers (all hooks) already `await` the call. No API change needed.

---

## Destructive-ops (deferred)

```python
# boson-agent/packages/basement/basement/context/conversation_api.py, lines 82-100

    async def remove_message(self, index: int) -> None:
        """Queue message removal. Applied at turn boundary via flush_pending()."""
        self._pending.append(PendingMutation(op="remove", index=index))

    async def replace_message(self, index: int, new_content: Any) -> None:
        """Queue message replacement. Applied at turn boundary."""
        self._pending.append(
            PendingMutation(op="replace", index=index, new_content=new_content)
        )

    async def trigger_compact(
        self, preserve_rules: dict | None = None
    ) -> None:
        """Queue compact. Applied at turn boundary."""
        self._pending.append(
            PendingMutation(
                op="compact", preserve_rules=preserve_rules or {}
            )
        )
```

These methods only append to `self._pending`. The index into the message list is captured at the time the hook runs, but applied later. This has a subtle implication: if a `PRE_TOOL_CALL` hook queues `remove_message(index=3)` and then the tool execution adds two more messages before `flush_pending` runs, index 3 still refers to what it referred to when the hook fired. Intermediate appends do not shift the deferred index because `flush_pending` applies removes in reverse index order (high index first) to prevent cascading shifts.

---

## flush_pending — the turn-boundary flush

```python
# boson-agent/packages/basement/basement/context/conversation_api.py, lines 104-142

    async def flush_pending(self) -> int:
        """Apply all queued mutations. Returns count of applied mutations.

        Removes are applied in reverse index order to prevent index shifting.
        """
        if not self._pending:
            return 0

        count = 0

        # Separate by type for correct ordering
        removes = [m for m in self._pending if m.op == "remove"]
        replaces = [m for m in self._pending if m.op == "replace"]
        compacts = [m for m in self._pending if m.op == "compact"]

        # Apply replaces first (indices still valid)
        for mutation in replaces:
            if self._manager._replace_at(mutation.index, mutation.new_content):
                count += 1

        # Apply removes in reverse index order (prevents index shifting)
        removes.sort(key=lambda m: m.index or 0, reverse=True)
        for mutation in removes:
            if mutation.index is not None and self._manager._remove_at(mutation.index):
                count += 1

        # Apply compacts last
        for mutation in compacts:
            messages = self._manager.get_messages()
            truncated = truncate_messages(
                messages,
                self._manager.get_system_prompt(),
                **(mutation.preserve_rules or {}),
            )
            self._manager._set_messages(truncated)
            count += 1

        self._pending.clear()
        return count
```

### Walkthrough — execution order within flush

`flush_pending` enforces a deterministic three-phase order regardless of the order mutations were queued:

1. **Replaces first** — replace operations use the original indices. Applying them before removes means the indices are still valid (no elements have been deleted yet).
2. **Removes in reverse index order** — removing index 10 before index 3 means index 3 is still pointing at the same element when its remove is applied. If you removed low indices first, all higher indices would shift down.
3. **Compacts last** — compaction rewrites the entire message list. Applying it after individual removes/replaces means it operates on the already-mutated list, not a stale snapshot.

**`self._pending.clear()` (line 142):** Called unconditionally after applying all mutations. If a mutation raises internally (e.g., `_remove_at` gets an out-of-range index), the error propagates up and `_pending` is not cleared — the mutations remain queued for the next flush. In practice `_remove_at` returns `False` on invalid index rather than raising, so this path is safe.

**Notice:** `flush_pending` is called by `agent_loop.py` at line 186 — after `ON_TURN_END` fires but before the function returns. This ordering means an `ON_TURN_END` hook can still queue destructive mutations and they will be applied in the same turn's flush. The window for queuing deferred mutations is: any time from `ON_TURN_START` through `ON_TURN_END`. After `flush_pending` runs, the pending list is empty.

---

## `message_count` property

```python
# boson-agent/packages/basement/basement/context/conversation_api.py, lines 149-152

    @property
    def message_count(self) -> int:
        """Get current message count."""
        return self._manager.message_count
```

The only read accessor exposed on `ConversationAPI`. Hooks can check `ctx.conversation.message_count` to make conditional decisions (e.g., trigger compact only when > 40 messages). Direct access to `self._manager.get_messages()` is technically possible (the attribute is prefixed with `_` but Python does not enforce access control), but hooks should use `message_count` and the inject methods only — accessing `_manager` directly bypasses the timing contract and is unsupported.

---
chapter: ch-03
course: boson-agent
phase: read
excerpt_for: packages/basement/basement/llm/base.py
created_at: "2026-04-17"
---

# Source: `basement/llm/base.py` — LLMProvider Protocol and StreamEvent Types

**One-line description:** Declares the `LLMProvider` structural Protocol and the five `StreamEvent` variant types that every provider must produce; the rest of the framework is coded against this file only, never against any concrete provider class.

---

## Code Excerpt

```python
# packages/basement/basement/llm/base.py, lines 20-63

class TextDelta(BaseModel):
    """Incremental text token from LLM response."""
    text: str


class ToolUseStart(BaseModel):
    """Start of a tool_use block in LLM response."""
    id: str
    name: str


class InputJsonDelta(BaseModel):
    """Incremental JSON fragment for tool input arguments."""
    partial_json: str


class ToolUseEnd(BaseModel):
    """End of a tool_use block."""
    id: str


class MessageEnd(BaseModel):
    """End of LLM response message."""
    stop_reason: str  # "end_turn" | "tool_use"


StreamEvent = TextDelta | ToolUseStart | InputJsonDelta | ToolUseEnd | MessageEnd


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider implementations."""

    async def stream(
        self,
        messages: list[Message],
        system: str,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamEvent]: ...
```

---

## Explanation

`base.py` is the **contract file** for the entire LLM subsystem. Everything it exports is a type definition — no logic, no I/O. Five Pydantic model classes define the complete vocabulary of streaming events that can cross the provider boundary. The union type `StreamEvent` is then the only type the agent loop imports from this module; when it does `isinstance(event, TextDelta)` in `agent_loop.py`, it is checking against these five classes.

The `LLMProvider` Protocol uses `@runtime_checkable`, which means you can call `isinstance(some_obj, LLMProvider)` at runtime without an explicit inheritance declaration — any class that has a method named `stream` with the right signature satisfies the protocol. This is Python's structural subtyping (duck typing with a compiler hint): Anthropic, OpenAI, and Google providers never inherit from `LLMProvider`; they just implement `stream`.

The signature of `stream` reveals an important design decision: `tools: list[ToolSpec] | None = None`. When `None`, the provider must not include any tool schema in the API request. This is not a suggestion — the agent loop passes `None` when there are genuinely no tools registered, and the Anthropic API will error if you pass an empty list instead of omitting the field. The `None` vs list distinction maps directly onto the API's `tools` field presence.

**Notice:** `system` is a bare `str`, not a `list[Message]` with role `"system"`. This forces adapters to handle the system prompt separately from the conversation history. Anthropic has a dedicated top-level `system` parameter; OpenAI/Google don't — so each adapter must route the system string differently. The Protocol exposes one uniform call shape; the adapters absorb the routing complexity.

**Connection to universal pattern:** This file is Step 1 and Step 5 of the universal pattern — it defines the shared event vocabulary that makes cross-provider substitution possible. No other module needs to know which concrete provider is running; they only see `StreamEvent`.

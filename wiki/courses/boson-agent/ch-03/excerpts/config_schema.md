---
chapter: ch-03
course: boson-agent
phase: read
excerpt_for: packages/basement/basement/schemas/config_schema.py
created_at: "2026-04-17"
---

# Source: `basement/schemas/config_schema.py` — LLMConfig and AgentConfig

**One-line description:** Pydantic models that define the validated shape of `config.yaml`; `LLMConfig` is the single object passed from config all the way to the provider constructor, carrying every field the adapter needs.

---

## Code Excerpt

```python
# packages/basement/basement/schemas/config_schema.py, lines 16-63

class LLMConfig(BaseModel):
    """LLM provider configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["anthropic", "openai", "google"] = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=200_000)
    api_key: str | None = None


class AgentConfig(BaseModel):
    """Top-level agent configuration loaded from config.yaml."""

    model_config = ConfigDict(extra="forbid")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    max_turns: int = Field(default=50, ge=1, le=1000)
    agent_dir: Path | None = None
    # v0.2 additions
    mcp_servers: dict[str, MCPServerConfig] = {}
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
    enable_tool_router: bool = False
    user_skills: bool = True
```

---

## Explanation

`LLMConfig` is a five-field Pydantic model with `extra="forbid"`, meaning any unknown field in `config.yaml` raises a `ValidationError` immediately at load time — not a silent no-op. This is the "zero hallucination" contract: an agent can't accidentally configure a field that doesn't exist and wonder why it has no effect.

The five fields map directly to how each adapter uses them:

| Field | Used by | Where in adapter |
|---|---|---|
| `provider` | `registry.py` | `PROVIDER_REGISTRY.get(config.provider)` — routes to the right factory |
| `model` | all providers | `kwargs["model"] = self._config.model` |
| `temperature` | all providers | conditional: `if self._config.temperature is not None` |
| `max_tokens` | all providers | Anthropic: `max_tokens`; OpenAI: `max_completion_tokens`; Gemini: `max_output_tokens` |
| `api_key` | all providers | `api_key=config.api_key` in the SDK client constructor |

The field name differences for `max_tokens` are entirely hidden inside each adapter's `kwargs` construction — the `LLMConfig` field is always `max_tokens`, regardless of what the target API calls it.

`api_key: str | None = None` is the default. This means you can ship a `config.yaml` with no `api_key` field and the config parses cleanly — the `None` value is the signal to `loader.py` that it should attempt environment variable resolution instead. `ConfigDict(extra="forbid")` does not prevent `None` fields; it only rejects keys that aren't in the schema at all.

`provider: Literal["anthropic", "openai", "google"]` uses `Literal` typing. Pydantic validates the value at parse time: if someone writes `provider: bedrock` in their config, the `ValidationError` fires at load time with a clear message, not at the first API call.

**Notice:** `temperature` has `ge=0.0, le=2.0` validation bounds. This is the union of valid ranges across the three providers (Anthropic and OpenAI both accept 0.0–2.0; Gemini also accepts this range). The schema uses the shared valid range rather than per-provider validation. A value that one API rejects but another accepts would still pass the schema check and fail at the API call — the adapter is responsible for clamping or forwarding.

**Connection to universal pattern:** `LLMConfig` is the single input object at Step 1 of the pattern. The `provider` field is the dispatch key; all other fields are forwarded unchanged into the adapter constructor. The schema's validation guarantees that when a factory function receives a `config`, the config's fields are already type-correct.

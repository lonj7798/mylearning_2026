---
chapter: ch-03
course: boson-agent
phase: read
excerpt_for: packages/basement/basement/llm/registry.py
created_at: "2026-04-17"
---

# Source: `basement/llm/registry.py` — Provider Registry (Dict-Dispatch)

**One-line description:** Maintains a `PROVIDER_REGISTRY` dict mapping string provider names to lazy factory functions, and exposes `get_provider(config)` as the single public function that turns an `LLMConfig` into a live `LLMProvider` instance.

---

## Code Excerpt

```python
# packages/basement/basement/llm/registry.py, lines 21-58

def _create_anthropic(config: LLMConfig) -> LLMProvider:
    from basement.llm.anthropic_provider import AnthropicProvider
    return AnthropicProvider(config)


def _create_openai(config: LLMConfig) -> LLMProvider:
    from basement.llm.openai_provider import OpenAIProvider
    return OpenAIProvider(config)


def _create_google(config: LLMConfig) -> LLMProvider:
    from basement.llm.google_provider import GoogleProvider
    return GoogleProvider(config)


PROVIDER_REGISTRY: dict[str, Callable] = {
    "anthropic": _create_anthropic,
    "openai": _create_openai,
    "google": _create_google,
}


def get_provider(config: LLMConfig) -> LLMProvider:
    """Get an LLM provider instance by config.

    Uses dict-dispatch to route to the correct factory.
    Raises ProviderError for unknown providers.
    """
    factory = PROVIDER_REGISTRY.get(config.provider)
    if not factory:
        raise ProviderError(
            f"Unknown provider: '{config.provider}'. "
            f"Available: {list(PROVIDER_REGISTRY)}"
        )
    return factory(config)
```

---

## Explanation

The registry uses **dict-dispatch** — a table of string keys to callable factories — rather than a chain of `if/elif` or class inheritance. `get_provider` does one dict lookup, one null check, then calls the factory. The entire routing logic is three lines of real work.

Each factory function wraps an import inside its body (`from basement.llm.anthropic_provider import AnthropicProvider`). This is a deliberate lazy import pattern: provider-specific SDK packages (the `anthropic`, `openai`, and `google-genai` libraries) are only imported when that provider is actually requested. An agent configured for Anthropic never loads the OpenAI or Google SDKs into memory.

The `ProviderError` on unknown provider is precise: it includes both the bad name and the list of known names. This is the LOD "zero hallucination" approach — the error message tells you exactly what the valid values are instead of making you hunt through documentation.

**Notice:** `PROVIDER_REGISTRY` is a module-level mutable dict. The README's "Adding a Custom Provider" section (lines 1244–1252) instructs you to add to this dict: `PROVIDER_REGISTRY["xai"] = _create_xai`. This is the designed extension point. There is no plugin system, no decorator registration — just one dict you edit. The LOD principle of "explicit over implicit" means the registry is a plain Python dict you can read in 40 lines, not a metaclass framework.

**Connection to universal pattern:** This file is Step 2 of the universal pattern — the dispatch layer that translates a config string into a concrete provider without leaking any provider-specific knowledge to its callers.

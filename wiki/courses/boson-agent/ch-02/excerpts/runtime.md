---
chapter: ch-02
course: boson-agent
phase: read
kind: excerpt
source: packages/basement/basement/schemas/runtime.py
created_at: 2026-04-17T00:00:00Z
---

# Excerpt: `runtime.py` — The AD3 Bundle

← Back to [[../read]]

---

## Source description

`schemas/runtime.py` is 36 lines. It contains exactly one thing: the `AgentRuntime` dataclass. No logic, no methods, no imports beyond stdlib and one local schema. Its placement under `schemas/` — not `loop/` — is intentional: it is a data container, not a behavior owner.

---

## Excerpt 1 — The full dataclass (lines 16-36)

```python
# packages/basement/basement/schemas/runtime.py, lines 16-36
@dataclass
class AgentRuntime:
    """Bundles all agent components into a single object.

    Simplifies run_agent_loop() signature and provides a natural
    home for v0.2 Gateway to hold.
    """

    config: AgentConfig
    provider: Any  # LLMProvider
    tool_registry: Any  # ToolRegistry
    hook_registry: Any  # HookRegistry
    context_manager: Any  # ContextManager
    conversation_api: Any  # ConversationAPI
    system_prompt: str
    # v0.2 additions
    skill_registry: Any = None  # SkillRegistry
    mcp_manager: Any = None    # MCPManager
    tool_router: Any = None    # ToolRouter (when enabled)
    permissions: Any = None    # PermissionChecker
```

**What this shows mechanically.** Seven required fields (no defaults) form the v0.1 core surface: `config`, `provider`, `tool_registry`, `hook_registry`, `context_manager`, `conversation_api`, `system_prompt`. Four optional fields (all default `None`) are v0.2 plugin additions: `skill_registry`, `mcp_manager`, `tool_router`, `permissions`.

This partitioning is load-bearing. The standalone CLI assembles a runtime with only the seven required fields. The Gateway assembles all eleven. The `run_agent_loop()` function checks `if runtime.tool_router:` and `if runtime.permissions:` — it branches on presence, not on a mode flag. This means plugins are opt-in without any registration: you don't call `loop.enable_permissions()`, you just populate the field.

The `Any` type annotations on all non-primitive fields are deliberate. Using `Any` instead of the concrete types avoids circular imports between the `schemas` layer (which must be importable by everything) and the `loop`, `tools`, and `hooks` layers (which import from `schemas`). The comments (`# LLMProvider`, `# ToolRegistry`) provide the intent without creating the import dependency.

**Notice.** The dataclass uses `@dataclass`, not `@dataclass(frozen=True)`. The runtime is mutable — `run_agent_loop()` sets `runtime.skip_user_append = True` (line 178 of `core.py`) as a dynamic attribute on an existing runtime instance. This works because `@dataclass` without `frozen=True` is just a class with `__init__` generated automatically. Setting `runtime.skip_user_append` adds a new attribute to the instance dict, which `getattr(runtime, "skip_user_append", False)` reads safely.

**Connection to universal pattern.** `AgentRuntime` is the "bundle" argument that makes the loop's signature `(runtime, user_input)` rather than `(config, provider, tool_registry, hook_registry, context_manager, conversation_api, system_prompt, skill_registry, mcp_manager, tool_router, permissions, user_input)`. AD3 is the design decision that collapses those eleven arguments into one.

---

## Excerpt 2 — Gateway assembly: `_build_agent_runtime()` (core.py lines 384-414)

```python
# packages/gateway/gateway/core.py, lines 384-414
def _build_agent_runtime(
    self, session: SessionState, shared_history: SharedHistory
) -> AgentRuntime:
    """Build an AgentRuntime for this session turn."""
    ctx = shared_history.create_context_manager()
    api = shared_history.create_conversation_api(ctx)

    provider = get_provider(self._agent_config.llm)
    permissions = load_permissions(self._agent_config)

    # Register use_skill per session (needs per-session ConversationAPI)
    # Re-register each turn to capture the new ConversationAPI instance
    if self._skill_registry and self._skill_registry.get_all():
        from basement.skills.injector import create_use_skill
        use_skill_spec = create_use_skill(
            self._skill_registry, api, self._hook_registry, permissions
        )
        self._tool_registry._tools[use_skill_spec.name] = use_skill_spec

    return AgentRuntime(
        config=self._agent_config,
        provider=provider,
        tool_registry=self._tool_registry,
        hook_registry=self._hook_registry,
        context_manager=ctx,
        conversation_api=api,
        system_prompt=self._system_prompt,
        skill_registry=self._skill_registry,
        permissions=permissions,
        tool_router=self._tool_router,
    )
```

**What this shows mechanically.** A new `AgentRuntime` is constructed on *every call to `handle_message()`* — that is, every user turn. The Gateway does not cache or reuse the runtime object across turns.

Two components are recreated fresh each turn: `ctx` (a `ContextManager` wrapping `SharedHistory`) and `api` (a `ConversationAPI` wrapping `ctx`). Both are turn-scoped. `provider`, `permissions`, and the skill spec registration also happen per-turn.

The `use_skill_spec` re-registration (lines 395-401) is the only mutable operation on a shared object: `self._tool_registry._tools[use_skill_spec.name] = use_skill_spec` directly mutates the registry's internal dict. This is necessary because `create_use_skill()` closes over the *current turn's* `api` instance. If the spec were registered once at `setup()` time, it would close over a stale `api` from a previous turn, causing skill injections to write to the wrong conversation.

**Notice.** The `provider` is also recreated each turn via `get_provider(self._agent_config.llm)`. This means a new Anthropic/OpenAI/Google client object is instantiated per turn. The cost is intentionally accepted: it keeps the runtime stateless and avoids connection-pooling complexity in v0.1/v0.2. Production deployments would typically override this with a provider that manages its own connection pool externally.

**Connection to universal pattern.** This assembly site is where the eleven fields of `AgentRuntime` get their concrete values. Comparing this to a hypothetical CLI `__main__` assembly makes the variant vs invariant distinction concrete: the CLI would use a single shared `ContextManager` across turns; the Gateway creates a per-turn one backed by `SharedHistory` so that the session's message list is always the source of truth.

---
calling_spec:
  purpose: "Full walkthrough of LayerPipeline, LayerContext, and layer discovery"
  parent: "[[../read.md]]"
  phase: read
  chapter: ch-12
  course: boson-agent
---

# Layer Pipeline — Full Walkthrough

> Sub-page of [[../read.md]]. Covers `gateway/layers/pipeline.py`,
> `gateway/layers/context.py`, and `gateway/layers/discovery.py`.

---

## What the Layer Pipeline Is

The layer pipeline is the outer shell that every user message passes through
before reaching the agent loop. It runs `N` named rule engines in numeric-prefix
order. Each engine returns a list of `Action` objects. The pipeline resolves
those into a single winning flow-control decision per layer, and either
short-circuits (Filter/Respond) or accumulates deferred actions (Inject,
StageTransition, Compact, PreTool).

The key invariant is **staged commit**: proposed injections and deferred actions
are never applied until all layers have passed. If any layer returns `Filter`
or `Respond`, all staged changes are discarded — the message never reaches the
agent loop.

---

## Action Priority Resolution

```python
# boson-agent/packages/gateway/gateway/layers/pipeline.py, lines 32-41

# Flow-control priority: filter > respond > inject > orchestration > pass
ACTION_PRIORITY = {
    "filter": 0,
    "respond": 1,
    "inject": 2,
    "stage_transition": 3,
    "compact": 3,
    "pre_tool": 3,
    "pass": 4,
    "continue": 4,
}
```

This dict drives `_resolve_actions()`. A layer's rule engine may return
multiple actions (e.g., an `Inject` and a `Continue` from two different checks
running in parallel mode). The pipeline picks the action with the lowest
priority number. `Filter` always wins. `Respond` beats `Inject`. `StageTransition`,
`Compact`, and `PreTool` share priority 3 — they are all "orchestration" actions
that do not block message flow and are processed together.

**Notice:** `inject` at priority 2 is *above* stage_transition at priority 3.
This means if a layer returns both `Inject` and `StageTransition`, the pipeline's
flow-control decision is `inject` — but the `StageTransition` is still processed
as a deferred action in the loop at lines 125–129. Priority only governs the
`_resolve_actions()` return value, which controls whether the message is
short-circuited. Non-flow-control actions are always processed regardless.

---

## The Main Processing Loop

```python
# boson-agent/packages/gateway/gateway/layers/pipeline.py, lines 70-188

async def process(
    self,
    session_id: str,
    content: str,
    session: SessionState,
) -> AsyncIterator[str]:
    """Run message through layers, then inner handler (staged commit)."""
    ctx = SharedLayerContext(
        session=session,
        messages=session.messages,
        user_message=content,
        signal_queue=self._signal_queue,
        get_agent_status=session.status_tracker.get_status,
    )

    proposed_injections: list[str] = []
    skill_fillers: list[str] = []
    session._in_pipeline = True

    for layer_name, engine in self._layers:
        ctx.layer_name = layer_name
        actions = await engine.evaluate(session.messages, content, ctx)
        decision = self._resolve_actions(actions)

        if decision == "filter":
            # ... log + signal + return (discard all staged changes)

        if decision == "respond":
            # ... yield fixed text + return (discard all staged changes)

        # Process all non-flow-control actions
        for a in actions:
            atype = self._action_type(a)
            if atype == "inject":
                proposed_injections.append(a.payload.get("content", ""))
            elif atype == "stage_transition" and self._on_stage_transition:
                target = a.payload.get("target_stage", "")
                fillers = await self._on_stage_transition(session, target)
                skill_fillers.extend(fillers)
            elif atype == "compact" and self._on_compact:
                await self._on_compact(session)
            elif atype == "pre_tool" and self._on_pre_tool:
                tool_name = a.payload.get("tool_name", "")
                tool_args = a.payload.get("arguments", {})
                await self._on_pre_tool(session, tool_name, tool_args)

    # All layers passed — commit
    parts: list[str] = []
    final_content = getattr(session, "_stripped_user_message", None) or content
    parts.append(final_content)

    for inj in proposed_injections:
        parts.append(f"<system-reminder>{inj}</system-reminder>")

    pending = getattr(session, "_pending_stage_injection", None)
    if pending:
        parts.append("---")
        parts.append(f"<system-reminder>{pending}</system-reminder>")
        parts.append("---")
        parts.append("The conversation has moved to a new stage. "
                     "Follow the updated instructions above and respond "
                     "to the customer accordingly.")
        session._pending_stage_injection = None

    session.messages.append(
        Message(role="user", content="\n".join(parts))
    )
    session._pipeline_appended = True
    session._in_pipeline = False
```

**Walking through the commit phase (lines 139–168):**

1. The final user message is assembled in `parts`: original content first, then
   any `Inject` contents wrapped in `<system-reminder>` tags, then the pending
   stage injection (if a `StageTransition` fired during this pipeline run).

2. All three components land in a **single `[user]` message**. This is critical
   for Anthropic API compliance: the API requires alternating user/assistant
   turns. If the stage injection were a separate message it would create two
   consecutive user messages.

3. `session._pipeline_appended = True` is the sentinel that tells
   `GatewayCore.handle_message()` to skip its own `session.messages.append()`
   (step 4) — the pipeline already did it.

4. `session._in_pipeline = False` releases the deferred-preload guard:
   after this line, `_run_stage_preloads` can append its synthetic tool pairs
   in the correct order (after the user message, before the next LLM turn).

---

## SharedLayerContext and `ctx.data`

```python
# boson-agent/packages/gateway/gateway/layers/context.py, lines 16-49

@dataclass
class SharedLayerContext:
    """Context object that flows through all layers during message processing.

    Created once per handle_message call. Ephemeral — not stored on session.
    Layer rules receive this as the ``session`` parameter via polymorphic
    convention.

    v0.6: ``data`` dict for inter-layer communication. Any layer can read/write.
    """

    session: Any  # SessionState
    messages: list
    user_message: str
    signal_queue: Any
    get_agent_status: Callable[[], Any]
    layer_name: str | None = None
    data: dict = field(default_factory=dict)  # v0.6: inter-layer data passing

    def __getattr__(self, name: str) -> Any:
        """Proxy unknown attributes to the real SessionState."""
        if name.startswith("_") or name in (
            "session", "messages", "user_message", "signal_queue",
            "get_agent_status", "layer_name", "data",
        ):
            raise AttributeError(name)
        session = object.__getattribute__(self, "session")
        if session is not None:
            return getattr(session, name)
        raise AttributeError(name)
```

`SharedLayerContext` is the object rule functions receive as their `session`
parameter. It is NOT `SessionState` — it is a wrapper that proxies unknown
attributes to the underlying session via `__getattr__`. This means rules that
access `session.active_stage` or `session.checklist_state` transparently reach
`SessionState` without needing to know they are in a pipeline context.

**`data: dict` (line 32):** This is the v0.6 inter-layer communication channel.
Layer 01 can write `ctx.data["intent"] = "purchase"` and Layer 03 can read
`session.data.get("intent")`. From Layer 03's rule function perspective,
`session` is the `SharedLayerContext` and `session.data` is the `ctx.data` dict.
This is the mechanism demonstrated in `demo-gateway/layers/03-orchestrator/rules/stage_manager.py`:

```python
# boson-agent/agents/demo-gateway/layers/03-orchestrator/rules/stage_manager.py, lines 23-38

@check("auto_stage_transition", mode="sequential", priority=10)
def auto_transition(messages, user_message, session):
    """Transition stages based on ctx.data intent from Layer 02."""
    # Read intent set by Layer 02
    intent = getattr(session, "data", {}).get("intent")
    active = getattr(session, "active_stage", None)

    if intent == "closing" and active != "closing":
        return StageTransition("closing")

    if active == "welcome" and session.turn_count > 1:
        return StageTransition("main")

    return Pass()
```

Layer 02 (analyzer) classifies the message intent and writes to `ctx.data`.
Layer 03 (orchestrator) reads that intent and decides whether to transition
stages. The two layers are decoupled — Layer 03 does not import or call
Layer 02 directly. `ctx.data` is their shared blackboard.

**Notice:** `ctx.data` is ephemeral — it lives only for the duration of one
`process()` call. It is not persisted to `SessionState`. If Layer 03 needs
information from Layer 02 across turns, it writes to `session.active_stage`
or a custom session attribute (like `session.turn_count` in the example).

---

## Layer Discovery

```python
# boson-agent/packages/gateway/gateway/layers/discovery.py, lines 37-95

def discover_layers(gateway_dir: Path) -> list[LayerDefinition]:
    """Discover layers from gateway_dir/layers/NN-name/rules/ convention."""
    layers_dir = gateway_dir / "layers"
    rules_dir = gateway_dir / "rules"

    if not layers_dir.exists():
        # v0.6 auto-wrap: flat rules/ → virtual inner layer
        if rules_dir.exists():
            return _auto_wrap_rules(rules_dir)
        return []

    definitions: list[LayerDefinition] = []
    for entry in sorted(layers_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = LAYER_PATTERN.match(entry.name)  # r"^(\d+)-(.+)$"
        if not match:
            continue
        rules_dir = entry / "rules"
        if not rules_dir.exists():
            logger.warning("Layer %s has no rules/ directory, skipping", entry.name)
            continue
        try:
            registry = CheckRegistry()
            count = registry.discover_checks(rules_dir)
        except Exception as exc:
            logger.warning("Layer %s: discovery failed (%s), skipping", entry.name, exc)
            continue
        definitions.append(LayerDefinition(
            name=entry.name,
            order=int(match.group(1)),
            checks=registry.get_all(),
            rules_dir=rules_dir,
        ))

    definitions.sort(key=lambda d: d.order)
    return definitions
```

The discovery convention is `layers/NN-name/rules/*.py` where `NN` is a
zero-padded integer that determines execution order. Sorting by `order` (the
`int(match.group(1))` parsed from the directory name) means:

- `01-guard` runs first — can drop spam before any expensive work.
- `02-analyzer` runs second — can classify intent for downstream layers.
- `03-orchestrator` runs last — can read Layer 02's classification and decide
  on stage transitions.

**Notice:** Discovery is fail-open per layer: a broken layer (import error,
zero checks) is skipped with a warning, not a crash. This means a typo in
one layer's rule file does not bring down the gateway. The remaining layers
still run. This is a deliberate production-safety choice.

**The `_auto_wrap_rules()` path** (lines 98–122) is a backwards-compatibility
shim. Gateways written for v0.4/v0.5 (which had a flat `rules/` directory
instead of numbered layers) are automatically wrapped as a single virtual
layer with `order=99`. They run last, after any real layers. Zero migration
required.

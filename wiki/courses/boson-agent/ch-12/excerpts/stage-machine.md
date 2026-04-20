---
calling_spec:
  purpose: "Full walkthrough of StageMachine — stateless transition enforcer"
  parent: "[[../read.md]]"
  phase: read
  chapter: ch-12
  course: boson-agent
---

# StageMachine — Full Walkthrough

> Sub-page of [[../read.md]]. Deep-dive into `gateway/stage/machine.py` and
> `gateway/stage/context.py`.

---

## The Shared-but-Stateless Design

`StageMachine` is instantiated **once** at process startup and shared across
every concurrent session. This is a deliberate design constraint: the machine
knows what stages exist and what transitions are legal, but it knows nothing
about any individual session's current stage. Per-session state lives
exclusively on `session.active_stage`.

```python
# boson-agent/packages/gateway/gateway/stage/machine.py, lines 17-44

class StageMachine:
    """Stateless transition enforcer and stage registry.

    Shared across all sessions. Holds stage definitions and validates
    transitions. Does NOT track current stage — that's per-session on
    session.active_stage.
    """

    def __init__(self) -> None:
        self._stages: dict[str, StageDefinition] = {}

    def register(self, stage: StageDefinition) -> None:
        """Register a stage definition."""
        self._stages[stage.name] = stage

    def has_stage(self, name: str) -> bool:
        return name in self._stages

    def get_stage(self, name: str) -> StageDefinition:
        """Get stage by name. Raises KeyError if not found."""
        if name not in self._stages:
            raise KeyError(f"Stage '{name}' not registered")
        return self._stages[name]

    def get_initial_stage(self, name: str) -> StageDefinition:
        """Get the initial stage definition. Raises KeyError if not found."""
        return self.get_stage(name)
```

**Line-by-line:**

- **`__init__`:** A plain `dict` is the entire "database." Keys are stage names;
  values are `StageDefinition` dataclass instances. No locks, no asyncio — reads
  are purely concurrent-safe because the dict is written only during startup
  (`load_stages()`) and never mutated afterward.
- **`register()`:** Called once per stage by `load_stages()`. At runtime, the
  registry is read-only.
- **`get_initial_stage()`:** Delegates to `get_stage()`. The separation exists
  for semantic clarity at call sites — callers reading the initial stage are
  expressing intent, not just doing a lookup.

---

## The `transition()` Method — Pure Validation

```python
# boson-agent/packages/gateway/gateway/stage/machine.py, lines 45-68

    def transition(self, from_stage: str, to_stage: str) -> TransitionResult:
        """Validate and return transition result. Stateless — does not mutate.

        Args:
            from_stage: Current stage name (from session.active_stage).
            to_stage: Target stage name.
        """
        current = self._stages.get(from_stage)
        if current is None:
            return TransitionResult(
                success=False, error=f"Stage '{from_stage}' not registered"
            )
        if to_stage not in current.transitions:
            return TransitionResult(
                success=False,
                error=f"Transition '{from_stage}' -> '{to_stage}' not allowed",
            )
        target = self._stages.get(to_stage)
        if target is None:
            return TransitionResult(
                success=False,
                error=f"Target stage '{to_stage}' not registered",
            )
        return TransitionResult(success=True, new_stage=target)
```

This method is a three-guard chain:

1. **Guard 1 (line 52):** Is `from_stage` registered at all? A rule returning
   `StageTransition("bogus")` from an unknown current stage would fail here.
2. **Guard 2 (line 56):** Is `to_stage` in the current stage's `transitions`
   allowlist? This is the core enforcement. The allowlist is a Python list, so
   the check is `to_stage not in current.transitions` — an O(n) list scan.
   Stage graphs are tiny (< 20 stages) so this is never a bottleneck.
3. **Guard 3 (line 61):** Is `to_stage` itself registered? This catches
   config typos (e.g., `transitions: ["prodcut_focused"]`) at the point of
   attempted transition rather than at startup.

**Notice:** The method does NOT mutate `session.active_stage`. That mutation
happens in `GatewayCore._apply_stage_transition()`:

```python
# boson-agent/packages/gateway/gateway/core.py, lines 317-323

result = self._stage_machine.transition(
    from_stage=session.active_stage, to_stage=target
)
if not result.success:
    return []
session.active_stage = target          # <-- mutation lives here, in core
self._inject_stage(session, result.new_stage)
```

The machine returns a `TransitionResult`; the caller decides what to do with
it. This separation means the machine can be unit-tested with zero session
mocking.

---

## StageContext — Tool/Skill Visibility

`StageContext` is the runtime view of a stage: it takes a `StageDefinition`
and provides filtering methods used by `GatewayCore`.

```python
# boson-agent/packages/gateway/gateway/stage/context.py, lines 16-50

@dataclass
class StageContext:
    """Computed context for the current stage."""

    visible_tools: set[str]
    visible_skills: set[str]
    prompt: str

    @classmethod
    def from_stage(cls, stage: StageDefinition) -> "StageContext":
        """Create context from a stage definition."""
        return cls(
            visible_tools=set(stage.tools),
            visible_skills=set(stage.skills),
            prompt=stage.prompt,
        )

    def filter_tools(self, all_tools: list[str]) -> list[str]:
        """Filter tool list to only stage-visible tools. Preserves order."""
        return [t for t in all_tools if t in self.visible_tools]

    def filter_skills(self, all_skills: list[str]) -> list[str]:
        """Filter skill list to only stage-visible skills."""
        return [s for s in all_skills if s in self.visible_skills]

    def should_filter_tools(self, tool_router_enabled: bool) -> bool:
        """Whether to filter tool registry based on ToolRouter mode.

        ToolRouter enabled: All tools loaded, stage prompt guides usage.
        ToolRouter disabled: Filter tool registry per stage.
        """
        return not tool_router_enabled
```

**Notice `should_filter_tools()`:** There are two modes for per-stage tool
visibility. When `ToolRouter` is enabled (v0.6+), all native tools stay in
the `ToolRegistry` and the LLM only sees `use_tool` / `use_skill` meta-tools;
`ToolRouter.set_allowed_tools()` then controls which native tools `use_tool`
can dispatch to. When `ToolRouter` is disabled (v0.4/v0.5 mode), the tool
registry itself is filtered per stage — only the stage-visible tools are loaded
into `AgentRuntime`. The `should_filter_tools()` predicate cleanly separates
these two code paths.

---

## The Injection Message

When a stage transition is confirmed, the LLM needs to learn about it. The
`build_stage_injection()` function formats the `StageDefinition` into a
human-readable system-reminder block:

```python
# boson-agent/packages/gateway/gateway/stage/context.py, lines 73-84

def build_stage_injection(stage: StageDefinition) -> str:
    """Build the injection message for a stage transition."""
    parts = [f"[Stage: {stage.name}]"]
    if stage.prompt:
        parts.append(stage.prompt)
    if stage.tools:
        tool_list = ", ".join(stage.tools)
        parts.append(f"Available tools: {tool_list}")
    if stage.skills:
        skill_list = ", ".join(stage.skills)
        parts.append(f"Available skills: {skill_list}")
    return "\n\n".join(parts)
```

This text is embedded inside `<system-reminder>` tags in the user message.
The LLM sees `[Stage: product_focused]\n\nAvailable tools: check_product_detail,
check_product_summary, lookup_faq` and updates its behavior accordingly — even
before any preload tool results are visible.

**Notice:** The tool and skill lists in the injection string are the same lists
that `ToolRouter.set_allowed_tools()` receives. The LLM's textual knowledge of
what it can do ("Available tools: …") and the router's enforcement of what it
*can* do are always in sync because both come from the same `StageDefinition`.

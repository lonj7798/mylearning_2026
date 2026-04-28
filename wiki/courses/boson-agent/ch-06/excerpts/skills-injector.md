---
calling_spec:
  purpose: Deep walkthrough of the skills subsystem — loader, registry, injector
  chapter: ch-06
  course: boson-agent
  phase: read
  excerpt_of: read.md
---

# Skills Subsystem — Deep Walkthrough

Three files cooperate to turn a plain `.md` file into a live conversation injection.
Reading them in dependency order — loader → registry → injector — reveals the full pipeline.

---

## `loader.py` — Filesystem to SkillSpec

```python
# packages/basement/basement/skills/loader.py, lines 20-50

def discover_skills(skills_dir: Path) -> list[SkillSpec]:
    """Discover all .md skill files in skills_dir.

    - Filename (without .md) becomes the skill name.
    - First line of file becomes the description.
    - Full content becomes the prompt_template.
    - Returns empty list if the folder doesn't exist.
    """
    if not skills_dir.exists():
        return []

    skills: list[SkillSpec] = []
    for md_file in sorted(skills_dir.glob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            lines = content.splitlines()
            description = lines[0].strip() if lines else ""
            skills.append(
                SkillSpec(
                    name=md_file.stem,
                    description=description,
                    prompt_template=content,
                    file_path=md_file,
                )
            )
            logger.debug("Discovered skill: %s", md_file.stem)
        except Exception as e:
            logger.error("Failed to load skill from %s: %s", md_file, e)
            continue

    return skills
```

**Mechanical explanation.** `discover_skills` is a pure filesystem scan. It calls
`sorted(skills_dir.glob("*.md"))`, which means discovery order is alphabetical and
deterministic across runs. For each file it reads the entire text, pulls line 0 as the
description (no heading stripping, no Markdown parsing — raw first line), and wraps
everything in a `SkillSpec` dataclass. The `file_path` field is stored on the spec so
tooling can report where a skill came from.

**Notice:** the function silently skips any file that raises during read (the bare
`except Exception: continue`). This is intentional fail-open design: a corrupt skill
file does not crash the agent startup. The error goes to the logger but the remaining
skills still load.

---

## `registry.py` — In-Memory Catalog

```python
# packages/basement/basement/skills/registry.py, lines 22-55

class SkillRegistry:
    """Discover and manage agent skills."""

    def __init__(self):
        self._skills: dict[str, SkillSpec] = {}

    def discover_skills(self, skills_dir: Path) -> int:
        """Discover .md files in skills_dir and register each.

        Returns count of discovered skills.
        """
        specs = _discover_skills(skills_dir)
        for spec in specs:
            self.register(spec)
        return len(specs)

    def register(self, spec: SkillSpec) -> None:
        """Register a skill. Raises ValueError on duplicate name."""
        if spec.name in self._skills:
            raise ValueError(f"Duplicate skill name: '{spec.name}'")
        self._skills[spec.name] = spec
        logger.debug("Registered skill: %s", spec.name)

    def get(self, name: str) -> SkillSpec:
        """Get skill by name. Raises SkillNotFoundError if not found."""
        if name not in self._skills:
            raise SkillNotFoundError(
                f"Skill '{name}' not found. Available: {list(self._skills)}"
            )
        return self._skills[name]

    def get_all(self) -> list[SkillSpec]:
        """Return all registered skill specs."""
        return list(self._skills.values())
```

**Mechanical explanation.** `SkillRegistry` mirrors `ToolRegistry` exactly (same pattern
as [[ch-04]]): a `dict[str, SkillSpec]` keyed by name, with `register` / `get` /
`get_all` as the only interface. The `discover_skills` method is a convenience wrapper
that calls the loader and feeds results to `register`. Duplicates raise immediately
(fail-fast) rather than silently overwriting, preventing accidental shadowing when
multiple directories are scanned.

**Notice:** `get_all()` returns `list(self._skills.values())` — a snapshot copy, not a
live view. Callers that iterate the result while `register` is being called from another
coroutine see a consistent list.

---

## `injector.py` — Activation and the `use_skill` Meta-Tool

```python
# packages/basement/basement/skills/injector.py, lines 25-82

async def inject_skill(
    api: ConversationAPI,
    skill: SkillSpec,
    hook_registry: HookRegistry | None = None,
) -> None:
    """Inject a skill's prompt template into the conversation.

    - Fires ON_SKILL_INVOKE hook if hook_registry is provided.
    - Injects skill.prompt_template via api.inject_system_reminder().
    """
    if hook_registry is not None:
        from basement.hooks.runner import fire_event

        ctx = HookContext(
            event=HookEvent.ON_SKILL_INVOKE,
            conversation=api,
            metadata={"skill_name": skill.name},
        )
        await fire_event(hook_registry, HookEvent.ON_SKILL_INVOKE, ctx)

    await api.inject_system_reminder(skill.prompt_template)
    logger.debug("Injected skill: %s", skill.name)


def create_use_skill(
    registry: SkillRegistry,
    api: ConversationAPI,
    hook_registry: HookRegistry | None = None,
    permissions: PermissionChecker | None = None,
) -> ToolSpec:
    """Create a 'use_skill' ToolSpec that injects skill prompts on demand."""

    async def handler(skill_name: str) -> str:
        if permissions is not None:
            permissions.check_skill(skill_name)
        skill = registry.get(skill_name)
        await inject_skill(api, skill, hook_registry)
        return f"Skill '{skill_name}' has been activated."

    return ToolSpec(
        name="use_skill",
        description="Activate a skill by name to inject its prompt template into the conversation.",
        input_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill to activate.",
                }
            },
            "required": ["skill_name"],
        },
        handler=handler,
    )
```

**Mechanical explanation.** Two separate concerns live here.

`inject_skill` is the actual injection side-effect. It fires `ON_SKILL_INVOKE` first
(giving hooks a chance to observe or veto skill activation), then calls
`api.inject_system_reminder(skill.prompt_template)`. That method appends the skill's
full Markdown content wrapped in `<system-reminder>` tags to the next LLM message —
the same channel [[ch-05]] hooks use. The LLM sees the injected text as a mid-turn
instruction.

`create_use_skill` is a factory. It closes over `registry`, `api`, `hook_registry`, and
`permissions` into an async `handler` function, then wraps that function as a `ToolSpec`
named `"use_skill"`. The resulting `ToolSpec` is registered in the tool registry like
any other tool — so from the LLM's perspective, `use_skill` is just another tool call.
The difference is that its side-effect is textual (prompt injection), not computational.

**Notice:** `permissions.check_skill(skill_name)` is called **inside** the handler,
at invocation time, not at registration time. This means permission changes (e.g., a
stage transition that narrows `allowed_skills`) take effect on the next `use_skill` call
without re-registering the tool.

---

## Real Skill Files

### `agents/demo/skills/explain.md` (entire file, 2 lines)

```
Explain Skill
You are now in explain mode. When the user asks about a topic, explain it simply as if talking to a beginner. Use analogies and examples.
```

The stem `explain` becomes `skill_name="explain"`. Line 0 (`"Explain Skill"`) becomes
the description. The full two-line string is the `prompt_template` injected verbatim.

### `agents/test-lina/skills/product_manager.md` (excerpt, lines 1-36)

```markdown
# Product Manager

Manages product comparison and switching during the sales conversation.

## Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `check_available_products` | — | List available products for this customer |
| `check_product_detail` | `keyword` (str, optional) | Coverage details |
| `set_product` | `plan` (str) | Switch plan |

## Scenarios

### A: Customer Asks for Cheaper Options

When the customer mentions cost concerns:

1. `use_tool("check_available_products", {})`
2. Present the options conversationally
3. Do NOT call `set_product` yet — wait for explicit choice

### B: Customer Agrees to Switch Products

1. `use_tool("set_product", {"plan": "cheap"})`
2. `use_tool("check_product_summary", {})`
3. Briefly summarize the key differences
```

This is a production-grade skill. Notice it instructs the LLM to call `use_tool(...)` —
skills and tools compose: the skill prompt teaches the LLM a workflow that involves
further tool calls. This is the pattern used in the Lina TMR sales agent (16 tools,
3 skills, 7 stages).

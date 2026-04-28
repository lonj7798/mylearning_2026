---
calling_spec:
  purpose: "Full walkthrough of stage_config.py format and StageDefinition schema"
  parent: "[[../read.md]]"
  phase: read
  chapter: ch-12
  course: boson-agent
---

# Stage Configuration — Full Walkthrough

> Sub-page of [[../read.md]]. Contains the complete per-implementation deep-dive
> for `stage_config.py` files. The parent read.md carries the overview, universal
> pattern, and synthesis.

---

## What a stage_config.py Is

Every gateway agent declares its conversational modes in a single Python module
named `stage_config.py` at the agent root. The file is **pure data** — no
framework imports, no conditionals. Its two exports are consumed by
`__main__.py` on startup:

- `initial_stage: str` — which stage a brand-new session starts in.
- `stages: dict[str, dict]` — the complete stage graph, one key per stage.

Each stage dict understands these keys:

| Key | Type | Effect |
|-----|------|--------|
| `tools` | `list[str]` | Tool names visible to the LLM in this stage |
| `skills` | `list[str]` | Skill names injectable in this stage |
| `transitions` | `list[str]` | Allowlist of target stages from here |
| `preloads` | `list[tuple[str, dict]]` | `(tool_name, args)` pairs auto-run on entry |
| `preload_skills` | `list[str]` | Skill names auto-injected on entry |

The allowlist in `transitions` is enforced by `StageMachine.transition()` —
any `StageTransition` action targeting a name not in the list returns
`TransitionResult(success=False)` and the transition silently no-ops.

---

## Demo Gateway: Minimal 3-Stage Config

```python
# boson-agent/agents/demo-gateway/stage_config.py, lines 1-29

"""Demo stage configuration — 3 stages: welcome → main → closing."""

initial_stage = "welcome"

stages = {
    "welcome": {
        "tools": ["get_time"],
        "skills": [],
        "transitions": ["main", "closing"],
    },
    "main": {
        "tools": ["calculate", "get_weather", "search_docs", "get_time"],
        "skills": ["explain", "summarize"],
        "transitions": ["closing"],
        # Preload: auto-run get_time on stage entry
        "preloads": [("get_time", {"timezone": "UTC"})],
        # Preload: auto-inject explain skill prompt
        "preload_skills": ["explain"],
    },
    "closing": {
        "tools": ["get_time"],
        "skills": [],
        "transitions": [],  # terminal stage
    },
}
```

**Line-by-line:**

- **Line 7** `initial_stage = "welcome"` — startup assigns `session.active_stage = "welcome"`
  via `GatewayCore._set_initial_stage()`. Every new connection begins here.
- **Lines 10–14** `"welcome"` stage allows two transitions but gives the LLM only
  `get_time`. The LLM cannot hallucinate calls to `calculate` or `search_docs`
  because `ToolRouter.set_allowed_tools(["get_time"])` hides everything else.
- **Lines 15–23** `"main"` is the richest stage. The `preloads` list causes
  `get_time(timezone="UTC")` to execute automatically on transition entry,
  injecting a synthetic `[assistant] tool_use + [user] tool_result` pair into
  history before the LLM's first turn in this stage. `preload_skills: ["explain"]`
  additionally injects the explain skill's prompt template as a `<system-reminder>`
  user message.
- **Lines 24–27** `"closing"` has an empty `transitions: []` — it is a terminal
  state. `StageMachine.transition("closing", anything)` will always return
  `TransitionResult(success=False)`.

**Notice:** the tool list in each stage is not a filter applied to the registry
at query time — it is the *allowed set* handed to `ToolRouter.set_allowed_tools()`
at the start of each turn, before `run_agent_loop` is called. The underlying
`ToolRegistry` always holds all tools; the router gates what the LLM API sees.
This means you can freely add tools to the registry without touching stage config,
and vice versa.

---

## Lina TMR Gateway: Production 8-Stage Config

```python
# boson-agent/agents/test-lina-gateway/stage_config.py, lines 1-74

initial_stage = "introduction"

_GLOBAL_TOOLS = []
_GLOBAL_SKILLS = ["consent_manager"]

stages = {
    "introduction": {
        "tools": _GLOBAL_TOOLS,
        "skills": [],
        "transitions": ["product_focused", "dnc_processing",
                        "reschedule", "escalate_to_human"],
    },
    "product_focused": {
        "tools": ["check_product_detail", "check_product_summary",
                  "lookup_faq"] + _GLOBAL_TOOLS,
        "skills": [] + _GLOBAL_SKILLS,
        "transitions": ["consultation", "reschedule",
                        "dnc_processing", "escalate_to_human"],
        "preloads": [
            ("check_product_summary", {}),
        ],
    },
    "consultation": {
        "tools": ["check_product_detail", "check_product_summary",
                  "lookup_faq", "get_consent_status", "record_consent"]
                 + _GLOBAL_TOOLS,
        "skills": ["product_manager"] + _GLOBAL_SKILLS,
        "transitions": ["reschedule", "dnc_processing", "escalate_to_human"],
        "preloads": [
            ("check_product_summary", {}),
        ],
    },
    "purchase": {
        "tools": [
            "check_product_detail", "set_product",
            "get_consent_status", "record_consent",
            "verify_personal_info", "get_disclosure_questions",
            "save_disclosure_answer", "save_payment_info", "save_address",
        ] + _GLOBAL_TOOLS,
        "skills": ["purchase_setup", "disclosure_manager",
                   "payment_manager"] + _GLOBAL_SKILLS,
        "transitions": ["end", "escalate_to_human"],
    },
    "reschedule": {
        "tools": ["check_existing_schedule", "check_available_schedule",
                  "reschedule"],
        "skills": ["schedule_manager"],
        "transitions": ["consultation", "end", "escalate_to_human"],
        "preloads": [
            ("check_available_schedule", {"date": "2026-04-07"}),
        ],
        "preload_skills": ["schedule_manager"],
    },
    "end": {
        "tools": [],
        "skills": [],
        "transitions": [],
        "preloads": [],
    },
}
```

**Key design observations:**

1. **`_GLOBAL_TOOLS` / `_GLOBAL_SKILLS` pattern (lines 3–4):** Lina factors out
   shared capabilities into module-level variables and appends them with `+` to
   each stage's list. This is a pure-Python DRY technique — no framework feature.
   The resulting lists are ordinary Python lists that the loader reads verbatim.

2. **`introduction` has zero tools (line 8):** The agent in the introduction stage
   cannot call any tools. This is intentional — the introduction is a scripted
   persuasion phase where tool calls would be a distraction. The LLM only has its
   system prompt and conversation history to work with.

3. **`preloads` on `product_focused` and `consultation` (lines 17, 27):**
   Both stages preload `check_product_summary` with empty args `{}`. When the
   transition fires, `GatewayCore._run_stage_preloads()` executes this tool
   immediately and injects the result into history. The LLM enters the stage
   already knowing the product summary — no wasted turn asking for it.

4. **`reschedule` has both `preloads` and `preload_skills` (lines 48–51):**
   The available-schedule tool runs (so the LLM has fresh slots), and the
   `schedule_manager` skill's prompt is injected (so the LLM knows how to
   present them). Two mechanisms, one entry point.

5. **`purchase` has no `preloads` key at all (lines 32–45):** Omitting the key
   is equivalent to `"preloads": []`. The loader uses `spec.get("preloads", [])`,
   so missing keys are safe.

6. **Transition graph enforces the sales funnel:** Notice `consultation` does NOT
   list `introduction` or `product_focused` in its transitions — you cannot go
   backward. The stage machine enforces this; a misbehaving rule returning
   `StageTransition("introduction")` from `consultation` would be silently
   rejected.

**Notice:** `_GLOBAL_TOOLS = []` is intentionally empty in Lina — no tools are
truly universal across all stages. The pattern is set up to make it trivially easy
to add a global tool later (e.g., `get_current_time`) by appending to one line.
This is a forward-compatibility design decision, not an oversight.

---

## How the Loader Reads the Config

```python
# boson-agent/packages/gateway/gateway/stage/machine.py, lines 82-101

def load_stages(
    config: dict[str, dict],
    prompts: dict[str, str],
) -> StageMachine:
    machine = StageMachine()
    for name, spec in config.items():
        stage = StageDefinition(
            name=name,
            prompt=prompts.get(name, ""),
            tools=spec.get("tools", []),
            skills=spec.get("skills", []),
            transitions=spec.get("transitions", []),
        )
        machine.register(stage)
    return machine
```

`load_stages` takes two separate dicts: `config` (from `stage_config.py`'s
`stages` variable) and `prompts` (from `.md` files in a `stages/` subdirectory
loaded by `load_stage_prompts()`). This separation lets you keep behavioral data
(tool lists, transitions) in Python and narrative data (stage-specific system
prompt additions) in Markdown files.

**Notice:** `preloads` and `preload_skills` are NOT loaded into `StageDefinition`
by this function — they are read directly from the raw `spec` dict by
`GatewayCore` during setup (`self._stage_preloads[name] = spec.get("preloads", [])`,
`self._skill_preloads[name] = spec.get("preload_skills", [])`). This means
`StageDefinition` stays minimal (name, prompt, tools, skills, transitions) and
the preload logic lives in `GatewayCore`, not in the data schema.

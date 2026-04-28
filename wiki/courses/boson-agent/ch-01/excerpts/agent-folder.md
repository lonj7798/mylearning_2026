---
chapter: ch-01
course: boson-agent
phase: read
kind: excerpt
source: boson-agent/agents
created_at: "2026-04-17T00:00:00Z"
---

# Agent Folder Convention — Deep Walkthrough

This sub-page covers the developer-facing interface: what files go where, what each file controls, and how the same convention scales from a 4-file demo agent to a production 8-stage insurance sales agent.

---

## The Minimal Agent: `agents/demo`

The simplest possible Boson agent has two files:

```
agents/demo/
├── BOSON.md      ← required
└── config.yaml   ← optional (shown here with full settings)
```

### `BOSON.md` — The System Prompt

```markdown
# agents/demo/BOSON.md, lines 1-22 (full file)

You are a helpful demo assistant for testing the Basement Agent Framework.

## Tools
You have access to tools. Use them when asked:
- `calculate` — evaluate math expressions
- `get_weather` — look up weather for a city
- `search_docs` — search documentation by keyword
- `get_time` — get current time for a timezone

When the user asks you to calculate something, use the calculate tool.
When the user asks about weather, use the get_weather tool.
When the user asks to search, use the search_docs tool.
When the user asks about time, use the get_time tool.

## Skills
- `explain` — detailed explanation mode
- `summarize` — concise summary mode

## Style
- Be concise and direct
- Always use tools when available instead of guessing
- Maximum 3 sentences per turn
```

`BOSON.md` is plain Markdown read as a string and passed verbatim as the LLM system prompt. There is no special parsing, no directives, no templating — the entire file becomes `system_prompt` in `AgentRuntime`. The tool and skill names listed here are **LLM instructions**, not framework registrations. The framework does not parse `BOSON.md` for tool names; it discovers them from `tools/*.py` independently.

> **Notice:** The file lists `calculate`, `get_weather`, `search_docs`, `get_time` — these must exactly match the function names (or `name=` override) in the corresponding `tools/*.py` files, or the LLM will try to call tools that the framework does not know about, producing tool-not-found errors. The human authoring `BOSON.md` is responsible for keeping this list consistent with the actual `tools/` directory. There is no automated check.

### `config.yaml` — LLM and Feature Configuration

```yaml
# agents/demo/config.yaml, lines 1-14 (full file)

llm:
  provider: anthropic
  model: claude-sonnet-4-6
  temperature: 0.3
  max_tokens: 1024

max_turns: 10

# v0.2: ToolRouter enabled (use_tool/use_skill meta-tools)
enable_tool_router: true

# v0.2: Permission example
permissions:
  denied_tools: []
```

This is the complete config for the demo agent. Notice `enable_tool_router: true` — with this flag, the LLM does not see the native tool schemas directly. Instead it sees only `use_tool` and `use_skill`. The agent calls `use_tool(tool_name="calculate", arguments={"expression": "2+2"})` and the ToolRouter dispatches to the real `calculate` function. This adds a layer of indirection that enables per-stage tool gating (see the Lina example below).

`permissions.denied_tools: []` is explicitly set to an empty list here, which is the same as the default. It is shown to illustrate the schema — you would add tool names here to block them.

**What `config.yaml` controls:**

| Key | Effect | Default if omitted |
|-----|--------|--------------------|
| `llm.provider` | Which LLM API to call | `"anthropic"` |
| `llm.model` | Model identifier | `claude-sonnet-4-20250514` |
| `llm.temperature` | Sampling temperature | `0.7` |
| `llm.max_tokens` | Max tokens per LLM response | `4096` |
| `max_turns` | Max inner-loop iterations per user message | `50` |
| `enable_tool_router` | Hide native tools, expose `use_tool`/`use_skill` | `false` |
| `permissions.allowed_tools` | Whitelist (null = all allowed) | `null` |
| `permissions.denied_tools` | Blacklist (overrides whitelist) | `[]` |
| `mcp_servers` | External MCP server configs | `{}` |

Every key is optional. `BOSON.md` + defaults is sufficient to run an agent.

---

## The Full Agent Folder: `agents/demo` (with tools, hooks, skills)

```
agents/demo/
├── BOSON.md
├── config.yaml
├── tools/
│   ├── calculate.py       ← @tool functions
│   ├── get_weather.py
│   ├── search_docs.py
│   └── get_time.py
├── hooks/
│   └── logger.py          ← @hook functions
└── skills/
    ├── explain.md         ← skill prompt files
    └── summarize.md
```

Each subdirectory is discovered independently by its registry. `tools/` → `ToolRegistry.discover_tools`. `hooks/` → `HookRegistry.discover_hooks`. `skills/` → `SkillRegistry.discover_skills`. None of the registries import each other. None require the directory to exist — if `hooks/` is absent, `discover_hooks` returns 0 and the hook registry is empty.

### A Tool File (`tools/calculate.py` pattern)

From `packages/basement/README.md` (lines 98-112), showing the canonical tool file shape:

```python
# packages/basement/README.md, lines 98-112 (Quick Start example)
from basement.tools.decorator import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Example: '2 + 3 * 4' returns '14'."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
```

One import, one decorator, one function, one docstring, type hints on all parameters. The framework requires nothing else. The `@tool` decorator plants `__tool_spec__` on the function object; `discover_tools` finds it.

### A Skill File (`skills/explain.md` pattern)

Skill files are plain Markdown. Their filename (without `.md`) is the skill name. Their content is the prompt injected into the conversation when the skill is activated. The `use_skill(skill_name="explain")` tool call triggers the injection.

```markdown
# Explain Skill

You are now in explain mode. When the user asks about a topic:
1. Explain it simply as if talking to a beginner
2. Use analogies and concrete examples
3. Avoid jargon

Keep your explanation to 2-3 paragraphs.
```

There is no metadata, no frontmatter, no configuration. The entire file is the prompt.

---

## The Gateway Agent Folder: `agents/demo-gateway`

When Gateway orchestration is added, the folder gains a new layer:

```
agents/demo-gateway/          ← Gateway config lives here
├── config.py                 ← GatewayConfig (Python, not YAML)
├── stage_config.py           ← 3 stages + preloads
├── stages/
│   ├── welcome.md
│   ├── main.md
│   └── closing.md
└── layers/                   ← rule layers (ordered by name prefix)
    ├── 01-guard/
    │   └── rules/
    │       └── spam_filter.py
    ├── 02-analyzer/
    │   └── rules/
    │       └── intent_detector.py
    └── 03-orchestrator/
        └── rules/
            └── stage_manager.py

agents/demo/                  ← Basement agent (unchanged)
    BOSON.md
    config.yaml
    tools/
    hooks/
    skills/
```

The Gateway agent folder and the Basement agent folder are separate directories. The Gateway's `config.py` points to the agent dir:

```python
# Conceptual — from packages/gateway/README.md (lines 147-162)
from gateway.schemas.config import GatewayConfig, CompactConfig

config = GatewayConfig(
    host="0.0.0.0",
    port=8765,
    agent_dir="./agent",          # ← points to the Basement agent folder
    compact=CompactConfig(
        enabled=True,
        threshold_messages=30,
        provider="openai",
        model="gpt-4o-mini",
        keep_recent=10,
    ),
    fail_open=True,
)
```

> **Notice:** Gateway config is Python (`config.py`), not YAML. This allows computed values, imports, and conditional logic in configuration — things YAML cannot express. The Basement agent's settings stay in `config.yaml` (processed by Pydantic). The split matches the layer split: declarative settings go to Pydantic-validated YAML; programmatic orchestration settings go to Python.

---

## The Production Agent: `agents/test-lina-gateway` Stage Config

The Lina agent demonstrates what the convention looks like at production scale. The `stage_config.py` file encodes the complete conversation flow:

```python
# agents/test-lina-gateway/stage_config.py, lines 1-74 (full file)

initial_stage = "introduction"

_GLOBAL_TOOLS = []
_GLOBAL_SKILLS = ["consent_manager"]

stages = {
    "introduction": {
        "tools": _GLOBAL_TOOLS,
        "skills": [],
        "transitions": ["product_focused", "dnc_processing", "reschedule", "escalate_to_human"],
    },
    "product_focused": {
        "tools": ["check_product_detail", "check_product_summary", "lookup_faq"] + _GLOBAL_TOOLS,
        "skills": [] + _GLOBAL_SKILLS,
        "transitions": ["consultation", "reschedule", "dnc_processing", "escalate_to_human"],
        "preloads": [
            ("check_product_summary", {}),
        ],
    },
    "consultation": {
        "tools": ["check_product_detail", "check_product_summary", "lookup_faq",
                  "get_consent_status", "record_consent"] + _GLOBAL_TOOLS,
        "skills": ["product_manager"] + _GLOBAL_SKILLS,
        "transitions": ["reschedule", "dnc_processing", "escalate_to_human"],
        "preloads": [
            ("check_product_summary", {}),
        ],
    },
    "purchase": {
        "tools": [
            "check_product_detail",
            "set_product",
            "get_consent_status",
            "record_consent",
            "verify_personal_info",
            "get_disclosure_questions",
            "save_disclosure_answer",
            "save_payment_info",
            "save_address",
        ] + _GLOBAL_TOOLS,
        "skills": ["purchase_setup", "disclosure_manager", "payment_manager"] + _GLOBAL_SKILLS,
        "transitions": ["end", "escalate_to_human"],
    },
    "reschedule": {
        "tools": ["check_existing_schedule", "check_available_schedule", "reschedule"],
        "skills": ["schedule_manager"],
        "transitions": ["consultation", "end", "escalate_to_human"],
        "preloads": [
            ("check_available_schedule", {"date": "2026-04-07"}),
        ],
        "preload_skills": ["schedule_manager"],
    },
    "dnc_processing": {
        "tools": ["register_dnc", "check_dnc_status"],
        "skills": [],
        "transitions": ["end", "escalate_to_human"],
    },
    "end": {
        "tools": [],
        "skills": [],
        "transitions": [],
    },
}
```

This is a plain Python dict. The `StageMachine` reads it at `GatewayCore.setup()` time and builds transition logic from the `transitions` allowlist. `preloads` are tool calls that execute automatically on stage entry — the Gateway runs them before the LLM turn, so the LLM sees the results as if it had called the tools itself.

> **Notice:** `_GLOBAL_TOOLS = []` is intentionally empty — the Lina agent uses `enable_tool_router: true`, so native tools are all hidden. The LLM calls `use_tool(tool_name="check_product_summary", ...)`. The per-stage `tools` list in `stage_config.py` controls which tool names `ToolRouter.set_allowed_tools` permits at each stage. A tool not in the current stage's list cannot be dispatched even if the LLM requests it — the ToolRouter will reject it.

**What the Lina stage config shows about the framework's design ceiling:**

- 8 stages, each with its own tool set, skill set, allowed transitions, and preloads — all in one file under 75 lines.
- The framework itself has zero Lina-specific code. Every Lina behaviour (Korean insurance sales flow, product coverage checks, scheduling) lives in the agent folder, not in any package.
- Adding a new stage means adding a dict key and dropping new `.py` tool files. No framework changes required.

---

## Convention Summary

| File / Directory | Required? | What it controls | Who reads it |
|---|---|---|---|
| `BOSON.md` | **Yes** | LLM system prompt (identity, instructions, tool list) | `config/loader.py` → `AgentRuntime.system_prompt` |
| `config.yaml` | No | LLM provider/model, max_turns, permissions, ToolRouter flag | `config/loader.py` → `AgentConfig` |
| `tools/*.py` | No | Tool definitions (`@tool` functions) | `ToolRegistry.discover_tools` |
| `hooks/*.py` | No | Hook observers (`@hook` functions) | `HookRegistry.discover_hooks` |
| `skills/*.md` | No | Skill prompts (injected on `use_skill` call) | `SkillRegistry.discover_skills` |
| `layers/*/rules/*.py` | No (Gateway only) | Rule checks (`@check` functions, per layer) | Gateway rule loader |
| `stage_config.py` | No (Gateway only) | Stage definitions, tool/skill sets, transitions, preloads | `StageMachine` |
| `stages/*.md` | No (Gateway only) | Stage-specific system prompt text | `build_stage_injection` |
| `config.py` | No (Gateway only) | `GatewayConfig` (host, port, compact settings) | `GatewayCore.__init__` |

The framework only **requires** `BOSON.md`. Everything else is optional and additive. You cannot break a working agent by adding files — the registries fail-open on broken imports and skip missing directories.

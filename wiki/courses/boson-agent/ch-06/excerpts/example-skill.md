---
calling_spec:
  purpose: Annotated real skill files showing the format and production patterns
  chapter: ch-06
  course: boson-agent
  phase: read
  excerpt_of: read.md
---

# Example Skills — Annotated Real Files

Two skills from the repository illustrate the full range of what a skill file can be:
the minimal `explain.md` (2 lines, demo agent) and the production-grade
`consent_manager.md` (60 lines, Lina TMR sales agent). Both follow exactly the same
loading contract.

---

## `agents/demo/skills/explain.md` — Minimal Skill

```
# agents/demo/skills/explain.md, lines 1-2

Explain Skill
You are now in explain mode. When the user asks about a topic, explain it simply as if talking to a beginner. Use analogies and examples.
```

**What the loader does with this file:**

| Field | Value |
|---|---|
| `name` | `"explain"` (stem of filename) |
| `description` | `"Explain Skill"` (raw line 0, no stripping of `#`) |
| `prompt_template` | Full 2-line string (entire file content) |
| `file_path` | `Path("agents/demo/skills/explain.md")` |

When `use_skill(skill_name="explain")` is called, the entire `prompt_template` string —
both lines — is passed to `api.inject_system_reminder()`. The LLM receives it wrapped
in `<system-reminder>` tags as part of its next context window.

**Notice:** There is no Markdown heading syntax here. Line 0 is just `"Explain Skill"`,
not `"# Explain Skill"`. The loader takes `lines[0].strip()` literally — the description
field in `SkillSpec` is whatever the first line says, no parsing applied. If the author
wrote `# Explain Skill`, the description would include the `#` character.

---

## `agents/demo/skills/summarize.md` — Minimal Skill (variant)

```
# agents/demo/skills/summarize.md, lines 1-2

Summarize Skill
You are now in summarize mode. When the user provides text, summarize it concisely in 2-3 sentences. Focus on the key points and main takeaway.
```

Same structure as `explain.md`. Two skills registered, two entries in `SkillRegistry._skills`:

```python
{
    "explain":   SkillSpec(name="explain",   description="Explain Skill",   ...),
    "summarize": SkillSpec(name="summarize", description="Summarize Skill", ...),
}
```

The `use_skill` tool's `input_schema` does not enumerate valid `skill_name` values in
the JSON Schema — it only types the field as `"string"`. The LLM must know which skills
exist from the BOSON.md system prompt or from the tool description.

---

## `agents/test-lina/skills/consent_manager.md` — Production Skill

```markdown
# agents/test-lina/skills/consent_manager.md, lines 1-60

# Personal Information Consent (사전동의)

개인정보 수집·이용 동의를 항목별로 순서대로 진행합니다.

## 사전동의 안내문 (반드시 먼저 읽어주세요)

> "다음 동의는 가입설계 및 맞춤형 보험상담..."

## Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `get_consent_status` | — | 현재 동의 상태 조회 |
| `record_consent` | `item_number` (int 1-4), `agreed` (bool) | 개별 항목 기록 |
| `verify_personal_info` | — | 본인 확인 |

## Flow

1. 사전동의 안내문을 고객에게 읽어줍니다
2. `use_tool("get_consent_status", {})` — 현재 상태 확인
3. 첫 번째 미완료 항목의 동의 문구를 그대로 읽어줍니다
...

## Constraints

- 한 번에 하나의 항목만
- 동의 문구는 tool 응답 그대로
- 명확한 답변 확인
- 순서 준수 — 1 → 2 → 3 → 4
```

**What this demonstrates.**

A production skill is a complete behavioral contract, not a personality switch. It
specifies:

- **Which tools to call** (`get_consent_status`, `record_consent`, `verify_personal_info`)
- **In what order** (numbered Flow section)
- **Under what conditions** (Scenarios / Constraints)
- **What not to do** ("한 번에 하나의 항목만" — one item at a time)

The LLM receives this entire document as a system reminder. After injection, the LLM
is expected to follow the protocol autonomously, calling `use_tool(...)` as instructed
by the skill's Flow section.

**Notice:** The skill instructs the LLM to call `use_tool(...)` (the meta-tool), not
the domain tools directly. This is the correct pattern when `enable_tool_router: true`
— the LLM never calls `record_consent` directly; it always goes through `use_tool`.
The skill author must know whether the router is enabled and write the skill accordingly.
This is a coupling point: skill files are not fully portable between router-enabled and
router-disabled agents.

---

## `agents/test-lina/skills/product_manager.md` — Tool-Orchestrating Skill

```markdown
# agents/test-lina/skills/product_manager.md, lines 1-15

# Product Manager

Manages product comparison and switching during the sales conversation.

## Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `check_available_products` | — | List available products |
| `check_product_detail` | `keyword` (str, optional) | Coverage details |
| `set_product` | `plan` (str) | Switch plan |

## Scenarios

### A: Customer Asks for Cheaper Options
1. `use_tool("check_available_products", {})`
2. Present the options conversationally
3. Do NOT call `set_product` yet — wait for explicit choice
```

This skill coordinates four domain tools across three scenario branches. It is
effectively a mini state machine expressed in natural language. When injected, it gives
the LLM a decision procedure that governs the entire product-discussion phase of the
sales call — without any code-level state machine implementation.

The key design insight: **skills are code-free workflows**. The framework's job is to
inject the Markdown text into context at the right moment (when `use_skill` is called).
The LLM's instruction-following capability executes the workflow. The skill file is the
source of truth for the behavior; the framework is a delivery mechanism.

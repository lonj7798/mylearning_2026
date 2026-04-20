---
calling_spec:
  purpose: "Annotated comparison of demo and Lina stage configs as worked examples"
  parent: "[[../read.md]]"
  phase: read
  chapter: ch-12
  course: boson-agent
---

# Example Stage Configs — Comparative Walkthrough

> Sub-page of [[../read.md]]. Side-by-side annotated comparison of the two
> concrete stage configurations in the repository.

---

## The Two Reference Implementations

The repository ships two gateway agents with stage configurations:

| Agent | Stages | Purpose |
|-------|--------|---------|
| `demo-gateway` | 3 (welcome, main, closing) | Minimal teaching example |
| `test-lina-gateway` | 8 (introduction, product_focused, consultation, purchase, reschedule, dnc_processing, escalate_to_human, end) | Production-scale Korean insurance sales agent |

Reading both side by side reveals what is fixed by the framework and what is
a free design choice.

---

## Stage Graph Topology

**Demo gateway — linear funnel:**

```
welcome ──► main ──► closing (terminal)
   └───────────────────────────────────►┘
```

Both `welcome` and `main` can transition directly to `closing`. There is no
cycle and no backward edge. This is a simple linear funnel — appropriate for
a demo where stage mechanics are the point, not conversational complexity.

**Lina — branching DAG with multiple terminals:**

```
introduction ──► product_focused ──► consultation ──► purchase ──► end
     │                │                   │               │
     ▼                ▼                   ▼               ▼
  reschedule ◄────────┘               reschedule    escalate_to_human
  dnc_processing                      dnc_processing
  escalate_to_human                   escalate_to_human
```

Several stages can reach `reschedule`, `dnc_processing`, or `escalate_to_human`.
These "exit" stages lead to `end` or `escalate_to_human`. There is no path
back to `introduction` from any downstream stage — the sales funnel is
strictly forward-moving.

**Notice:** The topology is entirely defined in `stage_config.py`. The
`StageMachine` enforces it but does not constrain its shape. You could define
a graph with cycles (e.g., `consultation → product_focused`) and the machine
would enforce those transitions faithfully. The framework imposes no topology
constraints — that is a product design decision.

---

## Tool Set Growth Across Stages

**Demo gateway:**

| Stage | Tools |
|-------|-------|
| welcome | `get_time` |
| main | `calculate`, `get_weather`, `search_docs`, `get_time` |
| closing | `get_time` |

Tools expand dramatically in `main` (4x more) and contract back in `closing`.
This mirrors an onboarding-style UX: greet the user first, then unlock full
capabilities.

**Lina:**

| Stage | Tool count | Notable tools |
|-------|-----------|---------------|
| introduction | 0 | None — scripted persuasion only |
| product_focused | 3 | Product lookup, FAQ |
| consultation | 5 | + consent tools |
| purchase | 9 | + personal info, disclosure, payment, address |
| reschedule | 3 | Schedule-specific tools only |
| dnc_processing | 2 | DNC registration only |
| escalate_to_human | 1 | `escalate_to_human` only |
| end | 0 | Terminal — no tools |

The tool count grows as the conversation deepens and the agent's role
becomes more transactional. `purchase` has the most tools (9) because it
must collect consent, disclosure answers, payment, and address in one
uninterrupted flow. `end` and `introduction` have zero — they are conversation
boundary stages, not work stages.

---

## Preload Strategy Comparison

**Demo gateway — `main` stage:**
```python
"preloads": [("get_time", {"timezone": "UTC"})],
"preload_skills": ["explain"],
```
Preloads the current time (cosmetic context-setting) and injects the `explain`
skill prompt. This is a demonstration of the mechanic, not a production necessity.

**Lina — `product_focused` and `consultation`:**
```python
"preloads": [("check_product_summary", {})],
```
Both product-presentation stages preload the product summary. This is
functionally necessary — without it, the agent's first turn in the stage
would be a tool call to fetch the summary, wasting a user-visible latency
slot.

**Lina — `reschedule`:**
```python
"preloads": [("check_available_schedule", {"date": "2026-04-07"})],
"preload_skills": ["schedule_manager"],
```
Preloads available schedule slots AND the scheduling skill prompt. When the
agent enters `reschedule`, it immediately knows what slots are open and how
to present them. The user hears "I can offer you the following times" on the
agent's very first turn in the stage.

**Key difference:** Demo preloads are illustrative. Lina preloads are
functionally necessary for turn-economy — each preloaded tool call saves one
user-visible round trip.

---

## Transition Detection: How Rules Drive the Stage Graph

Stage configs declare *what* transitions are legal. Layer pipeline rules
decide *when* to fire them. The two mechanisms are completely decoupled.

**Demo gateway — deterministic + turn-count heuristic:**

```python
# boson-agent/agents/demo-gateway/layers/03-orchestrator/rules/stage_manager.py, lines 22-38

@check("auto_stage_transition", mode="sequential", priority=10)
def auto_transition(messages, user_message, session):
    intent = getattr(session, "data", {}).get("intent")
    active = getattr(session, "active_stage", None)

    if intent == "closing" and active != "closing":
        return StageTransition("closing")

    if active == "welcome" and session.turn_count > 1:
        return StageTransition("main")

    return Pass()
```

Two conditions: intent from `ctx.data` (set by Layer 02), and turn count.
Simple, deterministic, no LLM call.

**Lina — keyword match + LLM fallback + checklist:**

```python
# boson-agent/agents/test-lina-gateway/layers/03-orchestrator/rules/transition_detector.py, lines 345-396

@check("stage_transition", mode="parallel", priority=20, check_type="llm")
async def detect_stage_transition(messages, user_message, session):
    stage = getattr(session, "active_stage", None)
    lower = user_message.lower().strip()

    # 1. Try deterministic keyword match (fast, free)
    target = _deterministic_check(stage, lower, messages)

    if target == "escalate_to_human":
        session.escalate_count = getattr(session, "escalate_count", 0) + 1
        if session.escalate_count < 2:
            return Inject(content="[Customer requested human agent …]")
        return StageTransition("escalate_to_human")

    if target:
        return StageTransition(target)

    # 2a. Product checklist — all items must be checked
    if stage == "product_focused":
        target = await _llm_checklist(stage, messages, user_message, session)
        if target:
            return StageTransition(target)
        return Continue()

    # 2b. Fall back to LLM evaluation
    target = await _llm_check(stage, messages, user_message)
    if target:
        return StageTransition(target)

    return Continue()
```

Three-tier detection: deterministic keywords (O(1), free) → cached checklist
LLM (batched, result cached on session) → full LLM evaluation. Each tier is
only reached if the previous tier fails. Escalation has a 2-request debounce
before firing — a single "I want a human" does not immediately terminate the
call.

**Notice:** Both sets of rules return `StageTransition(target_name)`. The
`target_name` is validated by `StageMachine.transition()` against the
`transitions` allowlist in `stage_config.py`. The rule does not need to
know which transitions are legal — it just names a target. The machine
silently rejects illegal transitions. This is a clean separation of
*detection logic* (rules) from *transition policy* (config).

---

## What You Can Copy Directly Into Your Own Agent

For a new gateway agent, a minimal viable `stage_config.py` is:

```python
initial_stage = "greeting"

stages = {
    "greeting": {
        "tools": [],
        "skills": [],
        "transitions": ["main"],
    },
    "main": {
        "tools": ["your_tool_name"],
        "skills": [],
        "transitions": ["closing"],
    },
    "closing": {
        "tools": [],
        "skills": [],
        "transitions": [],
    },
}
```

Add `preloads` only when you need the LLM to have fresh data at stage entry
without spending a turn. Add `preload_skills` only when the stage needs a
specific behavioral mode from the first turn. The rest is product design.

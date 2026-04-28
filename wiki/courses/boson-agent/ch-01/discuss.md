---
chapter: ch-01
course: boson-agent
phase: discuss
tactic: blend
created_at: 2026-04-18T00:40:00Z
---

# ch-01 Discussion Transcript

## Teacher's initial assessment of summary.md

**Strong:**
- Decorator structure captured cleanly, with both simple and parameterized forms
- "Nested function blocks direct call" — the deferral insight landed
- Async-for-multi-session reason captured

**Errors surfaced:**
- Claimed 3 decorators: `@tool`, `@hook`, `@rule`. Third is `@check`, not `@rule`.
- Async reduced to ONE reason (multi-session). Missed LLM-streaming and transitivity.
- Meta-tool claim: "cannot systematically limit tool calling." Conflates with `PermissionChecker` + `ToolRouter.set_allowed_tools()`.
- "rules, tools, skills are based on the folder system" — rules are NOT folder-scanned; they're wired via `config.py`.
- `count_time` example missing `return result` (regressed from earlier corrected version).

**Gaps (absent from summary):**
- Hook vs Tool distinction
- `GatewayCore.handle_message` per-turn flow
- system-reminder embedding and immutability invariant
- InterruptHandler (admitted)

---

## Q&A exchanges

### Meta-tool tradeoff — learner's defense

Learner clarified: "meta-tool's rejection happens AFTER the LLM attempts. Native tool calling prevents the attempt itself by hiding disallowed tools."

→ Teacher validated and retracted pushback. Sharp insight. Summary updated to reflect the actual tradeoff:
- Native = no wasted attempts, but baked into API request (less flexible)
- Meta-tool = flexible runtime + fewer schema tokens, but wasted attempts possible
- `PermissionChecker` works below either as a systematic limit.

### Async — learner's answer to "nine two remaining reasons"

Learner: "other functions are already async so it has to support async."

→ This is reason #3 (transitivity). Reason #2 (LLM streaming returning `AsyncIterator` forcing the initial async contract) was missing. Teacher restored full 3-reason stack in summary.

### Hook vs Tool — "tool" answer was wrong

Challenge: "Load user profile from DB at conversation start — tool or hook?"

Learner answered "tool." Incorrect — a tool would require LLM to self-initiate the call (unreliable, wastes a turn). Correct answer: `@hook(HookEvent.ON_TURN_START)` that auto-preloads and injects via system-reminder.

Follow-up: learner self-diagnosed the confusion — they had been thinking of "hook" in the general programming sense (extension point), then conflated when asked about boson-agent's hook system. Core insight: "LLM doesn't need to decide."

→ Two concepts now distinguished:
- General programming hook = extension point pattern (React useEffect, git pre-commit)
- boson-agent hook = specific instance of that pattern at 9 lifecycle points, with the framework (not LLM) deciding when to fire.

---

## Final state

| Topic | Coverage |
|-------|----------|
| Decorator structure (nested + parameterized) | confirmed |
| Closure / deferral | confirmed |
| `@check` vs `@rule` naming | corrected |
| Async 3 reasons | restored (1 → 3) |
| Meta-tool vs Native tradeoff | refined |
| Hook vs Tool trigger model | corrected and internalized |
| Permission/Router limits | confirmed |
| Rules discovery ≠ folder-scan | corrected |

Open / not deeply probed:
- `GatewayCore.handle_message` full per-turn flow
- InterruptHandler (Basement SIGINT vs Gateway barge-in)
- system-reminder embedding + immutability invariant

Learner opted to move to ch-02 rather than deepen these. The open items are adjacent to ch-07–09 (Gateway chapters) and will recur naturally.

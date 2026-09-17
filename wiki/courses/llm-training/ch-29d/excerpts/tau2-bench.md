---
chapter: ch-29d
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/**/tau2-bench.md on 2026-09-15)
source_url: https://arxiv.org/abs/2506.07982
source_version: arXiv:2506.07982v1 (2025-06-09)
created_at: "2026-09-15"
---

# Excerpt: τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment (Barres, Dong, Ray, Si, Narasimhan)

Facts used by [[read]] §1, §2, and the Recipe table. Read in the arXiv v1 PDF on 2026-09-15.

## Dual control and the constrained user simulator (§1, §3)
- The telecom domain is a Dec-POMDP: "both agent and user make use of tools to act in a shared, dynamic environment" (Abstract).
- Design goal for user tools: keep "complexity asymmetry" between agent and user. The authors constrain the simulator by "ensuring user tools yield only human-readable outputs, limiting user planning to reactive tool use based on agent requests, and tightly constraining user behavior through the environment" (§1).
- Personas: each telecom task was randomly assigned None, Easy, or Hard; Easy describes a user familiar with the domain, Hard a user with low technical knowledge (§3.2, App. A.1). Agents score higher on Easy than Hard tasks, and None is "on par or lower" than Hard (§4.2, Figure 7).

## User simulator prompt (App. C.2, quoted lines)
- "Never make up or hallucinate information not provided in the scenario instructions."
- "Disclose information progressively. Wait for the agent to ask for specific information before providing it."
- "Only call a tool if the agent has requested it."
- "If you find yourself in a situation in which the scenario does not provide enough information for you to continue the conversation, generate the '###OUT-OF-SCOPE###' token to end the conversation."

## Settings (§4.1)
- User simulator gpt-4.1-2025-04-14; agents gpt-4.1-mini, gpt-4.1, o4-mini, claude-3-7-sonnet; each task run four times at temperature 0.
- Task success criteria available: DB check, status assertions, natural-language assertions, communication info check, action matching; telecom uses assertion functions only (§3.3).

## Results used in the chapter
- Telecom pass^1: gpt-4.1 34%, o4-mini 42%, claude-3.7-sonnet 49% (§1).
- Moving from No-User mode (agent controls all tools) to Default dual control lowers pass^1 by "18% for gpt-4.1 and 25% for o4-mini" (§4.2).

## User simulator quality (§4.3, Table 2, App. E)
- Two annotators reviewed each conversation (gpt-4.1 as both user and agent) against four criteria: adherence to simulator guidelines, adherence to user instructions, correct user-tool use, natural continuation. Errors are task-critical (preclude completion) or task-benign.

| Domain | Conversations | Critical errors | Benign errors | Total |
|---|---|---|---|---|
| airline | 100 | 13 (13%) | 34 (34%) | 47 (47%) |
| retail | 50 | 6 (12%) | 14 (28%) | 20 (40%) |
| telecom | 50 | 3 (6%) | 5 (10%) | 8 (16%) |

- Source inconsistency: the Table 2 caption says telecom had "no critical errors", while the table and §4.3 text report 3 (6%).
- Retail error types (App. E.1, 20 errors): conversation-structure rule violation 11, missing constraint 4, premature termination 3, ungrounded reference 2. "Most task-critical errors stem from either premature termination or missing constraints." The text announces three failure modes and lists four.
- Airline error types (App. E.2, 47 errors): conversation-structure violation 19, ungrounded reference 15, missing constraint 11, premature termination 2.
- The authors attribute the lower telecom error rate to a domain design that constrains user behavior through tools and observable state (§4.3). They have not tested the same method on retail or airline (§5).

## Limitation stated (§5)
τ²-bench "does not explicitly model the expert-novice gap inherent to most customer support tasks."

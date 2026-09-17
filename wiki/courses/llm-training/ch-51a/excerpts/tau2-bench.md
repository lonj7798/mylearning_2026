---
chapter: ch-51a
course: llm-training
phase: read
excerpt_of: "τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment (arXiv:2506.07982v1, 2025-06-09)"
source_url: https://arxiv.org/abs/2506.07982
created_at: "2026-09-15"
note: "No library card exists at wiki/raw-data/llm-training/**/tau2-bench.md on 2026-09-15. Values read from the v1 PDF on 2026-09-15."
---

# Excerpt: τ²-bench — dual control, mode ablations, and simulator error rates

**Authors:** Victor Barres, Honghua Dong, Soham Ray, Xujie Si, Karthik Narasimhan (Sierra; University of Toronto
and Vector Institute). Used by [[read]] §2, §5, §7, and the Recipe table.

## What changes relative to τ-bench (§1, §3.1; Table 1)

- The telecom domain is a Dec-POMDP in which "both agent and user make use of tools to act in a shared, dynamic
  environment" (Abstract). In retail and airline only the agent holds tools.
- Domain statistics (Table 1): retail 115 tasks, 7 write / 6 read agent tools; airline 50 tasks, 6 write / 6 read;
  telecom 114 evaluated tasks (2,285 in the full generated set), 6 write / 7 read agent tools and 15 write /
  15 read **user** tools.
- Telecom tasks are composed from 15 atomic subtask groups over 3 user intents, each assigned a persona
  (None, Easy, or a low-technical-knowledge persona) (§3.2, §3.3, App. A.1).
- Task success in telecom is scored by assertion functions on the final world state. The general criteria
  available are database check, status assertions, natural-language assertions, communication-info check, and
  action matching (§3.3).

## Harness settings (§4.1)

- Models: gpt-4.1-mini-2025-04-14, gpt-4.1-2025-04-14, o4-mini-2025-04-16, claude-3-7-sonnet-20250219.
  User simulator: gpt-4.1-2025-04-14.
- "Each task is run four times, maintaining a consistent LLM temperature of 0 to promote deterministic outputs."
  Tools are supplied in the OpenAI tools format through LiteLLM.
- Cost with gpt-4.1 on both sides: about $0.086 per task for the agent and $0.059 for the user simulator; one
  trial over all domains is about $40.

## Results used in the chapter (§1, §4.2, Figs. 3-4)

- gpt-4.1 pass^1: retail 74%, airline 56%, telecom 34%. o4-mini telecom 42%; claude-3-7-sonnet telecom 49%.
- Telecom mode ablation, original policy (Fig. 4 left, pass^1): gpt-4.1 0.34 default / 0.67 no-user /
  0.88 oracle-plan; o4-mini 0.42 default / 0.73 no-user / 0.96 oracle-plan. §4.2 states the default-to-no-user
  drop as "18% for gpt-4.1 and 25% for o4-mini"; the abstract and §1 summarize it as "around 20% pass^1".
- claude-3-7-sonnet telecom pass^1 (49%) is close to its airline score, but its pass^k falls faster with k on
  telecom (§4.2).

## User-simulator error audit (§4.3, Table 2, App. E)

Two annotators reviewed every conversation (gpt-4.1 as both user and agent) against four criteria: adherence to
simulator guidelines, adherence to user instructions, correct user-tool use, natural continuation.

| Domain | Conversations | Critical errors | Benign errors | Total |
|---|---|---|---|---|
| airline | 100 | 13 (13%) | 34 (34%) | 47 (47%) |
| retail | 50 | 6 (12%) | 14 (28%) | 20 (40%) |
| telecom | 50 | 3 (6%) | 5 (10%) | 8 (16%) |

- Source inconsistency: the Table 2 caption says telecom had "no critical errors" while the table and the §4.3
  text report 3 (6%).
- The authors attribute the lower telecom rate to a domain design that constrains user behaviour through tools and
  observable state, and state they have not tested the same method on retail or airline (§4.3, §5).

## Limit stated (§5)

τ²-bench "does not explicitly model the expert-novice gap inherent to most customer support tasks." Only four
proprietary models are evaluated; no confidence intervals are reported over the 4 trials per task.

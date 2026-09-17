---
chapter: ch-46a
course: llm-training
phase: read
excerpt_of: "τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment (arXiv:2506.07982v1, 2025-06-09)"
source_url: https://arxiv.org/abs/2506.07982
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug tau2-bench). Values read from the v1 PDF on 2026-09-15. Code and data: github.com/sierra-research/tau2-bench."
---

# Excerpt: τ²-bench — dual-control agent evaluation and pass^k

**Authors:** Victor Barres, Honghua Dong, Soham Ray, Xujie Si, Karthik Narasimhan (Sierra; University of Toronto and Vector Institute).

## What it measures (§1, §2, §3.1)

- τ-bench evaluates an agent talking to a simulated user in retail and airline domains; only the agent can call tools. τ²-bench adds a telecom domain modeled as a Dec-POMDP in which **both** the agent and the simulated user hold tools that change a shared world state.
- pass^k is defined, following τ-bench, as "the fraction of k independent runs that succeed" (§2). It is not pass@k: it goes down, not up, with k.
- Domain statistics (Table 1):

| | retail | airline | telecom |
|---|---|---|---|
| Agent databases | 500 users, 50 products, 1,000 orders | 500 users, 300 flights, 2,000 reservations | 5 plans, 9 lines, 4 customers |
| Agent tools | 7 write, 6 read | 6 write, 6 read | 6 write, 7 read |
| User tools | — | — | 15 write, 15 read |
| Tasks | 115 | 50 | 114 (full generated set: 2,285) |

- Telecom tasks are composed from 15 atomic subtask groups covering 3 user intents, and each task is assigned a persona (None, Easy, or a low-technical-knowledge persona) (§3.2, §3.3, App. A.1).
- Task success in telecom is scored only by assertion functions on the final world state; the general criteria available are database check, status assertions, natural-language assertions, communication-info check, and action matching (§3.3).

## Evaluation settings (§4.1)

- Models evaluated: gpt-4.1-mini-2025-04-14, gpt-4.1-2025-04-14, o4-mini-2025-04-16, claude-3-7-sonnet-20250219. The user simulator is gpt-4.1-2025-04-14.
- "Each task is run four times, maintaining a consistent LLM temperature of 0 to promote deterministic outputs." Tools are supplied in the OpenAI tools format through LiteLLM.
- Cost with gpt-4.1 on both sides: about $0.086 per task for the agent and $0.059 for the user simulator; one trial over all domains is about $40 (§4.1).

## Results (§1, §4.2, Figs. 3-4)

- gpt-4.1 pass^1: retail 74%, airline 56%, telecom 34%. gpt-4.1-mini, o4-mini, and claude-3.7-sonnet reach about 50% on telecom; claude-3.7-sonnet telecom pass^1 is 49%, on par with its airline score, but its pass^k falls faster with k on telecom.
- Telecom mode ablation, original policy (Fig. 4 left): gpt-4.1 pass^1 0.34 default, 0.67 no-user, 0.88 oracle-plan; o4-mini 0.42 default, 0.73 no-user, 0.96 oracle-plan. The abstract and §1 summarize the default-to-no-user gap as "around 20% pass^1".
- The telecom user simulator shows a 16% error rate with 6% critical errors, against 40% and 12% for the τ-bench retail simulator (§1).

## Limits stated or observable

- Only four proprietary models are evaluated; no fine-tuned open model appears in the paper.
- Task counts per domain are 50 to 115, so a per-domain score has wide sampling error; the paper reports pass^k over 4 trials per task and does not report confidence intervals.
- The telecom full task set (2,285) is generated; the evaluated subset is 114 tasks.

## Used in

ch-46a §7 (the tool-calling and dual-control slice of the generality gate, and why pass^k is reported next to pass^1).

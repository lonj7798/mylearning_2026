---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: primary source (no library card existed on 2026-09-15)
source_url: https://arxiv.org/abs/2506.07982
created_at: "2026-09-15"
---

# Excerpt: τ²-bench — dual-control telecom domain and reasoning vs coordination ablations

**Paper:** Victor Barres, Honghua Dong, Soham Ray, Xujie Si, Karthik Narasimhan, "τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment", arXiv:2506.07982 v1 (2025-06-09). Read 2026-09-15.

## What changes from τ-bench (§1, §3, Table 1)

- The new telecom domain is modeled as a Dec-POMDP: the user simulator also has tools and acts on a shared environment (for example toggling airplane mode) (§1, §3.1).
- Telecom: agent database of 5 plans, 9 lines, 4 customers; 6 write and 7 read agent tools; 15 write and 15 read user tools; 114 tasks (full set 2,285). Retail keeps 115 tasks and airline 50.
- Tasks are composed programmatically from atomic subtasks defined by initialization, solution, and assertion functions; telecom has 15 atomic subtask groups for 3 user intents (§3.2). Telecom success uses assertion functions only.
- User personas None, Easy, Hard are assigned to telecom tasks (§3.2).

## Results

- pass^1 on telecom: gpt-4.1 34%, o4-mini 42%, claude-3.7-sonnet 49% (§1). gpt-4.1 scores 74% retail and 56% airline (§4.1, Fig. 3). For claude-3.7-sonnet, telecom pass^k falls faster with k than airline (§4.1).
- Modes (§4.2, Fig. 4): Default (dual control), No-User (the agent gets a ticket and controls all tools), Oracle Plan (the agent receives the required call sequence). Moving from No-User to Default lowers pass^1 by "18% for gpt-4.1 and 25% for o4-mini". A workflow-style policy document helps Default and No-User and hurts Oracle Plan for both models.
- User-simulator errors (Table 2): airline 47% of 100 conversations (13% critical); retail 40% of 50 (12% critical); telecom 16% of 50 (6% critical).

## Relevance for training data

The ablation separates tool-reasoning failures from failures of communicating with a user who shares control. A model trained only on single-control trajectories (where the user never acts) is not tested on the second skill by τ-bench retail or airline.

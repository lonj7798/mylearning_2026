<!-- scope: τ-bench (arXiv:2406.12045, Sierra) and τ²-bench (arXiv:2506.07982) — tool-agent-user benchmarks with LLM user simulator, database-state reward, pass^k; τ² dual-control telecom domain with compositional task generator and simulator error audit
     deps: [[apigen-mt]]
     see-also: [[agentscaler]], [[simulator-collapse]], [[bfcl]], [[multichallenge]], [[user-oriented-dialogue-generation]], [[diversity-axes]], [[rl-design-guide]]
-->

# τ-bench and τ²-bench: tool-agent-user interaction with a simulated user
- **Core Insight:** A task reward that compares the final database with a unique gold database (times a check that required strings appear in replies) makes multi-turn customer-service tasks gradable, and repeating each task k times exposes inconsistency: τ-bench's best agent had pass^1 61.2 on retail but pass^8 under 25% (τ-bench §3, §5.1, Table 2); τ²-bench shows that letting the user also act on the environment (dual control) costs 18% (gpt-4.1) and 25% (o4-mini) pass^1 relative to the agent acting alone (τ² §4.2, Fig. 4).
- **Guideline:** When building RL tasks for multi-turn agents, compose tasks from atomic subtasks that each have initialization, solution and assertion functions, so task difficulty (subtask and action count) is a controlled variable, and audit the user simulator on a sample of conversations, because τ²-bench pass^1 falls toward zero beyond 7 required actions (§4.2, Fig. 5) and simulator error rates ranged from 16% (telecom) to 47% (airline) of conversations (Table 2).
- **Authors:** τ-bench: Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan (Sierra). τ²-bench: Victor Barres, Honghua Dong, Soham Ray, Xujie Si, Karthik Narasimhan.
- **Year:** 2024 (τ-bench, arXiv v1 17 Jun 2024); 2025 (τ²-bench, arXiv v1 9 Jun 2025)
- **URL:** https://arxiv.org/abs/2406.12045 ; https://arxiv.org/abs/2506.07982
- **Source type:** paper (benchmark)
- **Relevant topics:** user simulators, state-based rewards, pass^k reliability, compositional task generation, task horizon, dual control

## Summary
τ-bench: retail (115 tasks, 7 write + 8 non-write tools) and airline (50 tasks, 6 write + 7 non-write tools); the agent receives a Markdown domain policy as its system prompt; a gpt-4-0613 user simulator receives a hidden task instruction and ends with "###STOP###" (user T = 1.0, agent T = 0.0). Reward r = r_action × r_output ∈ {0, 1}. pass^k = E_task[C(c, k) / C(n, k)], the probability that all k trials succeed. τ²-bench adds a telecom domain modeled as a Dec-POMDP where the user has tools (15 write, 15 read) as well as the agent (6 write, 7 read) (§3, Table 1).

## Key Contributions
- pass^k as a reliability metric distinct from pass@k (τ-bench §3).
- Database-state reward with output substring check (τ-bench §3).
- Compositional task generator: 15 atomic subtask groups over 3 intents; composing at most one subtask per group gives 2,285 tasks, 114 subsampled (τ² §3.2, Table 3).
- User-simulator error audit (τ² §4.3, Table 2).

## Key Figures/Tables to Study
- τ-bench Table 2 (pass^1), Fig. 6 (difficulty vs number of database writes), §5.2 failure analysis, Table 3 (policy ablation). τ² Fig. 4 (No-User vs Default), Fig. 5 (pass^1 vs actions), Tables 2–4.

## Technical Details
**τ-bench results.** gpt-4o pass^1: 61.2 retail, 35.2 airline (Table 2). Failure analysis on 36 gpt-4o agent failures (retail): wrong decision 25%, partial resolution of compound requests 19% (§5.2). Tasks with more database writes are harder (Fig. 6). Removing the policy document: retail 61.2 → 56.8, airline 33.2 → 10.8 (Table 3).

**τ²-bench results.** Telecom pass^1: gpt-4.1 34% (vs 74% retail, 56% airline), claude-3.7-sonnet 49% (§1, §4.2). Mean actions by intent: service_issue 2.31, mobile_data_issue 4.31, mms_issue 6.00 (Table 4). pass^1 trends down as distinct subtasks per task increase (Fig. 5).

**Simulator audit (Table 2; gpt-4.1 user).** Airline 100 conversations: 13% critical, 34% benign errors; retail 50: 12% / 28%; telecom 50: 6% / 10%. The authors attribute telecom's lower rate to user tools and observable state constraining the simulator.

## Verification
- Read τ-bench HTML v1 and τ²-bench HTML v1 (first 100K characters each) on 2026-11-24.
- τ-bench's "wrong information" share is garbled in the extracted text and is not used. o4-mini's telecom pass^1 is stated as 42% in τ² §1 while §4.2 says others are "around 50%"; not used.
- These are benchmarks; neither paper trains a policy. [[simulator-collapse]] measures RL against a τ²-bench user simulator.

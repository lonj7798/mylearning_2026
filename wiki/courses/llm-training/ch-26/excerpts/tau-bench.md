---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: primary source (no library card existed on 2026-09-15)
source_url: https://arxiv.org/abs/2406.12045
created_at: "2026-09-15"
---

# Excerpt: τ-bench — tool-agent-user tasks, state-based reward, and pass^k

**Paper:** Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan (Sierra), "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains", arXiv:2406.12045 v1 (2024-06-17). Read 2026-09-15.

## Setup (§3–§4, Table 1)

- An agent receives a domain policy (as system prompt) and API tools over a JSON database; a user is simulated by an LM from a task instruction.
- τ-retail: 500 users, 50 products, 1,000 orders; 7 write and 8 non-write tools; 115 tasks. τ-airline: 500 users, 300 flights, 2,000 reservations; 6 write and 7 non-write tools; 50 tasks.
- Instructions are written so that only one outcome is possible under the policy (§3).
- Reward `r = r_action × r_output ∈ {0, 1}`: the final database must equal the ground-truth database, and agent messages must contain the required information. The authors note that r = 1 is necessary but not sufficient (a return issued without explicit user confirmation still scores 1) (§3).
- At most 30 agent actions per task; agent temperature 0.0, user temperature 1.0; at least 3 trials per task for Table 2 (§5).

## pass^k (§3)

"pass^k (pass hat k), defined as the chance that all k i.i.d. task trials are successful, averaged across tasks." With n trials and c successes for a task:

```
pass^k = E_task[ C(c, k) / C(n, k) ]
pass@k = 1 − E_task[ C(n − c, k) / C(n, k) ]
```

pass^1 = pass@1 = E[c/n]. The same user prompt and database transitions are used across trials; variation comes from LM sampling of user and agent messages.

## Results (Table 2, Fig. 4)

- pass^1 (retail / airline / average by domain): gpt-4o 61.2 / 35.2 / 48.2; gpt-4-turbo 57.7 / 32.4 / 45.1; claude-3-opus 44.2 / 34.7 / 39.5; meta-llama-3-70B (text ReAct) 14.8 / 14.4 / 14.6.
- gpt-4o function calling on τ-retail: pass^8 below 25% while pass^1 is above 60% (§5.1, Fig. 4).
- Native function calling outperforms text ReAct and Act for the strongest models (Fig. 3). A "think" function did not improve FC agents (§5.1).
- Cost: gpt-4o agent with gpt-4 user on τ-retail, $0.38 agent and $0.23 user per task (§5.1).

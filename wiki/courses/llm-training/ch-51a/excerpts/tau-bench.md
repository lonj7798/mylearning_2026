---
chapter: ch-51a
course: llm-training
phase: read
excerpt_of: "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains (arXiv:2406.12045v1, 2024-06-17)"
source_url: https://arxiv.org/abs/2406.12045
created_at: "2026-09-15"
note: "No library card exists at wiki/raw-data/llm-training/**/tau-bench.md on 2026-09-15. Values read from the v1 PDF on 2026-09-15."
---

# Excerpt: τ-bench — state-based reward, pass^k, and the reliability gap

**Authors:** Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan (Sierra). Used by [[read]] §4 and §7.

## Setup (§3, §4, §5; Table 1)

- An agent holds a domain policy and API tools over a JSON database; a user is simulated by a language model
  from a written task instruction.
- τ-retail: 500 users, 50 products, 1,000 orders; 7 write and 8 non-write tools; 115 tasks.
  τ-airline: 500 users, 300 flights, 2,000 reservations; 6 write and 7 non-write tools; 50 tasks (Table 1).
- Reward is rule-based: the final database must equal the annotated goal database, and the agent's messages must
  contain required substrings. The authors state that "r = 1 might be a necessary but not sufficient condition for
  a successful episode e.g., the agent might issue the return without explicit user confirmation, which violates
  the policy" (§3).
- Harness settings (§5): at most 30 agent actions per task (tool calls or user responses); agent temperature 0.0,
  user temperature 1.0; at least 3 trials per task for Table 2.

## pass^k and pass@k (§3, verbatim definitions)

pass@k is "the chance that at least one out of k i.i.d. task trials is successful". pass^k is "the chance that all
k i.i.d. task trials are successful, averaged across tasks". With n trials per task and c successes:

```
pass^k  = E_task[ C(c, k)     / C(n, k) ]
pass@k  = 1 − E_task[ C(n−c, k) / C(n, k) ]
```

"By default, we report the average reward across tasks, pass^1 = pass@1 = E[r] = E[c/n], as the main metric."
For the same task the user prompt and database transitions are fixed; the stochasticity comes from language-model
sampling of the user and agent messages (§3).

## Results (Table 2, Fig. 3, Fig. 4)

- pass^1 via function calling (retail / airline / domain-weighted average, %): gpt-4o 61.2 / 35.2 / 48.2;
  gpt-4-turbo 57.7 / 32.4 / 45.1; gpt-4-32k 56.5 / 33.0 / 44.8; claude-3-opus 44.2 / 34.7 / 39.5;
  mistral-large 30.7 / 22.4 / 26.6; gpt-3.5-turbo 20.0 / 10.8 / 15.4; meta-llama-3-70B (text-ReAct)
  14.8 / 14.4 / 14.6. The average is weighted by domains, not by tasks (Table 2 caption).
- "Even for the best-performing gpt-4o function calling agent which has a > 60% average task success, pass^8 drops
  to < 25%" on τ-retail (§5.1, Fig. 4).
- Native function calling outperforms text-formatted ReAct and Act for the strongest models (Fig. 3). Adding a
  "think" function to the function-calling agent "did not boost performance" (§5.1).
- Cost: gpt-4o agent with a gpt-4 user simulator on τ-retail costs $0.38 (agent) and $0.23 (user) per task, about
  $200 for one trial over all tasks; 95.9% of agent cost is input tokens (§5.1).

## Limits stated or observable

- Figure 4 prints the pass^k curve only as a chart; per-k numeric values other than the "< 25% at k = 8" statement
  are not given in text.
- 165 tasks total across two domains, so a per-domain score has wide sampling error; the paper reports no
  confidence intervals.

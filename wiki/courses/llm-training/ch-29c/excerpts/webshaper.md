---
chapter: ch-29c
course: llm-training
phase: read
excerpt_of: "WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization, Tongyi Lab, arXiv:2507.15061v1 (2025-07-20)"
source_url: https://arxiv.org/abs/2507.15061
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug webshaper). Values read from the v1 PDF on 2026-09-15."
---

# Excerpt: WebShaper — formalization-driven synthesis of information-seeking tasks

**Authors:** Zhengwei Tao, Jialong Wu, Wenbiao Yin, Junkai Zhang, Baixuan Li, Haiyang Shen, et al. (Tongyi Lab, Alibaba Group).

## Problem (§1)
Information-driven synthesis (retrieve first, then write a question) can produce "inconsistency between information structure and reasoning structure, as well as between the question and the corresponding answer" (Abstract).

## Formalization (§2)
- Knowledge Projection (Eq. 2): R(V) = {u | ∃v ∈ V, (u, v) ∈ R or (v, u) ∈ R}, where E is the set of entities, R ⊆ E × E a relation, V ⊆ E.
- R-Union (Eq. 3) and Intersection (Eq. 4); a target set T = R_1(T_1) ∩ ... ∩ R_k(T_k) (Eq. 6); a task is q(T) ≜ ?T (Eq. 7).
- Example (Eq. 10): q(T) ≜ ?T s.t. [[V@T, playIn, V@X], [V@T, playAt, C@2004_05], [V@T, bornIn, C@90s], [V@X, foundIn, C@1966], [V@X, isA, C@East German football team]].
- Proposition 1: R(S_1) ∪ R(S_2) = R(S_1 ∪ S_2).

## Seeds (§3.1)
- Offline Wikipedia with preserved hyperlinks; random walks; an LLM writes QA pairs grounded only in the visited articles.
- Filter: 5 rollouts per question with the WebDancer framework on QwQ; keep if at least one rollout is correct. Result: 18k seed questions.

## Layer-wise expansion (§3.2.2-3.2.3)
- Random and sequential expansion create "Redundancy" (constants linked to constants) and "Reasoning Shortcut" (a constant linked directly to the target).
- Layer-wise: find all leaf constants; the Expander turns one constant into a sub-question whose answer is that constant, and merges it: q_{n+1}(T) = Expander(C, q_n(T)) (Eq. 11). "the q_{n+1}(T) always has the same answer as q_n(T)."
- Expander tools: Search (Google, with year filter), Summarize (visit URLs and form a union constant set), Validate (QwQ called once to check that the constant's type fits the sub-question, and once to answer the sub-question; if the prediction equals the constant, the sub-question is rejected as too simple).

## Trajectories and training (§3.3-3.4)
- QwQ-based ReAct agent with Search and Visit; 5 rollouts per question; a judge LLM keeps correct answers; trajectories with tool-call errors, guessed observations, or severe repetition removed. "We finally obtain 5,000 trajectories."
- SFT loss masks observation tokens (Eq. 12); RL with GRPO with ε_low and ε_high clipping (Eq. 13); values not printed.

## Results (Tables 1-2, Fig. 6-7)
- GAIA (LLM-judge Pass@1, text information-seeking subset): WebShaper 52.4 (Qwen-2.5-32B), 53.3 (QwQ-32B), 60.1 (Qwen-2.5-72B); WebWalkerQA 51.4 / 49.7 / 52.2.
- Same-size SFT data comparison, 5,000 samples per dataset, GAIA average (Table 2): Qwen-2.5-32B WebWalkerQA 32.0, E2HQA 39.8, MHQA 35.9, WebShaper 43.6; QwQ-32B 45.6 / 45.6 / 41.7 / 53.3.
- RL after SFT on GAIA: +7.8 (32B), +13.5 (72B) (§4.3.3).
- Formalized vs natural-language expansion and layer-wise vs sequential expansion: formalized and layer-wise better on all three backbones (Fig. 7; values only in the figure).

## Not reported
Overlap checks between the Wikipedia-seeded tasks and GAIA or WebWalkerQA.

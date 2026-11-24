<!-- scope: AgentScaler, "Towards General Agentic Intelligence via Environment Scaling" (arXiv:2509.13311, Tongyi Lab) — 30K+ APIs clustered into 1,000+ database-backed domains; tasks sampled as walks on tool-dependency graphs; state-match + exact-match trajectory filtering; two-stage SFT
     deps: [[kimi-k2-agentic-data]], [[apigen-mt]]
     see-also: [[agent-world]], [[toolace]], [[tau2-bench]], [[bfcl]], [[websailor-tongyi-deepresearch]], [[diversity-axes]], [[rl-design-guide]]
-->

# Towards General Agentic Intelligence via Environment Scaling (AgentScaler)
- **Core Insight:** Treating every tool as a read or write operation on a domain database makes environments generable and verifiable at scale: over 30,000 APIs are grouped by Louvain community detection into more than 1,000 domains, each with a generated schema and Python tool implementations, and a trajectory is accepted only if the final database state matches the gold state (or, for read-only tasks, the calls exactly match) (§2, §2.1, §3.1); accuracy on τ-bench falls as the number of tool calls per task grows ("clear negative correlation", Fig. 5).
- **Guideline:** When building agentic tasks for many domains, sample tool sequences as directed walks over a per-domain tool-dependency graph from a sampled initial database state, and grade by final-state match rather than by matching the call sequence, because state matching accepts different valid call orders while exact match is kept only for read-only tasks where state does not change (§2.2, §3.1); keep trajectories that hit tool errors but still reach the goal, which the authors do for robustness (§3.1).
- **Authors:** Runnan Fang, Shihao Cai, Baixuan Li, Jialong Wu, Guangyu Li, Wenbiao Yin, Xinyu Wang, Xiaobin Wang, Liangcai Su, Zhen Zhang, Shibin Wu, Zhengwei Tao, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou (Tongyi Lab, Alibaba Group)
- **Year:** 2025 (arXiv v1, 16 Sep 2025)
- **URL:** https://arxiv.org/abs/2509.13311
- **Source type:** paper
- **Relevant topics:** environment scaling, domain diversity, tool-graph task sampling, state-based verification, long-horizon tool calling

## Summary
AgentScaler builds simulated but executable environments: tools from ToolBench, API-Gen and an internal repository are linked when their parameter vectors exceed a cosine-similarity threshold, an LLM checks dependencies within each domain, and each domain is materialized as a database schema plus code. Tasks are produced by sampling an initial state and a tool walk, generating arguments and executing them. Agent trajectories are collected (simulated user, tool execution) and filtered in three stages, then used for two-stage fine-tuning (general domains first, vertical domains second). The training reported is SFT with loss only on tool calls and assistant responses.

## Key Contributions
- Database-as-environment abstraction for tool use (§2).
- Domain construction by tool-graph clustering, "exceeding 1,000" domains (§2.1).
- Three-stage trajectory filter: validity (role alternation, n-gram repetition), environment-state alignment, exact match for read-only tasks (§3.1).
- Two-stage training and results on τ-bench, τ²-Bench, ACEBench (Table 1).

## Key Figures/Tables to Study
- Fig. 1–2 (pipeline), Table 1 (main results), Table 2 (ACEBench-zh OOD), Fig. 3 (stage ablation), Fig. 4 (pass^k), Fig. 5 (accuracy vs number of tool calls).

## Technical Details
**Results (Table 1; pass^1 for τ / τ², accuracy for ACEBench-en).**

| Model | τ-Retail | τ-Airline | τ²-Retail | τ²-Airline | τ²-Telecom | ACE overall |
|---|---|---|---|---|---|---|
| AgentScaler-4B | 64.3 | 54.0 | 62.3 | 56.0 | 48.2 | 65.9 |
| AgentScaler-8B | 50.4 | 42.0 | 58.8 | 44.0 | 45.4 | 67.4 |
| AgentScaler-30B-A3B | 70.4 | 54.0 | 70.2 | 60.0 | 55.3 | 75.7 |

**OOD (Table 2, ACEBench-zh overall).** Qwen3-Thinking-4B 43.9 → 65.6; Qwen3-8B 71.3 → 73.7; Qwen3-Thinking-30B-A3B 74.2 → 81.5.

**Backbones.** Qwen3-Thinking-4B-2507, Qwen3-8B, Qwen3-Thinking-30B-A3B-2507.

## Verification
- Read arXiv HTML v1 on 2026-11-24. The similarity threshold τ is not given.
- Not in the paper: an ablation on the number of environments or domains, and any analysis of parallel vs sequential tool calls. The environment-count effect is measured in [[agent-world]] instead.
- Stage-ablation and Fig. 5 values are figure-only and not transcribed.

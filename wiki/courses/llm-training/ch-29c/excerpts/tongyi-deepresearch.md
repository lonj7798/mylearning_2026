---
chapter: ch-29c
course: llm-training
phase: read
excerpt_of: "Tongyi DeepResearch Technical Report, Tongyi DeepResearch Team, arXiv:2510.24701v3 (2026-05-18; v1 2025-10)"
source_url: https://arxiv.org/abs/2510.24701
created_at: "2026-09-15"
source_type: official technical report
note: "No library card exists for this source on 2026-09-15 (planned slug tongyi-deepresearch). Values read from the v3 PDF on 2026-09-15."
---

# Excerpt: Tongyi DeepResearch — environments and data for a 30B-A3B research agent

## Three environment forms (§2)
- **Prior World Environment:** "provides task elements, tools, and state definitions, allowing agents to autonomously mine interaction trajectories based on pretrained knowledge without receiving actual environmental responses. It offers perfect stability, zero interaction cost, and unlimited scalability, but lacks real-world feedback signals."
- **Simulated Environment:** "controlled, reproducible replicas of real-world interactions locally ... However, its data coverage is inherently limited, exhibiting a notable sim-to-real gap."
- **Real-world Environment:** "the most authentic data distribution and feedback signals ... the cost is expensive interactions, significant non-stationarity, and exploration risks."
- Mid-training uses the prior-world and simulated forms; post-training validates strategies in simulation and then trains in the real environment.

## Mid-training (§3.3)
- Base: Qwen3-30B-A3B-Base; 30.5B total, 3.3B activated parameters (Abstract).
- Two-stage Agentic CPT: 32K, then 128K with "a substantial corpus of long-sequence (64K-128K) agentic behavior data"; "a small proportion of general pre-training data is interleaved" (share not printed).
- Data synthesized for question synthesis, planning action, reasoning action, decision-making action; function-calling data from environment scaling with "each environment instantiated as a read–write database" (the AgentScaler method, cited as Fang et al., 2025).

## Post-training data (§3.4.1)
- Knowledge graph built by random walks with web search and real-world tables; subgraph sampling; "strategically increasing the uncertainty within the question"; set-theoretic formalization (WebShaper, cited as Tao et al., 2025) for controlled expansion and QA verification.

## SFT (§3.4.2)
- Trajectories from open-source models, rejection sampled. ReAct mode and Context Management mode samples. Stage 1 context 40K; stage 2 128K with ReAct samples of 40K-128K plus a small portion of 40K data.
- Over 20% of SFT samples exceed 32k tokens and involve more than 10 tool invocations (§4.4).

## RL (§3.4.3)
- Real environment: Search, Visit, Python Interpreter, Google Scholar, File Parser behind a sandbox with QPS limits, caching, timeout-and-retry, and failover to backup data sources.
- Simulated environment: "an offline environment based on the 2024 Wikipedia database" with local RAG tools; the reward curve there "closely matches the one observed in the real environment" (§4.4, Fig. 10b vs Fig. 8).
- GRPO variant (Eq. 4-5), strict on-policy, reward "a pure 0 or 1 signal of answer correctness", no format reward, token-level loss and clip-higher following DAPO, leave-one-out baseline.
- > "directly optimizing on an unfiltered set of negative rollouts significantly degrade training stability and can lead to policy collapse after extended training. To mitigate this, we selectively exclude certain negative samples from the loss calculation, for instance, those that do not yield a final answer because they exceed a length limit."
- Data curation: remove problems the SFT model always fails or always solves; refresh the active set with problems that became moderately difficult for later checkpoints, when a step count is reached or reward plateaus.
- > "the success of agentic RL depends more on the quality of the data and the stability of the training environment than on the specific algorithm being used." (authors' conclusion; no ablation table)
- Model merging: weighted parameter average of variants from the same base (Eq. 6); weights not printed.

## Evaluation (§4.1-4.2, Table 1)
- Temperature 0.85, repetition penalty 1.1, top-p 0.95, at most 128 tool calls, 128K context, Avg@3.
- Tongyi DeepResearch (30B-A3B): HLE 32.9, BrowseComp 43.4, BrowseComp-ZH 46.7, GAIA 70.9, xbench-DeepSearch 75.0, WebWalkerQA 72.2, FRAMES 90.6.
- General benchmarks (Fig. 11) compare the agent with tools against base models without tools; this does not measure forgetting.

## Not reported
RL learning rate, clip values, group size, number of RL steps beyond figures, data sizes per stage, overlap checks against the evaluation benchmarks.

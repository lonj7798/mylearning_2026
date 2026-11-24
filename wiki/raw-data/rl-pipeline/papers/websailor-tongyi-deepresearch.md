<!-- scope: Tongyi Lab web-agent line: WebSailor (SailorFog-QA graph+obfuscation synthesis, RFT cold start, DUPO on verl, synchronous, 50 RL steps) and the Tongyi DeepResearch 30B-A3B report (agentic CPT 32K→128K, SFT 40K→128K, step-level async on-policy GRPO variant on rLLM, real+simulated Wiki environments, dynamic data curation, model merging)
     deps: [[search-r1]], [[grpo]], [[dapo]]
     see-also: [[asearcher]], [[deepresearcher]], [[zerosearch]], [[gemini-2.5-deep-research]], [[rloo]], [[recipe-crosswalk]]
-->

# WebSailor and Tongyi DeepResearch: data synthesis + agentic RL for web agents
- **Core Insight:** Both Tongyi Lab reports put most of the pipeline effort in data and environment, not the RL algorithm. WebSailor synthesizes "Level 3" questions (high uncertainty, no predefined reasoning path) by sampling subgraphs of random-walk knowledge graphs and obfuscating facts, then runs RFT cold start → RL (WebSailor §3, §4). The Tongyi DeepResearch report states: "the success of agentic RL depends more on the quality of the data and the stability of the training environment than on the specific algorithm being used" (§3.4.3).
- **Guideline:** For a deep-research pipeline, (1) synthesize hard QA with controlled obfuscation and verify it, (2) cold-start with rejection-sampled, observation-masked trajectories (WebSailor keeps only correct trajectories ≤ 32k tokens with > 5 tool calls), (3) run group-relative RL with token-level loss and clip-higher, remove zero-variance groups, and exclude truncated negatives, (4) wrap real tools in a sandbox with QPS limits, caching, retry and failover, and keep a simulated Wiki environment for fast algorithm iteration (WebSailor §4.1-4.2; Tongyi §3.4.2-3.4.3).
- **Authors:** WebSailor: Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, et al. Tongyi DeepResearch: Tongyi DeepResearch Team. Both Tongyi Lab, Alibaba Group.
- **Year:** 2025 (WebSailor arXiv:2507.02592v1, 2025-07-03; Tongyi report arXiv:2510.24701, v3 2026-05-18)
- **URL:** https://arxiv.org/abs/2507.02592 ; https://arxiv.org/abs/2510.24701 (code: https://github.com/Alibaba-NLP/DeepResearch)
- **Source type:** paper + technical report
- **Relevant topics:** web-agent data synthesis, RFT cold start, DUPO, agentic mid-training, context management, asynchronous on-policy rollout, simulated vs real environments, dynamic data curation, model merging

## Summary
**WebSailor.** ReAct agent with `search` and `visit` tools, tool calls capped at 30 (App. A.3). Training: RFT cold start on reconstructed trajectories, then DUPO RL, on Qwen2.5-3B/7B/32B/72B (§4, §5.1). Megatron for SFT, verl for RL (App. A.4). RL is "limited to 50 steps" because of the synchronous framework (§5.4).
**Tongyi DeepResearch.** Qwen3-30B-A3B-Base (30.5B total, 3.3B active) → Agentic CPT stage 1 (32K) → stage 2 (128K, with 64K-128K agentic data) → agentic SFT → agentic RL → model merging (§3.2, Fig. 2; §3.4.4). Tools: Search, Visit, Python Interpreter, Google Scholar, File Parser (§3.1).

## Key Contributions
- SailorFog-QA: Wikidata-seeded knowledge graph grown by probabilistic node expansion; sampled subgraphs become QAs; obfuscation of dates, names, quantities raises initial uncertainty (WebSailor §3.1).
- Trajectory reconstruction: keep the expert LRM's action-observation trace, discard its verbose thoughts, regenerate short "why" thoughts per step with another LLM (WebSailor §3.2).
- DUPO: filter all-correct items before training; during training, fill slots of zero-std groups by duplicating non-zero-std groups in the same batch instead of resampling; ~2-3× faster than DAPO dynamic sampling (WebSailor §4.2).
- Tongyi: Markovian context management (condition on question, evolving report S_t, last action/observation) trained jointly with ReAct-mode data (§3.1, §3.4.2).
- Tongyi: unified tool sandbox, simulated 2024-Wikipedia environment, step-level asynchronous on-policy RL on rLLM, background data-refresh pipeline (§3.4.3).

## Key Figures/Tables to Study
- WebSailor Table 1 (BrowseComp-en/zh, xbench-DeepSearch, GAIA); Table 2 (SailorFog-QA pass@1 vs WebDancer-QA); Fig. 6 (direct RL vs RFT cold start).
- Tongyi Fig. 2 (pipeline), Fig. 5 (RL framework), Fig. 8 (reward and entropy), Fig. 9 (32k/48k/64k RL context), Fig. 10b (simulated-environment reward curve), Table 1.

## Technical Details
**Pipeline view.**
| Component | WebSailor | Tongyi DeepResearch |
|---|---|---|
| Prompt source | SailorFog-QA (graph + subgraph + obfuscation) (§3.1) | KG random walks + web tables, subgraph/subtable QAs, uncertainty "atomic operations", set-theory formalization; PhD-level question engine (§3.4.1) |
| Cold start | RFT: correct-only, ≤ 32k tokens, > 5 tool calls; observations masked (§4.1) | SFT on rejection-sampled trajectories, ReAct + context-management modes; 40K stage then 128K stage (§3.4.2) |
| Environment | search + visit, ≤ 30 calls (App. A.3) | real tools behind a sandbox (QPS limits, caching, timeout-retry, failover to backup search API) + simulated Wiki-2024 RAG env (§3.4.3) |
| Grader | R = 0.1·R_format + 0.9·R_answer; answer by LLM judge (§4.2, Eq. 5) | 0/1 answer correctness; no format reward because cold start already fixes format (§3.4.3) |
| Loss | GRPO-style group-normalized advantage, token-level loss, clip-higher; observations masked (§4.2, Eq. 4) | GRPO variant, token-level loss, clip-higher (ε_low, ε_high), leave-one-out baseline Â = R_i − mean; exclude some negatives, e.g., length-truncated with no answer (§3.4.3, Eq. 4-5) |
| Infra | verl, synchronous; group 8, temp 1.0, batch 128, mini-batch 32, LR 1e-6 (App. A.4) | step-level async, strict on-policy (ratio stays 1.0); separate inference and tool servers; rLLM (§3.4.3) |
| Data curation | drop items with 8/8 correct before training (§4.2) | drop always-fail/always-succeed under the SFT model; refresh active set from a backup pool sampled by intermediate checkpoints (§3.4.3) |

**Stated reasons for modifications (Tongyi §3.4.3).** Unfiltered negative rollouts "significantly degrade training stability and can lead to policy collapse after extended training"; the modifications target "a more efficient and stable training paradigm", not algorithmic novelty.

**Results.** WebSailor Table 1 (BrowseComp-en / BrowseComp-zh / xbench-DeepSearch / GAIA): WebSailor-7B 6.7 / 14.2 / 34.3 / 37.9; WebSailor-72B 12.0 / 30.1 / 55.0 / 55.4; WebDancer-32B 2.5 / 14.1 / 38.7 / 40.7. SailorFog-QA pass@1 before filtering: o4-mini 47.3, DeepSeek-R1 38.9, vs WebDancer-QA 90.2 / 84.4 (Table 2). Tongyi Table 1 (Avg@3): HLE 32.9, BrowseComp 43.4, BrowseComp-ZH 46.7, GAIA 70.9, xbench-DeepSearch 75.0, WebWalkerQA 72.2, FRAMES 90.6; eval with ≤ 128 tool calls and 128K context, temperature 0.85 (§4.1).

**Ablations and dynamics.** WebSailor Fig. 6: direct RL on Qwen2.5-32B-Instruct rises faster in Pass@1 but converges lower than RFT-then-RL, with fewer tool calls; the gap is widest on BrowseComp-en (§5.3). Tongyi Fig. 9: with curation done by a 64k model, 64k RL reaches the highest reward and lengthens responses, 48k stays flat, 32k shortens responses (§4.4). Tongyi Fig. 10b: the simulated-Wiki reward curve "closely matches" the real-environment curve (§4.4). Over 20% of SFT samples exceed 32k tokens and 10 tool calls (§4.4).

## Connections
- [[asearcher]]: the same "Beyond ten turns" problem solved with fully asynchronous (off-policy-tolerant) RL; Tongyi instead keeps strict on-policy and lists partial rollouts as future work (§5 limitations).
- [[zerosearch]]: simulated search for training; compare Tongyi's simulated Wiki environment.
- [[dapo]]: source of token-level loss, clip-higher, and dynamic sampling that DUPO replaces.

## Verification
- Created 2026-11-24 from PDFs of arXiv:2507.02592v1 (§1-5.4, App. A.3-A.4) and arXiv:2510.24701v3 (§3.1-4.4, §5 limitations).
- Not stated in either report: RL step count for Tongyi DeepResearch, RL GPU count, KL coefficient.

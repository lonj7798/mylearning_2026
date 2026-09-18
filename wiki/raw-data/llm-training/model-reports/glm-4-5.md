<!-- scope: GLM-4.5 technical report (arXiv:2508.06471): 355B-total / 32B-active MoE, 23T-token pre-training with 4K→32K→128K mid-training (repo code, synthetic reasoning, synthetic agent trajectories), expert-model iteration with self-distillation, reasoning / agentic / general RL, slime RL infrastructure
     deps: [[grpo]]
     see-also: [[glm-4-5-recipe]], [[glm-4]], [[deepseek-v3]], [[kimi-k2]], [[qwen-3]], [[kimi-k2-agentic-data]], [[apigen-mt]]
-->

# GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models
- **Core Insight:** GLM-4.5, a Mixture-of-Experts model with 355B total and 32B activated parameters, is trained on 23T tokens and post-trained by first training Reasoning, Agent, and General-chat expert models and then distilling them into one hybrid-reasoning model; it scores 70.1% on TAU-Bench, 91.0% on AIME 24, and 64.2% on SWE-bench Verified (Abstract, §3).
- **Guideline:** When SFT has already trained a model to produce 64K-token responses, run reasoning RL directly at the 64K output limit rather than in stages of increasing length, because in the report's smaller-model ablation the single-stage run reached 83.4% AIME 24 Avg@32 against 80.6% for the multi-stage run, whose shorter early stages reduced output length and accuracy (§3.2, Figure 6).
- **Authors:** GLM-4.5 Team, Zhipu AI & Tsinghua University (byline). §6 lists core contributors alphabetically: Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al.; tech leads Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou.
- **Year:** 2025 (arXiv v1 2025-08; no later version)
- **URL:** https://arxiv.org/abs/2508.06471 (models and code: https://github.com/zai-org/GLM-4.5)
- **Source type:** official technical report
- **Relevant topics:** MoE architecture, Muon optimizer, mid-training, repo-level code data, synthetic reasoning data, synthetic agent trajectories, long-context extension, expert-model distillation, rejection sampling, function-call templates, GRPO, difficulty curriculum, agentic RL, instruction-following RL, RL infrastructure

## Abstract
GLM-4.5 is an open-source MoE language model with 355B total and 32B activated parameters that supports a thinking mode and a direct-response mode. It is trained in multiple stages on 23T tokens and post-trained with expert model iteration and reinforcement learning. It scores 70.1% on TAU-Bench, 91.0% on AIME 24, and 64.2% on SWE-bench Verified, and ranks 3rd overall and 2nd on agentic benchmarks among the models evaluated on 12 ARC benchmarks (Figure 1, as of July 28, 2025). The authors also release GLM-4.5-Air (106B total parameters).

## Key Contributions
- A deeper and narrower MoE than DeepSeek-V3 and Kimi K2, with 96 attention heads at hidden size 5120, QK-Norm, and one MoE layer used as an MTP layer for speculative decoding (§2.1, Table 1).
- Pre-training followed by three mid-training stages that extend sequence length from 4K to 32K to 128K and add repo-level code, synthetic reasoning traces, long documents, and synthetic agent trajectories (§2.2-§2.3, Figure 3).
- Expert Model Iteration: Stage 1 trains Reasoning, Agent, and General-chat experts (cold-start SFT, then RL); Stage 2 distills them into one model with an SFT set of millions of expert samples (§3, §3.1).
- RL methods with ablations on a smaller model: GRPO without the KL loss term, a two-stage difficulty curriculum, single-stage 64K RL, dynamic sampling temperature, token-weighted loss for code, and expert-verified data for science (§3.2); agentic RL on web search and SWE with iterative self-distillation (§3.3); general RL in four tracks (§3.4).
- The slime RL framework with synchronous colocated and asynchronous disaggregated modes and FP8 rollouts (§3.5).
- A function-call template with XML-like tags that reduces character escaping in code arguments (§3.1, Figure 4).

## Key Figures/Tables to Study
- Table 1 (architecture vs DeepSeek-V3 and Kimi K2) and Figure 3 (stage token counts and sequence lengths).
- Figures 5, 6, 7: curriculum, output-length, loss-aggregation, and science-data ablations (smaller experimental model).
- Figure 8: BrowseComp accuracy vs test-time compute; Figure 9: instruction-following RL reward vs SysBench-ISR.
- Tables 3-6: agentic, reasoning, coding, and general-chat results; Table 7: SafetyBench; Tables 8-12 and Figures 12-13: human evaluations and CC-Bench.

## Technical Details
Recipe values with loci: [[glm-4-5-recipe]]. This section summarizes mechanisms.

**Architecture.** GLM-4.5 has 3 dense layers, 89 MoE layers, and 1 MTP layer; 160 routed experts with 8 active plus 1 shared; hidden dimension 5120; 96 heads with 8 key-value heads (Table 1). GLM-4.5-Air has 106B total and 12B activated parameters, 45 MoE layers, and 128 routed experts (Table 1). Routing uses loss-free balance routing with sigmoid gates; attention is grouped-query attention with partial RoPE (§2.1). The authors report that 2.5 times more heads did not lower training loss but improved MMLU and BBH, and that deeper models showed better reasoning (§2.1); no numbers are printed.

**Pre-training data.** Web pages are split into quality-score buckets following Nemotron-CC; the top bucket contributes over 3.2 epochs and the lowest is discarded (§2.2). SemDedup removes template-generated similar pages that MinHash deduplication misses (§2.2). Code is filtered by rules and language-specific quality models into three tiers, low-quality code is excluded, and Fill-In-the-Middle is applied to all source code (§2.2). Math and science documents are scored by an LLM for educational content, a small classifier is trained on those scores, and documents above a threshold are up-sampled (§2.2). Stage 1 trains mainly on general web documents; stage 2 up-samples code, math, and science (§2.2); Figure 3 labels these 15T and 7T tokens.

**Mid-training.** (1) Repo-level code: files from one repository are concatenated, and model-filtered GitHub issues, PRs, and commits are concatenated into one context with commits in a diff-like format, at 32K (§2.3). (2) Synthetic reasoning: questions and answers from webpages and books receive reasoning processes generated by a reasoning model (§2.3). (3) Long-context and agent: long documents are up-sampled and large-scale synthetic agent trajectories are added at 128K (§2.3). Figure 3 labels these stages 500B, 500B, and 100B tokens. Best-fit packing is used only in mid-training (§2.3).

**Optimization.** Muon (N = 5, μ = 0.95, update RMS 0.2) for all parameters except embeddings, biases, and RMSNorm weights; cosine LR from 2.5e-4 to 2.5e-5; batch 16M → 64M tokens over the first 500B tokens; weight decay 0.1; RoPE base 10,000 → 1,000,000 at 32K (§2.4). A cosine schedule was chosen because early WSD runs were worse on SimpleQA and MMLU, which the authors read as underfitting in the stable stage (§2.4).

**SFT.** Cold-start SFT uses a small set with extended chain-of-thought responses (§3.1). Overall SFT uses millions of expert samples (reasoning, general chat, agentic tasks, long-context understanding) at up to 128K tokens, with data with and without explicit thinking balanced to produce both modes (§3.1). Rejection sampling removes (1) repetitive, short, truncated, or badly formatted samples, (2) samples with wrong objective answers, (3) subjective responses filtered by reward models, and (4) tool trajectories that violate call protocol or miss the expected terminal state (§3.1). Agentic SFT data is built in four steps: collect frameworks, real APIs, MCP servers, and LLM-simulated tools; synthesize single-step and multi-step tasks; generate trajectories with existing LLMs, using an LLM user simulator to turn multi-step tasks into multi-turn dialogues; keep only trajectories that multiple judge agents mark as completed (§3.1).

**Reasoning RL.** Built on GRPO without the KL loss term (§3.2). Batches where all rewards are 1 or all are 0 give no gradient signal, so stage 2 of the curriculum switches to problems with pass@8 = 0 and pass@512 > 0 drawn from a pool with verified answers; this reached 83.4% vs 81.8% AIME 24 Avg@32 (Figure 5). RL stages with maximum lengths shorter than the SFT-trained 64K reduced average output length and caused a performance drop that the report calls "difficult to recover from" in the final 64K stage (§3.2, Figure 6). Temperature is raised when average reward stabilizes, up to the highest value that costs no more than 1% on a held-out set (§3.2).

**Agentic RL.** Web-search data combines multi-hop reasoning over knowledge graphs with human-in-the-loop extraction and selective obfuscation of web content; SWE data uses GitHub PRs and issues with executable unit tests in a sandbox (§3.3.1). For each problem, K traces are sampled from the previous policy and each is weighted by its reward minus the group-mean reward; only model-generated tokens enter the loss (§3.3.2). Rewards are final-answer accuracy (web search) and test results (SWE); a wrong tool-call format halts the trace with reward 0 (§3.3.2). Iterative distillation: after RL plateaus, the cold-start data is replaced by responses of the RL model, SFT is repeated, and RL continues on harder tasks (§3.3.2). BrowseComp accuracy rises with interaction turns (Figure 8).

**General RL.** Holistic RL: roughly 5,000 prompts over 7 primary, 33 secondary, and 139 tertiary categories, rewarded by a reward model trained on human preferences and by AI rubric scoring (§3.4). Instruction-following RL: 7 major and 151 minor constraint types, with rules, a reward model, and a critique model (§3.4). Function-calling RL: step-wise rule-based RL (reward 1 only for correct format and an exact match to the ground-truth call) plus end-to-end multi-turn RL on MCP-synthesized tasks, AgentGym, and LLM-simulated users; the end-to-end experts are distilled into the main model (§3.4). Pathology RL targets language mixing, excessive repetition, and formatting mistakes with prompts selected because they are likely to trigger them; the report describes general RL as the final stage of post-training (§3.4).

**Infrastructure.** slime uses Megatron for training and SGLang with a router for rollout (Figure 10). Synchronous colocated mode is used for general-purpose and reasoning RL; disaggregated asynchronous mode is used for agentic tasks such as SWE, with a Docker-based runtime and a unified HTTP endpoint writing to a shared data pool (§3.5).

**Evaluation.** GLM-4.5: TAU-Retail 79.7, TAU-Airline 60.4, BFCL V3 77.8, BrowseComp 26.4 (Table 3); the 70.1% TAU-Bench figure equals the mean of the two domains (derived: (79.7 + 60.4) / 2 = 70.05). GPQA 79.1, LiveCodeBench (2407-2501) 72.9, HLE 14.4 (Table 4); Terminal-Bench 37.5 (Table 5); IFEval 86.1, SysBench 81.0, SimpleQA 26.4 (Table 6); SafetyBench average 89.9 (Table 7). TAU-bench uses the authors' optimized user-simulator prompt (Figure 11).

## Recipe ledger
Full table (pre-training, mid-training, SFT, RL, evaluation settings; "not reported" rows included): [[glm-4-5-recipe]].

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality.** The authors state that RL on web-search and SWE tasks "leads to generalized performance improvements across other tasks and benchmarks, such as general tool usage and coding tasks like Terminal-Bench" (§3.3.2); no ablation or numbers are given (Result (single study), qualitative). They state that a trained LLM may overfit predefined benchmarks and add human evaluation on 660 real-scenario user prompts (392 English, 108 Chinese, 160 other), where GLM-4.5 scores 8.66 / 8.37 / 8.49 (§4.3.1, Tables 8-10), and a new logic set built to reduce contamination, where it scores 62.0 (§4.3.3, Table 11). The report attributes the lower SimpleQA and MMLU scores of early WSD runs to underfitting in the stable stage (§2.4); no numbers are printed.
- **Negative feedback.** In SFT data construction, failed samples are discarded (negative marginal value): rejection sampling (§3.1) and judge-agent filtering of agent trajectories (§3.1). Format violations receive reward 0 (§3.3.2); in a group whose mean reward is above 0 such a trace gets a negative weight (derived from the §3.3.2 objective). Pathologies occur in "often less than 1% of outputs", so penalizing them inside general RL is described as sample-inefficient (§3.4). Rules plus reward model plus critique model "mitigated reward hacking" in instruction-following RL up to about 1,000 steps (Figure 9).
- **Long context.** Sequence length goes 4K → 32K → 128K during mid-training (§2.3); Overall SFT runs at 128K (§3.1); RL stages shorter than the SFT output length cause the model to "unlearn" long outputs (§3.2).
- **Agentic training.** Synthetic agent trajectories enter at mid-training (§2.3), the four-step SFT pipeline follows (§3.1), and RL uses outcome rewards with a format penalty (§3.3.2). The XML-tag call template is reported not to lower function-call performance (§3.1); no numbers are printed.
- **Distillation.** Unified SFT distills three experts (§3.1); mid-training reasoning traces come from a reasoning model (§2.3); agent RL alternates RL and self-distillation (§3.3.2). Dropping the bottom 50% of prompts by response length gave +2%-4% on math and science with half the data, and 4 responses per prompt gave another +1%-2% (§3.1).

## Connections
- [[grpo]] — the GRPO objective that §3.2 uses without the KL loss term.
- [[deepseek-v3]] and [[kimi-k2]] — the MoE architectures compared in Table 1 and baselines in Tables 2-5.
- [[qwen-3]] — Qwen3-235B-A22B is a baseline in Tables 2 and 4; that report also combines thinking and non-thinking modes in one model.
- [[deepseek-r1]] — DeepSeek-R1-0528 is a baseline in Tables 4-6 and in the human evaluations (Tables 8-10).
- [[bfcl]] — BFCL V3 is one of the three agentic benchmarks (Table 3).
- [[kimi-k2-agentic-data]] and [[apigen-mt]] — other agentic data pipelines with simulated tools or users, for comparison with the §3.1 pipeline.
- [[sequence-packing]] — packing background for the §2.3 choice of best-fit packing in mid-training only.
- [[glm-4]] — the earlier GLM family report; GLM-4.5 is described as the team's first MoE model (§1).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2508.06471 (v1, 2025-08-08; a v2 identifier does not exist)
- Corrections to the previous card version:
  - Title "GLM-4.5" → exact report title; "Authors / Lab: Z.ai (Zhipu AI)" → GLM-4.5 Team, Zhipu AI & Tsinghua University (byline).
  - Core Insight "RL infrastructure choice should be made per task type" → §3.5 reports that different RL tasks benefit from different scheduling (synchronous for general-purpose and reasoning RL, asynchronous for agentic tasks); this is one infrastructure observation, so the Core Insight now states the main result (Abstract, §3).
  - "slime ... SGLang-native" → slime uses SGLang with a router for rollout and Megatron for training (Figure 10). "Sync is more effective for math/code where rollouts are similar-length" → the stated reason is reduced GPU idle time with dynamic sampling (§3.5).
  - "RL algorithm not specified" → GRPO without the KL loss term (§3.2); group-wise policy optimization for agentic RL (§3.3.2).
  - "Reward model not disclosed" → a reward model trained on human preference annotations (holistic RL) and rules + reward model + critique model (instruction-following RL) (§3.4). "KL / entropy handling not disclosed" → KL loss excluded; dynamic sampling temperature (§3.2).
  - "SFT data: concrete sizes not disclosed" → Overall SFT uses "millions of samples" at 128K (§3.1). "Hyperparameters: not disclosed" → pre-training and mid-training hyperparameters are in §2.4; SFT and RL optimizer settings are not reported.
  - "64K-vs-progressive ablation is described qualitatively without numbers" → Figure 6: 83.4% vs 80.6% AIME 24 Avg@32 on the smaller experimental model.
  - "exact agentic-task environments not disclosed" → web-search and SWE data construction (§3.3.1) and MCP / AgentGym tasks for function-calling RL (§3.4) are described.
- Removed as unsupported by the source: the "Innovations vs predecessors" section ("first GLM generation with open RL framework", "prior GLM generations ran monolithic RL", "GLM-4 targeted conversational + reasoning without explicit agentic leg", "up from GLM-4's smaller corpus"), because the report makes no comparison with GLM-4 beyond calling GLM-4.5 the team's first MoE model (§1); "a notable negative result on curriculum length scheduling" (evaluative wording); "slime's async mode is conceptually similar to Kimi's partial-rollout infra"; "Two-stage difficulty curriculum timeline" as a figure (Figure 5 is an accuracy curve).
- Not reported by the source: SFT and RL learning rates, batch sizes, epochs; group size K and samples per prompt for GLM-4.5 (only the smaller model's Figure 5 legend); clip ε; number of RL steps; cold-start data size; pre-training mixture percentages; warm-up length; gradient clipping; reward-model sizes; compute.

<!-- scope: Search-R1: outcome-reward RL (PPO/GRPO on verl) for interleaved reasoning and local-Wikipedia search; <search>/<information>/<answer> rollout protocol, retrieved-token loss masking, EM reward, 4-action budget
     deps: [[grpo]], [[ppo]]
     see-also: [[zerosearch]], [[asearcher]], [[deepresearcher]], [[websailor-tongyi-deepresearch]], [[gemini-2.5-deep-research]], [[verl-rollout]], [[recipe-crosswalk]]
-->

# Search-R1: Training LLMs to Reason and Leverage Search Engines with RL
- **Core Insight:** A search engine can be placed inside the RL environment with three additions to an R1-Zero-style loop: tag-delimited tool calls parsed at rollout time (Alg. 1), a loss mask that excludes retrieved tokens from the policy-gradient and KL terms (§3.1, Eq. 2-3), and an exact-match outcome reward with no format reward (§3.4, Eq. 4). Removing the mask lowers the 7-dataset EM average from 0.431 to 0.343 (Table 4).
- **Guideline:** In any search RL pipeline, mask tool-returned tokens from the loss, start from an outcome reward on a short-answer dataset, keep the action budget small for short-hop QA, and prefer PPO over GRPO when long runs collapse: the paper reports GRPO converges faster but shows reward collapse after many steps while PPO stays stable (§5.1, Fig. 2a).
- **Authors:** Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Ö. Arık, Dong Wang, Hamed Zamani, Jiawei Han (UIUC, UMass Amherst, Google Cloud AI Research)
- **Year:** 2025 (arXiv v1 2025-03; v5 2025-08-05; COLM 2025)
- **URL:** https://arxiv.org/abs/2503.09516 (code: https://github.com/PeterGriffinJin/Search-R1)
- **Source type:** paper
- **Relevant topics:** search-agent RL, tool-augmented rollouts, retrieved-token loss masking, outcome reward (EM), PPO vs GRPO stability, base vs instruct starting points

## Summary
Search-R1 trains Qwen2.5 3B/7B/14B (base and instruct) to interleave `<think>` reasoning with `<search>` queries against a local retriever (2018 Wikipedia dump, E5 retriever, top-3 passages) (§4.3). Training data is the union of the NQ and HotpotQA training sets; evaluation covers 7 QA datasets, 2 in-domain and 5 out-of-domain (§4.1, §4.3). The RL objective extends the KL-regularized objective to trajectories sampled with the retriever in the loop, π_θ(·|x; R) (§3.1, Eq. 1; App. A). Reported gains: 24% (7B) and 20% (3B) average relative improvement over RAG baselines (abstract; §4.4).

## Key Contributions
- Rollout protocol: generate until `</search>`, `</answer>` or `<eos>`; on a search tag, call R(q) and append `<information>d</information>`; on an answer tag, stop; otherwise append "My action is not correct. Let me rethink." and continue; stop after budget B actions (Alg. 1).
- Retrieved-token loss masking: I(y_t)=1 for LLM-generated tokens, 0 for retrieved tokens, applied in the PPO/GRPO normalization and in the KL term (§3.1).
- Template constrains structure only (think → search → answer) and avoids content-specific guidance, "so the model's natural learning dynamics during the RL process remain observable" (§3.3, Table 1).
- Empirical studies: PPO vs GRPO, base vs instruct, response-length dynamics, top-k passages, GRPO group size (§5; App. D-H).

## Key Figures/Tables to Study
- **Alg. 1:** the multi-turn rollout loop (the environment step for a search agent).
- **Table 2:** main EM results, 3B and 7B; **Table 5 (App. C):** 14B.
- **Table 3 / Fig. 2a:** PPO vs GRPO; **Table 4 / Fig. 3:** with vs without retrieved-token mask.
- **Fig. 2c-d:** response length decreases then increases; number of valid searches rises during training.
- **Table 7 (App. G):** top-k = 1/3/5; **Table 8 (App. H):** GRPO group size 1/3/5.

## Technical Details
**Pipeline view.**
| Component | Search-R1 setting | Location |
|---|---|---|
| Prompt source | NQ train ∪ HotpotQA train | §4.3 |
| Environment / tool | local retriever: 2018 Wikipedia, E5, top-3 passages, ≤500 tokens retrieved content | §4.3; App. B.2 |
| Rollout shape | interleaved think/search/answer, action budget B = 4; max response 500 tokens; max sequence 4,096 | App. B.2; Alg. 1 |
| Grader | rule-based EM of extracted `<answer>` vs gold; no format reward; no neural RM | §3.4, Eq. 4 |
| Loss masking | retrieved tokens excluded from policy loss and KL | §3.1 |
| Algorithm | PPO default (GAE λ=1, γ=1); GRPO with 5 samples/prompt | §4.3; App. B.2 |
| Infra | verl; 1 node × 8 H100; FSDP with CPU offload; vLLM rollout TP=1, GPU mem 0.6; synchronous | App. B.2 |
| Stage position | single RL stage from base or instruct (no SFT cold start) | §4.3-4.4 |

**Hyperparameters (App. B.2).** PPO: policy LR 1e-6, value LR 1e-5, 500 steps, warm-up ratio 0.285 (policy) and 0.015 (value). Batch 512, mini-batch 256, micro-batch 64. Temperature 1.0, top-p 1.0; KL β = 0.001; clip ε = 0.2. Checkpoints every 100 steps; when training diverges, the most recent stable checkpoint by train reward is evaluated.

**Main results (Table 2, EM average over 7 datasets).** Qwen2.5-7B: RAG 0.304, R1-base (RL without search) 0.276, rejection sampling with search 0.348, Search-R1-base 0.431, Search-R1-instruct 0.385. Qwen2.5-3B: RAG 0.270, Search-R1-base 0.303, Search-R1-instruct 0.325. Qwen2.5-14B (Table 5): RAG 0.281, Search-R1-base 0.479.

**PPO vs GRPO (Table 3).** 7B-base: PPO 0.431 vs GRPO 0.350; 7B-instruct: PPO 0.385 vs GRPO 0.396; 3B-base: PPO 0.303 vs GRPO 0.312; 3B-instruct: PPO 0.325 vs GRPO 0.336. The authors attribute faster GRPO convergence to PPO's critic warm-up and report GRPO reward collapse in long runs (§5.1).

**Loss masking (Table 4, 7B-base, PPO).** With mask: NQ 0.480, HotpotQA 0.433, avg 0.431. Without mask: NQ 0.388, HotpotQA 0.325, avg 0.343.

**Rollout length dynamics (§5.3, 7B-base).** First ~100 steps: response length drops sharply while reward rises slightly; after ~100 steps, length and reward both increase as the model calls search more often and retrieved passages lengthen the sequence.

**Top-k and group size (App. G-H).** top-k = 1/3/5 averages 0.375/0.431/0.400 (Table 7); top-k = 5 converges fastest but becomes unstable. Larger GRPO groups converge faster but risk collapse; group size 1 (REINFORCE) trains more stably and generalizes better (App. H).

## Source-internal conflicts
- Abstract and §4.4 report 24% (7B) and 20% (3B) relative improvement over RAG; the contributions list in §1 says 41% and 20%.

## Connections
- [[zerosearch]]: replaces the real retriever with a simulated search LLM during training.
- [[asearcher]]: long-horizon (up to 128 turns) asynchronous version of the same idea; reports that short turn limits like Search-R1's cap learned behaviour.
- [[deepresearcher]]: same outcome-reward, masked-observation recipe on live web search.
- [[recipe-crosswalk]]: search RL pipeline synthesis.

## Verification
- Created 2026-11-24 from arXiv:2503.09516v5 PDF (text extraction): §3.1-3.4, Alg. 1, §4.1-4.4, Tables 2-4, §5.1-5.4, App. B.2, App. C-H read directly.
- Not read: App. I-J case studies (only headings checked).

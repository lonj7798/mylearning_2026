<!-- scope: RL on GitHub pull-request data with a similarity-based rule reward; single-turn issue repair, no test execution during training
     deps: [[grpo]]
     see-also: [[swe-gym]], [[agentinstruct]], [[openhands-data]]
-->

# SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
- **Core Insight:** GRPO on 273k GitHub pull-request seeds with a reward equal to the `difflib.SequenceMatcher` similarity between the predicted and the oracle patch lifts Llama-3.3-70B-Instruct to a 41.0% solve rate on SWE-bench Verified under the Agentless Mini scaffold, without running any unit tests during training (§2, §3.2 Table 1).
- **Guideline:** When execution environments for the training issues are unavailable, use a continuous similarity reward against the human patch rather than a binary exact-match reward, because the continuous variant produced higher repair performance while the discrete reward's average stayed near zero at the end of training (§3.6, Figure 5).
- **Authors:** Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, et al. (Meta AI, UIUC, CMU)
- **Year:** 2025 (arXiv v1 2025-02; v2 2025-12); NeurIPS 2025
- **URL:** https://arxiv.org/abs/2502.18449
- **Source type:** paper
- **Relevant topics:** RL with rule-based rewards, GRPO, software engineering, SWE-bench, out-of-domain generalization

## Abstract
SWE-RL applies reinforcement learning to real-world software engineering using a rule-based reward: the similarity score between an LLM-generated patch and the developer's oracle patch taken from open-source pull requests. Trained on top of Llama-3.3-70B-Instruct, the resulting model Llama3-SWE-RL-70B solves 41.0% of SWE-bench Verified, which the paper reports as the best result for models below 100B parameters at the time. Although RL is run only on issue-solving data, the model improves on five out-of-domain task categories — function coding, library use, code reasoning, mathematics, and general language understanding — whereas a supervised fine-tuning baseline trained from the same base model declines on average.

## Key Contributions
- A rule-based reward for issue repair: −1 for a malformed response, otherwise the `difflib.SequenceMatcher` ratio between the predicted and oracle patch, in [0, 1] (§2.1, Eq. 1).
- A seed RL dataset of 273k pull-request instances selected from about 11M unique raw PR instances (§2, §A).
- Llama3-SWE-RL-70B, reaching 41.0% pass@1 on SWE-bench Verified with the Agentless Mini scaffold (§3.2, Table 1).
- Evidence that single-task RL on issue repair improves five out-of-domain benchmarks while an SFT baseline on the same base model declines on average (§3.5, Table 3).
- Agentless Mini, a pipeline scaffold that does file-level localization only and supports scaling repair samples and reproduction tests independently (§3.1, §B).

## Key Figures/Tables to Study
- Figure 1: the pipeline from GitHub PRs to seed RL dataset to GRPO.
- Figure 2: the prompt template, which asks for a `<think>` block followed by a `<solution>` block of search/replace edits.
- Table 1: SWE-bench Verified leaderboard comparison.
- Table 2: repair-only comparison against the base model and the SFT baseline.
- Table 3: five out-of-domain benchmarks for base, SFT, and RL models.
- Figure 4: scaling with repair samples and reproduction tests.
- Figure 5: continuous versus discrete reward.

## Technical Details
- Raw collection yields about 11M unique PR instances after bot, size, and hunk-level filters; 273k are selected as the seed RL dataset using heuristics such as having a linked bug-fix issue and touching programming files (§2, §A).
- The prompt contains the issue description plus code context, including all changed files and some relevant unchanged files identified by prompting Llama-3.1-70B-Instruct (§2, §A). The model must emit search/replace edits after a reasoning block (Figure 2).
- Reward: R(o) = −1 if o has the wrong format, otherwise compare(patch_pred, patch_gt), implemented as `difflib.SequenceMatcher` (§2.1, Eq. 1).
- Optimization: GRPO with the clipped objective and a KL term βD_KL(π_θ ‖ π_ref); ϵ and β are stated as hyperparameters but no numeric values are given (§2.1, Eq. 2).
- Training: 1,600 steps, 16k context window, global batch size 512 formed by 16 rollouts from each of 32 problems, one Adam step per global step, 512 NVIDIA H100 GPUs, about 32 wall-clock hours (§3.1).
- Training is single-turn and does not execute tests; tests are run only during SWE-bench evaluation (§2, §3.1).
- Main evaluation generates 500 patches per problem at temperature 1.0 and reranks using the top 30 reproduction tests; only the top-ranked patch is submitted (§3.1).
- SWE-bench Verified pass@1 (Table 1): Llama3-SWE-RL-70B 41.0 with Agentless Mini; Llama3-SWE-SFT-70B 36.2 with Agentless Mini. Reference rows in the same table include GPT-4o with Agentless 38.8 and Claude-3.5-Sonnet with Agentless 50.8.
- Repair-only comparison with oracle files in context and greedy decoding (Table 2): Llama-3.3-70B-Instruct 12.2% correct format and 5.4 repair performance; with 20-sample majority voting 44.6% and 16.6; SFT baseline 96.2% and 29.6; RL model 95.6% and 34.8.
- Sample scaling (§3.4, Figure 4): 33.6 at 20 repair samples rising to 40.0 at 160, then 41.0 at 500; reproduction tests improve the score from 38.8 to 41.0 up to 20 tests with no change between 20 and 30.
- Out-of-domain results, base / SFT / RL (Table 3): HumanEval+ 76.2 / 73.2 / 79.9; BigCodeBench-Hard (I) 28.4 / 25.7 / 28.4; BigCodeBench-Hard (C) 29.1 / 24.3 / 29.1; CRUXEval-I 60.5 / 68.4 / 71.6; CRUXEval-O 61.9 / 75.1 / 75.5; MATH strict 63.2 / 54.0 / 73.7; MATH lenient 70.9 / 71.7 / 73.7; MMLU 86.49 / 85.26 / 86.82.
- Reward ablation (§3.6, Figure 5): the discrete reward reaches similar format accuracy but lower repair performance, and its average stays near zero at the end of training because human patches are diverse and rarely matched exactly.
- The SFT baseline is trained on a 16k context window for 2B tokens using synthetic chain-of-thought editing data generated with Llama-3.3-70B-Instruct, mixed with Llama 3 coding and general SFT data (§3.1, §C).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama3-SWE-RL-70B | 70B | RL | base model | Llama-3.3-70B-Instruct | arXiv:2502.18449v2 §3.1 | verified 2026-09-18 | no ablation reported |
| Llama3-SWE-RL-70B | 70B | RL | algorithm | GRPO with clipped objective and KL to reference policy | arXiv:2502.18449v2 §2.1 Eq. 2 | verified 2026-09-18 | no ablation reported |
| Llama3-SWE-RL-70B | 70B | RL | reward | −1 on format failure, else `difflib.SequenceMatcher` ratio in [0,1] | arXiv:2502.18449v2 §2.1 Eq. 1 | verified 2026-09-18 | §3.6 Fig. 5: continuous beats discrete on repair performance |
| Llama3-SWE-RL-70B | 70B | RL | prompts (seed dataset) | 273k PR instances | arXiv:2502.18449v2 §2, §A | verified 2026-09-18 | no ablation reported |
| Llama3-SWE-RL-70B | 70B | RL | steps | 1,600 | arXiv:2502.18449v2 §3.1 | verified 2026-09-18 | no ablation reported |
| Llama3-SWE-RL-70B | 70B | RL | context window | 16k tokens | arXiv:2502.18449v2 §3.1 | verified 2026-09-18 | no ablation reported |
| Llama3-SWE-RL-70B | 70B | RL | global batch | 512 rollouts = 32 prompts × 16 rollouts | arXiv:2502.18449v2 §3.1 | verified 2026-09-18 | no ablation reported |
| Llama3-SWE-RL-70B | 70B | RL | optimizer | Adam, one step per global step | arXiv:2502.18449v2 §3.1 | verified 2026-09-18 | no ablation reported |
| Llama3-SWE-RL-70B | 70B | RL | compute | 512 NVIDIA H100 GPUs, about 32 wall-clock hours per run | arXiv:2502.18449v2 §3.1 | verified 2026-09-18 | no ablation reported |
| Llama3-SWE-RL-70B | 70B | RL | KL coefficient β, clip ϵ, learning rate | not reported (checked §2.1, §3.1, §A–§D) | — | not reported | — |
| Llama3-SWE-SFT-70B | 70B | SFT | tokens seen | 2B tokens, 16k context window | arXiv:2502.18449v2 §C | verified 2026-09-18 | Table 3: this baseline declines on out-of-domain tasks |
| Llama3-SWE-RL-70B | 70B | eval-gate | inference sampling | 500 repair samples at temperature 1.0, top 30 reproduction tests for reranking | arXiv:2502.18449v2 §3.1, §3.4 | verified 2026-09-18 | Fig. 4: 33.6 → 40.0 from 20 to 160 samples, 41.0 at 500 |

## Findings relevant to generality and agentic training
- **Result (single study).** RL on one task (issue repair from PR data) improved five out-of-domain benchmark categories relative to Llama-3.3-70B-Instruct, while the SFT baseline built from the same base model declined on average (§3.5, Table 3). The authors state this is the first such demonstration for real-world software data.
- The RL model also performs pipeline steps it was never trained on, such as file-level localization and reproduction-test generation, which the authors present as out-of-domain generalization inside the scaffold (§3.1).
- The training setup is single-turn and non-agentic: one prompt, one patch, no tool calls and no environment interaction (§2, §3.1). Multi-turn agent behavior is not trained.
- Format accuracy after RL (95.6%) is slightly below the SFT baseline (96.2%), while repair performance is higher (34.8 vs 29.6) (Table 2).

## Connections
- [[swe-gym]] — the execution-environment alternative; SWE-RL lists SWE-Gym among open baselines whose training data is distilled from GPT-4o or Claude-3.5-Sonnet.
- [[grpo]] — the optimization algorithm used, unchanged from its original formulation.
- [[agentinstruct]], [[agent-flan]], [[agenttuning]] — agent-trajectory training lines, contrasting with SWE-RL's single-turn setup.
- [[transferability-of-llm-reasoning]], [[front-loading-reasoning]] — related claims about reasoning transfer across domains.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2502.18449 (arXiv v2, 2025-12-01; NeurIPS 2025)
- Corrections to the previous card version:
  - "trains Llama-3.1-70B" → base model is Llama-3.3-70B-Instruct (§3.1).
  - "11M GitHub issue-PR-code triplets scraped and prepared as RL training data" → about 11M raw unique PR instances, from which 273k seeds are selected for RL (§2, §A).
  - "reward is a scalar in [0,1]" → the reward is −1 for a malformed response and otherwise the similarity ratio (§2.1, Eq. 1).
  - "GRPO with group size G=8, KL coefficient β=0.02, LR 1e-6" → group size is 16 rollouts per prompt; β, ϵ, and the learning rate are not reported (§2.1, §3.1).
  - "beats DeepSeek-Coder-V2-Instruct 18.0%, matches SWE-Gym-32B" → Table 1 does not contain those comparisons in that form; the listed reference points include GPT-4o 38.8 and Claude-3.5-Sonnet 50.8 with Agentless.
  - "HumanEval+ +6, MATH +4, BIG-Bench Hard +3" → the out-of-domain suite is HumanEval+, BigCodeBench-Hard, CRUXEval, MATH, and MMLU; BIG-Bench Hard and MBPP+ are not evaluated. Actual deltas over the base model: HumanEval+ +3.7, MATH strict +10.5, MMLU +0.33 (Table 3).
  - "issue + code context ~4K–20K tokens; output 100–1000 tokens" → not stated; the training context window is 16k (§3.1).
  - Evaluation scaffold added: Agentless Mini, a pipeline scaffold, not an agent loop (§3.1, §B).
- Removed as unsupported by the source: "~1M H100-hours for the 70B RL run"; "Patch modifies ≤ 10 files and ≤ 500 lines"; "Dedup near-identical issues with MinHash"; "authors experiment with binary thresholding vs continuous reward; continuous wins" stated as clamping/shaping (the ablation is continuous vs discrete reward, §3.6); "decontaminate by date and commit-hash filter" as described; "transfer to Go/Rust is untested" as a paper-stated limitation; the claim that a dense similarity reward beats an execution reward (no execution-reward baseline is run).
- Not reported by the source: β, ϵ, learning rate, warmup; the language distribution of the 273k seeds; an execution-based reward comparison.

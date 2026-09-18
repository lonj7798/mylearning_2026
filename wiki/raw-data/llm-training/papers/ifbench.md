<!-- scope: IFBench (58 unseen verifiable output constraints), IFTrain (29 training constraints), and IF-RLVR ablations on constraint count, variety, variable ranges, DPO vs GRPO, base vs instruct, multi-turn, and constraint over-optimization
     deps: [[ifeval]], [[rlvr-tulu3]]
     see-also: [[tulu-3]], [[grpo]], [[dpo]], [[wildchat]], [[instruction-hierarchy]]
-->

# Generalizing Verifiable Instruction Following
- **Core Insight:** Models that score well on IFEval's 25 constraint templates score much lower on 58 unseen verifiable constraints, with GPT-4.1 and Claude 3.7 Sonnet below 50% (§1, Fig. 1); GRPO training on varied constraints (IF-RLVR) raises a Tülu-3-8B-DPO policy from 81.1 to 92.2 on IFEval and from 25.2 to 44.6 on IFBench (Table 6).
- **Guideline:** When training precise instruction following with verifiable rewards, use many constraint types, several constraints per prompt, and variable ranges that include and extend the test range, because each of these improved in-domain or out-of-domain scores in the paper's ablations (§4.1-4.3, Table 1, Fig. 2-5); add a preference reward-model signal when general response quality matters, because pure constraint rewards lowered LLM-judge quality from 7 to 6.4 out of 10 and AlpacaEval 2 from 33.5 to 21.3 (§5; Table 3; App. E).
- **Authors:** Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, et al. (Allen Institute for AI; University of Washington)
- **Year:** 2025 (arXiv v1 2025-07; NeurIPS 2025 Datasets and Benchmarks Track)
- **URL:** https://arxiv.org/abs/2507.02833 (code: https://github.com/allenai/IFBench)
- **Source type:** paper
- **Relevant topics:** precise instruction following, benchmark overfitting, out-of-domain evaluation, RLVR, GRPO vs DPO, reward over-optimization, multi-turn instruction following

## Abstract
Instructions often carry output constraints such as "only answer with yes or no". The paper finds that most models overfit to the small set of verifiable constraints in existing precise instruction-following benchmarks and do not generalize to unseen constraints. It introduces IFBench, a benchmark of 58 new, diverse, out-of-domain verifiable constraints, and analyzes how and on what data models can be trained to generalize. With constraint verification functions as rewards, reinforcement learning with verifiable rewards (RLVR) improves instruction following. The authors release IFBench, 29 additional hand-annotated training constraints with verification functions, RLVR training prompts, and code.

## Key Contributions
- IFBench: 58 new constraints with Python verification functions in 7 categories (count, ratio, words, sentence, format, custom, copy), 300 prompts (§2; App. A Table 8).
- IFTrain: 29 new training constraints with verifiers, disjoint from IFBench test constraints (§2-3; App. B Table 9).
- IF-RLVR recipe with multi-constraint prompts and a weighted sum of per-constraint rewards (§3, Eq. 1).
- Ablations on constraints per prompt, seen versus unseen constraints, variable ranges, removed categories, DPO versus GRPO, base versus instruct policies, and multi-turn data (§4).
- Analysis of constraint over-optimization and a reward that combines verifiable and reward-model signals (§5; App. E).

## Key Figures/Tables to Study
- **Fig. 1:** IFEval versus IFBench for off-the-shelf models and IF-RLVR models.
- **Table 1 and Fig. 2-3:** constraints per instance (1-6) and instances per constraint (10-1000).
- **Table 3:** IF gains against AlpacaEval 2, GSM8K, MMLU, BBH.
- **Table 5:** DPO versus GRPO on the same prompts and verifiers.
- **Table 6 and Table 7:** base versus instruct policies; single-turn, multi-turn, and mixed training.
- **Fig. 7:** an over-optimized output that repeats required keywords instead of solving the task.

## Technical Details
- **IFEval baseline:** 25 constraint templates; many leading models score 80+% at sizes as small as 2B (§1).
- **IFBench construction:** constraints written manually from feedback of LM users beyond the authors or to cover core skills, kept only if verifiable in Python (§2). Constraints are added to held-out WildChat prompts that are not released, except the "custom" group, which has fixed prompts (§2; Table 8 caption). Each instance has 1 or 2 constraints and was human-checked for prompt-constraint compatibility (§2). Strict and loose accuracy follow IFEval; loose accuracy removes first/last lines and some font modifiers (§2).
- **Settings:** single-turn (task t plus constraints c in one prompt) and multi-turn (turn 1 task, turn 2 assistant response, turn 3 request to rewrite it under c) (§2). Mean input length 76 tokens single-turn and 408 multi-turn (App. D.1).
- **IF-RLVR data:** Tülu-3-SFT instructions with 1 to n constraints from IFTrain and IFEval (IFEval variable ranges expanded); a conflict dictionary blocks contradictory pairs; about 60k-100k prompts for most experiments (§3).
- **Reward (Eq. 1):** Instance Reward = Σ_{i=1..n} verifiable_reward_i · reward_multiplier_i · reward_weight_i, where i indexes the constraints in the prompt and multipliers and weights are generally 1 (§3).
- **Constraints per instance (Table 1, Qwen2.5 policy):** for n = 1…6, IFBench 48.9, 53.1, 59.5, 49.4, 55.8, 54.1 and IFEval 71.2, 79.9, 77.8, 79.5, 79.9, 85.8. With the Tülu-DPO policy, training on up to 5 or 6 constraints generalizes better than up to 3, although test prompts have at most 3 (IFEval) or 2 (IFBench) constraints (§4.1, Fig. 2).
- **Instances per constraint (Table 1):** 10, 50, 100, 500, 1000 give IFBench 48.6, 52.7, 51.7, 51, 48.6 (§4.1).
- **Seen versus unseen:** IFTrain plus n IFEval templates (n ∈ {5, 10, 15, 20, 25}); IFTrain plus all IFEval templates gives the highest IFEval score, while IFBench changes less with n (§4.2, Fig. 4).
- **Variable ranges:** training on a range disjoint from test scores lowest on IFEval; a wider range that includes the test range performs comparably or better than the same range (§4.3, Fig. 5).
- **Removed categories:** removing LENGTH or KEYWORDS constraints lowers IFEval most; removing CHANGE CASES or DETECTABLE FORMAT leaves IFEval at 89.65 (§4.4, Fig. 6).
- **Per-category IFBench (Table 2):** words 7.0 → 48.1 and sentence 13.3 → 36.7 for Tülu-DPO → IF-RLVR; the authors note that most IFEval categories exceed 90 after training while the IFBench words and sentence categories leave room for improvement (§4.5).
- **DPO versus GRPO (§4.6, Table 5):** prompts with up to 5 constraints; completions from Tülu-3-70B, Qwen-72B, Llama-3.1-405B, Llama3-8B, Yi-34B-Chat; all-constraints-correct rates 15%, 26%, 21%, 6%, 10% (Table 4). Chosen = passes all constraints; rejected = fails at least one; 46% of rejected completions pass none of 5 constraints; 31,751 prompts. IFEval/IFBench strict: DPO after SFT 76.89/25.2; DPO after DPO 79.67/29.3; GRPO after SFT 85.77/28.6; GRPO after DPO 89.65/30.6.
- **Base versus instruct (Table 6):** after IF-RLVR, base Llama3.1, Qwen2.5, OLMo2 reach IFBench 54.1, 53.7, 46.6; instruct Tülu3-DPO, Qwen2.5, OLMo2 reach 44.6, 45.9, 44.6. The authors conclude IF-RLVR from base with a reasoning chat template generalizes better to IFBench (§4.7). The §4.7 text names Qwen3-8B as a base model; Table 6 lists OLMo2 instead.
- **Multi-turn (Table 7, Qwen2.5-7B-Instruct, IFBench):** single-turn training 45.9 single / 71.7 multi-turn; multi-turn training 34.7 / 68.6; mixed training 54.8 / 72.9 (§4.8).
- **Source-internal conflicts:** §1 reports Tülu-3-8B IFBench 28.9 → 45.9 and Qwen2.5-7B base IFBench 54.7; Tables 2, 3, and 6 report 44.6 for IF-RLVR Tülu-DPO 8B and 53.7 for IF-RLVR Qwen2.5 base.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| IF-RLVR runs (Llama-3.1-Tulu-3-8B-DPO, Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-7B-Instruct, OLMo2, OLMo2-Instruct) | 7-8B (OLMo2 size not stated) | RL | algorithm; implementation; reward | GRPO with outcome supervision; open-instruct; Eq. 1 with multiplier and weight generally 1 | arXiv:2507.02833v3 §3 | verified 2026-09-14 | Table 5 (GRPO > DPO on same data) |
| same | 7-8B | RL | max token length; temperature; LR; samples per prompt; local mini-batch; hardware; wall time | 2048; 1; 5e-7; 16; 32; 8 H100; on average 1 day for 2000 steps | §3 | verified 2026-09-14 | no ablation reported |
| same, base models with reasoning chat template | 7-8B | RL | max token length; KL β; temperature | 10240; 0; 1 | §3; §4.7 | verified 2026-09-14 | Table 6 (from-base vs instruct) |
| same, runs from post-trained policies | 7-8B | RL | KL β; number of epochs | not reported | checked §3-4, App. E-G | not reported | none |
| IF-RLVR prompts | n/a | RL | prompts; constraints per prompt | about 60k-100k for most experiments; 1 to n | §3; §4.1 | verified 2026-09-14 | Table 1, Fig. 2 (n = 1-6) |
| DPO baseline from Tülu-3-8B-SFT or -DPO | 8B | preference | LR; β; batch; data | 5.0e-7; 5; 16; 31,751 strict prompts with chosen/rejected pairs | §4.6 | verified 2026-09-14 | Table 5 |
| IF-RLVR with reward-model mixing (policy not named) | not stated | RL | RM; threshold α; effective batch; samples per prompt; steps | Llama-3.1-Tulu-3-8B-RM; 7; 512; 8; 1100 | App. E, Eq. 2, footnote 5 | verified 2026-09-14 | IFEval 86.1, IFBench 30, AlpacaEval 2 31.6 (App. E) |
| Evaluation | n/a | eval-gate | decoding; metric | temperature 0 (DeepSeek-R1: 0.6, top-p 0.95; o3: 1); prompt-level loose accuracy | §3; App. G | verified 2026-09-14 | not applicable |

## Findings relevant to generality and negative feedback
- **Benchmark overfitting:** the authors attribute the IFEval-IFBench gap, with the same task and evaluation setup, to models overfitting a small constraint set (§2, Fig. 1). They cite targeted synthetic IF data built from the IFEval taxonomy as the common approach (§2). Result (single study).
- **Narrowing after IF-RLVR (Table 3, from Tülu-DPO):** AlpacaEval 2 33.5 → 21.3, GSM8K 84.3 → 83.2, MMLU 68.7 → 66.4, BBH 68.7 → 68.9. GPT-4.1 judge scores of responses with the constraint removed: 7 (base policy) versus 6.4 (IF-RLVR) out of 10 (§5). IF-RLVR responses average 210 tokens versus 2214 for frontier models (App. D.1).
- **Instruction hierarchy:** IF-RLVR models tend to prioritize the constraint over the task; among existing models, Qwen2.5-72B-Instruct IF accuracy is most negatively correlated with judge scores and Tülu3-70B most positively (§1, §5).
- **Negatives as gradient (DPO):** rejected completions fail at least one constraint (§4.6). **Reward shaping (App. E):** F_i = V_i + 1 if V_i > 0 and S_i > α; V_i − 0.5 if V_i > 0 and S_i ≤ α; V_i if V_i ≤ 0, where V_i is the verifiable reward and S_i the reward-model score. This run scores lower on IF (IFEval 86.1, IFBench 30) but higher on AlpacaEval 2 (31.6) than ground-truth-only training (App. E).
- **Limits:** only verifiable constraints, which may seem unnatural or contrived (§7).

## Connections
- [[ifeval]]: the 25-template benchmark that IFBench tests generalization beyond.
- [[rlvr-tulu3]], [[tulu-3]]: RLVR formulation, Tülu-3-SFT prompts, and Tülu policies used here.
- [[grpo]]: DeepSeekMath paper cited for GRPO.
- [[dpo]]: preference baseline in Table 5.
- [[wildchat]]: source of held-out IFBench prompts.
- [[nemotron-4-synthetic]]: cited example of synthetic IF data combining instructions with IFEval-taxonomy constraints (§2).
- [[instruction-hierarchy]]: cited framing for task versus constraint priority (§5).
- [[reinforcement-learning-with-one-training-example]]: cited for RLVR from base models (§4.7).
- [[reward-hacking-taxonomy]]: related card on reward over-optimization.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2507.02833 (v3, 2025-11-11; v1 2025-07-03).
- Audit claims not found in the source: "built specifically because IFEval-targeted training, RLVR included, overfit" (the paper does not say RLVR training caused the overfitting; it cites targeted synthetic data, §2).

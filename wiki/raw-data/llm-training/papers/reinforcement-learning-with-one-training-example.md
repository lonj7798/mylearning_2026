<!-- scope: 1-shot RLVR — data efficiency, example selection, and post-saturation generalization in RLVR for math reasoning
     deps: [[rlvr-tulu3]], [[deepseek-r1]]
     see-also: [[entropy-mechanism-llm-rl]], [[rlvr-beyond-base-model]], [[spurious-rewards-rlvr]], [[rloo-vs-grpo]], [[math-shepherd]]
-->

# Reinforcement Learning for Reasoning in Large Language Models with One Training Example
- **Core Insight:** RLVR on Qwen2.5-Math-1.5B with a single training example raises MATH500 from 36.0% to 73.6% and the average over six math benchmarks from 17.6% to 35.7%, matching RLVR on the 1.2k-example DeepScaleR subset that contains that example (Abstract; Fig. 1).
- **Guideline:** When an RLVR training set is being reduced, rank candidates by the historical variance of their per-epoch training accuracy and keep an entropy loss with a small negative coefficient (α = −0.001), because entropy loss adds 4.0% MATH500 and 2.5% AIME24 over the same run without it (§4.1, Tab. 5 rows 4 vs 5). Do not stop training when the single example is solved: test accuracy keeps rising for hundreds of steps after training accuracy saturates (§3.2.2, Fig. 2).
- **Authors:** Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, et al.
- **Year:** 2025 (arXiv v1 2025-04; NeurIPS 2025; text checked against v3, 2025-10-24)
- **URL:** https://arxiv.org/abs/2504.20571
- **Source type:** paper
- **Relevant topics:** RLVR, data efficiency, example selection, entropy loss, GRPO, PPO, math reasoning, post-saturation generalization

## Abstract
The paper shows that reinforcement learning with verifiable reward (RLVR) remains effective when the
training set is reduced to one example. On the base model Qwen2.5-Math-1.5B, one selected example
raises MATH500 from 36.0% to 73.6% and the six-benchmark average from 17.6% to 35.7%; the authors
report 8.6% of the MATH500 gain and 7.0% of the average gain as improvement beyond a format-reward
baseline. Two examples reach MATH500 74.8% and average 36.6%. The effect reproduces on
Qwen2.5-Math-7B, Llama-3.2-3B-Instruct, and DeepSeek-R1-Distill-Qwen-1.5B, and under both GRPO
and PPO. The authors report cross-category generalization, increased self-reflection frequency, and
post-saturation generalization (test accuracy still improving after training accuracy saturates). An
ablation attributes the gain mainly to the policy gradient loss rather than to weight decay, which
distinguishes it from grokking. Code, models, and data are released.

## Key Contributions
- Shows the RLVR training set can be reduced to one example without losing the bulk of the gain (Abstract; Fig. 1).
- Proposes a selection rule, the **historical variance score**: train on the full pool for E epochs, record each example's per-epoch training accuracy `s_{i,1..E}`, and rank by `v_i = var(s_{i,1},…,s_{i,E})` (§2, Eqn. 1-2).
- Names and measures **post-saturation generalization** (§3.2.2, Fig. 2).
- Separates the effect from grokking by loss-component ablation (§4.1, Tab. 5).
- Reports that math-only 1-shot RLVR also raises non-math ARC scores (§3.2, Tab. 1).

## Key Figures/Tables to Study
- **Figure 1** — 1-shot, 2-shot, 1.2k DSR-sub, 7.5k MATH, and format-reward curves on MATH500 and the six-benchmark average.
- **Table 1** — ARC-Easy / ARC-Challenge transfer.
- **Table 3** — MATH500 per-example results across high, medium, and low variance ranks.
- **Table 4** — other models and PPO.
- **Table 5** — loss-component and label-correctness ablation.
- **Figure 2 / Figure 3** — training-vs-test curves and the overfitted training response at step 1860.

## Technical Details
- Instance pool: 1209 examples randomly drawn from DeepScaleR-Preview-Dataset, called DSR-sub (§3.1). The comparison full set is the MATH training set, 7500 instances (§3.1).
- Selection ranking is computed once, from 500 training steps of Qwen2.5-Math-1.5B, and is reused unchanged in all experiments (§3.1).
- Reward is binary outcome-only: 1 when the parsed final answer matches ground truth (§2). A separate format-reward baseline gives 1 whenever a final answer is parseable (§3.1, App. C.2.3), and reaches MATH500 65.6 / AIME24 10.0 (Tab. 5 caption).
- The single example is duplicated to 128 rows because verl sets `drop_last=True` and the dataset must be at least the training batch size (§3.1, footnote 3).
- GRPO loss has three terms — policy gradient, KL to the reference model, entropy — combined as `L = L_PG + β·L_KL + α·L_Entropy` (App. B.1, Eqn. 3). β = 0.001, α = −0.001 (§3.1).
- Main result: Qwen2.5-Math-1.5B MATH500 36.0 → 73.6, six-benchmark average 17.6 → 35.7 with `{π13}`; 1.2k DSR-sub gives 73.6 / 35.9; `{π1, π13}` gives 74.8 / 36.6; 7.5k MATH gives 36.7 average (Abstract; Fig. 1 caption).
- Other models (Tab. 4, six-benchmark average): Qwen2.5-Math-7B 22.4 base → 40.2 with `{π1}` (+17.8; format-reward baseline 34.3); Llama-3.2-3B-Instruct 17.5 base → 19.0 with `{π1}` and 21.0 with `{π1, π13}`, above full DSR-sub at 19.8; Qwen2.5-Math-1.5B with PPO 17.6 base → 33.8 with `{π1}` vs 35.4 with DSR-sub; DeepSeek-R1-Distill-Qwen-1.5B at 32k eval 44.9 base → 46.3 with `{π1}` vs 48.6 with DSR-sub.
- Post-saturation: training accuracy on `{π1}` and `{π13}` saturates before step 100, yet the average improves 3.4% from step 100 to 1540 for `{π1}` and 9.9% from step 500 to 2000 for `{π13}` (§3.2.2). The model emits unintelligible multilingual training responses from step 1860 while test responses stay coherent at 74% MATH500 (Fig. 3).
- Ablation (Tab. 5, Qwen2.5-Math-1.5B, `{π1}`, best checkpoint per benchmark): policy gradient alone 71.8 / 15.4; + weight decay 71.4 / 16.3; + KL 70.8 / 15.0; + entropy 74.8 / 17.5; policy gradient + entropy only 75.6 / 17.1; weight decay + KL without policy gradient 39.0 / 10.0; entropy only 63.4 / 8.8. A larger entropy coefficient (−0.003) lowers results to 73.6 / 15.4.
- Label ablation (Tab. 5 rows 11-13): the near-correct label 12.7 gives 73.4 / 17.9; a wrong label 4 gives 57.0 / 9.2; a random large label gives 64.4 / 9.6.
- Evaluation: official Qwen2.5-Math pipeline; AIME24, AIME25, AMC23 repeated 8 times at temperature 0.6 and reported as avg@8 pass@1; the other three benchmarks at temperature 0 (§3.1). Default max generated tokens 3072 (App. B.5).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-1.5B | 1.5B | RL | RL algorithm | GRPO (verl); PPO also tested | arXiv:2504.20571v3 §3.1, Tab. 4 | verified 2026-09-18 | Tab. 4: PPO reaches 33.8 avg vs GRPO 35.7 |
| Qwen2.5-Math-1.5B | 1.5B | RL | KL coefficient β | 0.001 | §3.1 | verified 2026-09-18 | Tab. 5 rows 3 vs 4: removing KL changes MATH500 71.4 → 70.8 |
| Qwen2.5-Math-1.5B | 1.5B | RL | Entropy coefficient α | −0.001 (−0.003 tested) | §3.1; Tab. 5 row 6 | verified 2026-09-18 | Tab. 5 rows 4 vs 5: +4.0 MATH500, +2.5 AIME24 |
| Qwen2.5-Math-1.5B | 1.5B | RL | Rollout temperature | 0.6 (1.0 tested) | §3.1; §4.1 Fig. 5 | verified 2026-09-18 | §4.1: t = 1.0 adds 0.8% average |
| Qwen2.5-Math-1.5B | 1.5B | RL | Train batch / mini-batch (prompts) | 128 / 128 | §3.1 | verified 2026-09-18 | no ablation reported |
| Qwen2.5-Math-1.5B | 1.5B | RL | Samples per prompt | 8 (8 gradient updates per rollout step) | §3.1 | verified 2026-09-18 | no ablation reported |
| Qwen2.5-Math-1.5B | 1.5B | RL | Max prompt / response length | 1024 / 3072 tokens | §3.1 | verified 2026-09-18 | set by the 4096 context of Qwen2.5-Math (§3.1) |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | RL | Max response length | 8192 tokens | App. B.4 | verified 2026-09-18 | follows DeepScaleR stage 1 (App. B.4) |
| All models | — | RL | Learning rate; weight decay | 1e-6; 0.01 | App. B.4 | verified 2026-09-18 | Tab. 5 rows 2 vs 3: weight decay changes MATH500 71.8 → 71.4 |
| All models | — | RL | Training steps | 2000 / 1000 / 1000 / 1200 (Qwen2.5-Math-1.5B / -7B / Llama-3.2-3B-Instruct / R1-Distill-1.5B) | App. B.4 | verified 2026-09-18 | stopped early on a significant performance drop (App. B.4) |
| All models | — | RL | Compute | 8 A100 GPUs per experiment; checkpoint every 20 steps | App. B.4 | verified 2026-09-18 | no ablation reported |

## Findings relevant to generality
- Non-math transfer (Tab. 1, best checkpoint by six-benchmark math average): base ARC-Easy 48.0 / ARC-Challenge 30.2; `{π13}` 55.8 / 33.4; `{π1}` 52.0 / 32.2; `{π1, π13}` 52.1 / 32.4; MATH 7500 51.6 / 32.8. The 1209-example DSR-sub run **lowers** ARC-Easy to 42.2 and ARC-Challenge to 29.9, so the larger math set narrows non-math performance while the one-example run does not.
- Within MATH500, almost every tested single example gives at least a 30% improvement, except an example with an incorrect label and an extremely difficult one (§3.2.3, Tab. 3). Improvement does not concentrate in the training example's own category (§3.2.3).
- Part of the gain is format correction; the authors quantify the non-format part as 8.6% on MATH500 and 7.0% on the average (Abstract), and report that some examples give almost nothing beyond the format-reward baseline (§3.2.3).
- The result is demonstrated on math-pretrained base models, mainly Qwen2.5-Math. [[spurious-rewards-rlvr]] reports that RLVR effects measured on Qwen math models do not reproduce on other families; this paper's own Llama-3.2-3B-Instruct numbers show the smaller gain (17.5 → 19.0-21.0, Tab. 4).

## Findings relevant to negative feedback
- The reward is binary and the GRPO advantage is group-normalized, so below-average responses receive a negative advantage (§2). The paper does not measure the contribution of negative-advantage samples separately.
- With an incorrect label (Tab. 5 rows 12-13) the model receives reward almost never, and MATH500 falls to 57.0 and 64.4 against 74.8 with the correct label.

## Connections
- [[rlvr-tulu3]] — both report verifier-grounded RL gains; this paper sets the low-data bound of that family.
- [[entropy-mechanism-llm-rl]] — treats entropy as the controlling quantity in LLM RL; here entropy loss is the second-largest ablation term (Tab. 5).
- [[deepseek-r1]] — supplies the R1-Distill-Qwen-1.5B checkpoint tested in Tab. 4.
- [[spurious-rewards-rlvr]] — the caveat on measuring RLVR effects with Qwen2.5-Math base models.
- [[rlvr-beyond-base-model]] — argues RLVR raises pass@1 without extending pass@k; this paper does not report pass@k, so it does not settle that question.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2504.20571 (arXiv v3, 2025-10-24; NeurIPS 2025).
- Corrections to the previous card version:
  - "Authors: … Simon Shaolei Du, Yelong Shen" (all 14 listed) → first six then "et al.", per §9 (title page).
  - "the effect transfers across several base models and both GRPO and PPO" stated without numbers → Tab. 4 averages added, including the smaller Llama-3.2-3B-Instruct gain.
  - "entropy loss materially helps exploration" → quantified as +4.0 MATH500 and +2.5 AIME24 (Tab. 5 rows 4 vs 5).
  - "entropy alone can improve MATH500 even without outcome reward" left as a positive result → entropy-only reaches 63.4, which is below the 65.6 format-reward baseline (Tab. 5 rows 9-10 and caption).
  - "Non-math transfer table: shows math-focused 1-shot RLVR can also help unrelated reasoning tasks" → ARC numbers added, including the DSR-sub regression to 42.2 ARC-Easy.
  - "matching the reported 1.2k-example DeepScaleR subset" without its numbers → DSR-sub 73.6 / 35.9 stated; DSR-sub is a 1209-example random subset the authors drew, not a released 1.2k dataset (§3.1).
  - Training setup given in prose only → replaced by a Recipe ledger with loci (§3.1, App. B.4).
- Removed as unsupported by the source:
  - "High-variance examples are more informative for RLVR because they expose the model to reward-sensitive decision boundaries" — the paper gives no such mechanism; it cites reward-signal variance and states the criterion is "not necessarily optimal" (§2).
  - "the model keeps getting better on held-out math problems after it has already memorized the training example" phrased as memorization from the start — the paper dates overfitting to step 1400 (π1) and 1800 (π13) (§3.2.2).
  - "**Self-reflection increase:** downstream outputs contain more reflective language" as an unqualified claim — the paper counts three keywords and notes the base model already self-reflects (§3.2.4, Fig. 4).
- Not reported by the source: pass@k at large k; wall-clock cost; results on non-math training data; any contamination analysis of Qwen2.5-Math.

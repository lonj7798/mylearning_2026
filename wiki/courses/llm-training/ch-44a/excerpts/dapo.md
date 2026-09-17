---
chapter: ch-44a
course: llm-training
phase: read
excerpt_of: arXiv:2503.14476v2 (DAPO), §2.4, §3.3, §3.4, §4.1, §4.2 Table 1, §4.3 (chapter-local verified extract; no library card exists for this slug on 2026-09-15)
source_url: https://arxiv.org/abs/2503.14476
created_at: "2026-09-15"
---

# Excerpt: DAPO — An Open-Source LLM Reinforcement Learning System at Scale

- **Authors:** Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, et al. (ByteDance Seed; Institute for AI Industry Research (AIR), Tsinghua University; The University of Hong Kong)
- **Year:** 2025 (arXiv v1 2025-03; this extract read from v2, 2025-05-20)
- **Source type:** paper
- **Setting:** RL from the Qwen2.5-32B base model on DAPO-Math-17K (17K math prompts with integer answers), verl framework, AIME 2024 avg@32.
- **Used in:** ch-44a §1, §2, §3, Negative samples, Recipe.

## Outcome reward (§2.4, Eq. 7)
`R(ŷ, y) = 1 if is_equivalent(ŷ, y); −1 otherwise`, with y the ground-truth answer and ŷ the predicted answer.

## Token-level policy-gradient loss (§3.3, Eq. 12)
The DAPO objective divides the summed token losses by `Σ_i |o_i|` (all tokens of the group) instead of averaging inside each
sample first. Stated reasons: under sample-level averaging "tokens within longer responses ... may have a disproportionately
lower contribution to the overall loss", so (a) reasoning patterns in high-quality long samples are learned more slowly and
(b) "excessively long samples often exhibit low-quality patterns such as gibberish and repetitive words" that are not
penalized enough, which "leads to an unhealthy increase in entropy and response length" (Figures 4a, 4b).

## Overlong reward shaping (§3.4)
- "In RL training, we typically set a maximum length for generation, with overlong samples truncated accordingly. We find
  that improper reward shaping for truncated samples can introduce reward noise and significantly disrupt the training
  process."
- "By default, we assign a punitive reward to truncated samples. This approach may introduce noise into the training
  process, as a sound reasoning process can be penalized solely due to its excessive length."
- **Overlong Filtering**: "masks the loss of truncated samples. We find that this approach significantly stabilizes training
  and enhances performance, as demonstrated in Figure 5." Figure 5 plots AIME avg@32 and generation entropy with and without
  it (curves only; no table values).
- **Soft Overlong Punishment** (Eq. 13), added to the rule-based correctness reward:
  `R_length(y) = 0` for `|y| ≤ L_max − L_cache`; `((L_max − L_cache) − |y|) / L_cache` for `L_max − L_cache < |y| ≤ L_max`;
  `−1` for `L_max < |y|`.

## Training settings (§4.1)
AdamW, constant learning rate 1 × 10⁻⁶ with linear warm-up over 20 rollout steps; prompt batch size 512; 16 responses per
prompt; mini-batch size 512 (16 gradient updates per rollout step); expected maximum length 16,384 tokens plus a 4,096-token
soft punish cache, so maximum generation is 20,480 tokens; ε_low = 0.2, ε_high = 0.28; evaluation on AIME repeated 32 times
(avg@32) at temperature 1.0, top-p 0.7.

## Ablation ladder (§4.2, Table 1; Qwen2.5-32B base, AIME 2024 avg@32)
| Model | AIME24 avg@32 |
|---|---|
| DeepSeek-R1-Zero-Qwen-32B | 47 |
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling (DAPO) | 50 |

The rows are cumulative ("progressive techniques"), single runs, math only. For token-level loss the paper states it "brings
less performance improvement" but "enhances training stability and makes the length increase more healthily".

## Length as a training-dynamics metric (§4.3)
"The increase in length provides the model with a larger space for exploration ... However, it is important to note that
length does not always maintain a continuous upward trend during training. In some considerable periods, it can exhibit a
trend of stagnation or even decline ... We typically use length in conjunction with validation accuracy as indicators to
assess whether an experiment is deteriorating." The paper also reports that the final training reward "often exhibits little
correlation with the accuracy on the validation set, which indicates overfitting to the training set".

## Not reported
Truncation-rate values, response-length percentiles, the share of truncated samples at any step, seeds or variance for
Table 1, and any evaluation outside AIME 2024 / math.

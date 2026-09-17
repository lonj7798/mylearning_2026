---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: (no library card at time of writing; primary source)
source_url: https://arxiv.org/abs/2309.06256
revised_at: "2026-09-15"
---

# Excerpt: Mitigating the Alignment Tax of RLHF (Lin et al.)

Authors: Yong Lin, Hangyu Lin, Wei Xiong, Shizhe Diao, Jianmeng Liu, Jipeng Zhang, et al. arXiv v1 2023-09; checked against arXiv:2309.06256v4 (2024-10-13) on 2026-09-15. Used in read.md §6.4.

## Setting (§3)
- OpenLLaMA-3B instruction-tuned on ShareGPT gives θ_0; RLHF on HH-RLHF gives θ. Main algorithm: rejection-sampling fine-tuning (RSF); findings checked with PPO and DPO and extended to Mistral-7B.
- Tax benchmarks: ARC Easy/Challenge, RACE, PIQA (accuracy); SQuAD, DROP (F1); WMT14 FR→EN (BLEU). The tax is regression of θ relative to θ_0.

## Methods compared (§4.1)
- Early stopping; L1/L2 penalty λ‖θ − θ_0‖; LoRA; knowledge distillation toward π_θ0; model averaging π_{(1−α)θ_0+αθ}, α ∈ [0, 1]; stochastic moving averaging.
- "Notably, despite its simplicity, the Pareto-front of model averaging supersedes nearly all other methods across various hyper-parameters." (Figure 3; figure only)
- §4: as reward increased, translation and reading comprehension dropped; commonsense QA "increases first and then drops".

## Replay and KL penalty (App. C.1, C.2)
- Replay λ ∈ {0.25, 0.5, 1, 2, 4}, implemented as data proportion. Replay beat model averaging only on reading comprehension; "Despite maintaining extra pre-training data, which is four times larger than the RLHF data (400M token), ER under-performs model averaging in two out of three benchmarks", attributed to covering "about 0.03% of the pre-training data" (1.2T tokens). §2 of the paper gives "~0.01%".
- PPO with KL penalties {0.05, 0.1, 0.2}: "while a larger KL penalty can partially mitigate the forgetting issue, the model averaging is much more effective than the reward penalty in terms of the alignment-forgetting trade-off." (Figure 8)

## Layer-wise averaging (§5-6)
- Averaging the input (low-level) part of the transformer improved both NLP tasks and alignment reward; the effect is reported as consistent for DPO and PPO (App. E.2).
- Heterogeneous Model Averaging: θ^[k](K) = α_k θ^[k] + (1 − α_k) θ_0^[k], mean of α_k fixed to α, K = 3 by default; K = 6 and 9 gave slightly lower curves than K = 3 but stayed above vanilla averaging.

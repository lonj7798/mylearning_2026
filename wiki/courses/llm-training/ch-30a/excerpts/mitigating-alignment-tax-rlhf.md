---
chapter: ch-30a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/mitigating-alignment-tax-rlhf.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2309.06256
created_at: "2026-09-15"
---

# Excerpt: Mitigating the Alignment Tax of RLHF

**Authors:** Yong Lin, Hangyu Lin, Wei Xiong, Shizhe Diao, Jianmeng Liu, Jipeng Zhang, et al. (Princeton University; HKUST; UIUC; NVIDIA)
**Version read:** arXiv:2309.06256v4 (13 Oct 2024); v1 September 2023.
**Status:** no library card existed for this slug on 2026-09-15; quotes checked against the v4 PDF text.

## Setup (§3)
OpenLLaMA-3B, instruction-tuned on ShareGPT to give θ_0, then aligned on HH-RLHF with Rejection Sampling Fine-tuning (RSF, best-of-n) to give θ; extended to DPO, PPO, and Mistral-7B / Zephyr. Alignment tax is "the performance regression of θ with θ_0" on ARC Easy and Challenge, Race, PIQA (accuracy), SQuAD and DROP (F1), and WMT 2014 Fr→En (BLEU).

## Findings used in ch-30a
- §1: "as we gained a higher reward during RLHF ... the alignment tax also increased simultaneously."
- §4.1: early stopping, L1 and L2 penalties toward θ_0, LoRA, knowledge distillation, and stochastic moving averaging "effectively alleviate the alignment tax; however, they also result in a reduction in the RLHF reward"; "the Pareto-front of model averaging supersedes nearly all other methods across various hyper-parameters." Model averaging is the policy π_{(1−α)θ_0 + αθ} with α ∈ [0, 1].
- §5, Empirical Validation: averaging only the low-level layers gives "a 'magical' improvement in both the NLP tasks and alignment rewards" (the authors' wording).
- §6: Heterogeneous Model Averaging (HMA) splits the transformer into K parts with ratios α_k, θ^[k](K) := α_k θ^[k] + (1 − α_k) θ_0^[k], and maximizes reward with the mean ratio fixed. "α = 0.2 can consistently alleviate the alignment tax without hurting alignment performance."
- Table 1 (GPT-4 evaluation on the Alpaca benchmark): Zephyr-7B-β win rate 8.10%, reading 37.47, commonsense 66.34, translation 36.55; HMA 9.32%, 38.93, 66.55, 37.23.
- App. C.1, experience replay: the objective adds λ E_{(x,a)~D_pre} log π_θ(a | x), implemented by including up to 4 times the replay data over the RLHF data in a batch. "Despite maintaining extra pre-training data, which is four times larger than the RLHF data (400M token), ER under-performs model averaging in two out of three benchmarks. This may be attributed to the vast size of the pre-training data (1.2T token), such that even when replaying a subset four times larger than the RLHF data, it only covers about 0.03% of the pre-training data." (§2 of the same version prints "~0.01%" for this coverage; 400M / 1.2T = 0.033%.)
- Limitations: "Though our HMA significantly alleviates the alignment tax, it has not been fully eliminated."

## How ch-30a uses it
§5.1 (replay coverage), §5.5 (model averaging for LLM alignment), Recipe rows.

---
chapter: ch-07
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2309.06256
created_at: "2026-09-17"
---

# Excerpt: "Mitigating the Alignment Tax of RLHF"

**Artifact.** Yong Lin, Hangyu Lin, Wei Xiong, Shizhe Diao, Jianmeng Liu, Jipeng Zhang, et al. (Princeton;
HKUST; UIUC; NVIDIA). arXiv:2309.06256 (v1 2023-09; v4 2024-10-13; EMNLP 2024). Read on 2026-09-17. No
library card existed when ch-07 was revised.

## Setting (§3)

- Pipeline: pre-trained **OpenLLaMA-3B** → instruction tuning on ShareGPT (this checkpoint is `θ_0`) →
  RLHF (this checkpoint is `θ`). The alignment tax is defined as the performance regression of `θ` relative
  to `θ_0` on NLP benchmarks, following Ouyang et al. (2022).
- RLHF algorithms: Rejection Sampling Finetuning (RSF) for the main experiments, with PPO and DPO as
  additional checks. RSF samples `n` responses per prompt, keeps the highest-reward one, and fine-tunes on
  that set, iteratively.
- Alignment-tax benchmarks (§3): commonsense QA (ARC-Easy, ARC-Challenge, RACE, PIQA; accuracy), reading
  comprehension (SQuAD, DROP; F1), translation (WMT 2014 French→English; BLEU).
- Extension to 7B: Zephyr-7B-β (Mistral-7B-SFT-β aligned with DPO) and Zephyr-7B-Gemma, judged with the
  public PairRM preference model and AlpacaEval 2.0 (§6).

## Results quoted in ch-07

- During RLHF the reward rises while translation and reading comprehension fall continuously; commonsense
  rises first and then falls (§4, App. E.1, Fig. 12). The tax is therefore not visible from the reward
  curve, and not visible from a single end-of-run benchmark average either.
- Methods that reduce forgetting also reduce the RLHF reward: early stopping, L1/L2 weight-space
  regularization toward `θ_0`, LoRA, knowledge distillation from `θ_0`, model averaging (MA) and stochastic
  moving averaging (§4.1, Fig. 3). The authors report that the Pareto front of **model averaging**,
  `π_{(1−α)θ_0 + αθ}` with `α ∈ [0, 1]`, "supersedes nearly all other methods across various
  hyper-parameters" (§4.1).
- Experience replay and a KL reward penalty for PPO were also compared and under-performed model averaging
  (App. C.1–C.2).
- **Heterogeneous Model Averaging (HMA)** splits the Transformer into `K` blocks (default `K = 3`) and
  optimizes a separate ratio `α_i` per block under a fixed mean ratio; it pushes the Pareto front further
  than uniform averaging for RSF and DPO (§6, Fig. 5), and the trade-off curve degrades slightly as `K`
  grows from 3 to 6 to 9, which the authors attribute to overfitting the in-domain reward (§6).
- Figure axes give the scale of the trade-off measured: for RSF on OpenLLaMA-3B, reading comprehension F1
  moves over roughly 14.0–17.5 as the HH RLHF reward moves over roughly 6.0–7.0 (Fig. 5). Exact per-point
  values are not printed in the text.

## Limits

One 3B model for the main study; the 7B extension uses off-the-shelf aligned checkpoints rather than a
re-run pipeline; no pretraining-data replay baseline at scale.

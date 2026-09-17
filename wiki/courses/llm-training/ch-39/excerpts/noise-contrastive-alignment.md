---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: "primary source (no library card at the time of writing)"
source_url: https://arxiv.org/abs/2402.05369
created_at: "2026-09-15"
verified_against: "arXiv:2402.05369v3 (30 Oct 2024), NeurIPS 2024, cached plain text"
---

# Noise Contrastive Alignment of Language Models with Explicit Rewards (InfoNCA / NCA)

Huayu Chen, Guande He, Lifan Yuan, Ganqu Cui, Hang Su, Jun Zhu.

## Claims the chapter uses
- **DPO is a special case of InfoNCA** at `K = 2` responses and reward temperature `α → 0` (Table 1, §3.2).
- **Pairwise NCA loss (Table 1; App. B code).**
  `L_NCA = −log σ(r_θ(x, y_w)) − ½ log σ(−r_θ(x, y_w)) − ½ log σ(−r_θ(x, y_l))`, with
  `r_θ(x, y) = β (log π_θ(y|x) − log π_ref(y|x))` as in the released code
  (`chosen_rewards = (chosen_pi_logps - chosen_ref_logps) * beta`).
- **What differs (Table 1, §4.2).** InfoNCA/DPO optimise the *relative* value of the log-likelihood ratio;
  NCA optimises the *absolute* value, which the paper states "effectively prevents the likelihood of the
  preferred responses from decreasing" (§4.2, Fig. 5).
- **UltraInteract results (§5.2, Table 3).** Fine-tuning Mixtral-8×7B-SFT on UltraInteract pairs
  (β = 0.1, LR 5e-7, 1 epoch, cosine schedule, warmup ratio 0.1; App. C): average over the eight reasoning
  benchmarks 56.3 (SFT) → 50.1 (DPO) → 57.9 (NCA). HumanEval 61.0 → 47.6 (DPO) → 62.8 (NCA);
  GSM-Plus 57.6 → 55.8 (DPO) → 61.5 (NCA). For the 7B SFT model the averages are 24.5 → 25.5 (DPO) → 35.7 (NCA).
- **Stated scope (§5.2).** The authors state NCA is more suitable for reasoning tasks where high-quality
  responses are sparse, and that DPO may be more suitable for general instruction-following data that
  contains no "golden" answers; they call the difference "largely empirical".

## Not stated by the source
- No human evaluation; MT-Bench comparisons are on UltraFeedback, not UltraInteract.

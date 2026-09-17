---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: "primary source (no library card at the time of writing)"
source_url: https://arxiv.org/abs/2403.03419
created_at: "2026-09-15"
verified_against: "arXiv:2403.03419v2 (30 Sep 2024), cached plain text"
---

# Negating Negatives: Alignment with Human Negative Samples via Distributional Dispreference Optimization (D2O)

Shitong Duan, Xiaoyuan Yi, Peng Zhang, Yan Liu, Zheng Liu, Tun Lu, Xing Xie, Ning Gu.

## Claims the chapter uses
- **Motivation (§1, Fig. 1).** In the HH dataset, a fraction of the *positive* responses are scored as
  harmful by an off-the-shelf reward model, and 33.88% of Safe-RLHF positives versus 8.03% of HH positives
  are marked toxic by the classifier of Ji et al. (Fig. 1c), so the paper drops the positive side.
- **Loss (§3.2, Eq. 3).**
  `L_D2O = −E_{(x,y_l)∼D}[ log σ( (β/K) Σ_{i=1..K} log π_θ(y_i|x)/π_{r-}(y_i|x) − α log π_θ(y_l|x)/π_{r+}(y_l|x) ) ]`,
  with `y_i ∼ π_r(·|x)` self-generated responses, `K` the number of self-samples, `π_{r-}` a more harmful
  reference and `π_{r+}` a more helpful one; in the experiments `π_{r+} = π_{r-} =` the original Alpaca model.
- **Gradient (Eq. 4).** The weight `σ(r̂_θ(µ) − r̂_θ(π_r))` is distributional and the positive part of the
  update, `∇_θ E_{π_r}[log π_θ(y)]`, is averaged over the `K` self-samples, which the paper states smooths
  out the contribution of any single harmful sample.
- **Results (§4.2, Table 1; Alpaca-7B, PKU-SafeRLHF, K = 11, α = 0.1, β = 0.1 for all methods).**
  Harmfulness (lower is better) and GPT-4 win rate against Alpaca: Alpaca 1.36 / —; gradient ascent on
  negatives only (GA) 1.21 / 20.13; SimPO −0.57 / 28.70; DPO-Ori −1.02 / 32.43; DPO-Full (297K pairs)
  −3.16 / 40.20; D2O −4.27 / 61.82. MMLU: Alpaca 38.61, D2O 38.66, Safe SFT 33.20, Self-Align SFT 27.03.
- **Interpretation stated by the authors.** Training on negatives alone with plain gradient ascent (GA)
  shows "catastrophic unlearning"; replacing the single positive with a distribution of self-samples
  avoids it (§4.2).

## Conditions and limits
- Backbones are Alpaca-7B, Phi-3-mini-4k-instruct and Qwen2-1.5B; the target is harmlessness, not
  general instruction quality; helpfulness falls for every method tested (§4.2).

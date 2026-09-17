---
chapter: ch-32a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2505.12082v3 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2505.12082
created_at: "2026-09-15"
---

# Excerpt: Model Merging in Pre-training of Large Language Models

**Paper:** ByteDance Seed (project lead Yunshui Li; full list in Contributions). arXiv v1 2025-05; v3 2025-05-22 read.
Source type: paper. Models and pre-training data are internal and not released (§3).

## Definitions (§3, Eqs. 1–5)
- Pre-trained Model Average (PMA): merge N checkpoints from one training run, M_avg = Σ_i w_i M_i.
- Checkpoints are spaced by a constant token interval V = T_{i+1} − T_i, where T_i is cumulative tokens of checkpoint i.
- SMA: w_i = 1/N. WMA: w_i = i, normalized by Σ w_i. EMA: M_avg^(i) = α M_i + (1 − α) M_avg^(i−1).
- Schedule: warmup-stable-decay (WSD); LRs chosen by scaling-law guidelines; evaluation is a weighted average over
  ARC-C, BBH, DROP, WinoGrande, HellaSwag, MMLU, C-Eval, TriviaQA, Ape210K, GSM8K, MATH, MBPP, HumanEval, AGIEval, GPQA, MMLU-Pro.

## Results (§4)
- Stable phase (Fig. 1): HumanEval 31.1 → 36.6 for Seed-MoE-1.3B/13B and 54.3 → 61.6 for Seed-MoE-10B/100B after merging.
- Cosine annealing phase (Fig. 2): PMA at early annealing is comparable to the end of annealing; for larger models it
  sometimes exceeds the naturally annealed model.
- Substitute for annealing (Fig. 3): two forks of Seed-MoE-1.3B/13B at 1.4T tokens, each trained 250B more tokens, one at
  constant LR and one annealed. Merged constant-LR models "significantly outperformed" both early and were "comparable"
  to the annealed models later. Values are plotted only.
- Method (Fig. 4): WMA best at 204B tokens; differences shrink later; SMA used afterwards.
- Interval (Fig. 5): with N = 10, V = 16B and 32B underperform the baseline at 204B tokens; reported good intervals are
  about 8B tokens (1.3B/13B), 4B (0.7B/7B), and about 80B (10B/100B).
- Number of checkpoints (Fig. 5): at the end of training N = 3 is "nearly 1 point lower" than N = 15; N = 10 used.
- PMA-init for continued training (CT) (§4.4, Fig. 6): Seed-MoE-0.7B/7B merged after ~1T stable tokens; slightly lower
  initial loss, similar final loss; MMLU higher early; overall parity at the end across LR schedules. For SFT, gains are
  "not consistently observed" (App. B).
- Stability (§4.5, Fig. 7): a 330M/3.3B MoE at LR 6e-3 diverged; resuming from a PMA of three pre-collapse checkpoints
  continued past the spike.

## Mechanism (§4.6, Eqs. 6–15)
Second-order expansion around θ*: L(θ) ≈ L(θ*) + ½ δᵀHδ. The average of k checkpoints has lower loss than their mean
loss when Σ_i Σ_{j≠i} δ_iᵀ H δ_j < (k − 1) Σ_i δ_iᵀ H δ_i (Eq. 15), which is easier when the cross terms are negative.
The authors state that merging checkpoints annealed to a very low LR gives smaller gains (Interpretation).

## Verification
- Read on 2026-09-15 against arXiv:2505.12082v3 PDF text (Abstract, §1–§5, Figs. 1–8 captions).
- Not reported: forgetting or retention measurements for continued training on a new distribution; exact numeric
  values behind Figs. 2, 3, 6; model architectures and data.

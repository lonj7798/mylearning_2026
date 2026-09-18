---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: "Model Merging in Pre-training of Large Language Models (ByteDance Seed)"
source_url: https://arxiv.org/abs/2505.12082
created_at: "2026-09-17"
note: "No library card exists for the slug `merging-in-pretraining` as of 2026-09-17. Every claim below was read in the cached full text of arXiv:2505.12082v3."
---

# Excerpt: Pre-trained Model Average (PMA) — merging checkpoints inside pre-training

- **Authors:** ByteDance Seed (full author list in the paper's Contributions section); correspondence Yunshui Li
- **Year:** arXiv v1 2025-05 (dated May 18, 2025); read at v3 (2025-05-22)
- **Source type:** paper
- **Models:** dense models from 411M to 70B parameters and MoE models from 0.7B/7B to 20B/200B
  activated/total parameters, all trained from scratch under a warmup-stable-decay (WSD) schedule (§1, §3).

## Method (§1, §4.2)
**PMA** averages checkpoints from a single training trajectory. Three weighting schemes are compared, where
w_i is the weight of the i-th of N checkpoints:
- SMA (simple moving average): uniform weights.
- WMA (weighted moving average): w_i = i, so later checkpoints count more.
- EMA (exponential moving average): w_i = α(1 − α)^(N − i).

Two further knobs: **V**, the token interval between merged checkpoints, and **N**, the number of checkpoints
merged.

## Findings used by ch-06
- **Merging during the constant-learning-rate (stable) phase raises downstream scores** (§4.1, Fig. 1).
  Seed-MoE-1.3B/13B on HumanEval: 31.1 → 36.6. Seed-MoE-10B/100B on HumanEval: 54.3 → 61.6.
- **PMA in the stable phase predicts annealed performance** (§4.1, Figs. 2–3). Two runs were forked from the
  stable phase of Seed-MoE-1.3B/13B at 1.4T tokens and trained 250B tokens further, one at a constant learning
  rate and one annealed. Merged constant-LR checkpoints outperformed both early on and were comparable to the
  annealed model later. The authors' conclusion: PMA can act as a low-cost estimate of the annealed result
  during the stable phase (Interpretation by the authors).
- **Weighting scheme stops mattering as training proceeds** (§4.2, Fig. 4). At 204B tokens WMA is the best of
  the three; later the differences between SMA, WMA and EMA become negligible, and the authors use SMA
  thereafter.
- **Interval V scales with model size** (§4.3, Fig. 5): about 4B tokens for 0.7B/7B models, about 8B tokens for
  1.3B/13B models, and about 80B tokens for 10B/100B models. Large intervals early in training
  (V = 16B, V = 32B at 204B tokens) were worse than the unmerged baseline, because they pull in unstable early
  weights.
- **Number of checkpoints N** (§4.3): early in training more checkpoints hurt; once training is complete,
  merging more helps, with N = 3 nearly 1 point below N = 15 on the overall average. The authors use N = 10.
- **PMA-init** (§4.5, Fig. 7): initializing a continued-training or SFT stage from a merged checkpoint gives a
  smoother gradient-norm curve than initializing from the latest checkpoint, without harming final
  performance; and after a loss spike that did not recover, resuming from a PMA over the preceding N
  checkpoints restored the trajectory.

## Conditions and limits
- All results are from the authors' own models; no external replication is reported in the paper.
- The checkpoints merged come from one trajectory, so this is not the same operation as souping independently
  fine-tuned models.

## Verification
- Read on 2026-09-17 in the cached full text of arXiv:2505.12082v3, §1, §4.1–§4.5 and the captions of Figs. 1–7.

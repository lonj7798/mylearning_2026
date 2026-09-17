---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2506.02355
primary_version: arXiv:2506.02355v2 (20 Jun 2025)
created_at: "2026-09-15"
---

# Excerpt: Rewarding the Unlikely — Lifting GRPO Beyond Distribution Sharpening

He, Fried, Welleck (Carnegie Mellon University), 2025. Domain: formal theorem proving in Lean, where the
verifier is exact (`R(x, y) = 1{y proves x}`).

## Rank bias (§3.5)
- For each training problem the group's correct samples are sorted by their probability under the initial
  model `π_0`, and the "uplift rate" `u_j` is the fraction of correct samples at rank `j` whose probability
  under the GRPO-trained model exceeds their probability under `π_0`.
- Fig. 4: `u_j` rises with the sample's initial probability. "the low-probability positive samples – those
  most critical for improving pass@N at large N – are almost never uplifted." The authors call this rank bias
  and attribute it to the optimizer and clipping, not to the GRPO loss itself (§4).

## Unlikeliness reward (§4.1)
`r_i = R(x, y_i) · (1 − β_rank · (G − rank(y_i)) / G)`, where `rank(y_i)` is the rank of `y_i` under
`π_θ_old` within the group of `G` samples, with rank 0 the highest-probability sample. Incorrect samples keep
`r_i = 0`. Groups whose advantage is zero before the perturbation are still skipped, so the sign of the update
is still set by `R(x, y_i)`. The experiments use `β_rank = 0.25`. (The text says
`rank(y_i) ∈ {1, 2, …, G}` in the same sentence that defines rank 0 as the highest-probability sample.)

## PPO epochs (§4.2)
Increasing the number of optimization steps per batch (`ppo-epochs`) also mitigates rank bias: the first steps
push high-rank solutions past the clipping threshold, so later steps act mostly on low-rank samples. The
authors report that this is slower and can be unstable, and that diversity rises with `ppo-epochs` up to 4,
where training became unstable.

## Settings and results
- Variants (Table 1): GRPO-Default (`K = 1` PPO epoch, `β_KL = 0.02`); GRPO-Unlikeliness-1 (`K = 1`,
  `β_KL = 0.10`, `β_rank = 0.25`); GRPO-Unlikeliness-2 (`K = 2`, `β_KL = 0.10`, `β_rank = 0.25`);
  GRPO-Epochs-2 and -3 (`β_KL = 0.10`, no rank term). The KL penalty was raised "because we found that it
  helps prevent deteriorating pass@N", but the authors state that this change alone was not enough (§5,
  App. D).
- Fig. 5: on the validation set, unlikeliness reward raises pass@N at large N with a small loss at pass@1 and
  pass@2; raising PPO epochs also raises pass@N.
- Table 2 (training problems solved in one epoch, out of 9,600): static DeepSeek-Prover-V1.5-SFT 7,707;
  GRPO-Default 7,860; GRPO-Epochs-2 8,008; GRPO-Epochs-3 8,006; GRPO-Unlikeliness-1 8,023;
  GRPO-Unlikeliness-2 8,065.
- Diversity (§5.3, Fig. 7): the number of unique proofs per step declines monotonically for the other
  variants; for GRPO-Unlikeliness-2 it drops and then recovers.
- Table 3, final model trained on 11k Lean-Workbook theorems: MiniF2F-test pass@32 / pass@128 —
  DeepSeek-Prover-V1.5-SFT 47.1 ± 0.6 / 49.2 ± 0.6; V1.5-RL 49.2 ± 0.6 / 51.2 ± 0.3; this work (GRPO-
  Unlikeliness-2) 48.8 ± 0.7 / 50.6 ± 0.5. On the authors' own validation set: 78.3 / 83.1, 84.8 / 87.5, and
  84.3 / 88.8 respectively.

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2506.02355v2 (scratchpad `sources/unlikeliness-reward.txt`).
- Not reported: policy entropy; results outside formal theorem proving; an ablation of `β_rank`.

---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/trpo.md
source_url: https://arxiv.org/abs/1502.05477
revised_at: "2026-09-15"
---

# Excerpt: TRPO bound and trust-region constraint (Schulman et al. 2015)

Checked against arXiv:1502.05477v5 on 2026-09-15. Used in read.md §1. The library card's hyperparameter table (CG damping, line-search constants) was not checked and is not used.

## Bound (§3, Theorem 1, Eq. 8-9)
- η(π_new) ≥ L_πold(π_new) − (4εγ/(1 − γ)²) α², with α = D_TV^max(π_old, π_new) and ε = max_{s,a}|A_π(s,a)|.
- Using D_TV² ≤ D_KL: η(π̃) ≥ L_π(π̃) − C·D_KL^max(π, π̃), C = 4εγ/(1 − γ)².
- Maximizing the right-hand side gives a monotonically non-decreasing sequence η(π_0) ≤ η(π_1) ≤ … (Eq. 10).

## Constraint instead of penalty (§4, Eq. 11-12)
- "In practice, if we used the penalty coefficient C recommended by the theory above, the step sizes would be very small."
- maximize L_θold(θ) subject to D_KL^max(θ_old, θ) ≤ δ, approximated with the average KL over sampled states.

## Cost and settings
- "We used δ = 0.01 for all experiments" (§8; Table 2 lists stepsize D_KL 0.01).
- App. C: k conjugate-gradient iterations with Fisher-vector products; "We found k = 10 to be quite effective"; a naive implementation "would spend more than 90% of the computational effort on these Fisher-vector products"; computing them on 10% of the data makes the cost about that of the gradient.

---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/trpo.md (card unverified on 2026-09-15; the items below were checked against the primary PDF)
source_url: https://arxiv.org/abs/1502.05477
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; previous version replaced)"
---

# Excerpt: Trust Region Policy Optimization (TRPO)

- **Authors:** John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel
- **Year:** 2015 (arXiv v1 2015-02; ICML 2015)
- **Source type:** paper
- **Used in:** ch-37 §3 (from the vanilla estimator to a constrained surrogate). PPO and InstructGPT are covered in ch-38.

## Items used by ch-37, with loci
- **Surrogate objective (§2).** L_π(π̃) = η(π) + Σ_s ρ_π(s) Σ_a π̃(a|s) A_π(s, a). η is expected discounted return, ρ_π the discounted state visitation of the old policy π, A_π the advantage under π. L matches η to first order at π̃ = π.
- **Improvement bound (Eq. 9).** η(π̃) ≥ L_π(π̃) − C · D_KL^max(π, π̃), with C = 4εγ / (1 − γ)², ε = max_{s,a} |A_π(s, a)|, and D_KL^max the maximum over states of the per-state KL. Maximizing the right-hand side at each iteration gives a non-decreasing sequence η(π_0) ≤ η(π_1) ≤ … under exact advantage evaluation (Eq. 10, Algorithm 1).
- **Practical constraint (Eq. 12–13).** Because the penalty coefficient C gives small steps, the paper instead maximizes L_θold(θ) subject to an average KL constraint E_s[D_KL(π_θold(·|s) ‖ π_θ(·|s))] ≤ δ, estimated with importance sampling (Eq. 14).
- **Optimization (§6, App. C).** Conjugate gradient on Fisher-vector products followed by a line search that checks improvement of the surrogate and satisfaction of the KL constraint.
- **Step size used.** "We used δ = 0.01 for all experiments" (§8.1).

## Relation to the language-model setting (course interpretation)
- TRPO's constraint is on KL between consecutive policies (step size). The KL term in RLHF (ch-38) is to a fixed reference policy. The two quantities are different and are controlled separately.
- The paper contains no language-model experiments.

## Not used
The previous version of this excerpt listed CG damping, line-search constants, GAE λ, and a δ range 0.01–0.05; these were taken from the unverified card and are not used by ch-37.

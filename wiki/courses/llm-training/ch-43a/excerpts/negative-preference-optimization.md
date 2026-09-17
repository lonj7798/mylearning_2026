---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/negative-preference-optimization.md on 2026-09-15)
source_url: https://arxiv.org/abs/2404.05868
source_version: arXiv v2 (2024-10-10); v1 2024-04-08
created_at: "2026-09-15"
---

# Excerpt: Negative Preference Optimization — From Catastrophic Collapse to Effective Unlearning (Zhang, Lin, Bai, Mei; UC Berkeley and Salesforce AI Research)

Facts used by [[read]], read in the arXiv v2 PDF on 2026-09-15.

## Setting
- The task is unlearning: remove the influence of a forget set D_FG while keeping utility on a retain set. Gradient ascent (GA) maximizes the cross-entropy loss on D_FG.
- Catastrophic collapse (§2.1): under GA "the model utility quickly drops to zero, and the forget quality improves temporarily for a very short time horizon before quickly dropping too"; the model then "generates gibberish outputs" (Figure 2 shows an example output "narr narr narr narr").

## NPO loss and its weight (§3, Eqs. 3-5)
- `L_NPO,β = −(2/β) E_{D_FG}[log σ(−β log(π_θ(y|x)/π_ref(y|x)))] = (2/β) E[log(1 + (π_θ/π_ref)^β)]`: the DPO loss with the preferred term removed.
- Proposition 1: as β → 0, L_NPO reduces to the GA loss and its gradient converges to the GA gradient.
- Gradients: `∇L_GA = E[∇ log π_θ(y|x)]`, `∇L_NPO,β = E[W_θ(x,y) ∇ log π_θ(y|x)]` with `W_θ(x,y) = 2 π_θ^β(y|x) / (π_θ^β(y|x) + π_ref^β(y|x))`. The paper calls W an "adaptive smoothing weight": once `π_θ ≪ π_ref` on a forget example, W ≪ 1 and the NPO gradient is much smaller than the GA gradient. The GA loss is unbounded below; the NPO loss is lower-bounded for any finite β > 0.
- Theorem 2 (logistic regression, n_f ≤ d): GA's iterates diverge linearly in the number of steps while NPO's diverge logarithmically, so NPO "diverges exponentially slower".

## Experiments
- TOFU benchmark on Llama2-7b-chat, forget sets of 1%, 5% and 10% (Forget01/05/10), with β = 0.1 in all DPO-based experiments (App. D.1). NPO-based methods are the only ones reaching forget quality above 0.05 in Forget05 and Forget10; GA and GA+RT rise briefly and then fall (§5, §2.1).
- Extended tasks: NPO+RT with a tuned retain-loss coefficient is reported as the first method to give non-trivial results when forgetting 50% and 90% of the data (§5.4, Figure 10).

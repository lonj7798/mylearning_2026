<!-- scope: Identity Preference Optimization — ΨPO framework with identity link; avoids DPO over-confidence
     deps: [[dpo]]
     see-also: [[simpo]], [[kto]], [[orpo]]
-->

# A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO / ΨPO)
- **Core Insight:** DPO's sigmoid link drives the implicit reward gap to infinity whenever every pair is perfectly ranked in the dataset — replacing the sigmoid with the identity function yields a bounded squared-loss objective (IPO) that cannot overfit deterministic preferences.
- **Guideline:** Switch from DPO to IPO when the preference dataset has near-deterministic labels (distilled or BoN-gated); tune τ rather than β, and expect smaller log-prob drift on the chosen side.
- **Authors:** Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, Rémi Munos
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2310.12036
- **Relevant topics:** preference learning theory, overfitting, closed-form RLHF alternative, ΨPO family

## Abstract
We introduce ΨPO, a general objective for learning from pairwise preferences that expresses RLHF and DPO as special cases. By working directly on pairwise preferences we avoid two approximations present in prior work: reducing preferences to pointwise rewards and assuming reward-model generalization. We derive Identity-PO (IPO), a practical ΨPO instance with Ψ = identity, and show that IPO is provably robust to deterministic preferences where DPO collapses.

## Key Contributions
- Formalizes ΨPO — a family of preference-learning objectives parametrized by a non-decreasing Ψ.
- Proves that when the preference dataset has P(y_w ≻ y_l) → 1, DPO's loss becomes unbounded and optimization is equivalent to maximizing π(y_w)/π(y_l) without any regularization.
- Proposes **IPO** (Ψ = id) — yields a bounded mean-squared error loss.
- Connects τ to KL budget: larger τ → closer to π_ref.

## Key Figures/Tables to Study
- **Figure 2:** 2-action toy showing DPO policy drifting to assign probability 1 to y_w, while IPO remains bounded.
- **Section 4 / Equations for IPO:** the squared objective — the practical deliverable.

## Technical Details

### ΨPO general form
For a non-decreasing Ψ : [0,1] → R:
`max_π E_{x,y_w,y_l}[ Ψ( p*(y_w ≻ y_l | x) ) ] − τ · D_KL(π || π_ref)`
Choosing Ψ(p) = log(p / (1−p)) recovers DPO (logit link); choosing Ψ(p) = p recovers IPO.

### IPO loss (identity link)
`L_IPO(π_θ; π_ref) = E_{(x,y_w,y_l)~D} [ ( h_π(y_w, y_l, x) − (1 / (2τ)) )^2 ]`
where
`h_π(y_w, y_l, x) = log[ π_θ(y_w|x) / π_ref(y_w|x) ] − log[ π_θ(y_l|x) / π_ref(y_l|x) ]`
- Target is a fixed constant 1/(2τ) — not ±∞ like DPO.
- Bounded gradient even for deterministic data.

### Relationship to DPO
- Optimal h for DPO as dataset becomes deterministic: `h* → ∞`.
- Optimal h for IPO: `h* = 1/(2τ)` — fixed, finite.
- Same `log π_θ/π_ref` ratios, different loss shape (squared vs log-sigmoid).

### Hyperparameters
| Knob | Typical |
|------|---------|
| τ | 0.01–0.5 (paper: 0.005, 0.01, 0.1) |
| Learning rate | 5e-7 (same as DPO) |
| π_ref | SFT, frozen |
| Batch size | 32–64 pairs |
| Epochs | 1–3 |

Smaller τ → larger target margin → stronger preference enforcement (but higher variance).

### Empirical behavior
- Less sensitive to β/τ sweep than DPO.
- Better preserves chosen-side log-probabilities (DPO drops both sides, IPO mostly drops rejected).
- Slightly under-performs DPO on noisy preferences, outperforms on clean / distilled.

## Connections
- Direct predecessor: [[dpo]] (logit-link special case of ΨPO).
- Reference-free alternatives: [[simpo]], [[orpo]].
- Empirical survey of all PO variants: [[hf-dpo-zoo]].

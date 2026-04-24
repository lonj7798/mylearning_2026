---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ipo.md
source_url: https://arxiv.org/abs/2310.12036
created_at: "2026-04-23"
---

# Excerpt: IPO — bounded target via identity link

**Source library:** `wiki/raw-data/llm-training/papers/ipo.md`
**Authors:** Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, Rémi Munos
**Venue:** AISTATS 2024 (arXiv 2310.12036)
**Year:** 2023

---

## Why this source anchors ch-39 §3

DPO's derivation (see [[dpo-derivation]]) treats the pairwise preference as Bradley-Terry logistic — i.e., the link function is `Ψ(p) = log(p/(1−p))`. [[ipo]]'s critique is that when the preference data is deterministic (every pair perfectly labeled), the MLE under the logistic link pushes `p → 1`, which drives the implicit-reward margin `r̂_w − r̂_l → ∞`. Because the KL regularization is only implicit in DPO (the β factor plus the reference log-ratio), there is no finite force pulling the margin back in.

IPO's answer: **replace Ψ**. Use `Ψ(p) = p` (identity) instead of the logit. Then the objective is not an NLL but a squared-error penalty around a *finite* target `1/(2τ)`. Optimization can still prefer the chosen, but the margin cannot run away.

## The ΨPO family — where DPO and IPO live

Source lines 30–32:

```
max_π  E_{x, y_w, y_l} [ Ψ( p*(y_w ≻ y_l | x) ) ]  −  τ · D_KL( π || π_ref )
```

- `Ψ(p) = log[p/(1−p)]` → DPO (logit link).
- `Ψ(p) = p`             → IPO (identity link).

Both belong to the same family; they differ in one function.

## IPO loss verbatim

Source lines 35–37 (Equations for IPO):

```
L_IPO(π_θ; π_ref) = E_{(x, y_w, y_l) ~ D} [ ( h_π(y_w, y_l, x)  −  1/(2τ) )^2 ]

h_π(y_w, y_l, x) = log[π_θ(y_w|x) / π_ref(y_w|x)]  −  log[π_θ(y_l|x) / π_ref(y_l|x)]
```

Note `h_π` is exactly the quantity DPO's sigmoid takes as its argument (the implicit-reward margin). Swapping the sigmoid for a squared error around `1/(2τ)` is the entire algorithmic change.

## The deterministic-preference theorem

Source §Key Contributions (line 19): "when P(y_w ≻ y_l) → 1, DPO's loss becomes unbounded and optimization is equivalent to maximizing π(y_w)/π(y_l) without any regularization." IPO does not have this failure mode because the loss has a unique finite minimum at `h* = 1/(2τ)`.

This is why IPO is the right choice when:

- The preference labels come from distillation (teacher always prefers one response).
- Best-of-N sampling was used to manufacture pairs (the N=1 response was filtered in).
- A verifier with 0/1 correctness labelled pairs (e.g., GSM8K math).

In all three, DPO will saturate; IPO will converge to the finite target.

## Hyperparameters

Source lines 47–53:

| Knob | Typical | Notes |
|------|---------|-------|
| τ | 0.01–0.5 | Swap for β; larger τ → smaller target → closer to π_ref |
| LR | 5e-7 | Same as DPO |
| Batch | 32–64 | Same as DPO |
| Epochs | 1–3 | Same as DPO |
| π_ref | SFT, frozen | Same as DPO |

Note that smaller τ means a *larger* target margin `1/(2τ)`, which means stronger preference enforcement but higher variance. This is the opposite direction from DPO's β (where smaller β allowed more drift). Easy mental trap.

## How ch-39 uses this

§3 of `read.md` presents IPO as DPO's bounded-target counterpart. The comparison table in §8 pins it to "paired, near-deterministic" data. The framework code in §9 shows that IPO differs from DPO by one line: `losses = (logits - 1 / (2 * self.beta)) ** 2` vs `-F.logsigmoid(beta * logits)`. The loss landscape in `figures/dpo-landscape.html` visualizes the vertical dashed line at `m = 1/(2τ)` as the finite IPO target.

## Connections

- Predecessor and logit-link special case: [[dpo]] / [[dpo-derivation]].
- Reference-free cousins: [[simpo]], [[orpo]].
- Framework implementation: [[openrlhf-dpo]] with `ipo=True`.
- Survey of variants: [[hf-dpo-zoo]].

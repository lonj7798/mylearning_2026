---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dpo.md
source_url: https://arxiv.org/abs/2305.18290
created_at: "2026-04-23"
---

# Excerpt: DPO — the closed-form derivation ch-39 is built on

**Source library:** `wiki/raw-data/llm-training/papers/dpo.md`
**Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
**Venue:** NeurIPS 2023 (arXiv 2305.18290)
**Year:** 2023

---

## Why this source anchors ch-39

Every variant in ch-39 — IPO, KTO, SimPO, ORPO, RPO — perturbs exactly one assumption in DPO's derivation. To understand what each variant is trading, you have to know which step in the derivation it is attacking. This excerpt fixes the canonical walk so the chapter can say "IPO replaces Ψ in step 4" or "SimPO drops π_ref in step 2" without ambiguity.

The derivation has four load-bearing pieces:

1. **Gibbs form of the optimum.** For any reward `r`, the KL-constrained reward-max problem has a unique analytic solution `π*(y|x) = (1/Z(x)) π_ref(y|x) exp(r(x,y)/β)`. This is not approximate; it is the exact extremum of the Lagrangian. Source lines 33–35.

2. **Inversion for `r`.** Take logs: `r(x,y) = β log[π*/π_ref] + β log Z(x)`. Source line 37. The second term `β log Z(x)` depends only on the prompt — not on `y`. SimPO attacks step 1 by replacing the reference ratio with an average per-token log-prob; IPO attacks step 4 below by swapping the link; ORPO attacks by never computing `π_ref` at all and adding an SFT anchor instead.

3. **Bradley-Terry substitution.** `P(y_w ≻ y_l | x) = σ(r(x,y_w) − r(x,y_l))`. Because both sides share `x`, the `β log Z(x)` terms from step 2 cancel in the difference. This is the only reason DPO can skip fitting an explicit reward model: `Z(x)` is the single thing that made exact RLHF inference intractable, and it vanishes under the pairwise difference.

4. **NLL as classifier.** Fitting `π_θ` to the preference data under this model gives the DPO loss `L = −E[log σ(β(log π_θ(y_w)/π_ref(y_w) − log π_θ(y_l)/π_ref(y_l)))]`. Source line 41 (Equation 7).

## The equation verbatim

Source lines 40–45 (DPO loss, Equation 7 of the paper):

```
L_DPO(π_θ; π_ref) = − E_{(x, y_w, y_l) ~ D} [
    log σ( β · log[π_θ(y_w|x) / π_ref(y_w|x)]
         − β · log[π_θ(y_l|x) / π_ref(y_l|x)] )
]
```

And the implicit reward (line 48):

```
r̂_θ(x, y) = β · log[ π_θ(y|x) / π_ref(y|x) ]
```

## The gradient — why DPO is stable without a curriculum

Source line 52:

```
∇L_DPO = − β · E[ σ(r̂_l − r̂_w) · ( ∇ log π_θ(y_w|x) − ∇ log π_θ(y_l|x) ) ]
```

The scalar `σ(r̂_l − r̂_w)` sits *outside* the gradient. Its role is critical: when the current policy already ranks the pair correctly (`r̂_w > r̂_l`, large margin), `σ(r̂_l − r̂_w) ≈ 0` and there is effectively no gradient. When the policy ranks it wrong or barely right, `σ(r̂_l − r̂_w) ≈ 1` and full gradient flows. This is automatic hard-example mining. Ch-39's §2.5 is a one-line rephrasing of this.

## Hyperparameters anchoring ch-39's §2.6 and §11

Source lines 56–64:

| Knob | Paper | Ch-39 default | Industrial ([[llama-3]]) |
|------|-------|---------------|--------------------------|
| β | swept {0.05, 0.1, 1, 5} | 0.1 | 0.1 |
| LR | 5e-7 to 1e-6 | 5e-7 | 1e-5 (405B tolerates) |
| Batch (pairs) | 32–128 | 64 | per [[llama-3]] recipe |
| Epochs | 1–3 | 1 | 1 per round |
| π_ref | SFT, frozen | SFT, frozen | SFT, frozen |
| Length normalization | off | off | off (DPO base); use SimPO if length inflates |

The paper explicitly warns that length normalization is off — motivating [[simpo]]. Ch-39 §5 picks that up.

## Connections

- The RLHF objective this inherits: [[rlhf-instructgpt]].
- Identity-link successor attacking step 4: [[ipo]].
- Reference-free successor attacking step 2: [[simpo]], [[orpo]].
- Unary-label cousin (keeps π_ref, drops pairs): [[kto]].
- Chosen-logprob anchor: [[rpo]], [[llama-3]].
- Framework code implementing the exact equations above: [[openrlhf-dpo]].
- The chapter host: [[ch-39]] — §2 of `read.md`.

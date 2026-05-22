---
chapter: ch-20
course: model-quantization
phase: read
excerpt_of: "Statistically-Lossless Quantization of Large Language Models (Helcig, Kurtic, Alistarh 2026)"
source_url: https://arxiv.org/abs/2605.02404
created_at: "2026-05-21"
---

# Excerpt: Statistically-Lossless Quantization

**Authors:** Michael Helcig, Eldar Kurtic, Dan Alistarh
**Year:** 2026
**arXiv:** 2605.02404
**Raw-data source:** [[raw-data/statistically-lossless-quantization]]

---

## The three losslessness levels (paper §2)

The paper's central move is replacing the binary "lossy / lossless" framing with a three-level operational ladder. Each level corresponds to *which* aspect of model behavior is preserved.

| Level | What must be preserved | Typical bit budget |
|-------|------------------------|--------------------|
| **Task-lossless** | benchmark accuracy within run variance | below 4 bits/parameter |
| **Distribution-lossless** | next-token distribution effectively unchanged | about 5–6 average bits |
| **Exact lossless** | exact weights and logits | storage compression only |

The crucial empirical claim: the gap between task-lossless and distribution-lossless is roughly *one to two bits per parameter*. A model that aces MMLU at 4 bits can still fail speculative-decoding acceptance, distillation, or RL reward-modeling because its full next-token distribution has drifted.

---

## Expected Acceptance Rate (EAR)

EAR is the *interpretable* distribution-fidelity metric the paper introduces. For two next-token distributions `p` (FP16) and `q` (quantized) at the same context:

```math
\mathrm{EAR}(p, q) = \sum_{v \in V} \min(p(v), q(v))
```

EAR = 1 means `p = q` exactly. EAR = 0.99 means an optimal coupling could match samples from `p` and `q` 99% of the time — and this is *exactly* the maximum achievable acceptance rate for speculative decoding using `q` as the draft and `p` as the verifier.

Aggregated over a held-out corpus, EAR is the right scalar for any downstream consumer of the logit distribution (speculative decoding, KD, reward model, sampling-based generation with low temperature).

**Why this metric and not KL.** KL(p‖q) is unbounded; small KL doesn't directly map to "what fraction of tokens behave the same." EAR is bounded in [0, 1], has a direct operational interpretation, and is the right number to put in a deployment report.

---

## Why symmetric quantization can be insufficient at the distribution level

The paper proves (informal restatement) a gamma-squared variance law: symmetric scalar quantization of a tensor with sample skewness `γ` inflates the per-element variance by a factor proportional to `γ²` relative to asymmetric (zero-point) quantization at the same bit budget.

For weights of a well-trained FP16 base model, `γ ≈ 0` and the inflation factor is tiny — symmetric quant is fine.

For weights of an instruction-tuned or RLHF'd model, post-training reshapes the weight distribution and `γ ≠ 0`. The variance inflation under symmetric quant compounds across layers and shows up as next-token-distribution drift even when MMLU is preserved.

The practical corollary: if you are quantizing a base model, symmetric is OK. If you are quantizing a chat / instruct / RL model and you care about logit fidelity, use asymmetric (zero-point). This is exactly why GPTQ defaults to symmetric W4 but [[awq]] and the production AWQ recipes default to asymmetric.

---

## The SLQ method (paper §4)

SLQ is the method the paper proposes as a recipe for distribution-lossless quantization. Four ingredients:

1. **Layer-wise bitwidth search** — different layers get different bit budgets per a sensitivity score.
2. **Non-uniform quantization** — quantile-fit codes per channel (similar to NF4 but per-layer).
3. **Asymmetric** quantization (zero-point ≠ 0) — to control the variance-inflation issue above.
4. **Optimized kernels** — so the storage win translates to inference speedup (1.7–3.6× over FP16 reported).

The headline result: distribution-lossless at ~5–6 bits/parameter on Llama-class models. Strictly more than the 4-bit GPTQ/AWQ defaults, but with a *measurable guarantee* that the next-token distribution matches FP16.

---

## How to use this in an evaluation harness

For any quantized model deployed where logit consumers matter, the report should include:

- **PPL** (the long-standing default) with CI.
- **Task accuracy on a 5-task suite** with CI — for task-lossless verification.
- **EAR vs FP16 base** on a held-out 1K-token corpus — for distribution-lossless verification.
- **Speculative-decoding acceptance rate** if speculative decoding is the deployment target — the operational equivalent of EAR.

The asymmetry: task-lossless quantization is achievable at 4 bits and is the right target for memory-constrained deployment. Distribution-lossless quantization needs 5–6 bits and is the right target for any pipeline that consumes the logit distribution.

---

## Connections

- [[ch-20]] §2 — the chapter section that integrates this framework into the methodology.
- [[gptq]] / [[awq]] — the practical lossy baselines SLQ asks "are these statistically lossless?" of.
- [[rate-distortion-theory]] — the foundational lower bound this paper is reasoning against.
- [[aqlm]] / [[spqr]] / [[squeezellm]] — adjacent methods chasing near-lossless low-bit via non-uniformity, sparsity, or codebooks.

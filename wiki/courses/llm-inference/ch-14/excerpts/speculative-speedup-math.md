---
chapter: ch-14
course: llm-inference
phase: read
excerpt_of: "Speedup analysis synthesized from Leviathan 2023, vLLM speculative-decoding benchmarks, HF assisted-generation blog"
source_url: synthesis
created_at: "2026-05-21"
---

# Excerpt: Speculative-Decoding Speedup Math — Closed-Form + Production Numbers

**Sources synthesized:**
- Leviathan-Kalman-Matias 2023 (Theorem 3.8)
- vLLM `speculative_config` benchmarks
- HF assisted-generation blog (Joao Gante 2023)
- Production reports from Anthropic, Mistral, DeepSeek

---

## Expected accepted tokens per round

Per round, the drafter proposes K tokens. Acceptance at each position is Bernoulli with rate `α` (assumed independent for the basic analysis; not strictly true in practice — acceptances are correlated, but the approximation is close).

Expected accepted count (without the bonus):

```math
\mathbb{E}[\text{accepted}] = \sum_{i=1}^{K} \alpha^{i} = \frac{\alpha (1 - \alpha^K)}{1 - \alpha}
```

With the bonus token (free on all-accept):

```math
\mathbb{E}[\text{committed per round}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}
```

---

## Wall-clock speedup formula

Let `T_t` = target forward pass time, `T_d` = drafter forward pass time, `c = T_d / T_t`.

Per round:
- Drafter cost: `K · T_d = K · c · T_t`
- Target verification cost: `T_t` (one forward pass over K positions)
- Total round cost: `(1 + Kc) T_t`

Baseline (no spec-dec): one token per `T_t`. Speedup:

```math
S(\alpha, K, c) = \frac{1 - \alpha^{K+1}}{(1 - \alpha)(1 + Kc)}
```

---

## Speedup table — varied α and K

Assume drafter cost ratio `c = 0.05` (drafter is 20× cheaper than target — roughly Llama-3.2-1B vs Llama-3-70B).

| α \\ K | 2 | 4 | 6 | 8 |
|--------|---|---|---|---|
| 0.5 | 1.32× | 1.55× | 1.64× | 1.67× |
| 0.6 | 1.43× | 1.78× | 1.95× | 2.03× |
| 0.7 | 1.55× | 2.04× | 2.34× | 2.50× |
| 0.8 | 1.66× | 2.32× | 2.78× | 3.08× |
| 0.9 | 1.74× | 2.61× | 3.32× | 3.83× |

Key observations:
- At low α (0.5-0.6), more K barely helps — K=4 is near-optimal.
- At high α (0.8-0.9), more K helps significantly — K=8 is reasonable.
- **The speedup is bounded by `1/c = 20×` in the limit** — even perfect drafter can't help more than the drafter cost ratio allows. With `c = 0.05` the ceiling is `1 + 1/c ≈ 21×`, but you'd need α = 1 and K = ∞.

---

## Optimal K (rough)

Differentiating `S(α, K, c)` w.r.t. K (treating K as continuous) and setting to zero:

```math
K^* \approx \frac{\ln(c \ln(1/\alpha))}{\ln \alpha}
```

For `α=0.7, c=0.05`: `K* ≈ 4.5`. So K=4 or 5.
For `α=0.85, c=0.05`: `K* ≈ 7.5`. So K=7 or 8.

In practice, frameworks default to **K=5** because it's a sane middle ground for typical α=0.6-0.8.

---

## What `c` actually equals in practice

`c` depends on the drafter and the serving stack:

| Drafter | Target | c (approx) |
|---------|--------|------------|
| Llama-3.2-1B | Llama-3-70B | ~0.03 |
| Llama-3.1-8B | Llama-3-70B | ~0.15 |
| TinyLlama-1.1B | Llama-2-70B | ~0.04 |
| Distilled drafter (custom) | Llama-3-70B | ~0.02 |
| Medusa heads | self | ~0.02 (extra heads on shared trunk) |
| EAGLE feature predictor | self | ~0.02 |
| Prompt lookup (n-gram) | any | ~0.001 (CPU-only) |
| Lookahead Jacobi | self | ~0 (within target forward) |

The smaller `c`, the smaller the "cost penalty" per drafted token — which is why no-drafter and head-based methods (ch-15) win.

---

## What `α` actually equals in practice

From published benchmarks:

| Drafter / Target / Workload | α |
|----------------------------|---|
| Llama-3.2-1B / Llama-3-70B / ShareGPT chat | 0.65-0.75 |
| TinyLlama / Llama-2-70B / chat | 0.55-0.65 |
| Llama-3-8B / Llama-3-70B / chat | 0.70-0.80 |
| Llama-3-8B / Llama-3-70B / code | 0.80-0.90 |
| Llama-3-8B / Llama-3-70B / RAG | 0.85-0.95 (long copies) |
| Prompt lookup / Llama-3-70B / summarization | 0.80-0.95 |
| Prompt lookup / Llama-3-70B / open-ended chat | 0.10-0.30 |
| Medusa heads / target / chat | 0.70-0.85 |
| EAGLE feature / target / chat | 0.80-0.95 |

Code and RAG dominate the high-α regimes because token sequences are more predictable. Open-ended chat is the hardest case.

---

## End-to-end production speedups

vLLM benchmarks on Llama-3-70B with Llama-3.2-1B drafter at `num_speculative_tokens=5`:

| Workload | Baseline TPOT | Spec-dec TPOT | Speedup |
|----------|---------------|---------------|---------|
| ShareGPT chat | 35 ms | 18 ms | 1.94× |
| Code (HumanEval) | 32 ms | 12 ms | 2.67× |
| RAG (1k context, copy-heavy) | 40 ms | 14 ms | 2.86× |
| Creative writing | 35 ms | 28 ms | 1.25× |

Production-typical: 1.5-2.5× for chat, 2-3× for code/RAG, marginal for creative.

---

## The break-even threshold

Spec-dec helps only if the saved target passes exceed the drafter overhead. Per-round, this needs:

```math
\frac{1 - \alpha^{K+1}}{1 - \alpha} > 1 + Kc
```

For K=5, c=0.05: the LHS exceeds RHS when `α > ~0.25`. Below α=0.25, spec-dec is *strictly slower* than naive decoding.

**Implication**: monitor `α` per workload; if `α < 0.4` consistently, disable spec-dec. vLLM does this automatically via `--speculative-disable-by-batch-size`.

---

## Connections

- [[excerpts/leviathan-2023]] — the closed-form derivation of `S`.
- [[excerpts/hf-assisted-generation]] — the reference implementation that this math predicts.
- [[ch-15]] — head-based and feature-level methods that push c → 0 and α → 0.9.
- [[ch-14]] — parent chapter.

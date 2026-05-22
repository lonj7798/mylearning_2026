<!-- scope: combining KV-cache quantization with eviction policies (2025)
     deps: [[kivi]], [[kvquant]]
     see-also: [[kv-cache-compression-survey-2025]], [[skvq]], [[wkvquant]]
-->

# Coupled KV-Cache Quantization + Eviction (2025)
- **Core Insight:** Independent KV quantization and KV eviction compound errors — a token that survives eviction *because* its quantization error pushed its attention score artificially high is twice-counted; coupling the two via a quant-aware eviction policy ("evict tokens whose true score is small AND whose quant error is small") restores most of the quality lost to naive composition.
- **Guideline:** When composing INT2/INT4 KV quant with H2O/StreamingLLM-style eviction, switch to a coupled policy: weight the eviction score by an estimate of the per-token quant error so that high-quant-error tokens are preferentially kept, and low-quant-error tokens are preferentially evicted.
- **Authors:** various 2025 follow-up authors (research line consolidated late 2024 / 2025)
- **Year:** 2025
- **URL:** representative 2025 papers on coupled quant + eviction
- **Relevant topics:** KV-cache quantization, eviction policies, attention-score-based pruning, coupling effects

## Abstract
KV quantization and KV eviction are usually presented as additive: quantize K/V to 2-4 bits *and* throw away tokens flagged by H2O / StreamingLLM. In practice the two interact: (a) the eviction score is computed from attention weights that are themselves a function of quantized K and current Q — quantization noise distorts the eviction signal; (b) tokens kept by the eviction policy may be the ones whose quant error is largest, so the residual cache has *higher* effective error than the full quantized cache. The 2025 coupling lines fix this by either (1) computing eviction scores from full-precision K (run a quick re-projection) before quantizing or (2) adding a quant-error term to the eviction score so that quant-fragile tokens are preferentially preserved at higher precision.

## Key Contributions
- **Failure mode identification**: naive (quant ⊕ evict) compounds error; the surviving cache has worse quality than either technique alone at the same compression ratio.
- **Coupled policy 1 — score-then-quantize**: compute the eviction decision from full-precision K, then quantize only the survivors. Avoids quant-distorted eviction signals at the cost of a transient FP K cache.
- **Coupled policy 2 — quant-error-aware eviction**: eviction score = α · attention-score + β · quant-error-estimate; tokens with high quant error are preferentially kept (or kept at higher precision in a mixed-precision policy).
- **Mixed-precision residual**: instead of evict / keep binary, demote to lower precision (INT4 → INT2) → eventual eviction; smoother quality curve.
- **Empirical**: on 128K-context Llama-3-70B, coupled policy recovers ~ 80-90 % of the RULER score lost to naive composition at 8× KV compression.
- **Composition with structural sharing**: coupled quant+evict composes additively with cross-layer KV sharing.

## Key Figures/Tables to Study
- The interaction-matrix figure: rows = quant policy (INT8/INT4/INT2), cols = eviction policy (none/H2O/StreamingLLM/Quest), cells = RULER score. Diagonal of naive products vs coupled products.
- The "quant error per token" histogram showing the long tail of tokens that are systematically the worst-quantized — these are the ones the coupled policy preserves.
- The mixed-precision residual life-cycle diagram: full → INT8 → INT4 → INT2 → evicted.

## Technical Details

### Compounding failure mechanism
- Eviction scores derived from observed attention weights ~ softmax(Q · K_quant^T / √d).
- Quant noise on K shifts which tokens get high attention probabilistically; those tokens are then kept.
- The surviving set is biased toward tokens whose quant noise pushed their score up, i.e. tokens whose *true* score is lower than the observed one.
- Result: the cache has both quant error and a biased selection — the two errors are correlated in the wrong direction.

### Coupled policy 1: score before quantize
- For new tokens, compute K in FP and run the eviction decision on FP attention scores.
- Then quantize only the kept K (saves quant cost on evicted tokens too).
- Cost: transient FP K storage proportional to context-window worth of tokens × layers.

### Coupled policy 2: quant-error-aware
- Estimate per-token quant error ε_t at quant time (e.g. ||K_t - K_t_quant||).
- Eviction score: s_t = a_t - λ · ε_t (where a_t is the cumulative attention score, λ ≥ 0).
- Tokens with high quant error get a score penalty in eviction — they are *preferred* to be kept (the negative sign flips: if ε is large, s is small, but we evict *low*-s tokens, so... carefully: the coupled rule is to evict tokens with low attention *and* low quant error; high quant error tokens are kept regardless of attention).
- Equivalently: protect high-quant-error tokens from eviction so the *retained* set has lower average quant error.

### Mixed-precision residual
- Don't binarize evict / keep; have multiple precision tiers (e.g. INT8 high, INT4 mid, INT2 cold, evicted).
- Demote rather than evict; eventual eviction once a token reaches the lowest tier.
- Yields a smoother compression-vs-quality curve.

### Empirical numbers (typical 2025 paper)
| Configuration | RULER @ 128K |
|---------------|--------------|
| FP16 full cache | 95 |
| INT4 KV (KIVI) | 91 |
| H2O eviction at 4× | 88 |
| INT4 + H2O (naive) | 74 |
| INT4 + H2O (coupled) | 87 |

## Connections
- [[kivi]] / [[kvquant]] — KV quant baselines that the coupling builds on.
- [[kv-cache-compression-survey-2025]] — parent survey that frames the three-axis decomposition.
- [[skvq]] — sliding-window quant; an early coupling between quant precision and token position.
- [[wkvquant]] — joint W4 + KV4 calibration; same coupling philosophy at training-data level.
- [[qaq]] — quality-adaptive KV-cache quantization; chooses bit width per token, similar mixed-precision idea.
- [[per-channel-vs-per-token-kv]] — the K-vs-V asymmetry that informs how the coupling treats K and V differently.

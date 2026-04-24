# Excerpt: DeepSeek-V4 Innovations vs V3

<!-- source: [[deepseek-v4|report]], [[deepseek-v3|report]] -->

## The V3-to-V4 Transition: Five Axes of Change

DeepSeek-V4-Pro (1.6T total, 49B active) is not a simple scale-up of V3 (671B total, 37B active). Five new architectural components each target a specific bottleneck that V3 could not overcome by scaling alone.

---

## 1. Attention: MLA to Hybrid CSA + HCA

### V3: Multi-head Latent Attention

V3 stores a 576-dim compressed latent (512 KV + 64 RoPE) per token per layer. All heads reconstruct full keys and values from this shared latent via learned up-projections. The mechanism is uniform: every head attends to every position with the same resolution, and attention is still O(L^2) in sequence length.

### V4: Per-Head Sparse Pathway Assignment

V4 assigns each attention head to one of two pathways:

**CSA (Compressed Sparse Attention):**
- A lightweight FP8 "lightning indexer" scores all keys against each query
- GPU partial-sort selects top-k most relevant tokens per query
- Attention is computed only over these k selected tokens
- Complexity: O(Lk) instead of O(L^2) — at L = 1M with k = 1024, this is ~1000x cheaper

**HCA (Heavily Compressed Attention):**
- Extreme KV-cache compression — far beyond MLA's 576 dims
- Provides broad, coarse context awareness across the full sequence
- Suited for "what topic is being discussed" rather than "what exact token appeared"

**Why hybrid?** Empirical observation: attention heads serve fundamentally different roles. Some are "needle-in-haystack" lookup heads (best served by CSA's precise sparse access), others are "global summary" heads (best served by HCA's cheap broad coverage). Uniform attention forces all heads into one compromise; the hybrid lets each head specialize.

**Impact:** At 1M tokens, V4-Pro uses 27% of inference FLOPs and 10% of KV cache vs V3.2.

---

## 2. Residual Connections: Identity Addition to Birkhoff Polytope Mixing

### V3: Standard Residuals

$x_{l+1} = x_l + f_l(x_l)$. Simple, well-understood, but two failure modes at extreme depth: (a) gradient explosion as residual magnitudes accumulate, (b) representation collapse as the residual stream overwhelms layer contributions.

### V4: Manifold-Constrained Hyper-Connections

$x_{l+1} = M_l [x_l; f_l(x_l)]$ where $M_l$ is constrained to be doubly stochastic via Sinkhorn-Knopp projection onto the Birkhoff Polytope.

Key properties of doubly stochastic mixing:
- **L1-norm preserving:** signal magnitude is constant across depth
- **Identity as special case:** can recover standard residuals if optimal
- **Learnable mixing:** each layer chooses how much of the residual vs layer output to propagate
- **4x wider residual stream** with only 6.7% training overhead

The Birkhoff constraint is principled: it is the tightest convex set of mixing matrices that guarantees norm preservation. Any tighter constraint (e.g., orthogonal matrices) would be too restrictive; any looser constraint would permit explosion or collapse.

---

## 3. Memory: None to Engram Conditional Memory

### V3: No Dedicated Memory

V3 relies entirely on attention + MoE for all knowledge retrieval. Factual recall ("capital of France") and relational reasoning ("if X then Y because Z") both flow through the same attention and expert pathways.

### V4: Engram — O(1) Deterministic Lookup

Engram introduces a separate memory axis:
1. Multi-head hashing maps suffix N-grams (up to trigrams) into prime-sized embedding buckets
2. Depthwise convolution over N-gram context produces a gating scalar $g \in [0,1]$
3. Retrieved embedding is injected: $x' = x + g \cdot \text{Engram}(x)$

**Sparsity Allocation Law:** Under fixed sparse budget, optimal split is ~20-25% Engram, ~75-80% MoE. This is an empirically derived law from the Engram paper's ablations across multiple model sizes.

**Placement in V4:** Layers 2 (surface lexical patterns) and 15 (entity-level factual recall). Only two layers — Engram is surgical, not pervasive.

**Why it matters:** Attention is O(L^2) or O(Lk); Engram is O(1). Every fact that Engram handles is a fact that attention doesn't need to compute. At 1M tokens, this is a massive compute savings.

---

## 4. Expert Precision: FP8 to FP4

### V3: FP8 Expert GEMMs

V3 pioneered FP8 training with fine-grained tile-wise (1x128) and block-wise (128x128) quantization, achieving <0.25% relative loss error vs BF16.

### V4: FP4 Expert Parameters

V4 drops MoE expert precision to FP4 (4-bit floating point), while keeping other parameters in FP8. This halves expert memory and compute cost *again* — a 4x reduction from the BF16 baseline that V3 already halved.

The interaction with the MoE architecture compounds: V4 has 384 experts (vs 256) but activates only 6 (vs 8). More experts in FP4 means: more parameter capacity at lower storage cost, more routing diversity, lower per-token compute.

---

## 5. Optimizer and Scale

| Dimension | V3 | V4-Pro |
|-----------|-----|--------|
| Optimizer | AdamW | Muon |
| Pre-training tokens | 14.8T | >32T |
| Context (pre-training) | 4K → 128K | 32K → 1M |
| Post-training | SFT + RL (single pass) | Two-stage: domain experts then distillation |
| Model variants | Single | Pro (1.6T) + Flash (284B) |
| Hardware | NVIDIA H800 | H800 + Huawei Ascend 910B/950PR |

The Muon optimizer is a less-documented but significant change. V3's training stability was already exceptional (zero loss spikes across 14.8T tokens); Muon reportedly improves convergence speed, which matters when the training corpus more than doubles.

---

## Compound Effects

The five innovations are not independent — they interact multiplicatively:

- **CSA + Engram:** Engram handles factual recall, so CSA heads can focus on relational reasoning with smaller k budgets
- **FP4 + 384 experts:** More experts at lower precision = more routing diversity at lower cost
- **mHC + deeper/wider models:** Birkhoff-constrained mixing enables the wider residual stream that makes 1.6T parameters trainable
- **Muon + >32T tokens:** Faster convergence makes the larger training corpus practical

V3 proved that *integrating* five innovations (MLA, MoE, auxiliary-loss-free balancing, MTP, FP8) produces compound savings beyond any single technique. V4 extends this philosophy: each new component enables the others, and the whole is greater than the sum of its parts.

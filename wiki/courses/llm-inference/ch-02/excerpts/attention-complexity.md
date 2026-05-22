---
chapter: ch-02
course: llm-inference
phase: read
excerpt_of: "Attention Is All You Need (Vaswani et al. 2017) + Efficient Transformers: A Survey (Tay et al. 2020)"
source_url: https://arxiv.org/abs/1706.03762
created_at: "2026-05-21"
---

# Excerpt: Attention complexity — prefill `O(L²·d)`, decode `O(L·d)` per step

**Authors:** Ashish Vaswani et al. (Transformer); Yi Tay et al. (Efficient Transformers survey)
**Year:** 2017 / 2020
**URLs:** https://arxiv.org/abs/1706.03762 ; https://arxiv.org/abs/2009.06732
**Raw-data sources:** [[raw-data/attention-is-all-you-need]], [[raw-data/attention-complexity]]

---

## The two-phase complexity table

For a single layer with hidden size `d`, query heads `H_q`, KV heads `H_kv`, head dim `d_head = d / H_q`, sequence length `L`, batch `B`:

### Prefill (all `L` tokens at once)

```math
\begin{aligned}
\text{QKV projections} &: 3 B L d^2 \\
QK^\top &: B H_q L^2 d_{head} = B L^2 d \\
\text{softmax} &: B H_q L^2 \\
\text{attn} \cdot V &: B H_q L^2 d_{head} = B L^2 d \\
\text{output proj} &: B L d^2 \\
\text{FFN (gated, 8/3 factor)} &: \tfrac{16}{3} B L d^2 \\
\hline
\text{Total} &: \sim 7 B L d^2 + 2 B L^2 d
\end{aligned}
```

The `B L^2 d` terms come from the two `L × L` matmuls (`QK^\top` and `attn·V`). They dominate when `L > 3.5·d`. For `d = 4096`, attention overtakes FFN at `L ≈ 14336`.

### Decode (one new query token, `L` cached keys)

```math
\begin{aligned}
\text{QKV projections} &: 3 B \cdot 1 \cdot d^2 \\
QK^\top &: B H_q \cdot 1 \cdot L \cdot d_{head} = B L d \\
\text{softmax} &: B H_q L \\
\text{attn} \cdot V &: B L d \\
\text{output proj} &: B d^2 \\
\text{FFN} &: \tfrac{16}{3} B d^2 \\
\hline
\text{Total per step} &: \sim 7 B d^2 + 2 B L d
\end{aligned}
```

Same crossover: attention beats FFN at `L > 3.5·d`. But decode is bandwidth-bound at small batch, so the FLOPs accounting matters less than the *bytes moved* (next section).

---

## Memory traffic — the bandwidth-bound side of decode

Decode is rarely compute-bound at small batch. The bytes moved per step:

```
weight bytes/layer:      ~14 d² · bytes_per_elem  (Q,K,V,O,FFN_up,FFN_gate,FFN_down)
KV cache reads/layer:    2 L H_kv d_head · bytes_per_elem
                       = 2 L d_kv · bytes_per_elem
activation bytes:        << weight + KV bytes; usually ignorable
```

For Llama-3-70B (`d = 8192, n_layers = 80, d_kv = 1024`) at `B = 1`:

| Context `L` | Weights/layer (bf16) | KV reads/layer (bf16) | KV / Weight |
|---:|---:|---:|---:|
| 1k | 1.8 GB total weights | 4 MB | 0.002 |
| 8k | (same) | 32 MB | 0.018 |
| 32k | (same) | 128 MB | 0.073 |
| 128k | (same) | 512 MB | 0.29 |
| 1M | (same) | 4 GB | 2.3 |

At short context, decode bandwidth = weight loading; throughput improves by batching (weight load amortizes). At long context, KV reads scale per-sequence and *cannot* be amortized by batching — they grow with `B · L`.

---

## The roofline interpretation

Hardware (H100 SXM):
- BF16 compute: 1979 TFLOPS
- HBM bandwidth: 3.35 TB/s
- Roofline knee: `1979e12 / 3.35e12 ≈ 590 FLOPs/byte`

**Prefill arithmetic intensity** at `L = 4k, d = 4096`:
- FLOPs/layer: `7·4096·4096² + 2·4096²·4096 ≈ 4.7e11`
- Bytes/layer: `~7·4096²·2 (weights) + 2·4096·1024·2 (KV write) ≈ 2.5e8`
- Intensity: `4.7e11 / 2.5e8 ≈ 1900 FLOPs/byte` — **compute-bound** (above 590).

**Decode arithmetic intensity** at `L = 4k, d = 4096, B = 1`:
- FLOPs/layer: `7·4096² + 2·4096·4096 ≈ 1.5e8`
- Bytes/layer: `~14·4096²·2 + 2·4096·1024·2 ≈ 4.9e8`
- Intensity: `1.5e8 / 4.9e8 ≈ 0.31 FLOPs/byte` — **memory-bound** (way below 590).

**Two orders of magnitude apart.** Prefill operates near peak compute; decode operates near peak bandwidth. This is the entire reason for prefill-decode disaggregation (ch-09) and chunked prefill (ch-05).

---

## Causal mask and incremental decoding

The mask `M[i, j] = -∞` for `j > i` makes attention strictly causal:

```math
\mathrm{Attention}_{\text{causal}}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top + M}{\sqrt{d_k}}\right) V
```

During decode, the new query `q_{t+1}` is a single vector, and we attend over keys `k_1, ..., k_{t+1}` (no future tokens exist). The mask is trivially satisfied: just don't load future keys. Implementations skip the mask entirely in the decode kernel.

Without KV caching, decoding `N` tokens would cost `O(N²·d²)` (recompute history every step). With caching, it's `O(N·d²)` projections + `O(N²·d_kv)` attention reads. For `N=1024, d=4096, d_kv=1024`, cached is ~16× faster, growing linearly with `N`.

---

## Common pitfalls

- **Confusing `H_q` with `H_kv` in formulas.** `QKᵀ` has `H_q` parallel computations (one per query head), but `H_kv` distinct KV projections. Bandwidth scales with `H_kv`, compute with `H_q`.
- **Forgetting the `B·L²·d` term at long context.** The `7·B·L·d²` term dominates at short context, but `2·B·L²·d` takes over above `L ≈ 3.5·d`. For frontier 128k-context serving this is the binding cost.
- **Assuming decode is compute-bound.** It rarely is at `B < 64`. Profile your decode arithmetic intensity before optimizing kernels; you're usually bandwidth-bound and need MQA/GQA + PagedAttention more than you need a faster matmul.

---

## Connections

- [[excerpts/mha-mqa-gqa]] — `H_kv` is the only knob that affects the KV-bandwidth term.
- [[excerpts/rope]] — applied to `Q, K` per token; doesn't change the complexity, just the constants.
- [[raw-data/prefill-vs-decode]] — SARATHI's formalization of the two-phase asymmetry.
- [[ch-03]] — the KV bytes formula derived here becomes the chapter-level focus.
- [[ch-11]] — FlashAttention removes the `O(L²)` memory term by never materializing the attention matrix.

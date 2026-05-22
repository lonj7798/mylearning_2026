---
chapter: ch-02
course: llm-inference
phase: read
excerpt_of: "Fast Transformer Decoding (Shazeer 2019, MQA) + GQA: Generalized Multi-Query Transformer (Ainslie et al. 2023)"
source_url: https://arxiv.org/abs/1911.02150
created_at: "2026-05-21"
---

# Excerpt: MHA → MQA → GQA — one parameter, big consequences

**Authors:** Noam Shazeer (MQA, 2019); Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, Sumit Sanghai (GQA, 2023)
**Year:** 2019 / 2023
**URLs:** https://arxiv.org/abs/1911.02150 ; https://arxiv.org/abs/2305.13245
**Raw-data sources:** [[raw-data/multi-query-attention]], [[raw-data/grouped-query-attention]]

---

## The three configurations

Each attention layer projects an input `x ∈ ℝ^d` into `H_q` query heads and `H_kv` key/value heads. The three variants differ only in `H_kv`:

| Variant | `H_q` | `H_kv` | KV cache scaling | Quality vs MHA |
|---|---:|---:|---|---|
| **MHA** (Vaswani 2017) | `H` | `H` | `H` | baseline |
| **MQA** (Shazeer 2019) | `H` | `1` | `1/H` | drops ~1–2 ppl |
| **GQA-G** (Ainslie 2023) | `H` | `G` | `G/H` | matches MHA within noise (`G ≥ 8`) |

Llama-3-70B uses GQA-8: `H_q = 64, H_kv = 8`. Each KV head serves `64/8 = 8` query heads.

---

## The MQA bandwidth argument (Shazeer 2019)

Shazeer's [[raw-data/multi-query-attention]] one-page paper made the simplest possible argument: **decode is memory-bandwidth-bound, and the dominant memory traffic is reading the KV cache**. Reducing `H_kv` from `H` to `1` shrinks per-step KV reads by `H×` (typically 32× or 64×).

The forward-pass arithmetic is unchanged — you still have `H` query heads computing attention. The only change is that all `H` queries share one `K, V` projection:

```python
# MHA
K = x @ W_K.reshape(H, d_head)              # [B, L, H, d_head]
V = x @ W_V.reshape(H, d_head)              # [B, L, H, d_head]

# MQA
K = x @ W_K_shared.reshape(1, d_head)       # [B, L, 1, d_head]
V = x @ W_V_shared.reshape(1, d_head)       # [B, L, 1, d_head]
# broadcast K, V across H query heads in attention
```

At inference, MQA's KV cache occupies `1/H` of MHA's, allowing proportionally larger batches and proportionally less decode bandwidth.

---

## Why MQA hurts quality

A single shared `K, V` projection forces every query head to agree on what to "look for" and "retrieve". MHA's `H` distinct KV projections let different heads specialize on different aspects (syntactic, semantic, position-tracking). MQA collapses all of this into one channel.

Reported quality drops (PaLM 540B authors, post-MQA):
- HumanEval: −2.2 points
- GSM8K: −1.5 points
- MMLU: −0.6 points

Acceptable at the bandwidth savings — but not for safety-critical or reasoning-heavy applications.

---

## The GQA compromise (Ainslie 2023)

GQA introduces `G` intermediate KV heads. Each of the `H_q` query heads is grouped: query head `h` shares KV with `(h mod G)` other query heads, all reading from KV head `h // (H_q / G)`:

```python
# GQA-G example with H_q = 32, G = 8 → 4 query heads per group
K = x @ W_K.reshape(G, d_head)              # [B, L, G, d_head]
V = x @ W_V.reshape(G, d_head)              # [B, L, G, d_head]
# query head h reads from KV head h // (32/8) = h // 4
```

KV cache scales by `G/H`. For `H = 32, G = 8`, that's 4× shrinkage (vs 32× for MQA), recovering most of MHA's quality.

**The empirical sweet spot is `G = 8`**. Ainslie's ablation showed quality saturates around there for 13B–65B models. Frontier 2026 models almost universally use `G ∈ {4, 8, 16}`.

---

## Uptraining: converting MHA → GQA cheaply

Ainslie's secondary contribution: you don't need to train GQA from scratch. Take an existing MHA checkpoint, mean-pool its KV heads to match the target `G`, then **uptrain** with ~5% of original pretraining compute. The quality recovers to within ~0.1 perplexity of training from scratch.

```python
# Mean-pool MHA → GQA
for layer in model.layers:
    K_mha = layer.attention.W_K           # [d, H_q · d_head]
    K_grouped = K_mha.reshape(d, H_q, d_head).reshape(d, G, H_q//G, d_head)
    K_gqa = K_grouped.mean(dim=2).reshape(d, G * d_head)
    # same for V
```

This is how Llama 2 70B got its GQA-8 from a 70B MHA pretrain.

---

## Concrete impact: per-token KV bytes

For Llama-3-70B at bf16:
- `n_layers = 80`, `d_head = 128`
- MHA equivalent (`H_kv = 64`): `2 · 80 · 64 · 128 · 2 = 2.6 MB / token`
- GQA-8 actual (`H_kv = 8`): `2 · 80 · 8 · 128 · 2 = 320 KB / token`
- 8× reduction at the per-token KV footprint.

At 32k context: MHA = 83 GB / request; GQA = 10 GB / request. The MHA version literally would not fit one request on a single 80 GB H100. The GQA version fits ~8 concurrent 32k-context requests.

---

## Adoption table (late 2026)

| Model family | Attention variant | Notes |
|---|---|---|
| Llama 1 (7B/13B/65B) | MHA | pre-GQA |
| Llama 2 (7B/13B) | MHA | |
| Llama 2 70B | GQA-8 | first major GQA release |
| Llama 3 (8B/70B/405B) | GQA-8 | |
| Mistral 7B | GQA-8 | |
| Mixtral 8x7B / 8x22B | GQA-8 | MoE + GQA |
| Qwen 2/3 (7B/14B/32B/72B) | GQA | |
| Falcon 40B/180B | GQA-8 | |
| PaLM 540B | MQA | extreme bandwidth focus |
| DeepSeek V3 / R1 | MLA | learned low-rank latent — beyond GQA |

---

## Common pitfalls

- **Reading `n_heads` instead of `num_key_value_heads`** when sizing KV cache. The KV formula uses `n_kv_heads`, not `n_heads`. Off by 8× if you confuse them.
- **Assuming MQA is "always faster"**. It is at long context with small batch. At short context with large batch, weight loading dominates and the MQA bandwidth advantage is invisible.
- **Trying to mix MHA weights with a GQA kernel** (or vice versa). The KV head counts must match. PagedAttention kernels (ch-06) are parametrized on `H_kv`.

---

## Connections

- [[excerpts/attention-complexity]] — `H_kv` enters only the KV bandwidth term; query-side compute scales with `H_q` regardless.
- [[raw-data/kv-cache-memory-formula]] — `2 · L · H_kv · d_head · bytes` is the per-token cost.
- [[ch-03]] — fleet-scale implications of `H_kv` choice.
- [[ch-20]] — DeepSeek V3's MLA pushes this further with a learned low-rank latent KV.

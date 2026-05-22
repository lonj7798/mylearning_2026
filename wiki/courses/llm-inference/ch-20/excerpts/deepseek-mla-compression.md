---
chapter: ch-20
course: llm-inference
phase: read
excerpt_of: "DeepSeek-V3 Technical Report (2412.19437) — inference-side MLA detail"
source_url: https://arxiv.org/abs/2412.19437
created_at: "2026-05-21"
---

# Excerpt: DeepSeek MLA — what it actually compresses

**Authors:** DeepSeek-AI
**Year:** 2024 (V3 report, Dec 2024) / 2025 (R1, Jan 2025)
**URL:** https://arxiv.org/abs/2412.19437
**Raw-data source:** [[raw-data/deepseek-v3-inference]]

---

## The KV-cache cost story in one line

MLA (Multi-head Latent Attention) lets a 671B-parameter model cache ~70 KB/token. A naive MHA version of the same model would cache ~1.5 MB/token. That is the entire reason DeepSeek V3 is serveable.

---

## Standard MHA / GQA cache layout (the baseline)

In standard MHA / GQA, every decode step writes a per-head K and V vector to the cache:

```
K_cache[layer, token, head] : (head_dim,)
V_cache[layer, token, head] : (head_dim,)

Total bytes/token = 2 · layers · kv_heads · head_dim · dtype_bytes
```

For GQA-8 at 80 layers, head_dim 128, BF16: `2 · 80 · 8 · 128 · 2 = 327 680` bytes/token (this is the Llama-3-70B number). Scale to 61 layers and a hypothetical 128 KV heads (the dense equivalent at V3's hidden size): `2 · 61 · 128 · 128 · 2 = 4 MB/token`. That obviously does not fit a 128 k context.

---

## What MLA stores instead

MLA replaces the per-head `(K, V)` store with a single compressed latent vector per token per layer:

```
c_KV[layer, token] : (d_c,)      # latent KV vector, d_c ≪ kv_heads · head_dim
k_rope[layer, token] : (d_rope,) # small RoPE-coupled slice, kept un-compressed
```

For DeepSeek V3: `d_c = 512`, `d_rope = 64` (this is per-token, NOT per-head). So:

```
Bytes/token (BF16) = 2 · 61 · (512 + 64) · 2
                   = 2 · 61 · 576 · 2
                   ≈ 140 KB/token  (BF16)

Bytes/token (FP8)  = 2 · 61 · 576 · 1
                   ≈  70 KB/token  (FP8 weights, native)
```

70 KB/token at 128 k context = 9 GB per max-context request. That fits a single H100 with a 670 GB-weight model split via expert + tensor parallelism across the rest of the cluster.

The compression ratio vs naive MHA: `(128 heads · 128 head_dim) / 576 ≈ 28×`. At FP8 instead of BF16 you get another 2× → **~56× less KV memory** than naive MHA would require.

---

## How K and V are reconstructed at attention time

The trick is that the K and V tensors used inside the attention QK^T and softmax(QK^T)V computation are **reconstructed on the fly** from `c_KV`, with the reconstruction projections *absorbed* into adjacent matrices.

Schematically, with `W_DKV` the down-projection (input → c_KV), `W_UK` and `W_UV` the up-projections back to full K and V dims, and `W_Q`, `W_O` the query and output projections:

```
c_KV = W_DKV · h       # write to cache, ONCE per token, dim d_c
K    = W_UK  · c_KV    # reconstructed on read, fused into W_Q effectively
V    = W_UV  · c_KV    # reconstructed on read, fused into W_O effectively
```

Because `W_UK` and `W_UV` are fixed matrices (model weights, not per-request), the products `(W_Q · W_UK^T)` and `(W_UV · W_O)` can be precomputed offline and stored as the "absorbed" Q-projection and O-projection. At inference time the attention kernel reads `c_KV` (small), multiplies by absorbed Q to get a query in latent space, and the cache traffic per decode step is dominated by reading the latent — not full K/V.

This is what makes MLA an *inference* win: it reduces the bytes-per-token traffic out of HBM, which is the binding constraint at decode time per [[ch-03]].

---

## RoPE coupling — why `k_rope` stays separate

RoPE rotates K (and Q) by a position-dependent matrix. If the rotation were applied *before* the latent compression, the latent would mix position information across all dims and the compression would be position-specific (bad: defeats reuse). DeepSeek's solution: split each head's K into two parts —

- a "content" part that goes through the latent (no RoPE applied);
- a small "position" part of width `d_rope = 64` that gets RoPE'd and stored verbatim.

At attention time, the inner product `Q^T K` is the sum of (latent-reconstructed content) + (RoPE'd position part). This is why the cache cost is `(d_c + d_rope)`, not just `d_c`.

---

## FP8 native deployment

V3 was trained directly in FP8 (E4M3 for forward, E5M2 for backward) with per-block scaling: 1×128 activation tiles, 128×128 weight blocks. Each block carries one FP32 scale.

Storage overhead from scales:

```
Weight memory at FP8        = 671e9 × 1 byte    = 671 GB
Scales overhead             ≈ 1 % of above      ≈ 7 GB
Reference BF16 equivalent   = 671e9 × 2 bytes   = 1342 GB
```

The blocked-scale layout requires custom GEMM kernels that consume the block scales (DeepSeek released DeepGEMM for this; both vLLM and SGLang now ship integrations). Loading V3 at BF16 *can* be done by upcasting on load, but you should not: the model never saw the BF16 distribution during training, so naive upcast loses 0.2–0.5 points on most evals while doubling memory.

---

## Why MLA + MoE is the right pair

MoE only activates a subset of FFN experts per token — but attention is still shared across the whole model. As you scale MoE total params from 100B → 671B → 1T+, attention KV cost becomes the binding constraint (the FFN cost barely grew per-token). MLA explicitly attacks that cost.

This is why DeepSeek V3, the upcoming Qwen-3-Next, and several rumoured 2026 frontier MoEs converge on "MLA + MoE" as the architecture: MoE for cheap compute, MLA for cheap memory, both targeting frontier-scale serving.

---

## Connections

- [[excerpts/moe-param-accounting]] — V3 671B / 37B active follows the same accounting rule as Mixtral and Qwen-3-MoE.
- [[ch-13]] — expert parallelism for the 256 routed experts in V3; DeepEP all-to-all kernel.
- [[ch-22]] — the capstone does not include MLA, but the same "implement-from-paper" discipline applies if you ever try to reproduce V3's MLA on a smaller backbone.

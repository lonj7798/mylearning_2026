---
chapter: ch-02
course: llm-inference
phase: read
excerpt_of: "RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al. 2021)"
source_url: https://arxiv.org/abs/2104.09864
created_at: "2026-05-21"
---

# Excerpt: RoPE — rotation-based positional encoding for `Q, K`

**Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, Yunfeng Liu
**Year:** 2021
**Venue:** Neurocomputing 2024 (originally arXiv 2021)
**URL:** https://arxiv.org/abs/2104.09864
**Raw-data source:** [[raw-data/rope]]

---

## The rotation formula

RoPE pairs adjacent dimensions of `Q` and `K` and rotates each pair by a position-dependent angle. For a 2-D pair `(x_{2i}, x_{2i+1})` at position `m` with frequency `θ_i = 10000^{-2i/d}`:

```math
\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix}
=
\begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix}
\begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}
```

Equivalently in complex-number form: pair `z = x_{2i} + i \cdot x_{2i+1}` is multiplied by `e^{i \cdot m \theta_i}`.

The base `10000^{-2i/d}` is inherited from the original sinusoidal positional encoding ([[raw-data/attention-is-all-you-need]] §3.5). Low-index dims rotate fast (high frequency), high-index dims rotate slowly. The full `d`-dim vector spans many octaves of frequency.

---

## Why this gives relative position

For two positions `m, n`, the rotated query and key inner product is:

```math
\langle \mathrm{RoPE}(q, m), \mathrm{RoPE}(k, n) \rangle = \langle q, R_\Theta(n - m) \cdot k \rangle
```

The result depends only on `n - m` — relative position behavior without explicit relative tables. The proof is one line: `R(α)^\top R(β) = R(β - α)`.

This matters at inference because:

1. **Translation invariance**: `score(q_5, k_3) = score(q_15, k_13)`. The model behaves consistently as you shift the entire context window.
2. **Cache reusability**: rotated `K_j` stored at training time remains valid at inference, even if the relative positions of new queries change.
3. **Extrapolation hook**: shifting `θ` (NTK-aware scaling, YaRN) modifies relative-position behavior at long context.

---

## How RoPE interacts with the KV cache

**Critical implementation detail.** The cached `K` is the *post-rotation* `K'`:

```python
def attention_with_rope(q, k_new, v_new, kv_cache, position):
    # Apply RoPE to NEW query and new key, using current position
    q_rot = apply_rope(q, position)
    k_rot = apply_rope(k_new, position)
    # Append rotated k_new and unmodified v_new to cache
    kv_cache.k = torch.cat([kv_cache.k, k_rot], dim=seq_dim)
    kv_cache.v = torch.cat([kv_cache.v, v_new], dim=seq_dim)
    # Attention against ALL cached k (each rotated at its own position)
    scores = q_rot @ kv_cache.k.transpose(-2, -1) / sqrt(d_head)
    return softmax(scores) @ kv_cache.v
```

`V` is **never** rotated — only `Q` and `K`. This is the most common implementation bug for people writing RoPE from scratch.

---

## NTK-aware scaling and YaRN for context extension

A model trained at `L_train = 4096` typically fails when run at `L_target = 32768`. The high-frequency RoPE bands hit angles never seen during training, breaking attention behavior. Two practical fixes:

### NTK-aware scaling (Peng & Quesnelle, 2023)

Replace `θ_i` with `θ_i · (L_train / L_target)^{i/d}`:

- High-frequency bands (small `i`): unchanged.
- Low-frequency bands (large `i`): scaled down.

The result: long-range positional information stays "in-distribution" — the model never encounters never-trained angles. Works zero-shot; quality usually within 1–2 ppl of fine-tuned at 4× extension.

### YaRN (Peng, Quesnelle et al. 2023)

"Yet another RoPE extensioN" refines NTK-aware with three additions:
1. **Per-band ramp**: smooth transition between modified and unmodified bands.
2. **Attention temperature**: rescale `1/√d` term to compensate for changed entropy at long context.
3. **Brief fine-tuning**: 100–1000 steps at extended context to lock in quality.

YaRN can extend `8×` to `32×` context (e.g. Mistral 7B trained at 8k → 128k). Llama 3 used a YaRN-flavored recipe for 128k extension.

**Both are pure inference-time changes** to how `θ_i` is computed before rotation. The model weights are unchanged; the KV cache shape is unchanged.

---

## Compute cost

RoPE rotation is `O(d)` per token — two multiplies + one add per pair, `d/2` pairs per head, `H_q` heads per layer. Negligible compared to attention or FFN. Modern kernels fuse RoPE into the QKV projection.

---

## Adoption (late 2026)

Essentially universal among open decoder-only LLMs:

| Model | Position method |
|---|---|
| Llama 1/2/3 | RoPE |
| Qwen 1/2/3 | RoPE |
| Mistral / Mixtral | RoPE |
| Falcon | RoPE |
| DeepSeek V3 / R1 | RoPE (decoupled into MLA) |
| GPT-NeoX / Pythia | RoPE |
| Phi-3/4 | RoPE |
| Gemma | RoPE |
| BLOOM / MPT / Falcon-1B | ALiBi (legacy) |

---

## Common pitfalls

- **Rotating `V`**. Don't. `V` carries content, not position; rotating it scrambles values.
- **Forgetting position offset under cache resume**. If you serialize KV cache at position `t` and resume, new tokens must use position `t+1`, not `0`. Llama.cpp and vLLM track `seq_len` explicitly.
- **Mixing RoPE bases**. Llama-3's RoPE base is `500000` (not `10000`) for 128k context. Loading Llama-3 weights into a code path that hard-codes `10000` produces garbage at long context.
- **Applying RoPE after attention**. RoPE goes on `Q, K` *before* `softmax(QKᵀ/√d)·V`. Backwards order is silently wrong.

---

## Connections

- [[excerpts/attention-complexity]] — RoPE adds `O(d)` per token, negligible.
- [[excerpts/mha-mqa-gqa]] — `K` is rotated regardless of `H_kv` count.
- [[raw-data/alibi]] — the alternative; recency bias instead of rotation.
- [[ch-20]] — Llama 3 / Qwen 3 / DeepSeek V3 all use RoPE-scaled long context.

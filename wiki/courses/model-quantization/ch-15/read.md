<!-- chapter: ch-15
     track: 2024-maturation
     title: KV-Cache Quantization — KIVI / KVQuant / GEAR
     sources: [[kivi]], [[kvquant]], [[gear]], [[wkvquant]], [[skvq]], [[qaq]], [[coupling-kv-quant]], [[per-channel-vs-per-token-kv]]
     figures: figures/kv-asymmetry.html
-->

# Chapter 15 — KV-Cache Quantization: KIVI / KVQuant / GEAR

> **Core insight.** At long context (8K+), the KV cache — not the weights — dominates inference memory and HBM bandwidth. Naively quantizing K and V the same way fails: **K and V have different outlier geometries**. K inherits the residual-stream channel outliers (via W_k) and they're *persistent across tokens* → K must be quantized **per-channel**. V is structurally similar but attention reweighting averages out channel patterns → V should be quantized **per-token**. This K/V axis-asymmetry is the load-bearing finding of [[kivi]]; everything in this chapter (KVQuant, GEAR, WKVQuant, SKVQ, QAQ) builds on it.
>
> **Guideline.** Default recipe for 2-bit KV cache: per-channel K with group 32 along token axis, per-token V with group 32 along channel axis. Quantize K *pre-RoPE* (KVQuant) so the channel structure survives rotation. For sub-2-bit, add dense-and-sparse outlier isolation (KVQuant) or low-rank residual (GEAR). For million-token contexts, add a sliding window (SKVQ) keeping recent tokens at higher precision.

---

## Why this chapter exists

By ch-14 you can compress weights to 4 bits, activations to 4 bits, and the KV cache to 4 bits. The W4A4 result is the headline. But during long-context decode, the KV cache is *the* memory and bandwidth bottleneck:

**LLaMA-2-70B, 32K context, batch 8:**
- Weight memory (FP16): 140 GB → with W4: 35 GB.
- Activation memory: small at decode time.
- **KV cache (FP16)**: 32 layers × 8 heads × 128 dim × 2 (K, V) × 32K tokens × batch 8 × 2 bytes = **335 GB**.

The KV cache dwarfs the weights. Cut it from FP16 to INT2 (8×) and you fit a 70B model at 32K context on one H100. Cut V to 1.5 bits ([[skvq]]) and you fit million-token contexts on a single A100-80GB.

Three things you must walk away with:

1. **The K vs V asymmetry**: why per-channel K and per-token V — the empirical finding *and* the causal story (residual-stream outliers inherited by W_k, attention reweighting averaging away V channel structure).
2. **Why pre-RoPE K quant ([[kvquant]])**: RoPE rotates channels, entangling the per-channel structure; quantize before RoPE.
3. **The four-way design space**: per-axis (KIVI), non-uniform + dense-and-sparse (KVQuant), low-rank residual (GEAR), sliding window (SKVQ), joint W+KV (WKVQuant), adaptive bit allocation (QAQ). Know which to reach for under which constraint.

A companion HTML visualization shows the K vs V distributions side by side — see [`figures/kv-asymmetry.html`](figures/kv-asymmetry.html).

---

## 1. The bandwidth bottleneck

Modern decoding is **memory-bound**. Each new token reads:

- The full weight matrix W (mostly compute-bound).
- The full KV cache for all past tokens (purely bandwidth-bound — no compute reuse).

For batch 1, long context, on H100:
- HBM bandwidth: 3.35 TB/s.
- KV cache read per token = 2 × n_layers × n_heads × d_head × T × 2 bytes (FP16) = `4 · L · H · D · T` bytes.
- LLaMA-2-70B at 32K context: ~335 GB / step / batch.
- Even at full HBM bandwidth, this is 100ms / token / batch — *just for the KV read*.

At 4 bits per KV element, the read shrinks 4× → 25ms / token. At 2 bits → 12.5ms / token. KV-cache quantization is the **single largest lever** for long-context throughput.

The math:

```math
\text{KV memory} = 2 \cdot L \cdot H \cdot D \cdot T \cdot \text{bytes/elem}
```

For LLaMA-2-70B (L=80, H=8 GQA, D=128), per-token KV at FP16: `2 · 80 · 8 · 128 · 2 = 327 KB / token`. At 1M tokens = 327 GB. At 2-bit KV: 41 KB/token, 1M tokens = 41 GB → fits one A100.

---

## 2. K vs V — the asymmetry

This is the central empirical finding of [[kivi]] (Liu et al. ICML 2024). For any trained LLM:

- **K cache** has **channel-wise** outliers — a few specific channels are 10–100× larger than the rest, and *the same channels every token*.
- **V cache** has **token-wise** variation — no consistent channel-outlier pattern, but specific tokens dominate.

### 2.1 Why K is channel-wise

Recall `K_t = W_k · x_t`. The input `x_t` is the residual stream, which carries the LLM.int8() "outlier features" — a small number of consistent channels of magnitude 10–100× the bulk. `W_k` is a learned projection; empirically `W_k` mostly preserves this channel structure (its columns project the residual stream into per-head subspaces).

**Result**: across all tokens t, the *same* channels of K are large. By construction.

### 2.2 Why V is token-wise

`V_t = W_v · x_t` structurally inherits the same outliers. But V is *used* at attention time as a **weighted average**: `out_q = Σ_t a_{q,t} V_t`. The attention weights `a_{q,t}` re-mix the channel pattern of V; over the softmax average, channel-wise outliers wash out.

**Result**: what dominates V's contribution is **which token** is being attended to, not which channel. Token-wise variation dominates.

### 2.3 What goes wrong if you quantize along the wrong axis

If K is quantized **per-token** (one scale per token across all channels): the scale is dominated by the outlier channels at every token → crushes precision on the bulk channels. INT2 → garbage.

If V is quantized **per-channel**: the per-channel scale wastes precision on channels that attention will downplay anyway. Some loss but not catastrophic.

LLaMA-2-7B at INT2 KV (KIVI ablation, Wikitext-2):

| K axis | V axis | ppl |
|--------|--------|-----|
| per-channel | per-token | **7.0** |
| per-token | per-token | 12.4 (5+ ppl hit from wrong K axis) |
| per-channel | per-channel | 8.3 (1+ ppl hit from wrong V axis) |
| per-token | per-channel | 14+ (both wrong) |

The right K axis is worth 5+ ppl. The right V axis is worth 1+ ppl. **K axis is the bigger lever**; this is the one to get right.

---

## 3. KIVI — the canonical recipe

[[kivi]] formalises the asymmetric rule and provides the streaming implementation.

### 3.1 The quantization rule

For a KV tensor of shape `(n_tokens, n_heads, head_dim)`:

- **K cache**: quantize along the *token* axis but **group channels**. Each channel `c` has its own scale `s_K[c]` computed from the past tokens. INT2 representation per `(token, channel)`. Group size 32 along the token axis means stats refresh every 32 tokens.
- **V cache**: quantize along the *channel* axis but **group tokens**. Each token `t` has its own scale `s_V[t]`. INT2 per `(token, channel)`. Group size 32 along channel axis.

```
K[t, h, c] → INT2 with scale s_K[h, c, group(t)]
V[t, h, c] → INT2 with scale s_V[h, t, group(c)]
```

### 3.2 Streaming implementation

Decode-time KV cache grows by one token per step. KIVI maintains:

- A small **FP16 residual buffer** of the last `< g = 32` tokens.
- A **quantized INT2 packed past** for older tokens.

When the residual buffer fills to 32, the chunk is quantized (per-channel for K, per-token for V) using its own statistics and appended to the packed past.

```python
class KIVI_KV_Cache:
    def __init__(self, group_size=32):
        self.g = group_size
        self.k_quant = None       # INT2 packed
        self.v_quant = None
        self.k_scales = None      # FP16, per-channel
        self.v_scales = None
        self.k_fp_residual = []   # last <g tokens FP16
        self.v_fp_residual = []

    def append(self, k_t, v_t):
        self.k_fp_residual.append(k_t)
        self.v_fp_residual.append(v_t)
        if len(self.k_fp_residual) == self.g:
            self._flush_to_quantized()

    def _flush_to_quantized(self):
        chunk_k = torch.stack(self.k_fp_residual)         # (g, H, D)
        chunk_v = torch.stack(self.v_fp_residual)         # (g, H, D)
        # K: per-channel scale across the g tokens in this chunk
        s_K = chunk_k.abs().amax(dim=0) / 1.0             # (H, D)
        k_q = quantize_int2_symmetric(chunk_k, s_K)
        # V: per-token scale across channels
        s_V = chunk_v.abs().amax(dim=-1) / 1.0            # (g, H)
        v_q = quantize_int2_symmetric(chunk_v, s_V)
        append_to_packed(self.k_quant, k_q, self.k_scales, s_K)
        append_to_packed(self.v_quant, v_q, self.v_scales, s_V)
        self.k_fp_residual.clear()
        self.v_fp_residual.clear()
```

### 3.3 Attention math

```
attn_logits = Q · (dequant(K_int2, s_K))^⊤ / √d
attn_probs  = softmax(attn_logits)
out         = attn_probs · dequant(V_int2, s_V)
```

K dequant: one FP16 multiply per `(channel, group)` per chunk. V dequant: one FP16 multiply per `(token, group)`. Both are folded into the fused-attention kernel — no materialisation.

### 3.4 Numbers

LLaMA-2-7B at INT2 KV: WikiText-2 PPL increases by **<0.2** vs FP16. LongBench tasks within 1 point of FP16. Memory: **2.6× reduction**, throughput **2.35–3.47×** depending on batch / context.

### 3.5 Bit budget accounting

INT2 KV + FP16 scales:

```
2 bits + 16 / 32 = 2.5 bits/element
```

Two bits per value plus the amortised FP16 scale per 32-element group. INT4 + group32 → 4.5 bits/element.

---

## 4. KVQuant — sub-4-bit via pre-RoPE quant + non-uniform + dense-and-sparse

[[kvquant]] (Hooper et al. NeurIPS 2024) pushes KV cache to **3-bit and below**, with four jointly-applied techniques:

### 4.1 Pre-RoPE K quantization

Standard RoPE applies a token-dependent rotation to pairs of K channels: `K_t = RoPE_t(W_k · x_t)`. The rotation **entangles** pairs of channels into a mixture whose distribution drifts with token position.

**KVQuant's observation**: quantize *un-rotated* K, store INT2/3, apply RoPE on the dequant at attention time.

```python
def kvquant_kv_step(W_k, x_t, kv_cache, t):
    # Compute and store pre-RoPE K
    k_pre_rope = W_k @ x_t
    kv_cache.store_int_pre_rope(k_pre_rope)

def kvquant_attention(Q, kv_cache, t_query):
    # Dequant all past K, then apply RoPE per-token at use time
    K_past_fp = []
    for t_past in range(len(kv_cache)):
        k_pre_rope = kv_cache.dequant_k(t_past)
        k_post_rope = apply_rope(k_pre_rope, position=t_past)
        K_past_fp.append(k_post_rope)
    return softmax(Q @ stack(K_past_fp).T / sqrt(d))
```

**Why pre-RoPE matters**: post-RoPE K's per-channel outlier structure drifts (adjacent channels get rotated into each other by token-dependent angles). Pre-RoPE K has a stationary per-channel distribution → per-channel quant fits cleanly. Post-RoPE → mixed → per-channel quant *misses*.

KIVI (post-RoPE) at INT2: ppl 7.0. KVQuant (pre-RoPE) at INT2: ppl 6.0. The pre-RoPE trick is worth ~1 ppl at INT2.

### 4.2 Non-uniform per-channel codes

For each K channel `c`, compute calibration histogram and fit a **k-means codebook** of size `2^B` (B=2 or 3). Store the codebook (negligible amortised cost) and per-element index. Equivalent to SqueezeLLM's non-uniform weight quant ([[squeezellm]]) applied to KV.

V uses per-token uniform quant (consistent with KIVI's finding that V is token-wise).

### 4.3 Dense-and-sparse decomposition

Identify the top **1%** (by absolute value) of pre-RoPE K and per-token V elements per layer; store them in a **sparse FP16** vector (index + value). Dense path uses the non-uniform code at 2/3-bit. At attention time, dense path computes most of the dot product; sparse FP16 contributions are added as a correction.

Borrowed from [[spqr]] (which did this for weights).

```
K[t, c] = top_1_percent_sparse_FP16 ∪ dense_2bit_indexed
attn_logits = Q · dense_KK^⊤ + Q · sparse_KK^⊤
```

### 4.4 Q-norm quantization

The Query is *also* quantized so the QK^⊤ dot-product happens in low-bit arithmetic. Per-token symmetric INT4 or INT8.

### 4.5 The numbers

LLaMA-7B at 3-bit KV: <0.1 PPL degradation on Wikitext-2 and C4.

Context-length math: LLaMA-7B FP16 KV = 512 KB/token. 1M tokens = 512 GB. At 2-bit KVQuant: 64 KB/token → 64 GB for 1M tokens — fits in one A100-80GB. **Enables LLaMA-7B inference at 1M context on a single A100, and 10M context across 8 GPUs.**

---

## 5. GEAR — quant + low-rank residual + sparse outliers

[[gear]] (Kang et al. 2024) makes a different bet: the *residual* of uniform low-bit quantization has an **approximately low-rank structure** that captures most of the systematic error and can be cheaply represented with a rank-r SVD addendum.

### 5.1 The decomposition

For a per-head KV matrix `M ∈ ℝ^{T × d}` (T = #tokens, d = head_dim):

```math
M \approx Q + L + S
```

- **Q = dequant(quant_b(M))**: uniform b-bit quantization (b=4 typical).
- **L = A · B^⊤**: low-rank residual matrix, rank r=2–4. `A ∈ ℝ^{T × r}`, `B ∈ ℝ^{d × r}`.
- **S**: sparse correction holding the top-K largest residual entries that `L + Q` still misses, `K ≈ 1%` of T·d.

### 5.2 Computing L (truncated SVD)

```python
def gear_compute_L(M, Q, rank=2):
    R = M - Q
    U, S, V = torch.linalg.svd(R, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = V[:rank, :]
    A = U_r * S_r.sqrt()       # (T, r)
    B = V_r.T * S_r.sqrt()     # (d, r)
    return A, B
```

For streaming inference: update A, B incrementally as new tokens arrive — append to M, requantize the chunk to Q, compute residual delta, project onto current B and orthogonalise. Periodic full refresh (every ~256 tokens) prevents drift.

### 5.3 Sparse S

After Q + L, scan residual `M − Q − L`; pick top-K elements by absolute value (`K = ⌈0.01 T·d⌉`); store as `(token_idx, channel_idx, FP16 value)` tuples.

### 5.4 Bit budget

For b=4, r=2, K=1% T·d, T=2048, d=128:

```
Q:  4 bits/element
L:  2 · (T+d) · 16 bits / (T·d) = 32(T+d)/(Td) ≈ 32/T + 32/d ≈ 0.27 bits/element
S:  1% · 32 bits ≈ 0.32 bits/element
Total ≈ 4.6 bits/element
```

At b=2: ~2.6 bits/element with the same L, S overhead.

### 5.5 Attention math

```
attn_logits = Q · (Q_K + L_K + S_K)^⊤ / √d
            = Q · Q_K^⊤ + Q · L_K^⊤ + Q · S_K^⊤
```

Three matmuls and sum. Q-path uses INT4 GEMM (cheap). L-path is two small FP16 matmuls of rank `r × d` (very cheap). S-path is sparse SpMV (very cheap at 1% density).

### 5.6 Why all three components are non-redundant

GEAR ablations show:
- Q alone: ppl +1.5 vs FP16.
- Q + L: ppl +0.5.
- Q + S: ppl +0.8.
- **Q + L + S**: ppl +0.05.

The quantization residual has both **low-rank** structure (systematic error captured by L) and **sparse outliers** (a few entries L can't represent, captured by S). Removing either component loses ~0.3–1.0 ppl.

---

## 6. WKVQuant — joint W4 + KV4

[[wkvquant]] (Yue et al. 2024) makes a deployment-side observation: the **sweet spot** in LLM PTQ is *not* W4A4 (activations hurt accuracy) and *not* W4A16 (KV cache still dominates at long context) — it's **W4 + KV4** with activations at FP16.

### 6.1 Past-only KV quantization

At decode step t, attention uses K_{1..t}, V_{1..t}. WKVQuant stores K_{1..t-1}, V_{1..t-1} in INT4 but computes K_t, V_t in FP16 fresh; only writes the *previous* K_{t-1}, V_{t-1} to the INT4 cache (not the brand-new one).

```
attn = softmax([Q · K_{1..t-1}^{INT4} / √d, Q · K_t^{FP16} / √d]) · [V_{1..t-1}^{INT4}; V_t^{FP16}]
```

The newest contribution is **exact**; only the historical past is quantized. Cheap (no extra dequant for the current token) and helps a measurable amount on short prompts.

### 6.2 2D KV strategy

For each layer, calibrate variance of K along the channel axis vs the token axis; pick the lower-variance axis for grouping. Same for V. Generalises KIVI (which hard-codes K per-channel, V per-token) to a per-layer adaptive choice. ~0.1–0.2 ppl improvement.

### 6.3 Cross-block reconstruction loss

```math
\mathcal{L} = \sum_b \|h_b^{\mathrm{FP}} - h_b^{\mathrm{quant}}\|^2 + \lambda \sum_{b<b'} \|h_{b'}^{\mathrm{FP}} - h_{b'}^{\mathrm{quant}}\|^2
```

The second term jointly optimises later blocks given the propagated quant error of earlier blocks → addresses compounding error across the stack.

---

## 7. SKVQ — sliding-window for million-token contexts

[[skvq]] (Duanmu et al. 2024) exploits a long-context observation: **attention queries decay sharply across token distance**. The most-recent tokens carry the bulk of attention mass. So:

### 7.1 Sliding-window precision schedule

Last W tokens (W ≈ 128) kept at FP16 (or INT4 if memory pressured). Older history at INT2 K / **INT1.5 V**.

At each decode step, the window slides forward by 1 — the now-too-old (d = W) token gets re-quantized down to low-bit and appended to the compressed history buffer.

### 7.2 Channel reorder

For K (and similarly V): from calibration, compute per-channel max magnitude; sort channels in descending order. Permute the channel axis of K (and the matching weights W_k, W_q to preserve attention math) so similar-magnitude channels are adjacent. After reorder, group-of-32 per-group quant fits well because channels in each group have similar magnitudes.

The permutation is absorbed into adjacent weights → zero runtime cost. (Same trick as DuQuant ch-14.)

### 7.3 V at 1.5 bits

Pack pairs of V elements into a single 3-bit code from an 8-entry codebook (vector quant of pairs). Codebook fit per-token.

### 7.4 Throughput

LLaMA-7B at 1M context: KV at FP16 = 250 GB (impossible on 80 GB). KV at SKVQ-2/1.5 = ~30 GB, fits. **Decoding speedup 7×** because each attention call reads 8× less HBM.

---

## 8. QAQ — adaptive bit allocation

[[qaq]] (Dong et al. 2024) observes that within K and V, different tokens / heads contribute unequally to attention. Bit budget should be **allocated adaptively** rather than uniformly.

### 8.1 K vs V sensitivity

Per-bit removal cost (calibration):

```
s_K = ΔKL / Δbits_K
s_V = ΔKL / Δbits_V
```

Consistently `s_K > s_V` across LLaMA / Mistral. Optimal allocation gives K more bits (e.g. K at 3, V at 2 to average 2.5 bits/element).

### 8.2 Per-token bit allocation

For each token t, compute attention mass: `a_t = mean_q softmax(QK^⊤)_{q,t}`. Tokens with high `a_t` are attended often → more bits:

```
Top-10% by a_t:  INT4
Middle 80%:      INT2
Bottom 10%:      INT1 (or evict)
```

### 8.3 Sensitivity surrogate

For ranking elements:

```math
s_x = |\nabla_x \mathcal{L}| \cdot |x|
```

(gradient × value, first-order Taylor approximation of loss change from removing x).

---

## 9. The coupling problem — W4 and KV4 are not independent

[[coupling-kv-quant]] formalises an analytical issue: errors introduced by W4 quantization in `W_k`, `W_v` propagate into the cache, where they are then **re-quantized** to KV4, producing compound errors larger than the sum of individual contributions.

Let `Ŵ_k = W_k + ε_W`. Then:

```math
\hat{K}_t = \mathrm{Quant}(K_t + \varepsilon_W \cdot x_t)
```

Quantization is non-linear in its input (rounding boundary depends on a scale that depends on inputs), so combined error is **not** linear in ε_W. Empirically the combined PPL hit is 1.5–2× the sum.

**Fix 1 (joint calibration)**: single objective over a block.

```math
\min_{s_W, s_{KV}} \| y^{\mathrm{FP}} - y^{W4, KV4}(s_W, s_{KV}) \|^2
```

**Fix 2 (SmoothAttention)**: per-head invertible scaling Q · s, K / s ([[qserve]] ch-14). Distributes magnitude between Q and K so the KV4 path sees a flatter K distribution. Decouples without retraining W.

---

## 10. The KV-quant design space (consolidated)

| Method | Key idea | Bit floor | Cost |
|--------|----------|-----------|------|
| [[kivi]] | per-channel K + per-token V, tuning-free | INT2 | no calib |
| [[kvquant]] | pre-RoPE K + non-uniform + dense-and-sparse | INT2 (sub-2-bit avg) | calibration ~hrs |
| [[gear]] | quant + low-rank residual + sparse | INT4 (best near-lossless) | streaming SVD |
| [[wkvquant]] | joint W4 + KV4 with past-only | INT4 | joint calib |
| [[skvq]] | sliding window + channel reorder | INT2/1.5 (history) | small calib |
| [[qaq]] | adaptive bit allocation per token/head | flexible | sensitivity calib |

Decision tree:

```
Need 4-bit KV, no calibration?            → KIVI INT4
Need 2-bit KV, no calibration?            → KIVI INT2
Need <2-bit average KV?                   → KVQuant (non-uniform + sparse)
Need lossless 4-bit KV?                   → GEAR (low-rank residual)
Need million-token context?               → SKVQ (sliding window)
Targeting full W4-KV4 deployment?         → WKVQuant (joint calibration)
Need fine-grained bit budget?             → QAQ
```

---

## 11. Pitfalls

- **Per-token V quant breaks if attention masks aren't accounted for.** Tokens in a padded position get zero attention → their per-token scale is undefined. Mask out padded tokens before computing per-token scales.
- **Pre-RoPE K storage doubles the RoPE compute at attention time** because every dequant must apply RoPE per-position. Worth it at INT2 (KVQuant saves 1+ ppl); marginal at INT4.
- **Streaming SVD (GEAR) drifts.** Without periodic full refresh, the incrementally-updated A, B accumulate orthogonality loss. Refresh every 256 tokens; the refresh is amortised.
- **GQA changes the KV-cache shape.** For grouped-query attention (LLaMA-2-70B has 8 KV heads vs 64 Q heads), per-channel K is still right but the channel count is 8× smaller — re-check the group_size doesn't exceed channels per head.
- **Sliding window precision (SKVQ) has a discontinuity at boundary t = T - W.** The token transitioning from FP16 to INT2 K loses precision suddenly; some attention spikes can occur. If observed, use a graded precision schedule (FP16 → INT4 → INT2) over multiple boundaries.
- **Adaptive bit allocation (QAQ) requires per-token metadata.** Adds ~32 bytes/token of bookkeeping; for 1M-context this is ~30 MB of metadata. Usually fine.
- **K-axis re-check on new models.** The per-channel K outlier structure is empirical; verify it holds on Mistral, Mixtral, DeepSeek before deploying KIVI's recipe. Most LLMs follow the pattern but exceptions exist.
- **Don't quantize the K used for RoPE position embeddings.** RoPE uses sin/cos lookup tables — those stay FP. Only the projected K_t gets quantized.

---

## Connections and what's next

- **[[quantizer-design]] / ch-03** — the per-channel vs per-token axis choice generalises Lloyd-Max calibration for the asymmetric case; the KV asymmetry is structurally why the same calibration object can't serve both.
- **[[llm-int8]] / ch-07** — the residual-stream outlier-channel finding; KIVI's K-asymmetry is the long-context manifestation.
- **[[smoothquant]] / [[awq]] / ch-09** — outlier handling for activations; the SmoothAttention in ch-14 is the KV equivalent.
- **[[quarot]] / ch-14** — adds R2 (V → O) and R4 (K-cache after RoPE) Hadamards; *complements* per-channel/per-token KV quant — combines with KIVI cleanly.
- **[[squeezellm]] / [[spqr]] / ch-11** — non-uniform + dense-and-sparse for weights; KVQuant ports these tricks to KV.
- **[[turboquant]] / [[polarquant]] / [[qjl]] / ch-18** — data-oblivious KV-cache compression: instead of calibrating per-channel structure, *rotate* the KV cache to a regime where measure-concentration makes the channel structure go away. The eventual successor to KIVI / KVQuant once you accept that calibration is the bottleneck.

## Further reading

- [[kivi]] — Liu et al. ICML 2024, the K/V asymmetry finding + INT2 recipe.
- [[kvquant]] — Hooper et al. NeurIPS 2024, pre-RoPE quant + non-uniform + dense-and-sparse + Q-norm.
- [[gear]] — Kang et al. 2024, quant + low-rank residual + sparse.
- [[wkvquant]] — Yue et al. 2024, joint W4 + KV4 with past-only.
- [[skvq]] — Duanmu et al. 2024, sliding window + channel reorder.
- [[qaq]] — Dong et al. 2024, adaptive bit allocation.
- [[per-channel-vs-per-token-kv]] — analytical study of the axis asymmetry.
- [[coupling-kv-quant]] — analysis of W4–KV4 interaction.
- [[kv-cache-survey]] — broader KV-compression survey including eviction policies.

## Excerpts

- [excerpts/kivi.md](excerpts/kivi.md) — K/V axis asymmetry, streaming INT2 recipe, the load-bearing empirical finding.
- [excerpts/kvquant.md](excerpts/kvquant.md) — pre-RoPE K quant, non-uniform per-channel codes, dense-and-sparse decomposition.
- [excerpts/gear.md](excerpts/gear.md) — three-component decomposition Q + L + S, streaming SVD.
- [excerpts/skvq.md](excerpts/skvq.md) — sliding-window precision schedule, channel reorder, V at 1.5 bits.

## Companion visualization

[figures/kv-asymmetry.html](figures/kv-asymmetry.html) — interactive heatmaps showing the K vs V outlier geometry on a LLaMA-2-7B layer: K's persistent channel outliers vs V's token-wise variation. The single visual that locks in the chapter's main empirical finding.

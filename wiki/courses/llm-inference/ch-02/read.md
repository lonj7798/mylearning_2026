<!-- chapter: ch-02
     track: generation-foundations
     title: Transformer Inference Math — Attention Cost + MHA/MQA/GQA + RoPE
     sources: [[attention-is-all-you-need]], [[multi-query-attention]], [[grouped-query-attention]], [[rope]], [[alibi]], [[attention-complexity]]
     figures: figures/mha-mqa-gqa-bandwidth.html
-->

# Chapter 2 — Transformer Inference Math: Attention Cost + MHA/MQA/GQA + RoPE

> **Core insight.** A decoder layer at inference has two cost terms: attention scales as `O(L²·d)` during prefill but `O(L·d)` per decode step thanks to KV caching; FFN scales as `O(d²)` per token regardless of phase. The shape of `n_kv_heads` (MHA → MQA → GQA) is a *bandwidth* knob, not a *compute* knob — it sets how many KV bytes you read per decode step. RoPE applies to `Q, K` (never `V`) and is the only positional component that has to be re-applied per token at inference; ALiBi adds a fixed bias to scores instead.
>
> **Guideline.** When sizing inference workloads: pick `n_kv_heads` first (it sets KV memory and decode bandwidth — see ch-03), then choose context-extension strategy (RoPE-scaling vs ALiBi extrapolation), then think about which kernel will run attention (FlashAttention vs PagedAttention, ch-06 and ch-11). Stop reasoning about FFN unless it's >50% of your batch's total work — for small-batch decode at long context, attention dominates.

---

## Why this chapter exists

If ch-01 is "the loop", this chapter is "what the loop costs". Every later serving optimization is targeting some line of the cost equation: PagedAttention reduces KV waste, FlashAttention compresses the attention IO, MQA/GQA cuts `H_kv`, chunked prefill rebalances the `O(L²)` term, speculative decoding cheats on the `O(N)` decode steps. You cannot read those papers and know which optimization to deploy without the basic accounting.

Three things you should walk away with:

1. The exact attention complexity table — prefill is `O(L²·d)` (compute-bound), decode-per-step is `O(L·d)` (bandwidth-bound), and why the two phases live on opposite sides of the roofline.
2. The `n_kv_heads` formula — MHA, MQA, GQA differ only in `H_kv`; that one number sets KV-cache memory and per-step memory traffic; quality cost is ~0 above `H_kv ≈ 8`.
3. RoPE as a `(Q, K)`-only rotation applied per position; YaRN/NTK-aware scaling as a frequency-band rescaling; ALiBi as a `-m_h · (i - j)` additive bias that doesn't require any caching gymnastics.

Sources: [[attention-is-all-you-need]], [[attention-complexity]], [[multi-query-attention]], [[grouped-query-attention]], [[rope]], [[alibi]] in the raw-data library.

---

## 1. The attention complexity table

For a single layer with hidden size `d`, query heads `H_q`, KV heads `H_kv`, head dim `d_head = d / H_q`, sequence length `L`, batch `B`:

**Prefill — process all `L` tokens at once** ([[attention-complexity]]):

```
Q, K, V projections:    3·B·L·d²        FLOPs
QKᵀ matmul:             B·H_q·L²·d_head  FLOPs   ← the quadratic term
softmax + dropout:      B·H_q·L²         FLOPs
attn·V matmul:          B·H_q·L²·d_head  FLOPs   ← also quadratic
output projection:      B·L·d²           FLOPs
FFN (gated, factor 8/3): ~(16/3)·B·L·d²  FLOPs
─────────────────────────────────────────
Total per layer:        ~7·B·L·d² + 2·B·H_q·L²·d_head
                        = ~7·B·L·d² + 2·B·L²·d                  (since H_q·d_head = d)
```

The two `B·L²·d` terms (`QKᵀ` and `attn·V`) dominate once `L > ~3.5·d` — for `d = 4096`, attention overtakes FFN at `L ≈ 14k`. Memory: the attention matrix is `B·H_q·L²` elements per layer, naively `~bf16: 2·B·H_q·L²` bytes. FlashAttention (ch-11) eliminates this materialization.

**Decode — one new query token, `L` cached keys**:

```
Q, K, V projections:    3·B·1·d²            FLOPs
QKᵀ:                    B·H_q·1·L·d_head    FLOPs   ← linear in L now
softmax:                B·H_q·L              FLOPs
attn·V:                 B·H_q·1·L·d_head    FLOPs   ← linear in L
output projection:      B·1·d²               FLOPs
FFN:                    ~(16/3)·B·d²         FLOPs
─────────────────────────────────────────────
Total per layer:        ~7·B·d² + 2·B·L·d
```

For a single batch element (`B=1`), the per-step cost is `7·d² + 2·L·d`. At `d=4096, L=8k`, those terms are `~118M` vs `~65M` — FFN larger. At `d=4096, L=32k`, they're `~118M` vs `~262M` — attention larger. **The crossover where attention dominates decode is at `L ≈ 3.5·d`**, same as prefill. Long-context serving is attention-bound; short-context decode is FFN-bound.

**Memory traffic during decode** (the bandwidth-bound side of the roofline):

```
KV reads per layer per step: 2 · B · L · H_kv · d_head · bytes_per_elem
                           = 2·B·L·d_kv·bytes              where d_kv = H_kv·d_head
weight reads per layer:      ~14·d²·bytes  (Q,K,V,O,FFN_up,FFN_gate,FFN_down)
```

At small batch, KV reads scale with `L`; weight reads are constant. Crossover where KV bytes dominate weight bytes: `2·B·L·d_kv = 14·d²`, i.e. `B·L = 7·d²/d_kv`. For `d=4096, d_kv=1024` (GQA-8), that's `B·L = 7·4096²/1024 = 114k`. At `B=4, L=32k` you're well past it — KV bandwidth dominates and MQA/GQA matters hugely.

---

## 2. MHA → MQA → GQA: one parameter, big consequences

[[attention-is-all-you-need]] introduced **multi-head attention (MHA)** with `H_q = H_kv` heads. Each head has its own `K, V` projection. KV cache memory scales with `H_kv = H_q`.

[[multi-query-attention]] (Shazeer 2019) proposed **MQA**: `H_q = many query heads, H_kv = 1`. All query heads share one `K, V` projection. KV cache shrinks by `H_q`× (e.g. 32× for Llama-1-7B's 32-head config). Decode bandwidth shrinks by the same factor — the single dominant memory-traffic improvement of the late-2010s.

The problem with MQA: quality drops, especially on harder tasks (HumanEval, MMLU, GSM8K). The model loses representational capacity in its KV projections.

[[grouped-query-attention]] (Ainslie et al. 2023) split the difference: **GQA-G** uses `H_kv = G` groups of query heads sharing each KV head, where `1 < G < H_q`. Llama 2 70B used `H_q=64, H_kv=8` (GQA-8); Llama 3 8B uses `H_q=32, H_kv=8` (GQA-8); Llama 3 70B uses `H_q=64, H_kv=8` (GQA-8). Quality matches MHA within noise; KV cache drops 8× vs MHA.

**The single most important config field**: `num_key_value_heads`. Read it from any modern model's `config.json` before you reason about its serving cost.

| Model | `n_heads` | `n_kv_heads` | KV per token (bf16) | Variant |
|---|---:|---:|---:|---|
| Llama 1 7B | 32 | 32 | 524 KB | MHA |
| Llama 2 70B | 64 | 8 | 320 KB | GQA-8 |
| Llama 3 8B | 32 | 8 | 128 KB | GQA-8 |
| Llama 3 70B | 64 | 8 | 320 KB | GQA-8 |
| Qwen-3 32B | 64 | 8 | 320 KB | GQA-8 |
| Mistral 7B | 32 | 8 | 128 KB | GQA-8 |
| PaLM 540B | 48 | 1 | 24 KB | MQA |
| Falcon 40B | 64 | 8 | 256 KB | GQA-8 |

(KV per token = `2 · n_layers · n_kv_heads · d_head · 2 bytes`; numbers approximate, ignoring layer counts; see ch-03 §3 for the full formula.)

The pattern: **GQA-8 is the modern default**. MQA is too aggressive; full MHA is wasteful. Some MoE models go further (DeepSeek V3 uses MLA — Multi-head Latent Attention — which compresses KV via a learned low-rank projection; covered in ch-20).

---

## 3. The mask + KV reuse: why we don't recompute

In a causal Transformer, the attention mask `M` sets `M[i, j] = -∞` for `j > i`, so token `i` attends only to positions `1..i`. During prefill, the full `L × L` masked attention runs once. During decode at step `t+1`:

- The new query `q_{t+1}` needs to attend to keys `k_1, ..., k_{t+1}`.
- `k_1, ..., k_t` were already computed during prior steps (or during prefill for the prompt portion).
- We compute only `k_{t+1}, v_{t+1}` fresh, append to the cache, and run a `1 × (t+1)` attention.

Without caching, we'd recompute `k_1, ..., k_t` every decode step — `O(t·d²)` repeated work per step, summing to `O(N²·d²)` for an `N`-token output. With caching it's `O(N·d²)` for the new projections plus `O(N²·d_kv)` for the attention reads. For `d=4096, d_kv=1024, N=1024`: cached = `(N + N²/4)·d² ≈ 268M·d²`; uncached = `(N²·d²) ≈ 4M·d² · 1024 = 4G·d²`. ~16× speedup, growing quadratically with output length.

This is why **every production inference stack since GPT-1 caches**. There's no scenario where the recompute cost is justified.

---

## 4. RoPE: rotation-based positional encoding

[[rope]] (Su et al. 2021) injects position into attention by rotating `Q, K` (never `V`) by position-dependent angles. The rotation is applied per-head, pairing up adjacent dimensions:

For a 2-D pair `(x_{2i}, x_{2i+1})` at position `m` with base frequency `θ_i = 10000^{-2i/d}`:

```
x'_{2i}     =  x_{2i} · cos(m·θ_i) − x_{2i+1} · sin(m·θ_i)
x'_{2i+1}   =  x_{2i} · sin(m·θ_i) + x_{2i+1} · cos(m·θ_i)
```

Equivalently (complex-number form), pair `x_{2i} + i·x_{2i+1}` is multiplied by `e^{i·m·θ_i}`. The frequencies `θ_i` decay exponentially with dimension index, so low dims rotate fast and high dims rotate slowly.

**Why this gives relative position.** For two positions `m, n`:

```
⟨RoPE(q, m), RoPE(k, n)⟩ = ⟨q, R(n - m) · k⟩
```

The inner product depends on `n - m`, not `m, n` individually — a relative-position behavior without explicit relative-position tables.

**Implementation in cached decode.** Each cached `K_t` has been rotated by `θ_·t` *before* being stored. At step `t+1`, the new query gets rotated by `θ_·(t+1)`. The cache stores the *rotated* keys, so attention dot products `q'_{t+1} · k'_j` are correct for free.

### YaRN and NTK-aware: context extension at inference

A model pretrained at context length `L_train = 4096` typically fails at `L = 32768`: the high-frequency RoPE bands hit angles they never saw during training. Two common fixes:

- **NTK-aware scaling** (Peng & Quesnelle 2023): rescale `θ_i ← θ_i · (L_train/L_target)^{i/d}` — interpolate low frequencies more aggressively, leave high frequencies alone.
- **YaRN** (Peng et al. 2023): "Yarn" = "Yet another RoPE extensioN" — refines NTK with per-band ramp functions and a learned temperature. Allows `8×` to `32×` context extension with brief fine-tuning.

Both work at inference time as a function of how you compute `θ_i` before applying the rotation. The KV cache shape doesn't change.

---

## 5. ALiBi: linear bias on attention scores

[[alibi]] (Press et al. 2021) replaces positional embeddings with an additive bias on attention logits:

```
score(i, j) = (q_i · k_j) / √d − m_h · (i − j)    for j ≤ i
```

where `m_h` is a per-head slope (geometrically spaced, e.g. `m_h ∈ {1/2, 1/4, 1/8, ...}`). Recency is encoded directly: distant past tokens get a larger penalty.

Compared to RoPE:

| | RoPE | ALiBi |
|---|---|---|
| Mechanism | Rotate `Q, K` | Add bias to scores |
| Cache impact | Rotated K stored in cache | No K modification |
| Extrapolation beyond train length | Poor (needs YaRN/NTK) | Strong by design |
| Adoption | Llama 1/2/3, Qwen, Mistral, DeepSeek | BLOOM, MPT, Falcon-1B/7B |

**The 2026 winner is RoPE.** It outperforms ALiBi at trained context lengths and YaRN/NTK have closed most of the extrapolation gap. ALiBi remains relevant as the explanation of *why* extrapolation works — and is still active in some research stacks.

---

## 6. Per-layer breakdown: attention vs FFN dominance

For a decode step on one sequence at context `L`:

```
attention FLOPs/layer    ≈  7·d² + 2·L·d         (small-batch)
FFN FLOPs/layer          ≈  (16/3)·d²
attention/FFN ratio      ≈  (7·d² + 2·L·d) / (5.3·d²)
                         =  1.3 + 0.38·(L/d)
```

For `d = 4096`:
- `L = 1024` → ratio = 1.4 (FFN-comparable)
- `L = 8192` → ratio = 2.1 (attention 2× FFN)
- `L = 32768` → ratio = 4.4 (attention 4× FFN)
- `L = 131072` → ratio = 13.6 (attention dominates)

But the *bytes moved* matter more for decode (memory-bound). At small batch:

```
attention bytes/layer    ≈  2·L·H_kv·d_head·2     (KV read)
weight bytes/layer       ≈  14·d²·2               (weight load, bf16)
attention/weight ratio   ≈  (L·d_kv) / (7·d²)
```

For Llama-3-70B (`d=8192, d_kv=1024`):
- `L = 4096` → ratio = 0.073 (weights dominate, decode is weight-bound)
- `L = 32768` → ratio = 0.57 (attention bytes near parity)
- `L = 131072` → ratio = 2.3 (KV bandwidth dominates)

This is **why long-context serving is a totally different optimization problem** from short-context. At short context, throughput is set by `weight_bytes / hbm_bandwidth` and improves with larger batch (batch amortizes weight load). At long context, it's set by per-sequence KV reads, and batch size is *capped* by KV memory rather than compute — entirely the regime PagedAttention (ch-06) was built for.

---

## 7. Cheat-sheet

```
ATTENTION COMPLEXITY:
  prefill total:     7·B·L·d² + 2·B·L²·d        (compute-bound)
  decode per-step:   7·B·d² + 2·B·L·d           (small-batch, bandwidth-bound)
  attention vs FFN crossover:  L ≈ 3.5·d
  KV bytes per token:  2·n_layers·n_kv_heads·d_head·bytes

ATTENTION VARIANTS (only n_kv_heads changes):
  MHA:    H_kv = H_q              (full)
  MQA:    H_kv = 1                (PaLM, fastest, quality drop)
  GQA-G:  H_kv = G (1 < G < H_q)  (Llama 2/3, Qwen, Mistral — default)
  MLA:    learned low-rank KV     (DeepSeek V3)

ROPE:
  q'_m = R(m·θ) · q ; k'_n = R(n·θ) · k   (V unchanged)
  ⟨q'_m, k'_n⟩ depends only on (n - m)
  context extension: NTK-aware, YaRN — rescale θ_i frequencies

ALIBI:
  score(i, j) -= m_h · (i - j)     for j ≤ i
  no cache modification; stronger extrapolation; weaker in-distribution

CAUSAL MASK + KV REUSE:
  cache K, V for all prior positions; one fwd pass appends one new K, V.
  saves O(N²·d²) recompute over O(N) decode steps.

REGIME RULE OF THUMB:
  short context:   FFN-bound + weight-bandwidth-bound (batch up)
  long context:    attention-FLOPs + KV-bandwidth-bound (batch capped by KV)
```

---

## Connections and what's next

- **[[kv-cache-memory-formula]] / ch-03** — the `n_kv_heads · d_head · 2 · n_layers · L` formula derived in this chapter sets the per-request KV bytes; the next chapter does the fleet-level accounting.
- **[[prefill-vs-decode]] / ch-03** — formalizes the two regimes as TTFT vs TPOT and motivates the chunked-prefill / disaggregation work in ch-05 and ch-09.
- **[[flashattention]] / ch-11** — kills the `O(L²)` attention-matrix materialization with tiling + online softmax; the kernel that makes long-context prefill tractable.
- **[[pagedattention]] / ch-06** — when KV becomes the binding constraint (long context, this chapter), block-allocate it.
- **MLA (DeepSeek V3) / ch-20** — extends MQA/GQA with a learned low-rank latent KV, reducing memory by ~10× over GQA-8 at frontier scale.
- **[[rope]] frequency-scaling tricks / ch-20** — YaRN-scaled RoPE is in production for every long-context model; Llama 3 uses it for 128k extension.

## Further reading

- [[attention-is-all-you-need]] — the original scaled dot-product attention.
- [[attention-complexity]] — Tay et al. 2020 survey; the source of the `O(L²)` framing.
- [[multi-query-attention]] — Shazeer 2019.
- [[grouped-query-attention]] — Ainslie et al. 2023.
- [[rope]] — Su et al. 2021.
- [[alibi]] — Press et al. 2021.

## Companion visualization

**[figures/mha-mqa-gqa-bandwidth.html](figures/mha-mqa-gqa-bandwidth.html)** — interactive bar chart of KV bytes / token and per-step decode bandwidth for MHA / MQA / GQA-{2,4,8,16} on Llama-3 architectures. Adjust `n_heads`, `n_kv_heads`, `d_head` and see how decode bandwidth scales. Use it to internalize why GQA-8 is the universal default.

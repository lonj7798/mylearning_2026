# RoPE — Rotary Position Embedding as the Unique Relative-Position Solution
<!-- slug: rope-rotary-position-embedding · type: paper · source: wiki:llm-arch:wiki/courses/llm-arch/ch-06/excerpts/rope-complex-rotation-derivation.md -->

**Core Insight.** Demand one property — `⟨f(q,m), f(k,n)⟩ = g(q, k, m−n)`, i.e. the attention logit depends on positions **only through their difference** — plus `f(x,0) = x` and norm preservation, and the answer is forced: `f(q,m) = q·e^{imθ}` per complex dimension, i.e. rotate each 2D coordinate pair of `q` and `k` by angle `m·θ_i` with `θ_i = base^{−2i/d}`. RoPE is not a design among alternatives; it is *the* solution, which is why nothing has displaced it in five years.

**Guideline.** Apply RoPE to `Q` and `K` only, after the projections and immediately before the attention kernel — never to `V`, never to the residual stream. Then treat context extension as a frequency-domain problem: ask how many full rotations each pair `i` completes inside the training length `L` (`r_i = L·θ_i/2π`), leave the fast pairs alone, and interpolate only the pairs that never finished a revolution.

## Technical Details
- **Frequencies.** `θ_i = base^{−2i/d}`, pair index `i ∈ {0, …, d/2−1}`, `d = d_head`, `base = 10000` (Llama 1/2, Mistral, Qwen2), `500000` (Llama 3), up to `10⁶` for long-context finetunes. Same ladder as sinusoidal — the innovation is *where* it is applied, not the schedule.
- **Per-pair rotation (real form).** For pair `i` at position `m`:
  `[q'_{2i} ; q'_{2i+1}] = [[cos mθ_i, −sin mθ_i] ; [sin mθ_i, cos mθ_i]] · [q_{2i} ; q_{2i+1}]`
  Stacked over all pairs this is the block-diagonal orthogonal matrix `R_{Θ,m} ∈ ℝ^{d×d}` with `d/2` independent `2×2` blocks.
- **The central identity** (`Rᵀ_m = R_{−m}`, `R_{−m}R_n = R_{n−m}`):
  `(R_{Θ,m} q)ᵀ (R_{Θ,n} k) = qᵀ R_{Θ,m}ᵀ R_{Θ,n} k = qᵀ R_{Θ,n−m} k`
  **Absolute `m` and `n` vanish; only `n−m` survives.** Verified numerically: with `d=8`, random `q,k`, `(m,n)=(17,5)` gives `(R_m q)·(R_n k) = 3.6169521324913054` and `qᵀR_{n−m}k = 3.616952132491305`.
- **Per-pair closed form (verified — safe to animate).** Writing `Δ = m−n`:
  `⟨R_m q^{(i)}, R_n k^{(i)}⟩ = (q_{2i}k_{2i} + q_{2i+1}k_{2i+1})·cos(Δθ_i) + (q_{2i}k_{2i+1} − q_{2i+1}k_{2i})·sin(Δθ_i)`
  The relative angle mixes the pair's **dot product** (cos term) with its **2D cross product / determinant** (sin term). Complex form: `⟨f(q,m), f(k,n)⟩ = Re[Σ_i q_i·conj(k_i)·e^{iΔθ_i}] = Σ_i |q_i||k_i| cos(∠q_i − ∠k_i + Δθ_i)`.
- **Minimal worked example.** `q = k = (1, 0)` in one pair, `θ = 1.0 rad`. `(m,n) = (5,3)` → dot `= −0.416147`. `(m,n) = (7,5)` → `−0.416147`. `(m,n) = (100,98)` → `−0.416147`. Identical, because `Δ = 2` in all three. That single line *is* the relative-position property.
- **Uniqueness sketch** (complex space, one pair at a time): polar-decompose `f(q,m) = R_f(q,m)e^{iΘ_f(q,m)}`; setting `m=n` with `f(x,0)=x` forces `R_f(q,m) = |q|` (magnitude cannot depend on position → rotation, not scaling); the phase constraint `Θ_f(q,m) − Θ_f(k,n) = Θ_g(q,k,m−n)` forces `Θ_f(q,m) = Θ(q) + mθ` (the only continuous `φ` with `φ(m)−φ(n) = h(m−n)` is linear). Hence `f(q,m) = q·e^{imθ}`. Improving RoPE requires *relaxing a constraint* (which is exactly what iRoPE does by dropping PE from some layers), not solving the same one better.
- **Frequency bands, `d_head = 128, base = 10000, L = 8192`** — `r_i = L/λ_i` = rotations completed during training:

  | pair `i` | `θ_i` | `λ_i` | rotations in `L=8192` | band |
  |---|---|---|---|---|
  | 0 | 1.0 | 6.28 | 1303.8 | fast / local |
  | 16 | 1.0×10⁻¹ | 62.8 | 130.4 | fast |
  | 32 | 1.0×10⁻² | 628.3 | 13.0 | mid |
  | 48 | 1.0×10⁻³ | 6,283.2 | 1.30 | slow |
  | 63 | 1.1548×10⁻⁴ | 54,410.1 | **0.151** | slow / global — never completes a revolution |

  With YaRN's standard cutoffs (`r > β = 32` untouched, `r < α = 1` fully interpolated) the boundaries land at `i ≈ 25.8` and `i ≈ 49.8`: pairs 0–25 are left alone, 26–49 get a smooth ramp, 50–63 are scaled by `1/s`. NTK-aware alternative: `base' = base·s^{d/(d−2)}`; for `d=128, s=8` that is `10000 → 82,685`.
- **Implementation** (HF/GPT-NeoX style): `q' = q·cos + rotate_half(q)·sin`, `rotate_half(x) = cat(−x[…, d/2:], x[…, :d/2])`. **Pairing-convention gotcha:** original RoFormer/GPT-J pairs dims `(2i, 2i+1)` (interleaved); GPT-NeoX/HF-Llama pairs `(i, i+d/2)` (split-half). Mathematically equivalent — the same `d/2` planes, relabelled — *provided Q and K use the same convention*; mixing them silently destroys the model. Runtime overhead ≈ 1–3% of the forward pass.
- **Training-memory angle:** RoPE's whole positional state is a precomputed `cos`/`sin` cache of shape `[L, d_head/2]` each — `2·L·(d_head/2)·4 B` in fp32 = **4.00 MiB at `L=8192, d_head=128`** (16.00 MiB at 32K, 64.00 MiB at 128K), allocated **once** and shared by every layer and every head. Compare the learned absolute table it replaces: 384.00 MiB of AdamW state for GPT-3's `2048×12288`. RoPE has **zero trainable parameters, zero gradient, zero optimizer state**. It also adds zero saved-for-backward activations when fused, because its backward is just rotation by `−mθ` — an orthogonal map with constant coefficients, so no input tensor needs saving. Two real traps: (1) an **unfused** `q·cos + rotate_half(q)·sin` allocates fresh `q_rot`/`k_rot` of `B·H·T·d_head` each — **64.00 MiB apiece at `B=1, H=32, T=8192, d_head=128` in bf16**, so ~128 MiB/layer of avoidable transient if the pre-rotation copies are not freed before the kernel launches; (2) never cache `cos`/`sin` at `[L, d_head]` in fp32 for a 128K context unless you mean it — that is 128.00 MiB of pure duplication. The structural win is that RoPE lives **outside** the attention kernel: `Q` and `K` are rotated *before* the call, so FlashAttention/SDPA see an ordinary `(Q,K,V)` triple and keep their `O(T)` activation footprint, unlike logit-additive schemes that put a `B·H·T·T` object (4.000 GiB at `B=1,H=32,T=8192` bf16) into the one tensor the kernel refuses to materialize.

## Citation
Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864, 2021. https://arxiv.org/abs/2104.09864 — Peng, B., Quesnelle, J., Fan, H., Shippole, E. "YaRN: Efficient Context Window Extension of Large Language Models." ICLR 2024, arXiv:2309.00071. — Biderman, S. et al. "Rotary Embeddings: A Relative Revolution." EleutherAI blog, 2021. — Derivation and uniqueness argument restated from `wiki:llm-arch:wiki/courses/llm-arch/ch-06/excerpts/rope-complex-rotation-derivation.md`; band table and identities recomputed and numerically verified here.

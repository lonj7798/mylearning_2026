# Sinusoidal Absolute Positional Encoding — Frequency Ladder and Its Hidden Rotations
<!-- slug: sinusoidal-absolute-encoding · type: paper · source: wiki:llm-arch:wiki/courses/llm-arch/ch-03/excerpts/sinusoidal-encoding-frequency-analysis.md -->

**Core Insight.** The 2017 fix was to add a fixed, non-learned vector `PE_pos` to each token embedding, where dimension pair `i` is a `(sin, cos)` pair spinning at frequency `ω_i = 10000^{-2i/d}`. The ladder of geometrically spaced frequencies gives every position a unique fingerprint the same way binary place-values give every integer a unique bit pattern — and, as a mathematical accident nobody exploited until 2021, the map from `PE_pos` to `PE_{pos+k}` is a **block-diagonal rotation matrix that depends only on `k`**. RoPE is that accident, promoted from side-effect to design.

**Guideline.** Never read the sinusoidal formula as "sin and cos of position"; read it as "a bank of `d/2` clocks running at geometrically spaced speeds." Whether you are choosing a RoPE base, debugging a long-context failure, or sizing a positional table, the operative question is always "which clocks have completed a full revolution inside my training length `L`, and which have not?"

## Technical Details
- **Formula** (`pos` = token index, `i` = *pair* index, `i ∈ {0, …, d/2−1}`, pair `i` occupying dimensions `2i` and `2i+1`):
  `PE_{(pos, 2i)}   = sin(pos / 10000^{2i/d})`
  `PE_{(pos, 2i+1)} = cos(pos / 10000^{2i/d})`
  Frequency `ω_i = 10000^{−2i/d}`; wavelength `λ_i = 2π/ω_i`.
- **Corrected frequency ladder, `d_model = 512`** (recomputed; the llm-arch table mislabels rows by one step — it prints `ω = 0.1` against dims `(64,65)` when `ω = 0.1` actually belongs to dims `(128,129)`, and prints `λ = 62,832` for the last pair when the true value is `60,611`):

  | pair `i` | dims | `2i/d` | `ω_i` | `λ_i` (positions) |
  |---|---|---|---|---|
  | 0 | (0,1) | 0.0000 | 1.0 | 6.28 |
  | 32 | (64,65) | 0.1250 | 0.31623 | 19.87 |
  | 64 | (128,129) | 0.2500 | 0.1 | 62.83 |
  | 128 | (256,257) | 0.5000 | 0.01 | 628.3 |
  | 192 | (384,385) | 0.7500 | 0.001 | 6,283.2 |
  | 255 | (510,511) | 0.9961 | 1.0366×10⁻⁴ | **60,611.5** |
- **Why the last row is 60,611 and not 62,832.** `2π×10000 = 62,831.85` is the wavelength at exponent *exactly* 1.0, but the largest exponent any real pair reaches is `(d−2)/d = 510/512 = 0.9961`. The slowest clock is always slightly faster than `base·2π`. Same arithmetic at `d_head = 128`: slowest pair `i=63` has `2i/d = 126/128 = 0.9844`, `θ = 1.1548×10⁻⁴`, `λ = 54,410.1`.
- **Relative position is a rotation (the lemma).** By the angle-addition identities,
  `[PE_{(pos+k,2i)} ; PE_{(pos+k,2i+1)}] = [[cos ω_i k, sin ω_i k] ; [−sin ω_i k, cos ω_i k]] · [PE_{(pos,2i)} ; PE_{(pos,2i+1)}]`
  The `2×2` matrix depends on `k` and `i` only — **not on `pos`**. Full-vector form: `PE_{pos+k} = diag(R_k^{(0)}, …, R_k^{(d/2−1)}) · PE_pos`. Vaswani et al.'s stated motivation ("for any fixed offset k, `PE_{pos+k}` can be represented as a linear function of `PE_pos`") is exactly this.
- **Distance decay.** `PE_pos · PE_{pos+k} = Σ_{i=0}^{d/2−1} cos(ω_i k)`. At `k = 0` this is `d/2` (maximum, all cosines = 1); as `|k|` grows the fast clocks decorrelate and cancel, leaving only slow terms — a built-in locality prior.
- **Why it lost.** (1) *Semantic pollution*: `x_0 = Embed(tok) + PE_pos` mixes content and position from layer 0 onward and the model must spend capacity disentangling them; (2) *absolute is the wrong quantity* — "3 tokens before the verb" is linguistically real, "position 47" is not; (3) *extrapolation fails empirically* despite the lemma, because the learned `q·k` distribution, not the encoding, is what goes out of distribution past `L`.
- **Learned absolute PE (GPT-2/3) is worse on every memory axis.** A table `W_p ∈ ℝ^{L×d}` is *trainable*. GPT-3: `L=2048, d=12288 → 25,165,824 params`. Hard ceiling at `L`; position 2049 does not exist.
- **Training-memory angle:** sinusoidal PE is a **non-trainable buffer**: `4·L·d` bytes in fp32, materialized once and shared by every layer — `128.00 MiB` at `L=8192, d=4096`, with **zero** optimizer state and zero gradient. Learned absolute PE is the same tensor promoted to a parameter, so it now pulls the full mixed-precision AdamW tax of **16 B/param** (fp32 master + two fp32 moments + fp32 grad) or 18 B/param counting the bf16 working copy: GPT-3's table costs `25,165,824 × 16 B = 384.00 MiB` of resident optimizer state, and an `L=8192, d=4096` table costs `33,554,432 × 16 B = 512.00 MiB` — half a gigabyte of an A100-40GB spent on a function you could have computed in closed form. Neither scheme adds *activation* bytes: `PE` is summed into `x` before layer 0, so the tensor never grows. The genuinely expensive consequence of absolute PE is indirect and much larger: because it cannot extrapolate, extending context means **retraining at the longer `L`**, and activation memory is the term that scales with `T` (and the attention term with `T²`). RoPE's post-hoc extension via YaRN is a memory saving measured in training runs, not in bytes.

## Citation
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I. "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762. https://arxiv.org/abs/1706.03762 — frequency-ladder analysis and rotation lemma restated (with numeric corrections) from `wiki:llm-arch:wiki/courses/llm-arch/ch-03/excerpts/sinusoidal-encoding-frequency-analysis.md`.

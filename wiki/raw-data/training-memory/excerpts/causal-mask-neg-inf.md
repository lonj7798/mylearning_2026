# Causal Masking: −∞ Before Softmax, and What It Costs
<!-- slug: causal-mask-neg-inf · type: paper · source: wiki:llm-arch:wiki/courses/llm-arch/ch-04/excerpts/causal-masking-implementation.md + https://arxiv.org/abs/1706.03762 §3.2.3 -->

**Core Insight.** The causal mask is a single additive matrix applied to the scores *before* softmax: `0` on and below the diagonal, `−∞` above it. Because `e^{−∞} = 0`, future positions receive exactly zero weight — a hard, exact mask, not a soft penalty. This one line is what makes teacher forcing legal: it guarantees position `t`'s representation depends only on positions `1..t`, so `T` predictions computed in a single parallel forward pass are *identical* to `T` sequential autoregressive steps. It is also what makes the KV cache valid at inference. Half of the `N×N` score matrix is therefore structurally dead weight — and the kernel, not the model, decides whether you pay for it.

**Guideline.** Use `torch.finfo(dtype).min`, never a hard-coded `-1e9` and never literal `-inf`. `-1e9` overflows fp16 (max magnitude 65,504) and a fully-masked row of true `-inf` gives `0/0 = NaN`. And never materialise the mask at long context: a causal kernel (FlashAttention / SDPA `FLASH_ATTENTION` backend) applies it implicitly by skipping fully-future KV blocks, which both removes the mask tensor and halves attention FLOPs.

## Technical Details

- **Definition.** `M_ij = 0 if j ≤ i, else −∞`, applied additively:
  `Attention_causal(Q,K,V) = softmax((QKᵀ + M)/√d_k) · V`
  (equivalently `S_ij = q_i·k_j/√d_k` for `j ≤ i`, `−∞` for `j > i`).
  After softmax: `softmax(S+M)_ij → 0` wherever `M_ij = −∞`. Row `i` then has exactly `i+1` non-zero entries.
- **Paper text (§3.2.3, verbatim):** *"We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections."*
- **The mask must include the diagonal** (`j ≤ i`, not `j < i`). Off-by-one here silently removes each token's ability to attend to itself. `torch.tril(..., diagonal=0)` keeps it; `torch.triu(..., diagonal=1)` marks exactly the illegal region.
- **Mask value by dtype — the numbers that matter:**
  | dtype | `torch.finfo(dtype).min` | is `-1e9` representable? |
  |---|---|---|
  | fp16 | `-65504` | **no** — `-1e9` casts to `-inf` |
  | bf16 | `-3.3895×10³⁸` | yes |
  | fp32 | `-3.4028×10³⁸` | yes |
  GPT-2/HF historically used `-1e9`; modern best practice is dtype-aware:
  ```python
  # legacy (GPT-2 / HF): breaks under fp16
  attn = torch.where(causal_mask, attn, torch.tensor(-1e9, dtype=attn.dtype))
  # correct
  attn = attn.masked_fill(~causal_mask, torch.finfo(attn.dtype).min)
  ```
  Why `finfo.min` beats literal `-inf`: if an entire row is masked (a fully-padded row in a ragged batch), `-inf` everywhere gives `Σ e^{−∞} = 0` and softmax returns `0/0 = NaN`, which poisons the whole backward pass. `finfo.min` everywhere gives a finite uniform row — garbage that is later discarded, not NaN that propagates.
- **Causal mask ≠ padding mask.** They are logically distinct and must be **combined** in ragged batches: causal forbids attending to the future; padding forbids attending to non-existent tokens. Broadcasting shape is `[1, 1, N, N]` for the causal part against `[B, a, N, N]` scores; incorrect broadcast corrupts attention silently with no error.
- **Teacher-forcing coupling.** The mask enforces the autoregressive factorisation `P(x₁..x_T) = Π_t P(x_t | x_{<t})` architecturally. Without it, position `t` would see the ground-truth token at `t+1` and the loss would be trivially minimised. The training/inference hidden-state equivalence this buys is exactly what licenses KV caching.
- **At decode time the mask is free.** The single new query is always the latest position, so every cached key is legal; production decode kernels omit the mask entirely and just don't load future keys.
- **Training-memory angle:** Three separate costs, all avoidable.
  1. **The mask tensor.** A `[1,1,N,N]` mask is `N²` elements. At `N = 32768`: bool = **1.07 GB**, bf16 additive = **2.15 GB**, fp32 additive = **4.29 GB** — per model, but resident for the whole run and re-read from HBM on every layer.
  2. **The out-of-place copy.** `masked_fill` (not `masked_fill_`) allocates a second `(B, a, N, N)` tensor. At `B=1, a=32, N=32768`, bf16 that is another `32 × 32768² × 2 B = 68.7 GB` per layer — the same order as the score tensor it duplicates.
  3. **The structurally dead half.** Causal attention only needs the lower triangle, `≈ N²/2` entries, yet a dense implementation stores and computes all `N²`. FlashAttention-2 exploits this by skipping KV blocks entirely above the diagonal, **saving ~50% of attention computation** and materialising neither the mask nor the score matrix — masking on the diagonal blocks is applied inside the SRAM tile. Under the Korthikanti accounting the causal structure does *not* reduce the `5·a·s²·b` activation term for a dense kernel; only a causal-aware kernel converts the structural sparsity into real bytes saved.

## Citation
Ashish Vaswani et al. "Attention Is All You Need," NeurIPS 2017, §3.2.3. https://arxiv.org/abs/1706.03762 · Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019 · Tri Dao, "FlashAttention-2," 2023, https://arxiv.org/abs/2307.08691 · Implementation detail adapted from `llm-arch:wiki/courses/llm-arch/ch-04/excerpts/causal-masking-implementation.md`.

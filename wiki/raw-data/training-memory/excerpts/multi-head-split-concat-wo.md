# Multi-Head Attention: Split, Concat, W_O — Free in FLOPs, Linear in Memory
<!-- slug: multi-head-split-concat-wo · type: paper · source: https://arxiv.org/abs/1706.03762 §3.2.2 + wiki:llm-arch:wiki/courses/llm-arch/ch-02/excerpts/multi-head-redundancy.md -->

**Core Insight.** Multi-head attention runs `h` independent attention computations over `d_head = d_model/h`-dimensional slices, concatenates them, and merges with a single output matrix `W_O`. Because `h · d_head = d_model` exactly, the parameter count and the FLOP count are **identical** to one head of width `d_model` — the heads are a free reparameterisation. The reason to want them: a single head computes one softmax-weighted average per position, and *averaging destroys the ability to represent several distinct relations at once* ("with a single attention head, averaging inhibits this"). But the freeness is only true in FLOPs. The `N×N` score matrix is **per head**, so activation memory scales linearly in `h` while compute stays constant — multi-head attention is FLOP-neutral and memory-expensive.

**Guideline.** Choose `h` for representational diversity, not for compute: 4–16 heads captures nearly all the benefit and 1 head costs ~0.9 BLEU. But budget the score activation as `B·h·N²·2` bytes, not `B·N²·2` — doubling head count at fixed `d_model` doubles attention activation memory for free-of-charge zero FLOP change. If you are memory-bound at long context, the head count is a lever a dense kernel makes you pay for and a FlashAttention kernel makes free.

## Technical Details

- **The equations (§3.2.2, verbatim):**
  `MultiHead(Q,K,V) = Concat(head₁, …, head_h) W^O`
  `where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)`
  `W_i^Q ∈ ℝ^(d_model×d_k)`, `W_i^K ∈ ℝ^(d_model×d_k)`, `W_i^V ∈ ℝ^(d_model×d_v)`, **`W^O ∈ ℝ^(h·d_v × d_model)`**
- **Paper's config:** `h = 8`, `d_k = d_v = d_model/h = 64`, `d_model = 512`. Paper's own justification: *"Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality."*
- **Why heads at all (§3.2.2, verbatim):** *"Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."* One head produces **one** convex combination per query — one routing pattern per layer. Language needs several simultaneously (syntactic agreement, coreference, positional locality, delimiter structure), and a convex average of conflicting routings is not any of them.
- **Tensor shapes through the whole block** (batch `B`, seq `N`):
  ```
  X            (B, N, d_model)
  → W_Q/W_K/W_V, each (d_model, d_model)          # h heads fused into one matrix
  Q,K,V        (B, N, d_model)
  → .view(B, N, h, d_head).transpose(1, 2)
  Q,K,V        (B, h, N, d_head)                   # d_head = d_model / h
  S = Q @ Kᵀ   (B, h, N, N)      / √d_head
  P = softmax  (B, h, N, N)      row-wise, dim=-1
  O = P @ V    (B, h, N, d_head)
  → .transpose(1, 2).contiguous().view(B, N, h*d_head = d_model)   # the "Concat"
  → @ W_O      (B, N, d_model)                     # W_O is (d_model, d_model)
  ```
  The "concat" is not a copy of separate tensors — it is a transpose + reshape of one contiguous buffer. Heads are a *view*, not separate allocations.
- **Parameter arithmetic (the freeness):** `W_Q, W_K, W_V, W_O` are each `d_model × d_model` regardless of `h`. Attention block params = **`4·d_model²`**.
  `d_model = 512` → `4 × 512² = 1,048,576 ≈ 1.05M` per layer.
  `d_model = 4096` → `4 × 4096² = 67,108,864 ≈ 67.1M` per layer.
  Changing `h` from 8 to 32 changes this by **zero**.
- **FLOP arithmetic (also free):** `QKᵀ` costs `2·B·h·N²·d_head = 2·B·N²·d_model` — the `h` cancels. Same for `P·V`. Head count is FLOP-invariant *exactly*.
- **Table 3 row (A) — the real head-count ablation, verified from the paper** (EN-DE, newstest2013 dev; all rows have identical params by construction):
  | `h` | `d_k` | `d_v` | PPL (dev) | BLEU (dev) |
  |---|---|---|---|---|
  | 1 | 512 | 512 | 5.29 | 24.9 |
  | 4 | 128 | 128 | 5.00 | 25.5 |
  | **8 (base)** | **64** | **64** | **4.92** | **25.8** |
  | 16 | 32 | 32 | 4.91 | 25.8 |
  | 32 | 16 | 16 | 5.01 | 25.4 |
  Paper's summary, verbatim: *"While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads."* Best is `h = 8–16`; `h = 32` (`d_k = 16`) degrades because each head has too few dimensions to form a usable compatibility function.
- **Head redundancy.** Voita et al. (2019) gated a trained 6-layer 8-head (48-head) Transformer with an `L₀` penalty: keeping 38/48 heads costs −0.1 BLEU, 25/48 costs −0.3, 15/48 costs −0.6, 10/48 costs −1.0 on WMT EN-RU. **~60% of heads are prunable for <0.3 BLEU.** Surviving heads are disproportionately positional ("attend to `t−1`") and syntactic. This asymmetry — redundancy helps optimisation during training, wastes bandwidth during inference — is what MQA/GQA/MLA exploit.
- **Training-memory angle:** The score tensor is `(B, h, N, N)` — **`B·h·N²·2` bytes in bf16** — which is *linear in `h` at constant FLOPs*. Its arithmetic intensity is exactly **`2·d_head` FLOPs per stored *element*** — equivalently **`d_head` FLOPs per stored *byte*** in bf16 (2 bytes/element). (Under the multiply-add convention of 2 FLOPs per MAC, `QKᵀ` costs `2·B·h·N²·d_head` FLOPs and writes `B·h·N²` elements = `2·B·h·N²` bytes; e.g. `h = 32, d_head = 128, N = 8192, B = 1` → `5.498e11` FLOPs / `2.147e9` elements = `256 = 2·d_head` FLOP·element⁻¹ = `128 = d_head` FLOP·byte⁻¹. The ratio is independent of `B` and `N`.) So doubling `h` at fixed `d_model` (halving `d_head`) halves intensity and doubles memory for the same compute. This is the `5·a·s²·b` term in Korthikanti's per-layer activation formula `11·s·b·h + 5·a·s²·b` — `a` is the head count and it enters *linearly*, while `d_model` does not appear in that term at all. Worked case, 7B-class model (`d_model = 4096`, `a = 32`, `d_head = 128`, `L = 32`), bf16, `B = 1`:
  | `N` | `N²·2` bytes/head | per layer (×32 heads) | whole model (×32 layers) |
  |---|---|---|---|
  | 2,048 | 8,388,608 B = 8 MiB | 268 MB | 8.6 GB |
  | 8,192 | 134,217,728 B = 128 MiB | 4.3 GB | 137 GB |
  | 32,768 | 2,147,483,648 B = 2 GiB | 68.7 GB | 2.2 TB |
  (Decimal GB/TB. [[ch-04]] quotes the same three rows in binary units — 8 MB / 128 GB / 2 TB — so `2.2 TB` here and `2 TB` there are the identical 2 TiB number.)
  Compare `Q+K+V` at `N = 32768`: `3 × 32768 × 4096 × 2 B = 805 MB` per layer — **85× smaller** than the score tensor. Note the score memory does *not* shrink if you widen heads and reduce `h` at fixed `d_model`; it shrinks **proportionally to `h`**. And with FlashAttention the `(B,h,N,N)` tensor is never materialised at all, so head count becomes memory-free again — the kernel, not `h`, decides.

## Citation
Ashish Vaswani et al. "Attention Is All You Need," NeurIPS 2017, §3.2.2 and Table 3 row (A). https://arxiv.org/abs/1706.03762 · Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, Ivan Titov, "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned," ACL 2019, https://arxiv.org/abs/1905.09418 · Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," MLSys 2023, https://arxiv.org/abs/2205.05198.

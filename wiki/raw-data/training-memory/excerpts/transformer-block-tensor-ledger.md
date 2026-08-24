# The Transformer Block as a Dataflow: Every Tensor It Stores for Backward

<!-- slug: transformer-block-tensor-ledger · type: doc · source: wiki:course/training-memory:wiki/courses/training-memory/ch-03/figures/checkpointing.html -->

**Core Insight.** A transformer block is not "a layer" — it is an ordered chain of ~16 named tensors, and the backward pass needs a *specific, derivable subset* of them: every matmul must save its **input**, softmax must save its **output**, dropout must save its **mask**, and the residual add saves **nothing**. Korthikanti's coefficient `sbh(34 + 5as/h)` is not a magic constant: `34 = 11 (attention) + 19 (MLP) + 4 (two LayerNorms)` bytes per `s·b·h` element, and `5as/h` is the three s²-shaped tensors (softmax output 2, dropout mask 1, dropout output 2) that only the attention block produces.

**Guideline.** Before estimating any block's memory, walk the dataflow op by op and ask one question per op: *what does its VJP need?* Matmul → its input. GELU → its input. Softmax → its output. Dropout → its 1-byte mask. Add → nothing. LayerNorm → its input plus (mean, rstd). The list you produce IS the activation budget; every byte-per-element coefficient in the literature is that list summed.

## Technical Details

**Reference config used throughout this course (do not drift):** `B=1, T=s=4096, h=4096, a=32, d_head=h/a=128, 4h=16384, bf16=2 B/elem, L=80 blocks`. Derived units:

| Unit | Shape | Elements | Bytes | Human |
|---|---|---|---|---|
| `HID` | `[B, T, h]` | 16,777,216 | 33,554,432 | **33.55 MB** |
| `ATTN` | `[B, a, T, T]` | 536,870,912 | 1,073,741,824 | **1.07 GB** |
| `MLP` | `[B, T, 4h]` | 67,108,864 | 134,217,728 | **134.22 MB** |
| `LNST` | `[B, T] × 2 fp32` (mean, rstd) | 8,192 | 32,768 | 32.8 KB |
| `sbh` | scalar unit | `s·b·h` = 16,777,216 | — | 1 `sbh` byte-unit = 16.78 MB |

- **The ordered 16-tensor list (pre-LN block, one micro-batch, no recompute).** `→` marks what backward actually consumes.

| # | Tensor | Shape | Bytes | Backward needs it because |
|---|---|---|---|---|
| 1 | `x_in` — block input hidden state | `[B,T,h]` | 33.55 MB | **this is the checkpoint**; LN1's VJP needs its input |
| 2 | LN1 saved stats (mean, rstd) | `[B,T]×2 fp32` | 32.8 KB | avoids recomputing the reduction |
| 3 | LN1 output | `[B,T,h]` | 33.55 MB | shared input of the Q, K, V GEMMs → stored **once**, not 3× |
| 4 | Q projection | `[B,T,h]` | 33.55 MB | `dK = dSᵀ Q` |
| 5 | K projection | `[B,T,h]` | 33.55 MB | `dQ = dS K` |
| 6 | V projection | `[B,T,h]` | 33.55 MB | `dP = dO Vᵀ` |
| 7 | **attention probs** `P = softmax(QKᵀ/√d)` | `[B,a,T,T]` | **1.07 GB** | softmax VJP is `dS = P ⊙ (dP − rowsum(dP⊙P))` — needs the **output**, not the scores |
| 8 | context `P·V` | `[B,a,T,d]` | 33.55 MB | input to `W_O` GEMM |
| 9 | attention output (after `W_O`) | `[B,T,h]` | 33.55 MB | residual-1 branch value |
| 10 | residual-1 sum | `[B,T,h]` | 33.55 MB | LN2's input |
| 11 | LN2 saved stats | `[B,T]×2 fp32` | 32.8 KB | same as #2 |
| 12 | LN2 output | `[B,T,h]` | 33.55 MB | input of the MLP up-projection GEMM |
| 13 | MLP up-projection output | `[B,T,4h]` | **134.22 MB** | GELU's VJP needs its input — **4× wide** |
| 14 | GELU output | `[B,T,4h]` | **134.22 MB** | input of the down-projection GEMM — **4× wide** |
| 15 | MLP down-projection output | `[B,T,h]` | 33.55 MB | residual-2 branch value |
| 16 | residual-2 sum = **block output** | `[B,T,h]` | 33.55 MB | *this is the NEXT block's checkpoint* |

- **The residual add stores zero bytes.** `∂(x+f(x))/∂x = I`, so `y = x + f(x)` needs no saved tensor — its backward just copies the incoming gradient down both branches. The residual stream is the one operation in the block that is *free* in memory. Everything expensive lives inside the two sub-layers hanging off it.
- **Arithmetic (verified, reproduce exactly):** discarded per block = items 2–15 = `9×HID + 2×LNST + ATTN + 2×MLP` = **1,644,232,704 B = 1.64 GB**. Checkpoint = 1×`HID` = **33.55 MB**. Ratio = **49.0×**. Block total (items 1–15) = **1,677,787,136 B ≈ 1.68 GB**, ratio to checkpoint **50.0×**. Over `L=80`: **134.22 GB** without checkpointing → `2·s·b·h·L` = **2,684,354,560 B = 2.68 GB** with per-block checkpointing (**50×**).
- **Mapping onto Korthikanti `sbh(34 + 5as/h)` — the exact decomposition** (arXiv:2205.05198 §4.1, [[selective-recompute-korthikanti]]):

```
attention block   = 11 sbh + 5 a s² b
   2 sbh  shared QKV input (stored ONCE for all three projections)
   4 sbh  Q and K, retained for the QKᵀ matmul
   2 sbh  V, retained for the P·V matmul
   2 sbh  input of the W_O linear projection
   1 sbh  attention-dropout mask (1 byte/elem, not 2)
   2 a s² b  softmax OUTPUT
   1 a s² b  softmax-dropout MASK (1 byte/elem)
   2 a s² b  softmax-dropout OUTPUT
MLP block         = 19 sbh
   2 sbh  input of up-projection
   8 sbh  input of GELU        (4h wide → 4×2 = 8)
   8 sbh  input of down-projection (4h wide → 8)
   1 sbh  MLP-dropout mask
LayerNorms        =  4 sbh   (2 sbh saved input × 2 norms)
──────────────────────────────────────────────
TOTAL = 34 sbh + 5 a s² b  =  s·b·h·(34 + 5as/h)
```

- **Numbers for the reference config:** `11 sbh = 184.55 MB`, `19 sbh = 318.77 MB`, `4 sbh = 67.11 MB`, `34 sbh = 570.43 MB`; `5as/h = 5·32·4096/4096 = 160`, so `160 sbh = 5as²b = 2,684,354,560 B = 2.68 GB`. Per layer `(34+160)·sbh = 194 × 16,777,216 = 3,254,779,904 B = 3.25 GB`; ×80 layers = **260.38 GB**.
- **Why 3.25 GB (Korthikanti) ≠ 1.68 GB (the 16-tensor list):** the enumerated list counts the s²-tensor **once** at 2 B/elem (`2as²b` = 1.07 GB — the softmax output only), while Korthikanti counts all three s²-shaped tensors (`5as²b` = 2.68 GB) because he includes the dropout mask and dropout output. Conversely the list counts 36 `sbh` of block *outputs* vs Korthikanti's 34 `sbh` of saved *inputs*. Both are correct accountings of the same block at different granularity; quote whichever, never mix them in one sum.
- **Modern LLMs train with `dropout = 0`, which edits the coefficient (derived, not quoted).** Korthikanti's 2022 accounting assumes dropout is on. Delete the three dropout tensors — attention-dropout mask (`1 sbh`), MLP-dropout mask (`1 sbh`), softmax-dropout mask (`1as²b`) and softmax-dropout output (`2as²b`) — and the block becomes `sbh(32 + 2as/h)`: attention `10 sbh`, MLP `18 sbh`, LayerNorms `4 sbh`, s²-term `2as²b`. **That `2as²b` is exactly the 1.07 GB single attention-probability tensor in the 16-tensor list** — which is why the enumerated (dropout-free) block and Korthikanti's (dropout-on) block differ mainly on the s² term. Per layer at the reference config: `(32+64)·sbh = 96 × 16,777,216 = 1,610,612,736 B = 1.61 GB` vs Korthikanti's 3.25 GB. Cite `34 + 5as/h` when quoting the paper; use `32 + 2as/h` when modelling a real dropout-free 2026 run, and say which you are doing.
- **⚠ Coincidence trap at this config:** `5as²b` (one layer's attention term) `= 2.68 GB` and `2·s·b·h·L` (the full-recompute floor for **all 80 layers**) `= 2.68 GB` are numerically identical, because `5·a·s = 5·32·4096 = 655,360 = 2·h·L = 2·4096·80`. They are unrelated quantities. Do not let an animation label imply otherwise.
- **The 4h expansion is where the MLP's bytes are.** `d_ff = 4h` is the GPT-1→GPT-3 constant (768→3072, 1600→6400, 12288→49152 — all exactly 4×, [[wiki:llm-arch gpt-architecture-diff]]). Two tensors of shape `[B,T,4h]` (up-proj output and GELU output) = 268.44 MB, which is **8× the hidden-state checkpoint** and 84% of the MLP block's 318.77 MB. Modern SwiGLU MLPs use `d_ff ≈ (8/3)h` but with **three** matrices, landing at a similar byte count.
- **Where the `[B,a,T,T]` monster comes from is the whole reason [[ch-04]] exists.** At T=4096 it is 1.07 GB/block; at T=8192 it is 4.29 GB/block (4×, quadratic). It is simultaneously the *largest* saved tensor and the *cheapest to reconstruct* (one matmul + one softmax) — which is precisely the asymmetry selective recomputation and FlashAttention exploit.
- **Training-memory angle:** This list *is* the activation bucket of the [[ch-01]] six-item ledger. Every activation-reduction technique in the course is a rule for editing this exact table: full checkpointing keeps row 1 and deletes rows 2–15 (33.55 MB survives, 1.64 GB dies, 49×); selective recomputation deletes only row 7 and the two dropout tensors (kills the whole `5as²b` term, leaves `34 sbh`); FlashAttention never *creates* row 7 at all (stores only per-row logsumexp, `O(s)` not `O(s²)`); sequence parallelism divides the `4 sbh` LayerNorm term (and the rest) by `t`. For the learner's 27B/256-expert MoE: the 256 experts multiply the **weight** bucket, not this table — with top-`k` routing only `k` experts' `[tokens, 4h]` intermediates are saved, so rows 13–14 scale with `k`, not 256. For GDN linear attention, row 7 does not exist: no `[B,a,T,T]` tensor is materialized, the `5as²b` term is zero, and the block's activation cost collapses to the linear `34 sbh` plus a fixed-size recurrent state.

## Citation
Derived and verified against: Vijay Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," MLSys 2023, arXiv:2205.05198 (§4.1 activation-memory derivation), https://arxiv.org/abs/2205.05198; Quentin Anthony et al., "Transformer Math 101," EleutherAI, 2023, https://blog.eleuther.ai/transformer-math/; and the verified tensor table in `wiki/courses/training-memory/ch-03/figures/checkpointing.html` (header block, lines 11–18). Cross-refs: [[selective-recompute-korthikanti]], [[transformer-math-101]], [[gradient-checkpointing-chen]].

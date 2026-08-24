# COLLECTION-PLAN — `training-memory`

Target coverage for the source library. Status: `TODO` until the researcher
agent writes the excerpt; flip to `DONE` when `excerpts/<slug>.md` exists.

## Cluster A — Ledger / precision / activations (feeds ch-01..03)

| slug | artifact | status |
|------|----------|--------|
| `transformer-math-101` | EleutherAI, "Transformer Math 101" (memory & FLOPs accounting) | DONE |
| `ultrascale-playbook` | HuggingFace, "The Ultra-Scale Playbook" (nanotron) — memory chapter | DONE |
| `ml-engineering-memory` | Stas Bekman, `stas00/ml-engineering` — memory usage anatomy | DONE |
| `mixed-precision-training` | Micikevicius et al. 2017, "Mixed Precision Training" (fp32 master, loss scaling) | DONE |
| `fp8-training` | NVIDIA Transformer Engine / FP8-LM (Peng et al. 2023) — fp8 training memory | DONE |
| `liger-fused-ce` | Liger-Kernel / chunked cross-entropy — the loss-head logit-spike fix | DONE |
| `gradient-checkpointing-chen` | Chen et al. 2016, "Training Deep Nets with Sublinear Memory Cost" | DONE |
| `selective-recompute-korthikanti` | Korthikanti et al. 2022, "Reducing Activation Recomputation" | DONE |

## Cluster B — Attention kernels (feeds ch-04..06)

| slug | artifact | status |
|------|----------|--------|
| `self-attention-no-n2-memory` | Rabe & Staats 2021, "Self-attention Does Not Need O(n^2) Memory" | DONE |
| `online-softmax` | Milakov & Gimelshein 2018, "Online normalizer calculation for softmax" | DONE |
| `flash-attention-1` | Dao et al. 2022, "FlashAttention" (IO-aware exact attention) | DONE |
| `flash-attention-2` | Dao 2023, "FlashAttention-2" (better parallelism / work partitioning) | DONE |
| `flash-attention-3` | Shah et al. 2024, "FlashAttention-3" (Hopper async, FP8) | DONE |
| `pytorch-sdpa` | PyTorch `scaled_dot_product_attention` docs — backends (math/mem-eff/flash/cudnn) | DONE |
| `xformers-mem-efficient` | xFormers `memory_efficient_attention` | DONE |
| `sage-attention` | Zhang et al. 2024, "SageAttention" (INT8 quantized attention) 1/2 | DONE |
| `ring-attention` | Liu et al. 2023, "Ring Attention with Blockwise Transformers" (context parallel) | DONE |
| `paged-attention` | Kwon et al. 2023, "vLLM / PagedAttention" — inference KV-cache (contrast, NOT training) | DONE |

## Cluster C — Parallelism / formulas / OOM (feeds ch-07..09)

| slug | artifact | status |
|------|----------|--------|
| `zero-memory-optimization` | Rajbhandari et al. 2019, "ZeRO" (optimizer/gradient/parameter partitioning) | DONE |
| `megatron-tp-sp` | Shoeybi et al. 2019 "Megatron-LM" (tensor) + Korthikanti 2022 (sequence parallel) | DONE |
| `pipeline-parallelism-1f1b` | Huang et al. 2018 "GPipe" + PipeDream/Megatron interleaved 1F1B | DONE |
| `pytorch-fsdp` | Zhao et al. 2023, "PyTorch FSDP" (FullyShardedDataParallel) | DONE |
| `deepspeed-moe-ep` | Rajbhandari/Lepikhin — DeepSpeed-MoE / GShard expert parallelism | DONE |
| `memory-calculator-notes` | Per-GPU memory formula assembly (full-FT ZeRO-3 vs LoRA), safety margin, phase-peak | DONE |
| `training-oom-failure-modes` | OOM anatomy + the estimate->smoke->read-OOM->lever debugging loop | DONE |

## Gap log

- fp8 requires Hopper+; document but note A100-40GB cannot use it (capstone constraint).
- PagedAttention is inference-time; include only to draw the train-vs-serve memory boundary.
- If SageAttention training-time (not just inference) evidence is thin, say so explicitly in the excerpt.

### Harvest of 2026-08-18 (`ch-extra` — attention from scratch)

**Poisoned upstream: do NOT copy numbers from the `llm-arch` branch.** The three
items below were each **re-verified with `python3` during this harvest**; every
`llm-arch` value listed is wrong and must be replaced with the corrected value
if a later chapter reaches for it.

| # | `llm-arch` location | What it prints | Correct value (re-verified here) |
|---|---------------------|----------------|----------------------------------|
| a | `llm-arch:wiki/courses/llm-arch/ch-02/read.md` §6 (table L357-362, prose L364) and `.../ch-02/excerpts/multi-head-redundancy.md` §1 (table L15-21, prose L23) | head-ablation table `h=1→25.8, 4→26.3, 8→25.8, 16→25.7, 32→24.7` — argmax at `h=4`. `read.md`: *"The sweet spot is around 4-16 heads."* Excerpt: *"going from 1 head to 4 heads improves BLEU by 0.5 points, but going from 4 to 8 provides no improvement."* Both pages also mis-cite it as **"Table 2"**. | Vaswani et al. **Table 3 row (A)** (not Table 2): `h=1 → PPL 5.29 / BLEU 24.9`, `h=4 → 5.00 / 25.5`, `h=8 (base) → 4.92 / 25.8`, `h=16 → 4.91 / 25.8`, `h=32 → 5.01 / 25.4`. Argmax is `h = 8–16` (tied at 25.8), not 4; `h=8` is the *best* row, not a plateau after 4. |
| b | `llm-arch:wiki/courses/llm-arch/ch-03/excerpts/sinusoidal-encoding-frequency-analysis.md` §2 | frequency rows labelled **one step off**, plus `λ ≈ 62,832` for dims `(510,511)` | `ω=0.1` (`10000^-0.25`) belongs to dims **(128,129)**, not (64,65); `ω=0.01` → **(256,257)**, not (128,129); `ω=0.001` → **(384,385)**, not (256,257). Dims `(510,511)` have `λ = 2π·10000^(510/512) = 60,611.5`. `62,832 = 2π·10⁴` is the **unreachable asymptote** (it needs `2i = 512`, but `2i` maxes at 510). |
| c | `llm-arch:wiki/courses/llm-arch/ch-03/read.md` §5 figure (L297 and L315) | `dim 128–129  λ = 56` and `dim 510–511  λ = 62,832` | `dims (128,129)`: `ω = 10000^(-128/512) = 0.1` → `λ = 2π/0.1 = **62.83**`. `dims (510,511)`: `λ = **60,611.5**`, same asymptote error as (b). The figure's other two rows (`dim 0–1 λ = 2π`, `dim 256–257 λ = 628`) are **correct** — so this figure is *not* the shifted table of (b), it is two isolated bad values. |

Verification notes recorded at harvest time (all reproduced with `python3`):

- (a) is caught by the paper's own prose. Vaswani writes *"while single-head
  attention is 0.9 BLEU worse than the best setting, quality also drops off with
  too many heads."* The `llm-arch` table gives `best − h=1 = 26.3 − 25.8 = 0.5`;
  the true row gives `25.8 − 24.9 = 0.9`. Only the corrected row is
  self-consistent with the paper's sentence. The `llm-arch` table also drops the
  PPL column entirely, which is where the `h=8` vs `h=16` ordering actually lives
  (`4.92` vs `4.91` — BLEU ties at `25.8`).
- (b) the labels are shifted by exactly one row: `ω_i = 10000^(-2i/d_model)` with
  `d_model = 512`, so `ω = 0.1 ⇒ 2i/d = 0.25 ⇒ 2i = 128`. Dims `(64,65)` actually
  carry `ω = 10000^-0.125 = 0.3162`, `λ = 19.9` — a row `llm-arch` never prints.
  The `(510,511)` error is `+2,220.4` positions (`3.66 %` high).
- (c) is **not** the shift of (b): the figure's `dim 256–257 λ = 628` row is
  right (true `628.32`), which a shifted table could not produce. `56` is a bare
  wrong value — inverting it gives `ω = 2π/56 = 0.1122 ⇒ 2i = 121.6`, not an even
  integer, so it corresponds to **no** dimension pair at `d_model = 512`. The
  figure's `510–511 λ = 62,832` row *is* the same asymptote error as (b).
- The arithmetic-intensity line in `excerpts/multi-head-split-concat-wo.md` was
  also corrected in this harvest: it said `d_head` FLOPs per stored **element**;
  under the multiply-add convention (2 FLOPs/MAC) `QKᵀ` costs `2·B·h·N²·d_head`
  FLOPs against `B·h·N²` stored elements, so it is **`2·d_head` FLOPs per stored
  element** and `d_head` FLOPs per stored **byte** in bf16. Units are now explicit
  in the excerpt.

**Boson-specific gap neither library can fill.** The `O(N²)` score-matrix material
throughout Cluster B describes the **softmax baseline only**. Boson / Lina TMR uses
**GDN linear attention**, which never forms an `N×N` softmax score matrix, so
`5·a·s²·b` and every `B·h·N²` figure are upper bounds on a path the model does not
take. Worse, the boson model's actual `d_model`, head count and head dim are **not
recorded anywhere in the wiki** — the excerpts cannot be specialised to it without
first capturing that config. Action: record the boson attention config (d_model,
`h`, `d_head`, GDN state size, CP=1 assertion) as a source file before any chapter
tries to instantiate the memory formulas for boson.

**Future crawl target — QK-norm.** The `√d_k` scaling in scaled dot-product
attention rests on a variance assumption (unit-variance, uncorrelated `q`/`k`
entries) that **decays during training** as `q`/`k` norms grow, which is what drives
attention-logit blowup. **QK-norm** (RMSNorm/LayerNorm applied to `Q` and `K` before
the dot product) is the modern fix, and it is **uncovered by any source in either
`training-memory` or `llm-arch`**. Crawl target: Henry et al. 2020 "Query-Key
Normalization"; Dehghani et al. 2023 "ViT-22B" (§ attention-logit growth); Chameleon
(Meta 2024) and Gemma-2/OLMo-2 QK-norm ablations.

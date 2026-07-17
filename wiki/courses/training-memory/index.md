<!-- title: training-memory course index
     scope: navigable chapter list for the GPU Memory in LLM Training course
     deps: [[outline.json]]
     see-also: [[wiki/raw-data/training-memory/README.md]]
-->

# GPU Memory in LLM Training: The Ledger, Attention Kernels, and Parallelism

A 9-chapter course on where every byte of GPU memory goes during LLM training,
why jobs OOM, and which lever fixes it. Ends with a capstone MoE budget.

Raw-data library: `wiki/raw-data/training-memory/` (21 excerpts)

---

## Phase 1 — The Memory Ledger (ch-01 to ch-03)

| Chapter | Title | One-line summary |
|---------|-------|-----------------|
| [[ch-01]] | The Memory Ledger: What Fills a GPU | Six named residents (weights/grads/Adam/activations/logit-spike/overhead), Rule of 16 (16 B/param), full-FT vs LoRA at the ledger level |
| [[ch-02]] | Optimizer States, Precision, and the Loss-Head Spike | Why Adam costs 12 B/param (fp32 master required), BF16 vs FP16, FP8 on Hopper, the B·T×V logit spike and Liger chunked-CE |
| [[ch-03]] | Activations and Gradient Checkpointing | O(L·s·b·h) + O(L·a·s²·b/h) activation formula, Chen 2016 √n checkpointing (+33% compute), Korthikanti 2022 selective recompute (5× saving, <4% overhead), sequence parallelism |

---

## Phase 2 — Attention and Memory (ch-04 to ch-06)

| Chapter | Title | One-line summary |
|---------|-------|-----------------|
| [[ch-04]] | Attention Is a Memory Problem: O(N²) and Why the Kernel Decides | N×N score matrix materializes as O(N²) activations; Rabe & Staats 2021 prove O(1)/O(√N) exact streaming is possible; online softmax is the enabling primitive |
| [[ch-05]] | FlashAttention (1/2/3): IO-Aware Exact Attention | FA1 tiles Q/K/V in SRAM, never writes N×N to HBM (O(N) memory); FA2 warp-per-row doubles MFU (225 TFLOPs A100); FA3 TMA async + FP8 reaches 740 TFLOPs H100 |
| [[ch-06]] | The Attention Kernel Zoo: SDPA, xFormers, SageAttention, Ring, Paged | SDPA silent MATH-fallback trap (38× slower, O(N²)); xFormers FMHA (O(N) broader coverage); SageAttention INT8 (inference only); Ring Attention CP O(L/D); PagedAttention (inference serving only) |

---

## Phase 3 — Scaling the Ledger (ch-07 to ch-08)

| Chapter | Title | One-line summary |
|---------|-------|-----------------|
| [[ch-07]] | Parallelism Taxonomy: DP / ZeRO / FSDP / TP / SP / PP / EP / CP | What each primitive shards; ZeRO-3 = 16Ψ/N; FSDP AllGather transient; TP+SP gives true t× activation saving; 1F1B vs GPipe activation memory; EP for MoE; world_size = TP×PP×CP×DP |
| [[ch-08]] | Memory Formulas, the Calculator, and the OOM Debugging Loop | Full per-GPU formula assembly; LoRA node-invariance (activations not divided by N); step-2 OOM trap (Adam states materialize at end of step 1); 6-step debugging loop; lever priority order |

---

## Phase 4 — Capstone (ch-09)

| Chapter | Title | One-line summary |
|---------|-------|-----------------|
| [[ch-09]] | Capstone: Modeling a 27B MoE Memory Budget End-to-End | 27B 256-expert MoE on A100-40GB: Rule of 16, EP required (ZeRO-3 expert gather OOMs), GDN linear attention hard-bans CP, fused-CE mandatory at V=248k S=32k (~30 GB spike unfused), final plan TP=4 EP=8 DP=8 PP=1 (256 GPUs, ~23 GB/GPU) |

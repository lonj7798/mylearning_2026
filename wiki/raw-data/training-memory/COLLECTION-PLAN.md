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

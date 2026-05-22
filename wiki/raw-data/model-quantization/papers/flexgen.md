<!-- scope: FlexGen — high-throughput single-GPU LLM inference via tri-level offloading + 4-bit weight/KV quant
     deps: [[zeroquant]], [[gptq]]
     see-also: [[kvquant-2023]], [[qlora]], [[atom]]
-->

# FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU
- **Core Insight:** For batch-throughput-oriented LLM inference on a single commodity GPU, the bottleneck is the memory hierarchy (GPU HBM → CPU DRAM → NVMe) not the GEMM throughput; you can keep a 175B-class model serving by *jointly* solving an LP for offloading policy and *aggressively quantizing* (W4 + KV4) to minimise data movement, achieving the first 1 token/s on OPT-175B with a 16 GB GPU.
- **Guideline:** For throughput-oriented (latency-tolerant) batched inference on small GPUs, layer FlexGen-style tri-level offloading on top of W4A16 GPTQ + INT4 KV cache; effective batch size matters more than per-token latency.
- **Authors:** Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Ré, Ion Stoica, Ce Zhang
- **Year:** 2023 (ICML 2023)
- **URL:** https://arxiv.org/abs/2303.06865
- **Relevant topics:** tri-level memory offloading, throughput-oriented inference, W4 + KV4 quant, LP-based scheduling

## Abstract
FlexGen serves a 175B-class LLM on a single 16 GB consumer GPU at 1 token/s — the first system to do so — by combining (i) an LP-based offloading scheduler over GPU/CPU/disk, (ii) per-tensor INT4 weight quantization (GPTQ-style), and (iii) per-token INT4 KV cache quantization. The scheduler solves a small linear program over per-tensor placement (HBM / DRAM / NVMe) and batching parameters to maximise tokens/s subject to memory and bandwidth constraints; with effective batch sizes up to 144, the slow memory tiers' bandwidth is fully amortised. FlexGen targets throughput-oriented workloads (eval generation, data labelling, backfill) where latency per token is irrelevant.

## Key Contributions
- **Tri-level offloading** over GPU HBM, CPU DRAM, and NVMe SSD, with an LP solver picking per-tensor residency.
- **Aggressive quantization stack** integrated with offloading: W4 (GPTQ-like) + KV4 (per-token absmax) + FP16 activations.
- **First 1 token/s on OPT-175B on a single 16 GB GPU**, with effective batch size 144 — orders of magnitude better than DeepSpeed-Inference or Hugging Face Accelerate at the same hardware budget.
- Open-source system that became the template for offloaded LLM inference research.

## Key Figures/Tables to Study
- **Figure 2:** the three-level memory hierarchy diagram with bandwidth annotations.
- **Algorithm 1:** the LP formulation — variables for per-layer placement and batch policy.
- **Table 3:** throughput on OPT-175B — FlexGen vs DeepSpeed vs HF Accelerate, ~100× speedup.

## Technical Details

### Offloading LP
Variables (per layer):
- `p_w ∈ [0, 1]^3`: fraction of weights on (GPU, CPU, disk), `Σ = 1`.
- `p_a ∈ [0, 1]^3`: same for activations.
- `p_kv ∈ [0, 1]^3`: same for KV cache.
- Effective batch `b` and microbatch schedule parameters.

Objective: maximise tokens/s = total_tokens / (compute_time + transfer_time), with transfer time given by per-tier bandwidths and per-layer placement.

Constraints: GPU HBM capacity, CPU DRAM capacity, total layer memory.

The LP is solved once at startup per model + hardware combo; the resulting per-layer policy is materialised into the runtime scheduler.

### Quantization stack
- **Weights**: per-channel INT4 (GPTQ-equivalent), group_size = 128.
- **KV cache**: per-token INT4 absmax (K and V independently). Dequant on the fly inside FlashAttention.
- **Activations**: FP16 (the quant-effort budget went to the memory-bound tensors).

### Throughput numbers
| Hardware | Model | Throughput |
|----------|-------|------------|
| 1× T4 16GB | OPT-175B | 1.12 token/s @ batch 144 |
| 1× T4 16GB | OPT-30B | 6.50 token/s |
| 1× A100 40GB | OPT-30B | (resident) full-speed |

The "1 token/s with batch 144" means effective throughput of ~144 tokens/s aggregate.

### Why offloading + quantization compound
Each level of memory has limited bandwidth:
- HBM ~1 TB/s (GPU)
- PCIe ~32 GB/s (CPU↔GPU)
- NVMe ~3 GB/s (disk)

INT4 weights halve the bytes moved over PCIe / NVMe per layer fetch → halve the bottleneck time → roughly 2× throughput on top of the offloading gain.

### Hyperparameters
| Knob | Value |
|------|-------|
| Weight bits | 4 (GPTQ-style) |
| KV bits | 4 (per-token absmax) |
| Activation bits | 16 (FP16) |
| Effective batch | up to 144 |
| Hardware | T4 16 GB / A6000 48 GB |

## Connections
- KV quant lineage seeded by FlexGen: [[kvquant-2023]], [[kivi]], [[kvquant]].
- Companion W4 PTQ algorithms: [[gptq]], [[awq]].
- Successor offloaded systems: HF Accelerate + bitsandbytes; vLLM CPU offload.
- Single-GPU 65B fine-tune cousin (W4 + LoRA, but training-side): [[qlora]].
- Throughput-vs-latency contrast: [[atom]] (W4A4 + KV4 for *both* throughput and latency).

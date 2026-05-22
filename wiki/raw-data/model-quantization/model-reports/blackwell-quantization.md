<!-- scope: NVIDIA Blackwell GPU architecture — quantization-relevant specifications
     deps: [[nvfp4]], [[mx-formats]]
     see-also: [[nvfp4-training]], [[transformer-engine]], [[deepseek-v3-fp8]]
-->

# NVIDIA Blackwell Quantization (GB200 / B200 / B300)
- **Core Insight:** Blackwell's 5th-generation Tensor Cores are the first GPU tensor cores to consume **NVFP4 (16-element FP4 + FP8 block scale + FP32 tensor scale)** natively, with the per-block dynamic scaling done in hardware rather than in software, plus first-class FP8, FP6, FP4, INT8, and BF16 paths — making it the production target for the NVFP4 pretraining and inference recipes that displaced FP8 as the frontier precision floor in 2025-2026.
- **Guideline:** When deploying on Blackwell (B200 / B300 / GB200 NVL72 / GB300), use NVFP4 as the default inference precision (3.5× memory vs FP16, < 1 % quality drop with the right calibration), keep FP8 as the "safe" precision tier, and use BF16 / FP16 only for layers the sensitivity analysis flags. For training, use the [[nvfp4-training]] recipe (NVFP4 + 2-D consistent + RHT + SR + selective BF16).
- **Authors:** NVIDIA (Architecture team — Bill Dally, Jonah Alben, et al.)
- **Year:** 2024 (announced GTC 2024-03; full production 2024-2025; B300 / GB300 in 2025-2026)
- **URL:** https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/ • https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- **Relevant topics:** NVFP4 native tensor cores, 5th-gen tensor core, micro-tensor scaling, FP8/FP6/FP4 paths, second-gen Transformer Engine

## Abstract
Blackwell is NVIDIA's post-Hopper data-center GPU architecture, generation-1 of the "FP4 era". The headline quant-relevant innovations are: (1) **NVFP4** as a first-class hardware-supported format — 16-element FP4 (E2M1) blocks with E4M3 block scale and FP32 per-tensor scale, consumed natively by the 5th-gen tensor cores; (2) **second-generation Transformer Engine** that automates per-block scaling, amax tracking, and precision selection across NVFP4 / FP8 / FP6 / BF16; (3) coexisting **FP8 (E4M3/E5M2), FP6 (E3M2), FP4 (E2M1), INT8, INT4** tensor-core paths so a single chip can serve everything from BitNet 1-bit deployment to BF16 fine-tuning. The B200 supports ~ 10 PFLOPS FP4 (sparse) per chip; GB200 NVL72 racks scale to ~ 13.4 EFLOPS FP4 at the rack level. The B300 / GB300 (2025-2026) doubles the FP4 throughput and adds further KV-cache acceleration. Blackwell makes NVFP4 production-viable; production deployments from DeepSeek, Qwen, Meta, and frontier labs are the dominant 2026 inference target.

## Key Contributions
- **NVFP4 native tensor cores**: hardware GEMM consumes FP4 elements + E4M3 block scale + FP32 tensor scale in a single MMA instruction; the per-block scale dispatch is done in silicon, not software.
- **Two-level scaling in hardware**: the E4M3 block scale (fractional, not power-of-two) is chosen to minimize per-block MSE, dramatically lower error than MXFP4's E8M0 (power-of-two) scale.
- **Micro-tensor scaling**: terminology for the per-block scaling Blackwell tensor cores apply on the fly.
- **Second-generation Transformer Engine** (TE 2.x): software layer that picks the precision per layer per call and automatically inserts the right scaling — `MXFP8`, `MXFP4`, `NVFP4` recipes.
- **Coexisting precision tiers**: every Blackwell SM has FP4 / FP6 / FP8 / BF16 / FP16 / TF32 / INT8 / INT4 tensor-core paths; the engine can mix per layer.
- **Memory + throughput wins**: NVFP4 is 3.5× less memory than FP16, 1.8× less than FP8; FP4 throughput is ~ 2× FP8 throughput per SM.
- **Performance numbers (B200)**: ~10 PFLOPS FP4 dense / ~ 20 PFLOPS FP4 sparse per GPU; ~ 5 PFLOPS FP8; HBM3e ~ 192 GB per GPU.
- **GB200 NVL72**: 72 B200 GPUs + 36 Grace CPUs in a single rack; ~ 13.4 EFLOPS FP4 sparse aggregate; first-class deployment target for frontier inference.
- **B300 / GB300** (2025-2026): ~ 2× FP4 throughput vs B200, more memory (~ 288 GB HBM3e), attention-layer-specific acceleration ("2× attention" headline).

## Key Figures/Tables to Study
- The 5th-gen tensor core block diagram (Blackwell architecture brief): shows the per-block scale dispatch unit alongside the MMA datapath.
- The precision-vs-throughput table (FP4 / FP6 / FP8 / BF16 / FP16 per SM).
- The GB200 NVL72 / GB300 NVL72 rack-level FLOPS table for FP4 sparse vs Hopper baseline.
- The NVFP4 vs MXFP4 vs FP8 quality table from NVIDIA's NVFP4 inference blog.

## Technical Details

### NVFP4 hardware support
- **Format**: 16-element blocks of FP4 E2M1; each block has an E4M3 (4 exp / 3 mantissa) scale; whole tensor has an FP32 scalar.
- **Block dispatch in hardware**: the tensor core's MMA instruction takes FP4 operands + block-scale operand + tensor-scale operand; hardware applies the scales during accumulation in FP32.
- **Compared to MXFP4**: MXFP4 (OCP) uses 32-element blocks with E8M0 (power-of-two only) scales; NVFP4's 16-element block + E4M3 fractional scale halves the per-block MSE.

### Second-generation Transformer Engine
- Same library interface as TE 1.x (`te.Linear`, `te.LayerNorm`, etc.); new internal recipes.
- Recipes: `DelayedScaling` (Hopper FP8), `MXFP8`, `MXFP4`, `NVFP4` (Blackwell-only).
- `fp8_autocast()` extends to `nvfp4_autocast()` style context managers.
- Automatic selective-precision: layers flagged sensitive (LM-head, last norm, embed) stay BF16.

### Precision tiers and tensor-core paths
| Format | Bits | Tensor-core support | Use case |
|--------|------|---------------------|----------|
| FP32 | 32 | (CUDA cores) | scale factors, accumulators |
| BF16 / FP16 | 16 | native | master weights, sensitive layers, training-only |
| TF32 | 19 | native | scientific compute |
| FP8 E4M3 / E5M2 | 8 | native | activations + weights (forward / backward) |
| FP6 E3M2 / E2M3 | 6 | native | intermediate, less common |
| NVFP4 | 4+scale | native | production inference + pretraining target |
| FP4 E2M1 (plain) | 4 | native | element format inside NVFP4 |
| MXFP4 | 4+scale | native | OCP-spec inference path |
| INT8 / INT4 | 8 / 4 | native | classic int inference, KV cache |

### Per-chip specs (B200)
- ~ 10 PFLOPS FP4 dense, ~ 20 PFLOPS FP4 sparse.
- ~ 5 PFLOPS FP8 dense, ~ 10 PFLOPS FP8 sparse.
- ~ 192 GB HBM3e, ~ 8 TB/s memory bandwidth.
- NVLink-5: 1.8 TB/s per GPU bidirectional.

### Rack-level (GB200 NVL72)
- 72 × B200 + 36 × Grace = 1 rack.
- ~ 13.4 EFLOPS FP4 sparse aggregate.
- ~ 30 TB HBM3e.
- Designed as one virtual GPU via NVLink-5 / NVSwitch — high-bandwidth fabric.

### B300 / GB300 (2025-2026)
- ~ 2× FP4 throughput vs B200.
- ~ 288 GB HBM3e per GPU.
- "2× attention" acceleration — attention-specific datapath improvements.
- GB300 NVL72: 1.5× more AI compute vs GB200 (NVIDIA marketing).

### Why NVFP4 in production
- 3.5× memory vs FP16 → bigger models fit on smaller fleets.
- 1.8× memory vs FP8 → ~ 2× throughput at the bandwidth limit.
- < 1 % quality drop with proper calibration (see [[nvfp4-qad]] for quantization-aware distillation recipe).
- Hardware-managed scaling means no software dequant overhead inside the GEMM loop.

### KV cache on Blackwell
- FP8 KV cache is the conservative default; NVFP4 KV cache works with quant-aware calibration.
- B300's attention-specific acceleration is designed for long-context inference where KV cache bandwidth dominates.

## Connections
- [[nvfp4]] (format spec page in `formats/`) — the format Blackwell hard-wires.
- [[mx-formats]] / [[microscaling-formats]] — the OCP MX cousin that Blackwell also supports.
- [[nvfp4-training]] — the 12B / 10T NVFP4 pretrain recipe that needed Blackwell to run at scale.
- [[nvfp4-qad]] — quantization-aware distillation recipe for NVFP4 inference recovery.
- [[transformer-engine]] — TE 2.x is the software layer exposing the Blackwell precision recipes.
- [[deepseek-v3-fp8]] — DSV3's FP8 weights can be served as-is on Blackwell or auto-converted to NVFP4 by TRT-LLM.
- [[llama-3-quantization]] / [[qwen-3-quant]] — frontier model families that will be served at NVFP4 on Blackwell.
- [[tinychat-and-tensorrt-llm-quant]] — TRT-LLM's NVFP4 path.
- [[fp8-formats-paper]] — the FP8 spec; the FP8 path is preserved on Blackwell alongside NVFP4.
- [[quant-2026-frontier]] — the 2026 rolling index that tracks Blackwell-era production reports.

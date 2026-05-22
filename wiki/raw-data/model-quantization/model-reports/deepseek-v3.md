<!-- scope: DeepSeek-V3 model report — quantization angle (FP8 native training + W4 serving)
     deps: [[deepseek-v3-fp8]], [[fp8-formats-paper]]
     see-also: [[transformer-engine]], [[llama-3-quantization]], [[deepseek-r1-quantization]]
-->

# DeepSeek-V3 (Quantization Angle)
- **Core Insight:** DeepSeek-V3 is the first frontier-scale model both **trained in FP8 natively** (per-block scaling + FP32 promotion every 4 WGMMA — see [[deepseek-v3-fp8]]) and **deployed at W4 / W8 / FP8** across its serving stack; the full quant story for the model spans training, inference, and the MoE-specific FP8 communication path.
- **Guideline:** When studying DeepSeek-V3 quantization, treat training and serving as separate but coordinated decisions — the FP8 training recipe directly enables the FP8 / W4 serving recipe by keeping weight outliers predictable and limited.
- **Authors:** DeepSeek-AI (Liang et al.)
- **Year:** 2024 (released 2024-12-26)
- **URL:** https://arxiv.org/abs/2412.19437 • https://huggingface.co/deepseek-ai/DeepSeek-V3
- **Relevant topics:** FP8 native training, FP8 inference, W4 serving, MoE FP8 all-to-all, per-block scaling

## Abstract
DeepSeek-V3 is the 671B-total / 37B-active MoE that the broader DSV3 model report (`/llm-training/model-reports/deepseek-v3.md`) covers in full. From the quantization angle, three things are notable: (1) pretraining ran **natively in FP8** with per-block scaling — the first frontier-scale FP8 training, documented in §3.3 (see [[deepseek-v3-fp8]] for full details); (2) the released checkpoints ship in FP8 form (`deepseek-ai/DeepSeek-V3` is the FP8 build, no BF16 release at frontier size); (3) community W4 serving (via vLLM, SGLang) and Meta-style FP8 row-wise serving both work directly on the released FP8 weights with minimal additional calibration, because the per-block FP8 distribution is already quant-friendly. The model is a watershed case study for the "train in FP8, serve in W4-or-FP8" pipeline that the rest of the industry is now adopting.

## Key Contributions (quant-specific)
- **First frontier-scale FP8-native pretrain** with documented per-block scaling, per-channel scaling, and FP32 partial-sum promotion (see [[deepseek-v3-fp8]] for the algorithmic details).
- **FP8 checkpoint as the canonical release**: `deepseek-ai/DeepSeek-V3` is shipped at FP8 (E4M3 weights with per-block scales), not BF16; downstream finetunes either upcast to BF16 or stay in FP8.
- **Community W4 ports**: AWQ / GPTQ W4 versions calibrated from the FP8 base; minimal extra quant work because the FP8 weights are already smoothly distributed.
- **MoE FP8 all-to-all**: the expert-parallel dispatch sends activations in FP8, halving the all-to-all bandwidth — a quant-aware comm optimization.
- **Inference deployment**: SGLang / vLLM serve the FP8 build natively on H100/H200; throughput numbers from the report show ~ 2× FP16 baseline on the same hardware.
- **R1 derivatives**: DeepSeek-R1 inherits the FP8 base; its quant story is in [[deepseek-r1-quantization]].

## Key Figures/Tables to Study
- DSV3 §3.3 figures (covered in [[deepseek-v3-fp8]]): FP8 framework diagram, GEMM accumulation diagram, calibration ablation table.
- The cost table (§ training cost): 2.788M H800 hours — most of which is FP8 pretraining.
- The MoE FP8 all-to-all diagram (§3.2 / §3.4).
- Community W4 vLLM benchmark tables (off-paper).

## Technical Details

### Training quantization (summary, see [[deepseek-v3-fp8]] for full)
- E4M3 everywhere (forward + backward).
- 1×128 activation tile, 128×128 weight block per-block scales, online.
- BF16 master weights, FP32 first moment (Adam), BF16 second moment.
- FP32 promotion every 4 WGMMA inside H800 tensor cores.
- < 0.25 % relative loss gap vs BF16 baseline.

### Serving quantization
- **FP8 (canonical)**: the released checkpoint is FP8; serving frameworks consume it directly. H100 / H200 / B200 native FP8 tensor cores.
- **W4 (community)**: AWQ / GPTQ checkpoints maintained by the community; group_size=128; served via vLLM with Marlin (Ampere) / Machete (Hopper).
- **FP8 KV cache**: optional via vLLM / SGLang; some perf wins, no public detailed eval from DeepSeek itself.
- **MoE-aware quant**: routing gate stays BF16/FP32 (cannot be quantized — gates need to express tight score gaps); only the expert linears go to FP8 / W4.

### Storage and bandwidth
- FP8 671B = 671 GB → fits in 1×H100 8-GPU node (8 × 80 = 640 GB is *just* tight; in practice ~ 16-GPU 2-node deployment with TP / EP).
- BF16 equivalent would be 1.34 TB → 2× hardware.

### Hardware
- Trained on H800 (the China-specific H100 variant with reduced NVLink BW).
- Released for H100 / H200 / B200 inference; Blackwell can run the FP8 weights natively or convert to NVFP4 with TRT-LLM.

## Connections
- [[deepseek-v3-fp8]] — the FP8 *training* recipe in full (this is the deep-dive companion to this report).
- [[fp8-formats-paper]] — the E4M3 spec the training and serving stack uses.
- [[transformer-engine]] — NVIDIA's FP8 library; DSV3 used a custom variant that beat TE at frontier scale.
- [[llama-3-quantization]] — the dense-frontier-model counterpart; Llama 3 trained in BF16 then post-training-quantized to FP8 for serving; DSV3 went FP8-native.
- [[deepseek-r1-quantization]] — R1 inherits the FP8 base and adds reasoning-specific post-training quant choices.
- [[blackwell-quantization]] — DSV3 FP8 weights can be served on Blackwell as FP8 or auto-converted to NVFP4.
- [[deepseek-quant]] — lab summary page for the broader DeepSeek quantization lineage.

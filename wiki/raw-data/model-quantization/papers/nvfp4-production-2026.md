<!-- scope: production NVFP4 deployment reports from Blackwell-era 2026 — placeholder
     deps: [[nvfp4-training]], [[blackwell-quantization]]
     see-also: [[quant-2026-frontier]], [[transformer-engine]]
-->

# Production NVFP4 Reports (2026) — Placeholder
- **Core Insight:** Once the Blackwell B200 / B300 / GB200 fleet is at scale, NVFP4 ceases to be a research finding and becomes a deployment recipe: native NVFP4 tensor cores, automatic NVFP4 weight quantization at engine-build time, FP8 KV cache, and selective high-precision layers stay as a tight 5-recipe block applied across all major frontier-lab serving stacks (TensorRT-LLM, vLLM, SGLang).
- **Guideline:** Look for production NVFP4 reports from (a) NVIDIA's own MLPerf submissions, (b) DeepSeek / Qwen / Mistral inference releases with NVFP4 model cards, and (c) cloud serving providers (Together, Fireworks, Lambda, AWS Trainium-NVFP4-equivalent) when filling this placeholder.
- **Authors:** placeholder — to be populated as 2026 reports settle
- **Year:** 2026
- **URL:** see Connections
- **Relevant topics:** NVFP4 production deployment, Blackwell, TensorRT-LLM NVFP4 path, MLPerf

## Abstract
This is a placeholder for 2026 production NVFP4 deployment reports. As of the library freeze date, the canonical sources are the NVFP4 pretraining paper ([[nvfp4-training]]) and the Blackwell architecture marketing ([[blackwell-quantization]]); the per-frontier-lab production reports are not yet consolidated into a single citable artifact. Production reports typically include: end-to-end latency / throughput numbers per model size, the per-layer precision-mix decision (which layers stay BF16), the FP8 vs NVFP4 KV cache tradeoff, and the actual eval delta against an FP16 baseline at the same hardware.

## Key Contributions (to be populated)
- First per-lab Blackwell + NVFP4 serving recipes from DeepSeek / Qwen / Mistral / Anthropic / OpenAI (whichever disclose).
- NVFP4 MoE deployment: per-expert precision picks, NVFP4 routing decisions, NVFP4 all-to-all dispatch.
- Comparison tables: NVFP4 vs FP8 vs W4A16 throughput / latency / quality on the same hardware.

## Key Figures/Tables to Study
- (placeholder — populate when canonical 2026 production report exists)

## Technical Details
**Gap log entry**: no single canonical production-NVFP4 paper at library freeze; closest is the NVFP4 pretraining paper, which is research-focused. NVIDIA's MLPerf inference 2026 submission and Blackwell-era model cards from frontier labs will populate this in the next pass.

Expected production-NVFP4 recipe stub:
- Weight format: NVFP4 (16-element FP4 blocks + E4M3 scale + FP32 tensor scale).
- Activation: NVFP4 (online quant per token).
- KV cache: FP8 per-channel K, FP8 per-token V (or NVFP4 if quality holds).
- Selective high-precision: embed, head, final norm in BF16.
- Engine: TensorRT-LLM with NVFP4 path on Blackwell SM 10.x.

## Connections
- [[nvfp4-training]] — the research reference whose recipe productionized here.
- [[blackwell-quantization]] — hardware report; production NVFP4 depends on the 5th-gen tensor cores.
- [[transformer-engine]] — TE 2.x ships the NVFP4 recipe used in production.
- [[tinychat-and-tensorrt-llm-quant]] — the TRT-LLM NVFP4 path documentation.
- [[quant-2026-frontier]] — parent placeholder index for 2026.
- [[mx-formats]] / [[microscaling-formats]] — the OCP cousin format family.

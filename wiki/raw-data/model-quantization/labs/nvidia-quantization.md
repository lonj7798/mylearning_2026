<!-- scope: NVIDIA's quantization-research and runtime stack — FP8 / FP4 formats, Transformer Engine, TensorRT-LLM, NVFP4 / Blackwell
     deps: [[fp8-formats-paper]], [[transformer-engine]], [[nvfp4]]
     see-also: [[tensorrt-llm-quant]], [[marlin-kernel]]
-->

# NVIDIA Quantization — Hardware-Owned FP8 / FP4 + Production Runtime
- **Core Insight:** NVIDIA drives quantization by co-designing formats, tensor-core instructions, and runtime stacks, making FP8/FP4 a hardware-software contract rather than just an algorithm choice.
- **Guideline:** Use this lab track for Hopper FP8, Blackwell NVFP4, Transformer Engine, TensorRT-LLM, and ModelOpt deployment paths.
- **Authors:** NVIDIA research, architecture, and runtime teams
- **Year:** 2022–2026
- **URL:** https://github.com/NVIDIA/TransformerEngine ; https://github.com/NVIDIA/TensorRT-LLM ; https://github.com/NVIDIA/TensorRT-Model-Optimizer
- **Relevant topics:** FP8, NVFP4, MXFP4, Transformer Engine, TensorRT-LLM, ModelOpt, Blackwell

## Summary
NVIDIA's quantization work is **format-and-runtime first**: rather than producing standalone PTQ algorithms, the team has driven the existence of FP8 (E4M3 / E5M2), FP6, and FP4 (E2M1) **as first-class hardware formats** on H100 (Hopper) and B100 / B200 (Blackwell), and shipped the **Transformer Engine** + **TensorRT-LLM** runtime stacks that make these formats usable in mixed-precision training and inference. The joint Micikevicius-led 2022 "FP8 Formats for Deep Learning" paper (NVIDIA / Arm / Intel) is the spec the entire industry built on, and the 2024 OCP Microscaling spec (MX) — co-authored by NVIDIA — extended this to block-scaled FP6 / FP4 / INT8.

## Notable Works (NVIDIA-led or NVIDIA-anchored)
- [[fp8-formats-paper]] (Micikevicius 2022) — NVIDIA / Arm / Intel joint FP8 spec; the foundation for H100 tensor cores.
- [[transformer-engine]] — NVIDIA's open-source FP8 mixed-precision training library; per-tensor delayed scaling + amax history + E4M3-fwd / E5M2-bwd split.
- [[microscaling-formats]] (Rouhani 2023, NVIDIA + Microsoft + AMD + Intel + Arm + Meta + Qualcomm) — OCP MX spec; shared block exponent + low-bit element formats.
- [[nvfp4]] — NVIDIA-proprietary 4-bit format on Blackwell (FP8 block scale + FP32 tensor scale); native tensor-core acceleration.
- TensorRT-LLM ([[tensorrt-llm-quant]]) — the production inference runtime; integrates GPTQ, AWQ, FP8, SmoothQuant, and now NVFP4.
- Marlin / Machete kernels ([[marlin-kernel]], [[machete-kernel]]) — IST Austria + vLLM authored, but tightly co-designed with H100 / Blackwell tensor-core specifics.
- ModelOpt (NVIDIA's quant + sparsity + distillation toolkit) — the supported path from a HF model to a TensorRT-LLM engine.

## Recurring themes
- **Format ownership through silicon**: H100 FP8, B100 FP4, NVFP4 — NVIDIA is willing to add tensor-core instructions for new formats, which forces the rest of the ecosystem to follow.
- **Mixed precision is the default**: NVIDIA doesn't push "everything in FP4" — they push "FP4 forward, FP8 backward, FP32 accumulate." Transformer Engine encodes this discipline.
- **Toolchain integration**: every format gets a Transformer Engine + TensorRT-LLM + ModelOpt path; no algorithm-level work ships without a corresponding runtime story.

## Open Resources
- Transformer Engine: https://github.com/NVIDIA/TransformerEngine
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- ModelOpt: https://github.com/NVIDIA/TensorRT-Model-Optimizer
- Megatron-LM (FP8 mid-training): https://github.com/NVIDIA/Megatron-LM
- NVFP4 / Blackwell whitepaper: https://resources.nvidia.com/en-us-tensor-core (Blackwell tensor-core technical brief)

## Connections
- [[microsoft-bitnet]] — Microsoft co-authored MX spec; on sub-4-bit formats the two labs collaborate.
- [[frantar-alistarh-ist-austria]] — Marlin / Machete kernels run on NVIDIA silicon; deep co-design partnership.
- [[han-song-mit]] — AWQ + TinyChat → TensorRT-LLM integration is one of the main paths AWQ runs in production.
- [[intel-quantization]] — adjacent on MX spec; competitor on AMX (Intel's INT8 / BF16 / FP8 instruction family).
- [[deepseek-quant]] — DeepSeek V3 FP8 training validated the NVIDIA FP8 stack at frontier scale.

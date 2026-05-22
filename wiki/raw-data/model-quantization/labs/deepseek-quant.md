<!-- scope: DeepSeek's quantization line — FP8 native pretraining at frontier scale, R1 distill-and-quantize releases
     deps: [[deepseek-v3-fp8]]
     see-also: [[transformer-engine]], [[fp8-formats-paper]]
-->

# DeepSeek Quantization — FP8 Native Pretraining at Frontier Scale
- **Core Insight:** DeepSeek made low-precision training a frontier-scale result by training V3 with native FP8 and then distributing reasoning capability through distilled, quantized releases.
- **Guideline:** Use this lab track to connect FP8 training recipes with downstream distill-and-quantize deployment patterns.
- **Authors:** DeepSeek AI
- **Year:** 2024–2026
- **URL:** https://huggingface.co/deepseek-ai ; https://arxiv.org/abs/2412.19437
- **Relevant topics:** DeepSeek V3, DeepSeek R1, FP8 native training, per-block scaling, distillation, GGUF/AWQ/GPTQ releases

## Summary
DeepSeek (Hangzhou-based AI lab) operates one of the largest research-engineering teams pushing **quantization into the pretraining loop, not just inference**. The watershed contribution is [[deepseek-v3-fp8]] — the first frontier-scale (671B-parameter MoE, ~37B activated) **native FP8 training run** with per-block scaling, fine-grained accumulation, and BF16 master weights. This proved at scale what the FP8 specs ([[fp8-formats-paper]]) and [[transformer-engine]] had been promising in smaller settings: FP8 training is loss-stable and ~2× faster than BF16 without quality sacrifice. The R1 reasoning model line then established the **distill-and-quantize** release pattern — first distill into smaller dense models, then quantize each to W4 / GGUF for community deployment.

## Notable Works
- [[deepseek-v3-fp8]] — DeepSeek V3 technical report §3.3 (the FP8 mixed-precision training recipe).
- DeepSeek V3 / V3.1 / V3.2 — successive deployments of the FP8-trained base model; V3.2 introduced sparse attention with FP8 still as the storage format.
- DeepSeek R1 — pure-RL-reasoning model; the R1-distill family (Qwen-based, Llama-based) are typically released alongside official W4 / Q4_K_M gguf variants.
- DeepSeek R1-Zero — distinct from R1; the "no-SFT" RL training run that demonstrated emergent reasoning.

## Recurring themes
- **Quantization is a training problem, not just inference**: DSV3 is the practical proof that FP8 is not a post-training cast but a first-class pretraining precision; this set the agenda for the 2025-2026 NVFP4 / MXFP4 pretraining literature.
- **Distill-then-quantize for distribution**: R1's reasoning capability is shared with the community via small distilled models in GGUF / AWQ / GPTQ form, putting the quant releases on equal footing with the model releases.
- **Hardware-aware accumulation**: the DSV3 trick of accumulating in FP32 every 4 WGMMA instructions specifically targets H800 / H100 numerical limits; this is hardware-conscious design at the math level.

## Open Resources
- DeepSeek HF org: https://huggingface.co/deepseek-ai
- DeepSeek V3 report: https://arxiv.org/abs/2412.19437
- DeepSeek R1 report: https://arxiv.org/abs/2501.12948
- DeepSeek GitHub: https://github.com/deepseek-ai

## Connections
- [[nvidia-quantization]] — DeepSeek's FP8 training validates the Transformer Engine / Hopper stack at frontier scale.
- [[microsoft-bitnet]] — different end of the spectrum (1.58-bit weight, BF16 activation) vs DeepSeek's (FP8 weight + activation); both push training-time low-precision.
- [[dettmers-group]] / [[han-song-mit]] / [[frantar-alistarh-ist-austria]] — adjacent PTQ labs; DeepSeek's contribution is upstream of their work (training-time) rather than alternative.
- [[transformer-engine]] / [[fp8-formats-paper]] — the format + framework the DSV3 recipe rides on.

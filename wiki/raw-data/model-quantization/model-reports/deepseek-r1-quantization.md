<!-- scope: DeepSeek-R1 reasoning-model quantization releases
     deps: [[deepseek-v3-fp8]], [[deepseek-v3]]
     see-also: [[llama-3-quantization]], [[qwen-3-quant]]
-->

# DeepSeek-R1 Quantization
- **Core Insight:** R1 inherits DSV3's FP8 base, but reasoning-model quantization is more sensitive than chat-model quantization — long chains of thought magnify per-token quantization errors, so the R1 quant releases keep the FP8 base unchanged and add only conservative W4 community ports with KV cache held at FP8/FP16.
- **Guideline:** When deploying R1 (or any long-CoT reasoning model), be more conservative on quantization than for chat models: keep KV cache at ≥ FP8, prefer W4A16 over W4A8, and validate on reasoning evals (AIME, MATH-500, GPQA) not just MMLU — chat-model perplexity is a poor proxy for reasoning quality under quant.
- **Authors:** DeepSeek-AI (R1 = V3 + RL stage)
- **Year:** 2025 (R1 released 2025-01-20)
- **URL:** https://arxiv.org/abs/2501.12948 • https://huggingface.co/deepseek-ai/DeepSeek-R1
- **Relevant topics:** reasoning model quantization, long-CoT quant sensitivity, FP8 base, KV cache for reasoning

## Abstract
DeepSeek-R1 is the GRPO-trained reasoning descendant of DeepSeek-V3 — same MoE architecture, same FP8 base weights, additional RL stage focused on reasoning. The quantization releases follow DSV3's pattern: the canonical checkpoint is FP8 (inherited from V3's training format) and community W4 ports exist for prosumer hardware. The notable observation, surfaced across the community evals and DeepSeek's own deployment notes, is that *reasoning models are quant-sensitive in a way chat models are not*: a small per-token quant error that adds 0.01 to chat perplexity can compound across a 4096-token chain of thought to drop AIME pass@1 by several points. The practical consequence: R1 quant releases hold the line at FP8 base + FP8/FP16 KV cache and use W4A16 (not the more aggressive W4A4 / W4A8) for the cheaper deployment tier.

## Key Contributions
- **FP8 canonical release**: `deepseek-ai/DeepSeek-R1` is the FP8 build (E4M3 weights with per-block scales, inherited from V3 training).
- **R1-Distill series**: smaller dense distilled models (Qwen / Llama bases, 1.5B-70B) are released in BF16 and quantized aggressively by the community; the quality-vs-bit-budget curve is steeper for the distilled-reasoning models than for chat-only counterparts.
- **Quant-sensitivity observation**: community studies show that the same W4 recipe that loses ~ 0.5 MMLU pt on Llama loses 3-5 AIME pt on R1.
- **KV cache held at higher precision**: most R1 serving stacks keep KV at FP8/FP16 even when weights are W4 — reasoning depends on accurate attention over long CoT.
- **Hardware deployment**: same as V3 — H100 / H200 / B200, FP8 native; AWQ W4 community ports run on Ampere via Marlin.

## Key Figures/Tables to Study
- R1 paper benchmark tables (AIME, MATH-500, GPQA, Codeforces) — these are the evals that surface quant degradation.
- Community quant-eval tables for R1 W4 vs FP8 vs FP16 on the same reasoning benchmarks.
- R1-Distill model-size-vs-eval curves; useful for picking the right quant target.

## Technical Details

### Why reasoning models are quant-sensitive
- Reasoning trace = autoregressive generation of L tokens (L can be 1K-10K for hard problems).
- Per-token quantization error compounds in two ways:
  1. Each token's logits are perturbed; the wrong-token-pick probability per step grows linearly with quant noise.
  2. Wrong-token picks compound multiplicatively across the trace; one wrong intermediate step can derail the whole proof.
- Chat models stop at ~ 50-200 tokens and the error doesn't compound far; reasoning models go to thousands.

### R1 quant recipe (community + DeepSeek deployment)
- **Base**: FP8 (inherited from V3); no separate R1 post-training quant.
- **W4 community**: AWQ / GPTQ group_size=128; quality drop visible on AIME / GPQA but tolerable for low-priority queries.
- **KV cache**: FP8 or FP16; INT4 KV is not recommended for R1 (too much attention precision loss for long CoT).
- **Activations**: FP8 in the FP8 serving path; FP16 in the W4A16 community path.
- **Selective precision**: same as V3 — embeddings, head, final norm, routing gate at BF16/FP32.

### R1-Distill quantization
- The distilled models (`DeepSeek-R1-Distill-Llama-70B`, `-Qwen-32B`, etc.) are BF16 dense.
- Community W4 / W8 quant works but with the same reasoning-sensitivity caveat.
- Smaller distills (1.5B / 7B) are more quant-sensitive in absolute terms — proportional drop is larger.

### Serving stacks
- **vLLM**: native FP8 path on H100; AWQ/GPTQ W4 via Marlin/Machete.
- **SGLang**: similar FP8 + W4 support; popular for reasoning serving because of structured-output features.
- **TensorRT-LLM**: FP8 path; NVFP4 path on Blackwell (with the reasoning-sensitivity caveat doubly applying).

## Connections
- [[deepseek-v3-fp8]] — the FP8 training recipe inherited by R1.
- [[deepseek-v3]] — DSV3 quantization story this builds on.
- [[grpo]] (in llm-training raw-data) — the RL algorithm that produced R1 from V3.
- [[llama-3-quantization]] — counterpart for the dense-frontier line; Llama 3 chat models tolerate W4 better than R1 does.
- [[qwen-3-quant]] — Qwen 3's reasoning-mode models likely face the same sensitivity.
- [[kv-cache-compression-survey-2025]] — why KV cache precision matters more for reasoning.

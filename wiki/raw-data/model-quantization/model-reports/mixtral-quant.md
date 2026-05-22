<!-- scope: Mixtral MoE quantization studies (community + QMoE precedent)
     deps: [[qmoe]], [[awq]], [[gptq]]
     see-also: [[qwen-3-quant]], [[deepseek-v3]]
-->

# Mixtral MoE Quantization
- **Core Insight:** MoE quantization differs from dense quantization in two ways: (1) the routing gate cannot be quantized aggressively without destabilizing expert selection; (2) per-expert quantization opens up new memory savings via per-expert bit allocation (cold experts at lower bits) but at the cost of routing-decision-time decompression overhead.
- **Guideline:** When quantizing Mixtral (or any MoE), keep the routing gate at BF16/FP16, quantize expert linears uniformly at W4 (AWQ/GPTQ) as the default, and only attempt per-expert variable-bit-width if you have a kernel that can handle mixed-precision expert serving (QMoE / TRT-LLM-MoE).
- **Authors:** Mixtral team (Mistral AI); QMoE for MoE-specific quantization (Frantar + Alistarh, IST-Austria)
- **Year:** 2024 (Mixtral 8x7B 2024-01, Mixtral 8x22B 2024-04; QMoE 2023-10)
- **URL:** Mixtral https://arxiv.org/abs/2401.04088 • QMoE https://arxiv.org/abs/2310.16795
- **Relevant topics:** MoE quantization, routing gate precision, per-expert quant, expert-parallel FP8

## Abstract
Mixtral 8x7B (47B total / 13B active) and Mixtral 8x22B (141B total / 39B active) are Mistral's MoE releases; both are sparse MoE with 8 experts and top-2 routing. Mistral itself did not ship lab-blessed quantized checkpoints (unlike Qwen / Meta / DeepSeek); the quantization story is community-driven (AWQ / GPTQ / GGUF for the dense expert linears) plus the academic QMoE precedent for sub-1-bit MoE compression. The two consistent findings across the literature are: (a) the routing gate cannot tolerate W4 or W8 — the per-token score gaps between experts are too small, and quant noise on the gate changes routing decisions; (b) per-expert variable bit allocation (cold experts at lower bits, hot experts at higher) is feasible but requires a custom kernel and a real expert-frequency profile.

## Key Contributions
- **Community AWQ / GPTQ / GGUF Mixtral builds**: standard W4 group_size=128 for the expert linears, keeping the routing gate at BF16.
- **QMoE precedent** (IST-Austria, [[qmoe]]): sub-1-bit compression of SwitchTransformer-c2048 (1.6T params) via per-expert codebook + custom decode kernel; runs on 4×A6000 or 8×3090. Showed MoE-specific compression can go below the dense-model floor.
- **Per-expert hot/cold observation**: in production traffic, expert activation counts are heavily skewed — 80/20 or worse; per-expert bit allocation is a real win in principle.
- **Routing-gate precision rule**: gate must stay at BF16/FP16; W8 already perturbs routing enough to drop quality.
- **All-to-all in FP8** (from DeepSeek-V3 precedent): the dispatch step of expert-parallel routing can send activations in FP8, halving bandwidth.

## Key Figures/Tables to Study
- The Mixtral paper Figure showing per-token expert activation distributions across layers — informs which experts could go to lower bits.
- The QMoE compression-ratio-vs-quality table for SwitchTransformer; the recipe transfers to Mixtral with adjustment for the smaller expert count.
- Community AWQ-Mixtral eval tables on MMLU / GSM8K / HumanEval.

## Technical Details

### What can / cannot be quantized in MoE
- **Quantizable**: expert linear layers (Q, K, V, O, FFN gate, up, down).
- **NOT-quantizable (without breaking routing)**: routing gate weights, routing softmax temperature.
- **Sensitive**: attention output projection (sometimes routed differently per expert in shared-expert variants).

### Why the routing gate is precision-sensitive
- Gate score = W_gate · h, top-2 routing.
- Per-token score gap between expert 2 and expert 3 is often O(0.01); W8 quantization noise can push that gap negative → expert selection flips.
- Mass flipping changes which experts see which tokens during training (impossible to retrain post-quant) and during inference (load imbalance, quality drop).
- Empirical rule: gate at BF16 always, expert linears at W4-W8 as needed.

### Per-expert bit allocation (QMoE direction)
- Cold experts can absorb more compression — they're seen by few tokens, so per-token error amortizes.
- Hot experts need higher precision — they're the active path for most tokens.
- Kernel implication: a single GEMM call now has heterogeneous-precision operands across experts; requires a kernel that can dispatch per-expert dequant.
- vLLM and TRT-LLM-MoE have partial support; full per-expert variable-bit is still research.

### Mixtral-specific
- 8 experts, top-2 routing → average activation per expert is ~ 25 % of tokens.
- Per-expert bit allocation pays less here than for 64-expert Switch-style models; uniform W4 across experts is the practical choice.

### Storage
| Model | FP16 | W4 uniform | QMoE 0.8 bpw (hypothetical) |
|-------|------|-----------|----------------------------|
| Mixtral 8x7B | 94 GB | 27 GB | 5 GB |
| Mixtral 8x22B | 282 GB | 81 GB | 14 GB |

### Serving
- vLLM: AWQ / GPTQ for expert linears; routing gate at FP16; expert-parallel sharding for the large variant.
- SGLang: same support.
- TRT-LLM: official MoE path; W4 expert quant + FP16 gate.

## Connections
- [[qmoe]] — sub-1-bit MoE compression; the academic baseline for MoE-aware quantization.
- [[awq]] / [[gptq]] — algorithms used for community Mixtral builds.
- [[deepseek-v3]] — frontier MoE quant precedent; uses FP8 expert linears + BF16 routing + FP8 all-to-all comm.
- [[qwen-3-quant]] — Qwen 3 MoE (30B-A3B, 235B-A22B) follows the same routing-gate-at-BF16 rule.
- [[fp8-formats-paper]] — FP8 spec used in DeepSeek-V3-style MoE FP8 deployment.
- [[blackwell-quantization]] — NVFP4 MoE quant is a 2026 frontier (placeholder for now).

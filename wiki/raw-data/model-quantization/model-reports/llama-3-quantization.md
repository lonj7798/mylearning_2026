<!-- scope: Meta's official quantized Llama 3 / 3.1 / 3.2 releases — INT4 (AWQ + GPTQ), FP8 QAT for 405B, SpinQuant for the Llama-3.2 edge tier
     deps: [[awq]], [[gptq]], [[fp8-e4m3]], [[spinquant]]
     see-also: [[autoawq]], [[autogptq]], [[transformer-engine]], [[marlin-kernel]]
-->

# Llama 3 / 3.1 / 3.2 Official Quantized Releases (Meta)
- **Core Insight:** Meta ships official quantized checkpoints — Llama-3-8B/70B in W4 (AWQ + GPTQ) for community deployment, Llama-3.1-405B in FP8 (per-row block-wise) trained with a QAT phase for the production release, and Llama-3.2-1B/3B in W4-SpinQuant for on-device.
- **Guideline:** For server: pull `meta-llama/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (group_size=128, AWQ via AutoAWQ) or the GPTQ sibling; for 405B inference start with the official FP8; for mobile/edge use the Llama-3.2 SpinQuant releases (`SpinQuant` + `QLoRA`-trained adapters).
- **Authors:** Aaron Grattafiori et al. (Meta) — Llama 3 Herd of Models paper + Meta AI blog posts
- **Year:** 2024 (Llama 3, 3.1); 2024-09 (Llama 3.2 edge tier)
- **URL:** https://arxiv.org/abs/2407.21783 (main paper); https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/ (3.2 quant tier); https://huggingface.co/meta-llama (model org)
- **Relevant topics:** AWQ, GPTQ, FP8 QAT, SpinQuant, on-device LLM, official quant releases

## Abstract
Meta's Llama 3 family ships with first-party quantization across the entire sizing range. For server-class deployment, the 8B and 70B models are released in W4 form via both AWQ and GPTQ (group_size=128); the 405B "production" release is FP8 — per-row blockwise E4M3 weights and activations, with a QAT phase on top of the BF16-trained base to recover the small loss from naive post-training FP8 casting. For the on-device tier (Llama-3.2-1B / 3B), Meta released SpinQuant-quantized W4A16KV8 checkpoints — learnable rotation + W4 GPTQ — alongside a QLoRA-trained "lightweight" variant. The technical report (§9) and the official Meta AI blog post describe the methodology and provide MMLU / HumanEval / GSM8K numbers within ~1 point of the BF16 reference.

## Key Contributions
- First-party W4 AWQ + W4 GPTQ checkpoints for 8B and 70B — the community no longer needs to re-quantize the BF16 release.
- FP8 405B as the production deployment format — proves frontier-scale FP8 inference is the default, not a research artifact.
- QAT phase for Llama-3.1-405B's FP8 release — recovers ~0.5-pt average benchmark drop that would occur from pure post-training FP8 casting.
- SpinQuant + QLoRA combo for Llama-3.2 edge tier — first on-device-targeted official quant from a frontier lab.
- Quality table (Llama 3.2 blog): MMLU / IFEval / GSM8K / HumanEval — quant within 1-2 points of BF16 across the board.

## Key Figures/Tables to Study
- Llama 3 paper §9 "Inference" — FP8 deployment recipe diagram and quality table.
- Llama 3.2 blog post quant table — W4 SpinQuant vs BF16 on 8 standard benchmarks for 1B and 3B.
- HF model cards on `meta-llama/*-AWQ-INT4` and `meta-llama/*-GPTQ-INT4` — exact `quantization_config` JSON (group_size, desc_act, sym).

## Technical Details

### Server tier (8B / 70B): W4 AWQ + W4 GPTQ
- **Method**: AWQ ([[awq]]) and GPTQ ([[gptq]]) ran in parallel as independent recipes; both shipped.
- **group_size**: 128 (per-group scale + zero point)
- **Symmetric**: False (asymmetric for INT4 — uses zero point)
- **Calibration**: ~128–512 sequences from a held-out instruction-tuning dataset.
- **Kernels**: Marlin ([[marlin-kernel]]) for W4A16 GEMM on Ampere/Hopper, Machete ([[machete-kernel]]) on Hopper.
- **Quality** (8B Instruct, AWQ-INT4 vs BF16, from HF card): MMLU 68.4 → 68.0; GSM8K 84.5 → 84.0; HumanEval 62.2 → 60.4 — within ~2 pts.

### 405B production: FP8 with QAT
- **Format**: E4M3 weights + E4M3 activations (forward); BF16 master weights for QAT.
- **Per-row blockwise scaling**: 128-element block along the K axis; one FP scale per block (mirrors the [[deepseek-v3-fp8]] recipe).
- **QAT phase**: ~200B-token additional training in fake-quant mode (BF16 master → FP8 cast → BF16 grad). Recovers ~0.5-pt average benchmark gap vs pure PTQ FP8.
- **Kernels**: NVIDIA Transformer Engine ([[transformer-engine]]) — fused FP8 matmul + delayed scaling + amax history.
- **Throughput**: ~2× over BF16 on H100; the only economically deployable form of 405B for most production stacks.

### Edge tier (Llama-3.2-1B / 3B): SpinQuant + QLoRA
- **SpinQuant** ([[spinquant]]): learnable orthogonal rotations folded into the weight matrices to flatten activation outliers; W4A16 PTQ.
- **KV cache**: INT8 per-head (so W4A16KV8 — server-class memory profile).
- **QLoRA variant**: NF4 base ([[nf4]]) + LoRA adapters trained on the same SFT data Meta used for the BF16 reference. Released as the "lightweight" line.
- **On-device target**: Qualcomm and MediaTek SoC chains (ExecuTorch + QNN/MTK backends).
- **Quality** (3B, Llama 3.2 blog): MMLU 63.4 → 60.5 (SpinQuant); IFEval 77.4 → 73.5 — 2-4 pt drop in exchange for ~3× memory savings and ~2× decode latency.

### Recipe summary
| Tier | Model | Format | Method | Quality drop vs BF16 |
|------|-------|--------|--------|----------------------|
| Server | 8B / 70B | W4A16 | AWQ-INT4 (group=128) | ≤2 pt MMLU |
| Server | 8B / 70B | W4A16 | GPTQ-INT4 (group=128) | ≤2 pt MMLU |
| Production | 405B | FP8 (E4M3) | per-row block + QAT | ≤0.5 pt avg benchmark |
| Edge | 1B / 3B | W4A16KV8 | SpinQuant | 2-4 pt MMLU |
| Edge | 1B / 3B | NF4 | QLoRA-finetuned | similar to SpinQuant |

## Connections
- [[awq]] / [[gptq]] — the two W4 PTQ methods Meta shipped in parallel for 8B/70B.
- [[spinquant]] — the learnable-rotation method behind the Llama 3.2 edge release; this is the most prominent production deployment of SpinQuant to date.
- [[deepseek-v3-fp8]] — the FP8-training cousin; Meta's QAT-on-BF16 recipe is the conservative variant of DSV3's native-FP8-training recipe.
- [[transformer-engine]] — the kernel stack underpinning the 405B FP8 inference.
- [[marlin-kernel]] / [[machete-kernel]] — the W4A16 GEMM kernels powering AWQ/GPTQ deployments.
- [[nf4]] — the QLoRA storage code in the edge tier.

## Notes
This release is the canonical industry proof point that quantization has moved from "research artifact" to "first-party production format" — across three tiers (server, frontier production, edge) and three formats (W4A16, FP8, NF4), all officially supported by the model provider.

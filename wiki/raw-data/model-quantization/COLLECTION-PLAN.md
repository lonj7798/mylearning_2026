<!-- scope: master checklist of topics + target sources for model-quantization raw library
     deps: [[README]]
     see-also: [[insights]]
-->

# Collection Plan — Topic Checklist (Model Quantization)

Every bullet below is a target file. Mark `[x]` when the file lands in the right subdirectory with the required structure (see `[[README]]`). Gaps are filled in a second sweep after initial collection.

Legend: `P` = paper, `B` = blog, `R` = model report, `F` = framework code, `S` = format spec, `L` = lab summary. Files live in the subdir matching their type.

The course is **theory-first**. The first six buckets (`0`, `1`, `2`, `3`, `4`, `5`) are all foundations and must be filled before the bulk year-by-year paper crawl in buckets `6`–`10`.

---

## 0. Math Foundations (theoretical bedrock)

Why here: the learner asked for theoretical depth up front. These sources establish *why* quantization works at all — independently of any LLM-specific trick.

- [x] **P** `classics/rate-distortion-theory.md` — Shannon 1948 / Cover & Thomas Ch. 10; R(D) bound; why D ∝ 2^{−2R} for Gaussian source; the floor every quantizer is fighting
- [x] **P** `classics/lloyd-max-quantizer.md` — Lloyd 1957 / Max 1960; optimal scalar quantizer for a given distribution; centroid + nearest-neighbour conditions
- [x] **P** `classics/uniform-quantization-noise.md` — Bennett 1948; uniform-noise model; σ² = Δ²/12; high-resolution assumption
- [x] **P** `classics/companding-mu-law.md` — Smith 1957 µ-law / A-law companding; non-uniform quantization for skewed distributions; analog for log-scale weights
- [x] **P** `classics/vector-quantization.md` — Linde-Buzo-Gray (LBG) 1980; k-means for VQ; bit-rate vs distortion in higher dimensions; precursor to product quantization
- [x] **P** `classics/product-quantization.md` — Jégou 2011; product-quantization (PQ) for ANN search; foundation for AQLM-style additive quantization
- [x] **P** `classics/information-theoretic-bounds.md` — Gish-Pierce 1968 high-rate optimal density; why optimal density ∝ p(x)^{1/3} for Lp distortion
- [x] **P** `classics/stochastic-rounding.md` — Gupta 2015 (IBM); unbiased rounding; why it preserves gradients at low precision
- [x] **P** `classics/round-to-nearest-even.md` — IEEE-754 RNE; banker's rounding; how it interacts with quantization bias

## 1. Numerical Formats (what bits actually represent)

Why here: every quantization paper presupposes you know what FP8/FP4/MX/INT4 actually mean at the bit level. These are reference cards.

- [x] **S** `formats/ieee-754.md` — IEEE-754 binary32/binary16; exponent + mantissa; subnormals; rounding modes
- [x] **S** `formats/bf16.md` — Google Brain bfloat16; same exponent as FP32, truncated mantissa; why it dominated training
- [x] **S** `formats/fp16.md` — half-precision; narrower dynamic range; loss-scaling necessity
- [x] **S** `formats/fp8-e4m3.md` — OFP8 / NVIDIA H100 E4M3; 4 exp + 3 mantissa; for activations and weights
- [x] **S** `formats/fp8-e5m2.md` — E5M2; 5 exp + 2 mantissa; wider range for gradients
- [x] **S** `formats/fp6.md` — FP6 (E3M2 / E2M3); intermediate-precision proposals
- [x] **S** `formats/fp4-e2m1.md` — FP4 E2M1; sub-8-bit float; Blackwell-era native support
- [x] **S** `formats/int8.md` — INT8 symmetric / asymmetric; per-tensor vs per-channel scale
- [x] **S** `formats/int4.md` — INT4 packing layouts (W4A16, W4A8); group-wise scale
- [x] **S** `formats/nf4.md` — Dettmers' Normal Float 4-bit; quantile-based code for Gaussian-distributed weights (introduced in QLoRA)
- [x] **S** `formats/af4.md` — Abstract Float / Asymmetric Float 4-bit variants
- [x] **S** `formats/mx-formats.md` — OCP Microscaling 2023 spec; shared block exponent + element format (MXFP8 / MXFP6 / MXFP4 / MXINT8)
- [x] **S** `formats/nvfp4.md` — NVIDIA NVFP4 (Blackwell); block-scaled FP4 with FP8 block scale + FP32 tensor scale; the production 4-bit format
- [x] **S** `formats/gguf-k-quants.md` — llama.cpp gguf k-quant family (q2_k, q3_k, q4_k, q5_k, q6_k, q8_0); block layout + super-block scale
- [x] **S** `formats/bitnet-w158.md` — BitNet ternary {-1, 0, 1} weight format; 1.58 bits/weight = log2(3)

## 2. Classical Quantization Theory (pre-LLM, still load-bearing)

Why here: the entire PTQ vs QAT distinction, STE, learned step size, and integer-only inference were established years before LLMs.

- [x] **P** `classics/straight-through-estimator.md` — Bengio 2013; STE as a biased-but-useful gradient through non-differentiable quantizers
- [x] **P** `classics/bnn.md` — Courbariaux 2016 Binary Neural Networks; the original 1-bit weight & activation training
- [x] **P** `classics/xnor-net.md` — Rastegari 2016; XNOR-Net; binary conv via XNOR + popcount
- [x] **P** `classics/dorefa-net.md` — Zhou 2016; arbitrary-bit weights, activations, gradients; quantizer family
- [x] **P** `classics/wage.md` — Wu 2018 WAGE; weight + activation + gradient + error all quantized
- [x] **P** `classics/lq-nets.md` — Zhang 2018; learnable quantization basis vectors
- [x] **P** `classics/pact.md` — Choi 2018 PACT; learned clipping threshold for activations
- [x] **P** `classics/lsq.md` — Esser 2020 Learned Step Size Quantization; learn the quantizer scale itself; SOTA QAT baseline
- [x] **P** `classics/lsq-plus.md` — Bhalgat 2020 LSQ+; asymmetric extension for ReLU+
- [x] **P** `classics/quantization-mapping.md` — Krishnamoorthi 2018 (Google whitepaper); the canonical PTQ playbook; per-tensor / per-channel / symmetric / asymmetric taxonomy
- [x] **P** `classics/data-free-quantization.md` — Nagel 2019 DFQ; weight equalization + bias correction without data
- [x] **P** `classics/adaround.md` — Nagel 2020 AdaRound; per-weight learned rounding direction; closed-form via Hessian; the parent of GPTQ
- [x] **P** `classics/brecq.md` — Li 2021 BRECQ; block-wise reconstruction PTQ; cross-layer dependency
- [x] **P** `classics/qdrop.md` — Wei 2022 QDrop; randomly dropping quantization during PTQ calibration
- [x] **P** `classics/integer-only-inference.md` — Jacob 2018 (Google); INT8-only quantized inference; the foundational mobile-CPU quant paper
- [x] **P** `classics/i-bert.md` — Kim 2021; integer-only BERT; INT-GELU, INT-Softmax, INT-LayerNorm approximations
- [x] **P** `classics/q8bert.md` — Zafrir 2019; 8-bit BERT QAT
- [x] **P** `classics/zeroq.md` — Cai 2020; distill calibration data via BN statistics
- [x] **P** `classics/hawq.md` — Dong 2019 HAWQ; Hessian-aware mixed-precision bit allocation

## 3. Calibration & Error Analysis (the bridge to LLM PTQ)

- [x] **P** `classics/obs-obd.md` — Hassibi 1993 Optimal Brain Surgeon / LeCun 1989 Optimal Brain Damage; second-order weight removal; the direct ancestor of GPTQ
- [x] **P** `classics/obc.md` — Frantar 2022 Optimal Brain Compression; Hessian-based pruning + quantization unified
- [x] **P** `classics/mse-vs-kl-calibration.md` — survey of calibration objectives; when MSE vs cosine vs KL matter
- [x] **P** `classics/percentile-clipping.md` — clip-to-percentile vs full-range; outlier handling in calibration
- [x] **P** `classics/quantization-error-propagation.md` — error compounding across a transformer block; analytic bounds

## 4. Survey & Overview (orient before the year-by-year deep dive)

- [x] **P** `papers/survey-gholami-2021.md` — Gholami 2021 "A Survey of Quantization Methods for Efficient NN Inference"; canonical reference
- [x] **P** `papers/survey-llm-quantization-2024.md` — recent LLM-quant-only survey (e.g. Zhu 2024 or equivalent)
- [x] **P** `papers/survey-low-bit-llm-2024.md` — Gong / Liang survey of <8-bit LLM quant
- [x] **P** `papers/survey-efficient-llm-inference-2024.md` — Zhou 2024 efficient-inference survey with quantization section
- [x] **B** `blogs/qualcomm-quantization-whitepaper.md` — Qualcomm AI Research PTQ/QAT whitepaper; production reference

## 5. Pre-2022 LLM-era Bridges

These are 2020–2021 papers that already saw quantization stress on transformer-scale models and seeded everything that followed.

- [x] **P** `papers/q-bert.md` — Shen 2020 Q-BERT; mixed-precision Hessian-aware quant for BERT
- [x] **P** `papers/bibert.md` — Qin 2022 BiBERT; binary BERT
- [x] **P** `papers/oscar.md` — Outlier Suppression with Equalization; precursor to SmoothQuant outlier idea
- [x] **P** `papers/outlier-channel-splitting.md` — Zhao 2019 OCS; splits outlier channels; lineage for SmoothQuant
- [x] **P** `papers/bit-pruning.md` — pre-LLM mixed-precision bit-budget allocation

## 6. LLM Quantization — 2022 (the year LLM PTQ was born)

- [x] **P** `papers/llm-int8.md` — Dettmers 2022 LLM.int8(); mixed-precision INT8 + FP16 outlier path; first to make 6.7B+ models work at 8-bit; the outlier paper
- [x] **P** `papers/gptq.md` — Frantar 2022 GPTQ; one-shot Hessian-based PTQ for OPT/BLOOM; 4-bit at scale; OBS lineage
- [x] **P** `papers/zeroquant.md` — Yao 2022 ZeroQuant; INT8 group-wise weight + token-wise activation
- [x] **P** `papers/zeroquant-v2.md` — Yao 2023 ZeroQuant-V2; low-rank compensation for harder regimes
- [x] **P** `papers/zeroquant-fp.md` — Wu 2023 ZeroQuant-FP; FP8 / FP4 for LLM PTQ
- [x] **P** `papers/nuqmm.md` — Park 2022 nuQmm; non-uniform LUT-based weight quant kernel
- [x] **B** `blogs/dettmers-llm-int8-blog.md` — Tim Dettmers' explanatory blog for LLM.int8() and outlier features

## 7. LLM Quantization — 2023 (the explosion year)

### 7a. Activation-aware & equivalent-transformation methods

- [x] **P** `papers/smoothquant.md` — Xiao 2023 SmoothQuant; migrate quantization difficulty from activations to weights via per-channel scaling; W8A8 LLM PTQ
- [x] **P** `papers/awq.md` — Lin 2023 AWQ (Activation-aware Weight Quantization); per-channel scaling driven by activation magnitude; weight-only PTQ default
- [x] **P** `papers/omniquant.md` — Shao 2023 OmniQuant; learnable equivalent transformations + learnable weight clipping; gradient-based PTQ
- [x] **P** `papers/rptq.md` — Yuan 2023 RPTQ; reorder channels into similar-range clusters

### 7b. Non-uniform & sensitivity-based

- [x] **P** `papers/squeezellm.md` — Kim 2023 SqueezeLLM; sensitivity-weighted non-uniform quantization + dense-and-sparse decomposition
- [x] **P** `papers/spqr.md` — Dettmers 2023 SpQR; sparse-quantized representation; keep outlier weights at high precision
- [x] **P** `papers/owq.md` — Lee 2023 OWQ; outlier-aware weight quantization

### 7c. QAT / parameter-efficient

- [x] **P** `papers/qlora.md` — Dettmers 2023 QLoRA; 4-bit NF4 base + LoRA adapters + paged optimizer; democratized 65B-model fine-tuning
- [x] **P** `papers/qa-lora.md` — Xu 2023 QA-LoRA; QAT for LoRA-style adapters
- [x] **P** `papers/loftq.md` — Li 2023 LoftQ; LoRA-fine-tuning-aware quantization initialization
- [x] **P** `papers/llm-qat.md` — Liu 2023 LLM-QAT; data-free QAT via teacher self-generation
- [x] **P** `papers/peqa.md` — Kim 2023 PEQA; parameter-efficient quantization-aware adaptation

### 7d. Incoherence processing (precursor to rotations)

- [x] **P** `papers/quip.md` — Chee 2023 QuIP; incoherence processing via random rotations; first principled 2-bit LLM PTQ
- [x] **P** `papers/quip-sharp.md` — Tseng 2024 QuIP#; lattice-based 2-bit with E8 codebook
- [x] **P** `papers/quik.md` — Ashkboos 2023 QUIK; INT4 weights + INT8 activations PTQ

### 7e. FP8 emerging

- [x] **P** `papers/fp8-formats-paper.md` — Micikevicius 2022/2023 FP8 Formats for Deep Learning; the joint NVIDIA/Arm/Intel FP8 spec
- [x] **P** `papers/fp8-llm-inference.md` — early FP8 LLM inference studies
- [x] **P** `papers/fp8-lm.md` — Peng 2023 FP8-LM; FP8 mixed-precision LLM training

### 7f. KV-cache quantization debut

- [x] **P** `papers/kvquant-2023.md` — earliest KV-cache-only quant attempts
- [x] **P** `papers/flexgen.md` — Sheng 2023 FlexGen; quantization as part of offloaded inference

### 7g. BitNet line begins

- [x] **P** `papers/bitnet.md` — Wang 2023 BitNet; 1-bit transformer trained from scratch
- [x] **P** `papers/bitnet-b158.md` — Ma 2024 BitNet b1.58; ternary {-1, 0, 1} weights; "the era of 1-bit LLMs"

## 8. LLM Quantization — 2024 (rotations, sub-2-bit, KV-cache maturation)

### 8a. Rotation-based outlier removal (the year's signature move)

- [x] **P** `papers/quarot.md` — Ashkboos 2024 QuaRot; Hadamard rotations fold outliers into weight space; W4A4 LLM PTQ
- [x] **P** `papers/spinquant.md` — Liu 2024 SpinQuant; learnable rotations replacing random Hadamards
- [x] **P** `papers/duquant.md` — Lin 2024 DuQuant; dual rotation + permutation for outlier elimination
- [x] **P** `papers/flatquant.md` — Sun 2024 FlatQuant; affine transformations that flatten weight/activation distributions
- [x] **P** `papers/rotation-and-quantization.md` — theory paper unifying rotation-based methods

### 8b. Sub-2-bit and additive quantization

- [x] **P** `papers/aqlm.md` — Egiazarian 2024 AQLM; additive quantization (PQ lineage) for 2-bit and lower
- [x] **P** `papers/quip-sharp-2024.md` — QuIP# expanded results
- [x] **P** `papers/pv-tuning.md` — Malinovskii 2024 PV-Tuning; fine-tuning sub-2-bit models
- [x] **P** `papers/vptq.md` — Liu 2024 VPTQ; vector PTQ
- [x] **P** `papers/gptvq.md` — van Baalen 2024 GPTVQ; vector quantization with GPTQ-style updates

### 8c. Fast / data-free / training-free

- [x] **P** `papers/hqq.md` — Badri 2024 HQQ; Half-Quadratic Quantization; data-free, very fast, optimization-based
- [x] **P** `papers/affinequant.md` — Ma 2024 AffineQuant; affine transformations for activation quantization
- [x] **P** `papers/atom.md` — Zhao 2024 Atom; W4A4 + KV4 with mixed sub-channel reorder
- [x] **P** `papers/qserve.md` — Lin 2024 QServe; W4A8KV4 serving with progressive quantization

### 8d. KV cache quantization (matured)

- [x] **P** `papers/kivi.md` — Liu 2024 KIVI; per-channel K quant + per-token V quant; W16A16KV2 viable
- [x] **P** `papers/kvquant.md` — Hooper 2024 KVQuant; ultra-low-bit KV cache with non-uniform quant + dense-and-sparse
- [x] **P** `papers/gear.md` — Kang 2024 GEAR; KV-cache quantization with error compensation
- [x] **P** `papers/wkvquant.md` — Yue 2024 WKVQuant; joint W4+KV4 with calibration
- [x] **P** `papers/qaq.md` — Dong 2024 QAQ; quality-adaptive KV-cache quantization
- [x] **P** `papers/skvq.md` — Duanmu 2024 SKVQ; sliding-window KV quant
- [x] **P** `papers/coupling-kv-quant.md` — analysis of coupling between weight and KV quantization

### 8e. Training & FP8 mid-training

- [x] **P** `papers/quest.md` — Quest data-free QAT
- [x] **P** `papers/efficientqat.md` — Chen 2024 EfficientQAT; block-wise QAT for LLMs
- [x] **P** `papers/bitdistiller.md` — Du 2024 BitDistiller; self-distillation for sub-4-bit
- [x] **P** `papers/lq-lora.md` — Guo 2024 LQ-LoRA; low-rank quantization-aware decomposition

### 8f. BitNet line continues

- [x] **P** `papers/bitnet-a48.md` — BitNet a4.8; 4-bit activations on top of 1.58-bit weights
- [x] **P** `papers/era-of-1bit-llms.md` — survey-style follow-up consolidating BitNet results
- [x] **P** `papers/onebit.md` — Xu 2024 OneBit; 1-bit weight LLM via SVID

### 8g. MX format adoption

- [x] **P** `papers/microscaling-formats.md` — Rouhani 2023 (Microsoft); microscaling shared-exponent formats; the OCP MX spec basis
- [x] **P** `papers/llm-fp4.md` — Liu 2023 LLM-FP4; FP4 quantization for LLMs
- [x] **P** `papers/mxfp-training.md` — training experiments with MXFP8 / MXFP6 / MXFP4

## 9. LLM Quantization — 2025 (production & training-time)

### 9a. FP8 / FP4 native training

- [x] **P** `papers/deepseek-v3-fp8.md` — DeepSeek V3 FP8 native training; per-block scaling + fine-grained casting; first frontier-scale FP8 training run
- [x] **P** `papers/nvfp4-training.md` — NVIDIA NVFP4 native pretraining studies on Blackwell
- [x] **P** `papers/mxfp4-pretraining.md` — Microsoft / academic studies on MXFP4 pretraining
- [x] **P** `papers/fp8-vs-bf16-scaling.md` — scaling-law studies comparing FP8 to BF16
- [x] **P** `papers/transformer-engine.md` — NVIDIA Transformer Engine FP8 mixed-precision recipes

### 9b. Production W4 + KV quant deployments

- [x] **P** `papers/marlin-kernel.md` — Frantar 2024/2025 Marlin; W4A16 GEMM achieving near-FP16 throughput on Ampere/Hopper
- [x] **P** `papers/machete-kernel.md` — vLLM/NeuralMagic Machete; W4A16 GEMM for Hopper
- [x] **P** `papers/tinychat-and-tensorrt-llm-quant.md` — production-system quant integration notes
- [x] **P** `papers/qoq.md` — quantization-on-quantization composition studies
- [x] **P** `papers/squeezellm-followups.md` — 2025 refinements to sensitivity-based quant

### 9c. Scaled BitNet / 1-bit training

- [x] **P** `papers/bitnet-b158-2b.md` — BitNet b1.58 2B / 4B trained models with full evals
- [x] **P** `papers/bitnet-scaling-laws.md` — scaling-law studies for 1-bit LLM pretraining

### 9d. Rotation theory and unification

- [x] **P** `papers/rotation-unification-2025.md` — 2025 unified frameworks consolidating QuaRot / SpinQuant / DuQuant
- [x] **P** `papers/orthogonal-finetuning-quant.md` — orthogonal-fine-tuning interactions with rotated quant
- [x] **P** `papers/learnable-rotation-2025.md` — learnable rotations beyond Hadamard

### 9e. KV-cache compression beyond quantization (boundary)

- [x] **P** `papers/kv-cache-compression-survey-2025.md` — survey including quant + sparsity + eviction
- [x] **P** `papers/coupled-quant-eviction.md` — combining KV quant with eviction policies

## 10. LLM Quantization — 2026 (most recent)

- [x] **P** `papers/turboquant.md` — Zandieh 2025/ICLR 2026 (Google); online vector quantization with near-optimal distortion rate; random rotation + per-coordinate scalar quant + 1-bit QJL residual; 2.5–3.5 bit KV cache with no calibration
- [x] **P** `papers/polarquant.md` — Han/Kacham/Zandieh 2025/AISTATS 2026 (Google); recursive pair-wise polar transform after random preconditioning; closed-form angle distribution → no per-block scale; 4.2× KV compression
- [x] **P** `papers/qjl.md` — Zandieh 2024 (Google); 1-bit Quantized JL transform for KV cache; sign-bit sketch with asymmetric inner-product estimator; zero per-block metadata overhead; 5× KV reduction at 3-bit equivalent
- [x] **P** `papers/quartet-ii.md` — Panferov/Alistarh 2026; NVFP4 pretraining with MS-EDEN unbiased microscaling quantizer; lower-error alternative to stochastic rounding; Blackwell kernels
- [x] **P** `papers/mxfp4-native-hardware-2026.md` — Cim/Kandemir 2026; native MXFP4 training on AMD MI355X; Wgrad quantization is the main failure mode; deterministic Hadamard stabilizes
- [x] **P** `papers/nvfp4-qad.md` — NVIDIA/HAN Lab 2026; quantization-aware distillation for NVFP4 LLM/VLM inference recovery; near-BF16 recovery after SFT/RL/model-merge pipelines
- [x] **P** `papers/fp4-inference-diagnosis.md` — Cim/Topcu/Kandemir 2026; NVFP4 vs MXFP4 sensitivity map across Qwen2.5; MLP up/down projections and early blocks need protection
- [x] **P** `papers/statistically-lossless-quantization.md` — Helcig/Kurtic/Alistarh 2026; task-lossless vs distribution-lossless quantization; EAR metric; asymmetric non-uniform SLQ
- [x] **P** `papers/quant-2026-frontier.md` — placeholder for additional 2026 frontier papers (filled during crawl)
- [x] **P** `papers/nvfp4-production-2026.md` — production NVFP4 reports from Blackwell-era deployments
- [x] **P** `papers/sub-bit-llm-2026.md` — fractional-bit research consolidation
- [x] **R** `model-reports/blackwell-quantization.md` — NVIDIA Blackwell quantization recipes

## 11. KV-cache Quantization Dedicated Bucket

(Cross-references items in §7f, §8d, §9e — assemble dedicated track.)

- [x] **P** `papers/adaptive-kv-cache-quant.md` — CVPR 2026; learned per-token KV precision policy for on-device LLMs; selects 2-bit/4-bit/8-bit/FP16 by token importance
- [x] **P** `papers/kvtc.md` — ICLR 2026; KV-cache transform coding for reusable caches; PCA decorrelation + adaptive quantization + entropy coding
- [x] **P** `papers/kv-cache-survey.md` — Park 2024/2025 dedicated KV-cache compression survey
- [x] **P** `papers/per-channel-vs-per-token-kv.md` — analytical study of K vs V quant asymmetry

## 12. Frontier Model Reports (quantization specifics)

- [x] **R** `model-reports/gpt-oss-mxfp4.md` — OpenAI GPT-OSS 120B/20B model card; MoE weights post-trained to MXFP4 at 4.25 bits/parameter; production model-release case
- [x] **R** `model-reports/llama-3-quantization.md` — Llama 3 official quantization deployment (8B/70B/405B in INT4/INT8/FP8)
- [x] **R** `model-reports/deepseek-v3.md` — DeepSeek V3 FP8 training & deployment
- [x] **R** `model-reports/deepseek-r1-quantization.md` — R1 quantization releases
- [x] **R** `model-reports/qwen-2.5-quant.md` — Qwen 2.5 GPTQ/AWQ official releases
- [x] **R** `model-reports/qwen-3-quant.md` — Qwen 3 quant release notes
- [x] **R** `model-reports/mixtral-quant.md` — Mixtral MoE quantization studies
- [x] **R** `model-reports/gemma-quant.md` — Gemma official quant
- [x] **R** `model-reports/phi-quant.md` — Microsoft Phi quant releases
- [x] **R** `model-reports/llama-cpp-gguf-releases.md` — community gguf release patterns (k-quant ladder)
- [x] **R** `model-reports/bitnet-models.md` — official 1-bit/1.58-bit model releases

## 13. Hardware Vendor & Whitepaper

- [x] **B** `blogs/nvidia-h100-fp8.md` — NVIDIA H100 FP8 deep-dive
- [x] **B** `blogs/nvidia-blackwell-fp4.md` — Blackwell NVFP4 architecture posts
- [x] **B** `blogs/intel-amx.md` — Intel AMX INT8/BF16 instructions
- [x] **B** `blogs/amd-mi300-fp8.md` — AMD MI300 FP8 support
- [x] **B** `blogs/qualcomm-quantization-whitepaper.md` — covered in §4 but referenced here for hardware angle
- [x] **B** `blogs/ocp-mx-spec.md` — OCP Microscaling spec discussion / blogs explaining MX
- [x] **B** `blogs/transformer-engine-blog.md` — TE FP8 usage patterns

## 14. Frameworks (code-level reference)

- [x] **F** `frameworks/bitsandbytes-int8.md` — bitsandbytes LLM.int8() implementation
- [x] **F** `frameworks/bitsandbytes-nf4.md` — bitsandbytes 4-bit NF4 / FP4 kernels
- [x] **F** `frameworks/autogptq.md` — AutoGPTQ implementation; quantize() + pack
- [x] **F** `frameworks/autoawq.md` — AutoAWQ; quantization + kernel
- [x] **F** `frameworks/hqq-framework.md` — HQQ reference code
- [x] **F** `frameworks/llama-cpp-ggml.md` — llama.cpp ggml/gguf quant kernels (q4_k_m, q5_k_m, …)
- [x] **F** `frameworks/tensorrt-llm-quant.md` — TensorRT-LLM quant pipeline
- [x] **F** `frameworks/vllm-quant.md` — vLLM integration of GPTQ/AWQ/FP8/Marlin
- [x] **F** `frameworks/torchao.md` — PyTorch torchao quant utilities
- [x] **F** `frameworks/hf-quanto.md` — HuggingFace Quanto
- [x] **F** `frameworks/transformer-engine-fp8.md` — NVIDIA Transformer Engine FP8 implementation
- [x] **F** `frameworks/megatron-fp8.md` — Megatron-LM FP8 integration

## 15. Blogs & Postmortems

- [x] **B** `blogs/dettmers-llm-int8-blog.md` — Dettmers' explanatory blog
- [x] **B** `blogs/lilianweng-quantization.md` — if Lilian Weng has a quant overview post
- [x] **B** `blogs/hf-quantization-fundamentals.md` — HF quantization explainer series
- [x] **B** `blogs/hf-fp8-deep-dive.md` — HF FP8 post
- [x] **B** `blogs/sebastian-raschka-quant.md` — if Raschka has a quant post
- [x] **B** `blogs/maxime-labonne-quant-guide.md` — Maxime Labonne's gguf / quant practical guides
- [x] **B** `blogs/answer-ai-qlora-followups.md` — Answer.AI QLoRA-era practical posts
- [x] **B** `blogs/character-ai-quant-deployment.md` — production-deployment war stories where available
- [x] **B** `blogs/groq-quant-deployment.md` — Groq quant approach (if disclosed)

## 16. Labs & Researchers

- [x] **L** `labs/dettmers-group.md` — Tim Dettmers (UW / Allen AI alumnus): LLM.int8(), QLoRA, SpQR
- [x] **L** `labs/frantar-alistarh-ist-austria.md` — Elias Frantar + Dan Alistarh (IST Austria): GPTQ, Marlin, AQLM, SparseGPT
- [x] **L** `labs/han-song-mit.md` — Song Han (MIT HAN Lab): AWQ, SmoothQuant, SqueezeLLM, QServe, TinyChat
- [x] **L** `labs/microsoft-bitnet.md` — Microsoft Research BitNet team
- [x] **L** `labs/nvidia-quantization.md` — NVIDIA Applied Deep Learning Research; FP8/FP4/Transformer Engine
- [x] **L** `labs/intel-quantization.md` — Intel Neural Compressor team
- [x] **L** `labs/qualcomm-ai-research.md` — Markus Nagel + AdaRound, BRECQ lineage
- [x] **L** `labs/deepseek-quant.md` — DeepSeek FP8 / quant deployment lineage

---

## Gap log

After the first collection pass, list any bullet above that could not be filled (source paywalled, no good extract, contradictions) here. The planner uses the gap log to decide whether a chapter needs to be cut or scope-narrowed.

- 2026-05-21 sweep: added concrete 2026 replacements for several frontier placeholders: [[quartet-ii]], [[mxfp4-native-hardware-2026]], [[nvfp4-qad]], [[fp4-inference-diagnosis]], [[statistically-lossless-quantization]], [[adaptive-kv-cache-quant]], [[kvtc]], and [[gpt-oss-mxfp4]].
- Remaining open gap: `papers/quant-2026-frontier.md`, `papers/nvfp4-production-2026.md`, `papers/sub-bit-llm-2026.md`, and `model-reports/blackwell-quantization.md` are still broad placeholders rather than single canonical artifacts.

<!-- scope: aggregated cross-source insights for the model-quantization raw library
     deps: [[README]], [[COLLECTION-PLAN]]
     see-also: (filled after pass 1)
-->

# Model Quantization — Insights Index

This page is the cross-source map for the raw library. It is **built last**, after the COLLECTION-PLAN checklist has reached a high fill rate. Each section below is a placeholder for the synthesis that will be written once the underlying source pages exist.

> Status: skeleton plus 2026 freshness notes. Re-populate the full synthesis after the first crawl pass.

## 1. Theoretical Foundations

(Will summarize how rate-distortion, Lloyd-Max, and uniform-noise theory bound what any quantizer can achieve, and how the bound interacts with the empirical distribution of LLM weights and activations.)

## 2. Numerical Formats

(Will summarize the FP / INT / block-format design space — IEEE-754 vs bfloat vs FP8 vs FP4 vs MX / NVFP4 — and the practical recipe of "which format for which tensor".)

## 3. Classical PTQ / QAT Lineage

(Will summarize the STE → DoReFa → LSQ → AdaRound → BRECQ → GPTQ chain, plus the integer-only-inference branch through I-BERT / Q8BERT.)

## 4. LLM PTQ Era (2022–2026)

### 4a. Outlier handling
(LLM.int8 mixed-precision → SmoothQuant equivalent scaling → AWQ activation-aware scaling → SpQR/SqueezeLLM sparse-and-dense → QuaRot/SpinQuant/DuQuant rotation-based.)

### 4b. Sub-4-bit weight quantization
(GPTQ → AWQ → OmniQuant → QuIP/QuIP# → AQLM → PV-Tuning chain; the move from per-weight scalar quant to vector / additive / lattice-based codebooks.)

### 4c. Activation quantization
(SmoothQuant W8A8 → Atom W4A4 → QuaRot W4A4 → MXFP4 native — the gradual move down in activation precision.)

### 4d. KV-cache quantization
(Per-channel-K vs per-token-V asymmetry; KIVI / KVQuant / GEAR / WKVQuant / SKVQ; the route to W4A4KV2.)

### 4e. 1-bit and ternary LLMs
(BitNet → BitNet b1.58 → BitNet a4.8; the "era of 1-bit LLMs" claim, what it actually rests on, and where it breaks.)

### 4f. FP8 / FP4 native training
(FP8-LM → DeepSeek V3 FP8 → MXFP4 / NVFP4 pretraining; what scales, what doesn't.)

## 5. Hardware / Format Spec

(Will summarize how H100 FP8 → Blackwell FP4 → MX formats co-evolved with the algorithm literature, and how kernel work — Marlin, Machete, Transformer Engine — closed the gap between "paper PTQ" and "production serving".)

## 6. Open Gaps

- The 2026 sweep added concrete frontier sources for NVFP4 training ([[quartet-ii]]), native MXFP4 training diagnostics ([[mxfp4-native-hardware-2026]]), NVFP4 production recovery ([[nvfp4-qad]]), FP4 inference sensitivity ([[fp4-inference-diagnosis]]), statistically-lossless evaluation ([[statistically-lossless-quantization]]), adaptive KV-cache precision ([[adaptive-kv-cache-quant]]), KV-cache transform coding ([[kvtc]]), and a model-release case study for MXFP4 ([[gpt-oss-mxfp4]]).
- Remaining placeholders worth filling later: true Blackwell production deployment reports from multiple labs, fractional/sub-bit LLM consolidation beyond BitNet/AQLM, and official Qwen/Gemma/Llama quantization release reports if they include enough method detail.

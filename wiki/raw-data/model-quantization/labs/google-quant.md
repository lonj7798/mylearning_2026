<!-- scope: Google Research's quantization line — Zandieh/Mirrokni KV-cache trilogy, Jacob 2018 integer-only, Gemma quant releases, bfloat16 origin
     deps: [[qjl]], [[polarquant]], [[turboquant]], [[integer-only-inference]], [[bf16]]
     see-also: [[gemma-quant]]
-->

# Google Quantization — KV Trio + Integer-Only Inference + bfloat16
- **Core Insight:** Google's quantization lineage spans training formats, mobile integer inference, and the 2024–2026 data-oblivious KV-cache trilogy built on high-dimensional geometry.
- **Guideline:** Read this lab track for bfloat16, integer-only inference, and QJL/PolarQuant/TurboQuant as the theory-heavy KV-cache branch.
- **Authors:** Google Brain / Google Research contributors, including Zandieh and Mirrokni collaborators
- **Year:** 2017–2026
- **URL:** https://research.google/blog/ ; https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- **Relevant topics:** bfloat16, integer-only inference, QJL, PolarQuant, TurboQuant, Gemma quantization

## Summary
Google Research's quantization output spans three distinct lineages: (1) the **Brain group's bfloat16 invention** (~2017) which became the dominant training-precision format; (2) the **Jacob et al. 2018 integer-only inference** paper which established the canonical mobile/edge INT8 PTQ playbook (Krishnamoorthi taxonomy); and (3) the **Zandieh + Mirrokni KV-cache trilogy** ([[qjl]] → [[polarquant]] → [[turboquant]], 2024-2026) which establishes data-oblivious KV quant with zero per-block metadata overhead. The Gemma model releases ([[gemma-quant]]) ship with first-party QAT and PTQ variants in W4 / W8 form.

## Notable Works
- bfloat16 (Google Brain ~2017) — same 8-bit exponent as FP32, truncated mantissa; the format BF16-master-weight training rides on. ([[bf16]])
- [[integer-only-inference]] (Jacob et al. 2018) — INT8-only inference; the requant pipeline `M = S_w·S_x/S_y` approximated as `M₀·2^{-n}`; the foundational mobile-CPU quant paper.
- [[quantization-mapping]] (Krishnamoorthi 2018 Google whitepaper) — the canonical PTQ taxonomy (sym/asym, per-tensor/channel, min-max/MSE) the industry still references.
- [[qjl]] (Zandieh 2024) — 1-bit Quantized JL transform for KV cache; zero per-block metadata overhead.
- [[polarquant]] (Han / Kacham / Zandieh, AISTATS 2026) — recursive polar transform after random preconditioning; closed-form angle distribution.
- [[turboquant]] (Zandieh, ICLR 2026) — random rotation + per-coordinate scalar quant + 1-bit QJL residual; matches rate-distortion bound up to constant.
- Gemma quant releases ([[gemma-quant]]) — official W4 / W8 variants of Gemma 2 / Gemma 3 with QAT phase.

## Recurring themes
- **Data-oblivious / no-calibration**: the KV trio's signature is "no calibration set needed" — exploit high-dimensional concentration after random rotation. This contrasts with the Hessian-based Western labs (Frantar/Alistarh) and the activation-aware Western labs (Han Lab, Dettmers).
- **Format invention as enabler**: bfloat16 in 2017 set up an entire training-precision agenda; the same lab DNA produced the integer-only-inference template a year later.
- **Theoretical grounding**: Zandieh / Mirrokni / Kacham anchor their KV quant work in rate-distortion theory and the Johnson-Lindenstrauss lemma; the proofs are part of the contribution.

## Open Resources
- Google Research blog (quantization tag): https://research.google/blog/
- TurboQuant blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- Gemma HF org: https://huggingface.co/google
- JAX / TPU bf16 + int8 conventions: https://github.com/google/jax

## Connections
- [[microsoft-bitnet]] — co-author overlap on MX-format work via Rouhani 2023; otherwise different directions (sub-2-bit training vs data-oblivious KV PTQ).
- [[nvidia-quantization]] — different hardware target (TPU vs GPU) but converging formats (bf16, FP8); the bf16 lineage is shared.
- [[dettmers-group]] / [[han-song-mit]] / [[frantar-alistarh-ist-austria]] — Western LLM-PTQ labs; Google's KV-trio is the orthogonal data-oblivious alternative to their calibration-based methods.

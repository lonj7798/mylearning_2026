<!-- scope: raw source library for course/model-quantization
     deps: [[COLLECTION-PLAN]]
     see-also: [[insights]], [[wiki/courses/model-quantization/outline]]
-->

# Model Quantization — Raw Source Library

This directory holds the primary source material the `course/model-quantization` course is built from. Every page here is an **extracted summary** of one artifact (paper, blog, tech report, framework module, hardware format spec). Course chapters cite these pages via wikilinks.

The course is **theory-first**: the early chapters lean on `classics/` (math + numerical-format fundamentals + pre-LLM quantization theory) before touching any LLM-specific algorithm. Empirical / production work in `papers/` and `model-reports/` is sequenced afterward.

## Scope

End-to-end LLM quantization with full theoretical grounding:

- **Math foundations**: entropy, rate-distortion, numerical precision, fixed-point vs floating-point representation, IEEE-754, block formats (MX/NVFP4), round-to-nearest-even, stochastic rounding, error propagation in matmuls.
- **Classical quantization theory**: uniform vs non-uniform quantizers, scalar vs vector quantization, k-means / Lloyd-Max, Bradley-Terry style sensitivity, KL-minimizing calibration, quantization noise modelling.
- **Pre-LLM neural-net quantization**: PTQ (post-training quantization) and QAT (quantization-aware training) lineage from CNNs and BERT-era transformers — straight-through estimator (STE), learned step size (LSQ), DoReFa, integer-only inference (I-BERT).
- **LLM quantization 2022–2026**: GPTQ → AWQ → SmoothQuant → QLoRA → OmniQuant → SqueezeLLM → SpQR → QuIP/QuIP# → AQLM → QuaRot → SpinQuant → HQQ → KV-cache line (KIVI/KVQuant/GEAR → QJL/PolarQuant/TurboQuant → adaptive KV / KVTC) → 1-bit line (BitNet, BitNet b1.58, BitNet a4.8) → FP8 training (DeepSeek V3) → MXFP4 / NVFP4 training and deployment (NVFP4 pretraining, Quartet II, NVFP4 QAD, native-hardware MXFP4 diagnostics).
- **Hardware + formats**: INT8/INT4 GEMM, FP8 (E4M3 / E5M2), FP6, FP4, MX (microscaling) formats, NVFP4 (Blackwell), Marlin / TensorRT-LLM / llama.cpp gguf k-quants.
- **Frameworks**: `bitsandbytes`, `AutoGPTQ`, `AutoAWQ`, `HQQ`, `llama.cpp`, `TensorRT-LLM`, `vLLM` quant integration, HuggingFace `quanto`, `torchao`.
- **Frontier reports**: DeepSeek V3 (FP8 native training), GPT-OSS MXFP4 release, Llama 3 (post-training quant), Qwen 2.5/3, Gemma, BitNet-scale 1-bit models, MiniMax-01.
- **Surveys / consolidations**: end-of-year quantization surveys, hardware vendor whitepapers (NVIDIA, AMD, Intel).

## Directory layout

```
raw-data/model-quantization/
├── README.md             this file
├── COLLECTION-PLAN.md    master topic checklist + source targets
├── insights.md           aggregated core-insights index (built last)
├── classics/             math + pre-LLM quantization theory
├── formats/              numerical format specs (FP8/FP4/MX/NVFP4)
├── papers/               arxiv + conference papers (flat; filename = slug)
├── model-reports/        frontier-model technical reports with quant details
├── blogs/                practitioner blogs, hardware vendor posts, lecture notes
├── frameworks/           OSS quant framework code excerpts
└── labs/                 per-lab capability summaries
```

## File-naming convention

- Slug-cased, no prefixes: `gptq.md`, `awq.md`, `smoothquant.md`, `bitnet-b158.md`.
- One artifact per file. Framework code goes in `frameworks/<framework>-<module>.md`.
- Format specs go in `formats/<format>.md` (e.g. `formats/fp8-e4m3.md`, `formats/mxfp4.md`).

## File format (required for every source page)

```markdown
<!-- scope: one-line description of what this source covers
     deps: prereq-source (optional)
     see-also: related-source
-->

# <Artifact title>
- **Core Insight:** one sentence — the thing this source is famous for
- **Guideline:** one sentence — what a practitioner should actually do
- **Authors:** ...
- **Year:** ...
- **URL:** ...
- **Relevant topics:** ...

## Abstract
(for papers) verbatim or faithful paraphrase

## Key Contributions
- 3–6 bullets

## Key Figures/Tables to Study
- which figure + one-line why

## Technical Details
(varies by source type — for quant algos include the quantization rule,
calibration objective, error metric, group/block size, kernel layout;
for formats include exponent/mantissa bits, dynamic range, special values,
block-scale rules; for frameworks include file paths + line references)

## Connections
- where this connects to other sources
```

## How this library is used

1. **Planner** reads `COLLECTION-PLAN.md` + this library to decide chapter granularity.
2. **Course chapters** (`wiki/courses/model-quantization/ch-*/read.md`) quote these pages via wikilinks and lift real equations / code from them.
3. **Insights index** (`insights.md`) is built last — one row per source, core insight + guideline, organized by theme.

Do not edit these pages to match course narrative. If a source changes interpretation during course writing, add a `Notes` section — don't overwrite the primary extract.

## Theory-first sequencing rule

The learner has explicitly asked for theoretical depth at the start. When the planner sequences chapters, the first chapters MUST be sourced from `classics/` and `formats/` only. No LLM-specific paper (GPTQ, AWQ, etc.) appears in the first ~5 chapters; those chapters establish:

1. Why quantization works at all (rate-distortion, the information-theoretic bound).
2. What numerical formats actually represent (FP / INT / block formats).
3. How quantization error propagates through a linear layer and an attention block.
4. The classical PTQ vs QAT distinction with STE.
5. The standard calibration objectives (MSE, KL, Hessian-weighted).

Only after these foundations does the course enter the 2022–2026 LLM literature.

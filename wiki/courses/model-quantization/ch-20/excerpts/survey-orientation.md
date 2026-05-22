---
chapter: ch-20
course: model-quantization
phase: read
excerpt_of: "The four 2024-era quantization surveys — when to read which"
created_at: "2026-05-21"
---

# Excerpt: The Four 2024 Surveys — Orientation Map

**Sources:** [[raw-data/survey-gholami-2021]], [[raw-data/survey-llm-quantization-2024]], [[raw-data/survey-low-bit-llm-2024]], [[raw-data/survey-efficient-llm-inference-2024]]

---

## Why four surveys, not one

The four surveys overlap but each has a distinct organizing principle. Reading the wrong one for your question wastes a day; reading the right one orients you in an hour.

| Survey | Year | Organizing axis | Coverage | Read this when … |
|--------|------|-----------------|----------|------------------|
| Gholami et al. | 2021 | uniform/non-uniform × symmetric/asymmetric × PTQ/QAT | pre-LLM CV+NLP foundations | learning vocabulary; cite for taxonomy definitions |
| Zhu et al. | 2024 | isolation / equivalent transformation / non-uniform code / vector quant | 2022–2024 LLM PTQ explosion (30+ methods) | navigating across LLM PTQ methods |
| Gong et al. | 2024 | bit-width cliff (uniform vs non-uniform vs VQ vs QAT) | sub-8-bit specific | planning sub-4-bit deployment |
| Yuan / Wan et al. | 2024 | inference roofline (compute-bound vs memory-bound) | quant + KV + speculative + kernels | prioritizing optimization work across multiple levers |

---

## Survey 1: [[survey-gholami-2021]] — pre-LLM canonical taxonomy

The pre-LLM canonical reference. Established the vocabulary every modern paper uses:

- **Uniform vs non-uniform** code.
- **Symmetric vs asymmetric** mapping.
- **PTQ vs QAT** calibration.
- **Per-tensor vs per-channel vs per-group** granularity.
- **Static vs dynamic** activation quantization.

Read this first if you're new to quantization. Cite this when you write "per-channel asymmetric INT8 with min-max calibration" — Gholami is where that vocabulary was canonicalised.

Blind spots: pre-dates outlier handling at scale, activation-aware methods, rotations, KV-cache quant, sub-2-bit, native low-precision training. Don't expect to learn anything LLM-specific from it.

---

## Survey 2: [[survey-llm-quantization-2024]] — Zhu et al., the LLM PTQ map

The most-cited LLM quantization survey of 2024. Organizes 30+ methods into four classes that map cleanly to the field's actual methodological branches:

1. **Isolation methods** — keep outliers in higher precision: LLM.int8, SpQR, OWQ.
2. **Equivalent transformation** — pre-multiply to flatten: SmoothQuant, AWQ, OmniQuant, QuaRot, SpinQuant.
3. **Non-uniform / vector code** — better-than-uniform codes: NF4, SqueezeLLM, QuIP, AQLM, GPTVQ.
4. **KV cache methods** — separate track: KIVI, KVQuant, GEAR.

When you read a new LLM-quant paper, place it in this 2×2×2 (target × calibration × code) plus outlier-handling class. Half the methods turn out to be the same idea with cosmetic differences once you've classified them.

Use the bit-width-vs-perplexity comparison table for orientation. Use the activation-outlier histogram figures (per-channel max-abs) to internalise *why* every equivalent-transformation method exists.

---

## Survey 3: [[survey-low-bit-llm-2024]] — Gong et al., the bit-width cliff

The sub-8-bit specialist. The headline finding is the **bit-width cliff** — a discontinuous quality drop at ~3 bits for uniform scalar quantization, regardless of method or model size:

```
PPL gap vs FP16 (Llama-7B, GPTQ uniform scalar):
  8 bits → 0.05  (negligible)
  4 bits → 0.15  (deployable)
  3 bits → 0.70  (cliff begins)
  2 bits → catastrophic (NaN-class divergence)
```

The cliff is *fundamental* to scalar quantization; better algorithms can shave 0.5 bits off the cliff edge but can't cross it. Crossing the cliff requires:
- **Vector quantization** (QuIP, AQLM, QuIP#) — d > 1 codes recover the 1.53 dB space-filling loss bound.
- **End-to-end QAT** (BitNet, BitNet b1.58) — the model learns representations that work natively at 1-bit.

Read this when planning anything below 4-bit. The cliff is the right mental model.

Also notable: documents the calibration-source-shift effects (used in [[excerpts/calibration-design]]).

---

## Survey 4: [[survey-efficient-llm-inference-2024]] — Yuan / Wan et al., the deployment roofline

The broader inference-optimization survey. Quantization is one of four orthogonal levers; the survey's value is in showing **which lever matters in which regime** via roofline analysis:

```
Llama-70B decode at batch-1 on H100:
  weight reads ~140 GB/token, FLOPs ~140 GFLOPs/token
  arithmetic intensity ~1 FLOP/byte
  H100 ridge ~30 FLOP/byte
  ⇒ memory-bandwidth bound
  ⇒ weight-only quantization (W4A16) is the dominant lever
```

But at large batch or in prefill, the bottleneck shifts to compute, and W4A16 stops helping — that's where W4A4 (rotation methods) or FP8/FP4 native precision matter.

The lever-stacking framing is the practical takeaway:

```
W4 weight (AWQ/GPTQ)  → 4× weight memory
KV4 (KIVI/KVQuant)    → 4× KV memory
PagedAttention         → batch-efficient
Speculative decoding   → ~2× decode speedup
FlashAttention 3       → ~30% prefill speedup
                       ──────────────
Net                    → 10–15× throughput vs FP16
```

Read this when you're deciding *what to optimize next* in a production deployment, not when you're studying one quantization algorithm.

---

## How to use the surveys together

Use case: you need to deploy a 70B model with 4× memory reduction, low latency, in production.

1. **Yuan / Wan (Survey 4)** — confirm the bottleneck is memory bandwidth → W4A16 is the right lever.
2. **Zhu (Survey 2)** — navigate to the W4A16 algorithm class (GPTQ vs AWQ vs OmniQuant); pick AWQ for OOD-robustness.
3. **Gong (Survey 3)** — verify 4 bits is on the safe side of the bit-width cliff for your model size; don't try 3 bits without VQ.
4. **Gholami (Survey 1)** — cite for the per-tensor/per-channel/per-group vocabulary in your design doc.

Use case: you need a 1.58-bit deployment at frontier scale.

1. **Gong (Survey 3)** — the bit-width cliff says you cannot do this with PTQ; must use end-to-end QAT.
2. **Zhu (Survey 2)** — confirms by listing only BitNet/BitNet b1.58/OneBit as the sub-2-bit end-to-end candidates.
3. Read the BitNet papers directly; the surveys cover them but the recipe details live in the papers.

---

## What none of the surveys give you

The surveys are *catalogs*. They don't give you:

- A reproducible PPL-or-better number for any specific (model, method, bit-width) combination — calibration-set-shift makes their cited numbers ±0.3 PPL.
- The exception policies (which layers to keep BF16 — see [[fp4-diagnosis]]).
- The distribution-lossless threshold (see [[statistically-lossless]]).
- The long-context failure modes under KV-quant (see [[long-context-eval]]).

The methodology of [[ch-20]] §1–§5 fills these gaps. The surveys orient; this chapter equips.

---

## Connections

- [[ch-20]] §7 — the chapter section that uses these surveys.
- [[gptq]] / [[awq]] / [[smoothquant]] — the most-cited methods across all four surveys.
- [[bitnet-b158]] — the QAT-cliff-crosser from Survey 3.
- [[kvquant]] / [[kivi]] — KV-cache lever from Survey 4.
- [[statistically-lossless-quantization]] — the methodology paper that exposes what the surveys leave unsaid.

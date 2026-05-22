<!-- chapter: ch-20
     track: eval-lab-capstone
     kind: content
     title: Quantization Evaluation Methodology
     deps: [ch-08, ch-15, ch-17]
     sources: [[statistically-lossless-quantization]], [[fp4-inference-diagnosis]], [[survey-gholami-2021]], [[survey-llm-quantization-2024]], [[survey-low-bit-llm-2024]], [[survey-efficient-llm-inference-2024]]
-->

# Chapter 20 — Quantization Evaluation Methodology

> **Core insight.** "Same perplexity" is not "same model." A single PPL number on Wikitext-2 is a lossy projection of model behavior onto one corpus's next-token distribution — it can move by <0.1 while MMLU drops 4 points, IFEval tanks, GSM8K loses two digits of math, or speculative-decoding acceptance falls off a cliff. **Statistically-lossless quantization** (Helcig–Kurtic–Alistarh 2026, [[statistically-lossless-quantization]]) makes the asymmetry explicit by separating *task-lossless* (benchmark scores within run variance) from *distribution-lossless* (Expected Acceptance Rate ≥ 0.99 against the FP16 next-token distribution). The two thresholds usually sit at different bit budgets: task-lossless ~4 bits, distribution-lossless ~5–6 bits.
>
> **Guideline.** For any deployment claim, report (1) PPL on at least two corpora — one in-domain, one out — with a confidence interval, (2) a multi-task suite (MMLU + GSM8K + IFEval + HumanEval + TruthfulQA at minimum), (3) per-layer or per-block sensitivity ablations a la [[fp4-inference-diagnosis]], (4) long-context probes (NIAH + RULER) when KV-quant is in play, and (5) the calibration set provenance (size, source, deduplication). If any one of these is missing, the claim is "vibes."

---

## Why this chapter exists

The previous twelve chapters introduced algorithms — [[gptq]] (ch-08), [[awq]] (ch-09), [[omniquant]], [[squeezellm]], [[qlora]], [[quip]], rotations (ch-14), [[kivi]]/[[kvquant]] (ch-15), BitNet (ch-16), [[deepseek-v3-fp8]] (ch-17), data-oblivious KV (ch-18), production kernels (ch-19). Each algorithm paper reports a number; many of those numbers don't survive contact with a multi-task harness or a long-context probe.

This chapter is the methodology stack that lets you (a) tell which reported numbers are reproducible, (b) catch the most common pathologies before they ship, and (c) build an evaluation harness that ranks methods on the actual deployment workload rather than on Wikitext-2 perplexity.

The four 2024 surveys ([[survey-gholami-2021]], [[survey-llm-quantization-2024]], [[survey-low-bit-llm-2024]], [[survey-efficient-llm-inference-2024]]) are the orientation reads — they catalog 30+ methods and give the bit-width × method × model perplexity tables. Use them for navigation. Use the methodology in this chapter for evidence.

---

## §1 Why perplexity lies

PPL is `exp(−(1/N) Σ log p(token_i | context))` averaged over a held-out corpus. The expectation is over *tokens*, not over downstream behaviors. Three failure modes are well-documented.

**Failure mode 1 — LayerNorm gain drift.** A small per-layer scaling error inside a quantized linear projection feeds into the LayerNorm/RMSNorm that follows it. Modern transformers use RMSNorm with a *learned scale*; that scale gets re-normalised at runtime, so the next-token distribution looks roughly unchanged for the bulk of tokens — PPL on Wikitext-2 barely moves. But the LayerNorm output magnitude shift compounds across 32 blocks. On in-context reasoning (chain-of-thought, multi-step arithmetic) where the model needs to copy specific tokens out of context, the rank of the correct token drops out of the top-k more often than the PPL drift suggests. Concrete shape from [[survey-llm-quantization-2024]]: GPTQ-W4 at `group_size=128` on Llama-2-7B gives PPL gap ~0.15 vs FP16, but `group_size=−1` (per-channel) gives PPL gap ~0.10 *and yet* MMLU drops 2–4 points more under the latter — the per-channel scale is too coarse for the LayerNorm dependency.

**Failure mode 2 — Sharpening / temperature drift.** Quantized models often have systematically more peaked or flatter logit distributions than their FP16 parents. PPL is insensitive to this (cross-entropy on the *true* next token does not care about the rank of the other vocab entries). But sampling-based generation does. Two attested symptoms from [[statistically-lossless-quantization]]: (i) speculative-decoding acceptance rates drop from 0.85 → 0.78 when the draft model is W4 and the verifier is FP16; (ii) instruction-following benchmarks like IFEval (which require exact-string format compliance) regress 3–5 points where MMLU regresses by <1.

**Failure mode 3 — Long-context calibration mismatch.** Calibration is almost always done on short sequences (512–2048 tokens). Quantization error grows roughly proportional to context length on the attention KV path. A model that PPL-matches FP16 at 2K tokens can degrade catastrophically at 32K — see §4 below for the NIAH evidence.

The taxonomy from [[statistically-lossless-quantization]] formalises this:

| Level | What is preserved | Typical bit budget | Right metric |
|-------|-------------------|--------------------|--------------|
| **Task-lossless** | Benchmark accuracy within run variance | ≤ 4 bits/parameter | multi-task suite ± CI |
| **Distribution-lossless** | Next-token distribution effectively unchanged | ~5–6 bits/parameter | **EAR ≥ 0.99** |
| **Exact lossless** | Exact weights and logits | storage compression only | bit equality |

The intermediate "distribution-lossless" level is the one that matters for downstream consumers of the logits — speculative decoding, knowledge distillation, RL reward modeling — and is the level that traditional quantization papers underreport.

---

## §2 Statistically-lossless quantization

The 2026 framework by Helcig–Kurtic–Alistarh [[statistically-lossless-quantization]] gives a principled answer to "is this quantization lossless?" by treating it as a *coupling* question between the FP16 and quantized next-token distributions.

### 2.1 Expected Acceptance Rate

Given two probability distributions over the vocabulary `p` (FP16) and `q` (quantized) for the same context, the optimal coupling probability — the probability that a sample from one distribution could equal a sample from the other under the best joint distribution — is

```
EAR(p, q) = Σ_v min(p(v), q(v))   ∈  [0, 1]
```

Aggregated over a held-out corpus:
```
EAR = E_context [ Σ_v min(p(v|context), q(v|context)) ]
```

EAR ≥ 0.99 means the two models can be coupled to sample the same token 99% of the time — i.e. for speculative decoding, this is exactly the maximum achievable acceptance rate.

### 2.2 Confidence intervals

For a quantization claim to be *statistically* lossless, the gap on each metric must be smaller than the natural run-to-run variance of the metric. From [[statistically-lossless-quantization]]:

- MMLU on Llama-2-7B has a run-to-run variance of ~0.3 percentage points across resamples of the test split. A quantization that drops MMLU by 0.4 pp is *statistically detectable*, even if the absolute drop sounds small.
- Wikitext-2 PPL on the same model has run-to-run variance ~0.02. PPL gaps of 0.05+ are real signal.
- EAR variance is ~0.003 at the 99% confidence band; an EAR drop from 0.995 → 0.987 is real.

The discipline: report `metric ± CI` from at least 5 calibration-seed reruns. Single-number "0.15 PPL gap" claims hide whether the gap is signal or noise.

### 2.3 Why asymmetric matters at the distribution level

A side result of the [[statistically-lossless-quantization]] paper: symmetric quantization (zero-point fixed at 0) inflates the next-token-distribution variance by a factor tied to the asymmetry of the weight distribution. For task-lossless results, symmetric quant is usually fine. For distribution-lossless results, asymmetric is necessary on instruction-tuned models whose post-RLHF weight distributions are skewed.

This is *not* an "asymmetric > symmetric" statement universally. It is a statement that the choice between them depends on which lossless level you care about. If you're consuming the logits (speculative decoding, distillation, RL), pay the asymmetric cost.

---

## §3 Per-layer / per-component sensitivity diagnostics

A model is not a uniform quantization target. The [[fp4-inference-diagnosis]] paper (Cim–Topcu–Kandemir 2026) establishes the now-canonical sensitivity ordering for FP4 inference on Qwen2.5:

```
MLP up/down projections  >>  gate projection  >  attention QKV/output projections
```

Concretely, on Qwen2.5-7B with MXFP4: quantizing only the MLP up/down projections to FP4 (and leaving everything else BF16) recovers ~80% of the FP4-everywhere degradation. Quantizing only the attention projections to FP4 (and leaving MLP BF16) recovers <10% of the degradation. The MLP is where FP4 hurts.

This generalises: the sensitivity ordering for any aggressive quantization on any transformer is approximately

```
1.  Embedding lookup            (low sensitivity — quantize freely)
2.  Attention output projection (very sensitive at W4A4)
3.  FFN up / gate               (most sensitive at FP4 / sub-4-bit)
4.  FFN down                    (most sensitive at FP4 / sub-4-bit)
5.  LM head                     (very sensitive — usually kept in BF16)
6.  RMSNorm                     (tiny FLOP share, keep BF16)
```

### 3.1 The diagnostic recipe

Per the [[fp4-inference-diagnosis]] methodology:

1. **Single-component swap**: quantize *only one component class* (e.g. only `mlp.down_proj` across all layers) to the target precision; leave everything else FP16. Measure PPL + MMLU. Repeat for each component class.
2. **Single-block swap**: quantize *only one transformer block* (all linear projections in block `i`) to the target precision; leave everything else FP16. Sweep `i = 0..L−1`. Plot the per-block sensitivity curve.
3. **Greedy mixed-precision**: starting from FP16, quantize the least-sensitive component first, then the next, until PPL exceeds a budget. The resulting mixed-precision recipe usually beats uniform quantization at the same average bit budget.

### 3.2 What [[fp4-inference-diagnosis]] documents

- MLP up/down projections dominate FP4 degradation (component ranking above).
- Early transformer blocks (block 0–2) are *more* sensitive than middle blocks. "Keep the last N layers high precision" — a common folk heuristic — is *incomplete*; early blocks need protection too.
- NVFP4 (16-element block, FP8 scale) and MXFP4 (32-element block, E8M0 scale) have *different* sensitivity profiles. MXFP4 amplifies sensitivity when a block contains a few high-energy channels; NVFP4 is more tolerant. They are not interchangeable in mixed-precision recipes.

### 3.3 The mixed-precision exception policy

Combining the [[fp4-inference-diagnosis]] findings with the [[survey-low-bit-llm-2024]] outlier characterization gives a practical exception policy for aggressive quantization:

```
Default            → FP4 / W4 everywhere (per the algorithm's recipe)
Raise to FP8/W8    → MLP up/down projections
Raise to BF16      → LM head, embedding, RMSNorm
Inspect            → blocks 0–2 (early-block sensitivity per fp4-diagnosis)
Optional           → attention output projection (W4A4 case; rotation usually handles it)
```

Document the exceptions. A reproducible quantized model checkpoint should ship with the layer-by-layer bit-budget map alongside the weights.

---

## §4 Long-context-specific evaluation

KV-cache quantization (chs 15 + 18) and weight quantization interact differently with long context. Three benchmarks have become standard.

### 4.1 NIAH — Needle in a Haystack

A single fact ("the magic password is 7392") is inserted at a known position in a long context, and the model is asked to retrieve it. The metric is binary accuracy as a function of (context length, needle depth).

KV-quant failure mode: under W16 + KV-INT2 (per-token), the model loses needles in the middle of long contexts. This is the "lost in the middle" phenomenon amplified by KV quantization noise. [[kivi]] reports >95% accuracy at 32K context with INT2 KV per its asymmetric (per-channel K, per-token V) scheme; naive per-token INT2 KV drops to <40% accuracy by the middle of the same context.

### 4.2 RULER — Realistic Universal Language Eval

13 task variants spanning retrieval, multi-hop, aggregation, and QA at context lengths from 4K to 128K. A model that aces NIAH can still fail RULER's multi-hop (needs to chain two facts) and aggregation (count or sum across the context) tracks.

KV-quant relevance: RULER's multi-hop track is the most sensitive to KV quantization noise because errors compound across the chained lookups. [[kvquant]]'s pre-RoPE quant + dense-and-sparse outlier handling holds up here; naive per-token KV quant collapses.

### 4.3 LongBench — Realistic Long-Context Tasks

A 21-task suite covering single-doc QA, multi-doc QA, summarization, few-shot learning, synthetic, and code completion at context lengths typically 4K–32K.

Use LongBench as the "is this deployment-ready?" check after KV quant. A model that loses >2 points on LongBench average vs FP16 KV is *not* deployment-ready, regardless of its NIAH score.

### 4.4 Failure-mode taxonomy under KV-quant

From [[kivi]], [[kvquant]], [[turboquant]] cross-referenced:

| Failure mode | Symptom | Likely cause | Diagnostic |
|--------------|---------|--------------|------------|
| Mid-context blindness | NIAH drops in the middle, fine at start/end | per-token K quant + RoPE; outlier channels get crushed | Switch to per-channel K (KIVI) or pre-RoPE K quant (KVQuant) |
| Multi-hop chain breaks | RULER multi-hop tanks; single-hop fine | KV error compounds across attention layers | Reduce KV bit-aggression; add sparse outlier path (KVQuant dense-and-sparse) |
| Position drift | NIAH accuracy a function of needle depth | RoPE-rotated K stored at low bits; angle quantization noise | Pre-RoPE quant; or data-oblivious rotation (TurboQuant) |
| Generation length cliff | Quality fine to N tokens, then collapses | per-token quant scales don't refresh fast enough on streaming | Smaller streaming group (KIVI g=32) |

Always pair KV-quant with long-context evals. PPL on Wikitext-2 won't catch any of these.

---

## §5 Calibration set design

Every PTQ algorithm consumes a calibration set. The choice of calibration set is the single most underdocumented hyperparameter in the field.

### 5.1 Size

Empirical defaults across the algorithms surveyed:

| Method | Default calibration size | Why |
|--------|-------------------------|-----|
| [[gptq]] | 128 sequences × 2048 tokens (~260K tokens) | Hessian estimation; saturates around 128 |
| [[awq]] | 128 sequences × 512 tokens (~65K tokens) | Per-channel activation magnitude; saturates fast |
| [[smoothquant]] | 512 samples | Activation amax per channel |
| [[omniquant]] | 128 sequences × 2048 tokens | Block-wise gradient optimization; needs full forward passes |
| [[quarot]] / [[spinquant]] | 128 sequences (Hadamard fixed; SpinQuant learns rotation) | Rotation calibration; size matters for SpinQuant only |
| QAT (BitNet, etc.) | Full pretraining corpus | Trains from scratch / extends pretraining |

**Why 128 sequences suffices for GPTQ.** GPTQ estimates a per-layer Hessian `H = 2 X Xᵀ`. The Hessian rank is bounded by `min(d_in, N_calib)` where `N_calib` is the number of activation samples. For `d_in = 4096` and `2048 tokens/sequence`, 128 sequences gives `128 × 2048 = 262144 ≫ d_in`; the Hessian is fully ranked and well-conditioned for percdamp=0.01. Above 128 sequences, marginal returns are <0.02 PPL.

**Why 128 sequences sometimes does NOT suffice for activation-aware methods.** AWQ's grid search over `α` is robust to small calibration sets because it estimates a *scalar* per layer. OmniQuant's learnable per-channel scales need more — the underlying gradient optimization has more parameters than GPTQ's closed-form update. Empirically, OmniQuant benefits from 256+ sequences on Llama-13B; SpinQuant from 512+.

### 5.2 Source

The biggest reproducibility gap. [[survey-low-bit-llm-2024]] documents:
- **C4 calibration → C4 evaluation:** 4-bit GPTQ PPL gap ~0.10.
- **C4 calibration → Wikitext-2 evaluation:** PPL gap ~0.15.
- **C4 calibration → MMLU evaluation:** sometimes 2 pp worse than Wikitext calibration.
- **Wikitext calibration → MMLU evaluation:** typically 0.5–1.5 pp better than C4 calibration, even though MMLU is multiple-choice not language modeling.
- **Domain-specific calibration (code, math, multilingual):** essential when the deployment domain differs from C4/Wikitext.

The rule: calibrate on a distribution that *covers* the deployment distribution. For a general-purpose chat model, a mix of C4 + alpaca-style + code is more robust than pure C4. For a code model, calibrate on code.

### 5.3 Distribution shift

A subtle pathology: instruction-tuned and RLHF'd models have *different* activation distributions than their base-model parents. GPTQ calibrated on Wikitext-2 against an instruction-tuned base will produce a model whose chat behavior regresses more than its Wikitext PPL suggests. The fix: calibrate on instruction-format text (alpaca / ultrachat / wildchat samples) when quantizing chat models.

The [[awq]] paper generalises better here than GPTQ exactly because it uses calibration only to estimate a scalar per layer — it can't overfit to its calibration corpus the way GPTQ's per-column Hessian update can.

### 5.4 Reproducibility checklist

When reporting a quantized model's numbers, the calibration set must be specified by:
1. Source corpus name and version (e.g., `allenai/c4 en validation 2023-09 snapshot`).
2. Number of sequences and tokens per sequence.
3. Sampling seed.
4. Pre-processing (tokenizer, deduplication, length filter).

Anything less is non-reproducible. The 0.1–0.5 PPL differences between published numbers and reproductions are usually traceable to calibration-set choices that the paper didn't fully specify.

---

## §6 The evaluation harness — what to actually report

Compose §1–§5 into a deliverable. For any quantized model claim, the minimum reportable matrix is:

| Axis | Required entries |
|------|-----------------|
| **PPL** | Wikitext-2 + C4 + one in-domain corpus, each with 5-seed mean ± CI |
| **Knowledge** | MMLU (5-shot) ± CI |
| **Math** | GSM8K (8-shot CoT) + minerva-math sample |
| **Reasoning** | BBH (3-shot) or ARC-challenge |
| **Code** | HumanEval (0-shot) + MBPP |
| **Instruction following** | IFEval (strict) |
| **Calibration** | TruthfulQA (multiple-choice) |
| **Long context** (if KV-quant) | NIAH + RULER subset + LongBench average |
| **Distribution fidelity** | EAR vs FP16 base on a 1K-token held-out corpus |
| **Inference** | Decode latency at batch 1 + 16 + 64; peak VRAM; tokens/sec |
| **Calibration spec** | Per §5.4 |
| **Sensitivity ablation** | Per-component map per §3 |

This is a lot. The point of the long list is that **every one of these axes catches a different failure mode**. If a method's report covers only PPL + MMLU, you cannot tell whether IFEval will hold. If it omits long-context, you cannot tell whether the model works past 8K tokens. If it omits EAR, you cannot tell whether speculative decoding will benefit.

The [[ch-21]] lab implements this harness end-to-end. The [[ch-22]] capstone uses it to grade a reproduction against a paper's reported numbers.

---

## §7 The four 2024 surveys — when to read each

A short field guide. Use these for orientation, not for evidence.

- **[[survey-gholami-2021]]** — pre-LLM canonical taxonomy. Read first if you're new to quantization vocabulary. Does not cover anything from the LLM era. Cite for any per-tensor/per-channel/per-group/symmetric/asymmetric definitions.
- **[[survey-llm-quantization-2024]]** (Zhu et al.) — 2022–2024 LLM quantization consolidated. The 4-class taxonomy (isolation / equivalent transformation / non-uniform code / vector quant) is the right mental model. Use for navigation across 30+ methods.
- **[[survey-low-bit-llm-2024]]** (Gong et al.) — sub-8-bit specific. The bit-width cliff figure (uniform scalar hits a wall at 3 bits; vector quant crosses it; QAT crosses 1-bit) is the headline. Use when planning a sub-4-bit deployment.
- **[[survey-efficient-llm-inference-2024]]** — broader inference optimization survey; quantization is one chapter. The roofline analysis (decode is memory-bound; weight-only quant is the dominant lever) is the right framing for production deployment. Use when prioritizing optimization work across quant + KV + speculative + kernels.

All four surveys agree on one thing: the gap between "PPL number in the paper" and "what the model does in production" is the unsolved problem. This chapter is the methodology stack you use to close that gap on your own deployments.

---

## §8 Practitioner's checklist

A one-page version of this chapter:

```
[ ] Calibration set documented (corpus + size + seed + preprocessing)
[ ] PPL on ≥ 2 corpora with 5-seed CI
[ ] MMLU + GSM8K + IFEval + HumanEval + TruthfulQA, all with CI
[ ] Per-component sensitivity map (which layers were exceptions)
[ ] If KV-quant: NIAH + RULER + LongBench at deployment context length
[ ] If consumer is speculative-decoding / distillation / RL: EAR ≥ 0.99 reported
[ ] Inference: latency + VRAM + throughput at deployment batch size
[ ] Statistical claim: every gap reported with confidence interval
[ ] Reproducibility: calibration seed + harness version + tokenizer version
```

Anything checked → defensible. Anything missing → revisit before shipping.

---

## Connections

- **Back to [[gptq]] (ch-08), [[awq]] (ch-09)** — the W4 PTQ workhorses whose numbers this methodology was built to audit.
- **Back to [[kivi]] (ch-15), [[kvquant]] (ch-15)** — KV-cache methods that fail in long-context-specific ways §4 is designed to expose.
- **Back to [[deepseek-v3-fp8]] (ch-17)** — frontier FP8 pretraining; the <0.25% relative loss claim is exactly the kind of statistically-lossless argument §2 formalises.
- **Forward to [[ch-21]]** — the lab applies this harness to four W4 methods head-to-head.
- **Forward to [[ch-22]]** — the capstone uses §1–§5 to grade a paper reproduction against its reported numbers.
- [[statistically-lossless-quantization]] — the 2026 paper that gave the field a vocabulary for what "lossless" actually means.
- [[fp4-inference-diagnosis]] — the per-component sensitivity recipe §3 codifies.

## Excerpts

- [[excerpts/statistically-lossless]] — EAR, the three losslessness levels, and the asymmetric-variance argument.
- [[excerpts/fp4-diagnosis]] — the component-sensitivity heatmap and the early-block fragility finding.
- [[excerpts/long-context-eval]] — NIAH / RULER / LongBench methodology under KV-quant.
- [[excerpts/calibration-design]] — size / source / distribution-shift recipes.
- [[excerpts/survey-orientation]] — when to read which of the four 2024 surveys.

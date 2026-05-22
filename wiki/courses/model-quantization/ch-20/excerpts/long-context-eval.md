---
chapter: ch-20
course: model-quantization
phase: read
excerpt_of: "Long-context evaluation methodology synthesized from KIVI / KVQuant / TurboQuant / RULER / LongBench"
created_at: "2026-05-21"
---

# Excerpt: Long-Context Evaluation under KV-Quantization

**Sources synthesised:** [[raw-data/kivi]], [[raw-data/kvquant]], [[raw-data/turboquant]], RULER/NIAH/LongBench standard methodology
**Year:** 2024–2026

---

## The three benchmark families

### NIAH — Needle in a Haystack

A controlled retrieval test. A short, deterministic fact ("the magic password is 7392") is inserted at a known position (depth) inside a long context, and the model is asked to retrieve it. Metric: binary accuracy as a function of `(context_length, needle_depth)`.

The output is a 2-D heatmap: rows = context length (4K, 8K, 16K, 32K, 64K, 128K), cols = needle depth (0%, 10%, ... 100%). A FP16 model under healthy KV typically shows full-green at all positions. A model with KV quantization pathology shows a "lost in the middle" band — green at the edges, red in the middle 30–60% depth range, worsening with context length.

### RULER — Realistic Universal Language Eval

13 task variants spanning four categories:
- **Retrieval** — single-needle, multi-needle, multi-key variants of NIAH.
- **Multi-hop** — chain two retrievals (find fact A, then use A to find fact B).
- **Aggregation** — count, sum, or aggregate something across the context.
- **QA** — answer a question requiring information from the long context.

Tested at context lengths 4K, 8K, 16K, 32K, 64K, 128K. Multi-hop is the most KV-quant-sensitive track because retrieval errors compound across the chain.

### LongBench — Realistic Long-Context Tasks

21 tasks across 6 categories: single-doc QA, multi-doc QA, summarization, few-shot learning, synthetic, code completion. Typical context 4K–32K. Closest to "is this deployment-ready?" because it mirrors real long-context use cases.

---

## KV-quant failure-mode taxonomy

Cross-referenced from [[kivi]], [[kvquant]], [[turboquant]]:

| Failure mode | Symptom | Likely cause | Fix |
|--------------|---------|--------------|-----|
| **Mid-context blindness** | NIAH drops in the middle, fine at start/end | per-token K quant with RoPE; outlier channels in K get crushed by per-token scale | Switch to per-channel K (KIVI); or pre-RoPE K quant (KVQuant) |
| **Multi-hop chain breaks** | RULER multi-hop tanks; single-hop fine | KV error compounds across attention layers | Reduce KV bit-aggression; add sparse outlier path (KVQuant dense-and-sparse) |
| **Position drift** | NIAH accuracy a strong function of needle depth | RoPE-rotated K stored at low bits; angle quantization noise | Pre-RoPE quant (KVQuant); or data-oblivious rotation (TurboQuant) |
| **Generation length cliff** | Quality fine to N tokens, then sharply collapses | per-token quant scales don't refresh fast enough on streaming | Smaller streaming group (KIVI g=32) |
| **Aggregation fail** | RULER count/sum tasks degrade; retrieval OK | quantization noise on V accumulates in attention-weighted sum | Per-token V at higher bits; or non-uniform V code (KVQuant) |

---

## Why naive per-token KV-INT2 fails NIAH

The mechanism: K cache has *channel-wise* outliers (a few persistent channels carry magnitudes 10–100× the bulk — the RoPE-induced + residual-stream pattern). Per-token quantization picks one scale per token, dominated by these outlier channels. The bulk channels get crushed to a tiny portion of the INT2 dynamic range, losing all retrieval-relevant precision.

[[kivi]]'s asymmetric scheme fixes this: K per-channel (each problem channel gets its own scale), V per-token (V doesn't have channel-wise outliers). INT2 viable, NIAH accuracy >95% at 32K context.

[[kvquant]] pushes further to sub-2-bit by adding pre-RoPE quantization (RoPE mixes channels and destroys the per-channel structure) and dense-and-sparse decomposition (top 1% outliers in FP16).

[[turboquant]] eliminates calibration entirely by random-rotation + per-coordinate scalar quant + 1-bit QJL residual — data-oblivious, online-feasible, hits the rate-distortion bound up to a constant.

---

## What to actually run

For a KV-quant evaluation:

```
1. NIAH heatmap   at deployment context lengths × 11 depth points (0%–100% in 10% steps)
2. RULER 4-task subset (single-needle, multi-needle, multi-hop, aggregation)
   at deployment context length only
3. LongBench average across 6 categories at deployment context length
4. Baseline: same model at FP16 KV
5. Compare: per-category accuracy delta, not just average
```

Mid-context blindness shows up in NIAH but is invisible in per-category LongBench averages. Multi-hop degradation shows up in RULER but is partially hidden in LongBench (LongBench's multi-doc QA is a weaker version of RULER multi-hop). Both matter; both have to be measured.

---

## What FP16-PPL hides

A model that PPL-matches FP16 on Wikitext-2 at 2K context can:
- Lose 30 percentage points on NIAH at 32K context.
- Drop from 80% to 40% on RULER multi-hop.
- Lose 5–10 points on LongBench multi-doc QA.

PPL on short corpora simply does not test the long-context attention pathway that KV quantization stresses. The evaluation harness must include long-context probes when KV-quant is in play. No exceptions.

---

## Connections

- [[ch-20]] §4 — the chapter section that codifies this methodology.
- [[kivi]] — the asymmetric per-channel-K / per-token-V baseline.
- [[kvquant]] — sub-4-bit via pre-RoPE + non-uniform + dense-and-sparse.
- [[turboquant]] — data-oblivious KV quant via random rotation + QJL residual.
- [[ch-15]] — KV-cache quantization chapter (parent).
- [[ch-18]] — data-oblivious KV chapter.
- [[ch-22]] — the capstone that reproduces one of these methods end-to-end.

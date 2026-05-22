---
chapter: ch-18
course: model-quantization
phase: read
excerpt_of: "Keep the Cost Down: A Review on Methods to Optimize LLM's KV-Cache Consumption (Shi et al., COLM 2024) + KV-Cache Compression Survey 2025"
source_url: https://arxiv.org/abs/2407.18003
created_at: "2026-05-21"
---

# Excerpt: KV-cache compression — three-axis taxonomy + composition guidance

**Authors:** Luohe Shi, Hongyi Zhang, Yao Yao, Zuchao Li, Hai Zhao (Shi 2024); various survey authors (2025 update)
**Year:** 2024 (COLM); 2025 update
**URL:** https://arxiv.org/abs/2407.18003 + representative 2025 surveys
**Raw-data source:** [[raw-data/kv-cache-survey]] + [[raw-data/kv-cache-compression-survey-2025]]

---

## The three-axis taxonomy

KV-cache optimization fractures into **three orthogonal axes**. The 2024–2026 surveys converge on the same operational rule: pick one technique from each axis and compose, rather than pick the single best technique on each axis alone.

### Axis 1 — Architectural (pre-training)

Requires retraining. Not retroactively applicable to existing FP models.

| Method | What it does | KV reduction |
|--------|--------------|--------------|
| **MQA** (Multi-Query Attention) | all heads share one K and one V | 32× for 32-head model |
| **GQA** (Grouped-Query) | `g` heads share one K/V pair | factor-`g` reduction; Llama-3 uses GQA-8 |
| **Multi-Latent Attention (MLA)** | project K/V into low-dim latent | 8–16×; DeepSeek V2/V3 |

### Axis 2 — Deployment (post-training)

Quantization is the bulk of this axis. This whole chapter (and ch-15) is about Axis 2.

| Method | Bit budget | Calibration |
|--------|-----------|-------------|
| **KIVI** | 2-bit K (per-channel) + 2-bit V (per-token) | yes |
| **KVQuant** | sub-2-bit V + non-uniform 4-bit K + sparse outliers | yes |
| **GEAR** | W4 + low-rank residual | yes |
| **QJL** | 1-bit JL sketch | **no** (data-oblivious) |
| **PolarQuant** | < 4 bits via angle quant | **no** |
| **TurboQuant** | 2.5–3.5 bit | **no** |

Also:
- **Low-rank decomposition** (GEAR's residual; LESS factorization).
- **Distillation into smaller K/V** (DistillKV — trains compact replacement KV head).

### Axis 3 — Inference (runtime)

| Method | What it does |
|--------|--------------|
| **H2O** (heavy-hitter) | keep tokens with high cumulative attention score |
| **StreamingLLM** | sink tokens (first 4) + sliding window; no retraining |
| **Quest** | query-aware page-level top-k retrieval from KV cache |
| **Sliding window** | Mistral's SWA, Longformer fixed-size context |

---

## Composition guidance

The surveys' practical contribution: how axes interact.

### Orthogonal pairs (compose freely)

- **Quant + Architectural:** GQA + INT4 KV ≈ 16× cumulative, little interaction.
- **Eviction + Architectural:** GQA + StreamingLLM compose cleanly.

### Interactive pairs (need care)

- **Quant + Eviction:** *can compound errors*. Quantization noise affects eviction's heavy-hitter detection — H2O scores on quantized K may differ from FP K. The 2025 fix: **quant-aware eviction policies** — evict tokens whose quantized representation has small quant error (sacrifice precision-redundant tokens, keep precision-critical ones).
- **Architectural reduction (GQA) + Quant:** GQA reduces redundancy quant relied on for averaging out noise. May need slightly more bits on GQA models than on MHA models at the same quality target.

### Example budget (7B / 1M context / 80 GB GPU)

| Axis | Choice | Reduction |
|------|--------|-----------|
| 1: Architectural | GQA-8 | 8× |
| 2: Quantization | TurboQuant 3-bit (or KVQuant 2-bit) | ~5× |
| 3: Eviction | StreamingLLM (4K sink + 4K recent) | variable |
| **Cumulative** | | **> 100×** vs full FP16 KV |

Enables the target context on one 80 GB GPU.

---

## Long-context evaluation — the methodological warning

**Perplexity is dominated by near-context tokens** → doesn't surface long-context KV compression errors.

NIAH (Needle In A Haystack), RULER, LongBench are the standard suite. The 2025 surveys consistently show:

- 2-bit KV without quant-aware eviction can **lose 20+ points on RULER** where perplexity moves by < 0.1.
- LongBench / NIAH / RULER are required for any 2026 KV-quant claim.
- Calibration-set design (size, source, distribution shift) matters more for activation-aware methods than for GPTQ-style (128 sequences usually sufficient for GPTQ but not for activation-aware KV methods).

This is one of the most important methodological points in the chapter — **report long-context numbers, not just PPL**.

---

## K vs V asymmetry — still true in 2026

Old observation from [[kivi]] / [[kvquant]]:

- **K** has channel-aligned outliers (RoPE-induced; channel `i` may have systematically larger magnitude than channel `j` across all tokens).
- **V** is roughly Gaussian per token (channel structure less pronounced).

Per-channel scale for K + per-token scale for V is the calibration-based fix.

**The data-oblivious (this chapter's) twist:** rotation absorbs this asymmetry. After a random Hadamard, both K and V coordinates concentrate on the same Beta distribution — one quantizer fits both. The asymmetry was an artifact of working in the raw coordinate frame.

This is another reason rotation-based KV-quant generalizes better than per-channel calibration — fewer assumptions about which tensor needs which granularity.

---

## What surveys flag as open in 2026

- **Coupling between axes:** quant noise vs heavy-hitter detection vs sliding-window decisions — under-studied.
- **Dynamic-vs-static bit allocation:** Adaptive KV ([[adaptive-kv-cache-quant]]) is one entry; the design space is wide open.
- **Cross-batch sharing:** can KV from one request inform compression decisions for another? Mostly unexplored.
- **Persistent vs live storage:** KVTC ([[kvtc]]) is an early entry; how to split the bit budget between live attention and offload remains active.

---

## Connections

- Quant-axis primaries: [[kivi]] / [[kvquant]] / [[gear]] / ch-15; [[qjl]] / [[polarquant]] / [[turboquant]] / this chapter.
- [[per-channel-vs-per-token-kv]] — analytical companion to the K vs V asymmetry.
- [[wkvquant]] / [[skvq]] / [[qaq]] / [[coupling-kv-quant]] / [[coupled-quant-eviction]] — additional 2024–2025 KV-quant variants.
- Architectural compression (out of chapter but referenced): MQA, GQA, MLA.
- Eviction (out of chapter but referenced): H2O, StreamingLLM, Scissorhands, Quest.
- [[adaptive-kv-cache-quant]] / [[kvtc]] / [[excerpts/kvtc-and-adaptive]] — the 2026 orthogonal-axes additions.
- [[ch-18]] — parent synthesis.

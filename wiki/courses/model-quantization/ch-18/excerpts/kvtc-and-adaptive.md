---
chapter: ch-18
course: model-quantization
phase: read
excerpt_of: "KVTC (Staniszewski, Lancucki, ICLR 2026) + Adaptive KV-Cache Quantization (Boroujeni et al., CVPR 2026)"
source_url: https://arxiv.org/abs/2511.01815 + https://arxiv.org/abs/2604.04722
created_at: "2026-05-21"
---

# Excerpt: KVTC + Adaptive KV — the 2026 orthogonal axes

**Authors:** Konrad Staniszewski, Adrian Lancucki (KVTC); Sayed Pedram Haeri Boroujeni et al. (Adaptive)
**Year:** 2025 / 2026
**URL:** https://arxiv.org/abs/2511.01815 + https://arxiv.org/abs/2604.04722
**Raw-data source:** [[raw-data/kvtc]] + [[raw-data/adaptive-kv-cache-quant]]

---

## KVTC — transform coding for reusable KV caches

[[kvtc]] reframes KV-cache compression as **media-codec-style transform coding** rather than plain scalar quantization. The pipeline borrows directly from JPEG / Opus:

1. **Decorrelate** features via PCA-like transform.
2. **Quantize** transformed coefficients with adaptive precision.
3. **Entropy-code** for compact storage.
4. **Decode** when the cache is reused.

### What it's for

KVTC is **not** a drop-in replacement for live low-bit attention kernels (KIVI / TurboQuant). The decode cost makes it wrong for the per-token attention path. It's the right tool when:

- Large prefixes repeat across requests (chat history, system prompt, iterative code editing).
- Caches need to be stored compactly *between* turns or requests (offload to CPU / disk).
- The compression budget can amortize a one-shot decode cost.

### Result

- **Up to 20× compression** while maintaining reasoning + long-context accuracy.
- Benchmark coverage: AIME25, GSM8K, LiveCodeBench, LongBench, MATH-500, MMLU, Qasper, RULER.
- Tested across Llama 3, Mistral NeMo, R1-Qwen 2.5.
- Higher compression in selected use cases.

### Why it matters for the chapter

KVTC is the *classical-compression* angle on KV caches. The triad "decorrelate, quantize, entropy-code" is the same skeleton as JPEG / MP3 / Opus / H.264 — proven over decades on media signals. KV-cache compression in 2026 borrows from this lineage when the deployment problem is *persistent* storage rather than live decode.

Connects ch-18 to ch-01's rate-distortion thread on the practical side: rate-distortion theory predicts that decorrelation + scalar quant + entropy coding is the right *order* for compression, and KVTC instantiates this for KV caches.

---

## Adaptive KV — per-token bit allocation

[[adaptive-kv-cache-quant]] takes an orthogonal axis: **token-dependent bit-width**.

### The controller

A compact controller selects per-token precision from `{2-bit, 4-bit, 8-bit, FP16}` during decoding. Inputs:

- **token frequency**
- **token quality score**
- **attention variance**
- **entropy / uncertainty signal**

These features estimate whether a token's cached K/V will matter enough to justify extra bits.

### Precision menu

| Precision | Intended use |
|-----------|--------------|
| 2-bit | low-impact tokens |
| 4-bit | default compressed cache |
| 8-bit | moderately important tokens |
| FP16 | highly important tokens |

### Where it fits

Tested at on-device / edge scale (SmolLM-135M / 360M / 1.7B on CVPR 2026). Improved accuracy-latency frontier vs static KV quantization and rule-based baselines.

Not a universal KV quantizer. But a useful pointer: 2026 KV compression moved from "pick one bit-width" to "*allocate* bits dynamically."

### How it differs from KIVI / KVQuant / TurboQuant

KIVI, KVQuant, and TurboQuant are *fixed-method* quantizers — they prescribe a specific layout (per-channel K / per-token V, dense-and-sparse, rotation + scalar + QJL) and apply it uniformly. Adaptive KV is a **bit-allocation policy** layered *over* the quantizer choice. The two compose: TurboQuant at 3-bit on most tokens, FP16 on attention-variance-flagged tokens.

---

## The orthogonal-axes story

The 2026 KV-cache compression frontier has split into orthogonal axes:

| Axis | Example | Compose with |
|------|---------|--------------|
| **Quantization** (per-element bit-width) | TurboQuant, KIVI, KVQuant | — |
| **Bit allocation** (per-token) | Adaptive KV | any quantizer |
| **Eviction** (drop tokens) | H2O, StreamingLLM, Quest | any quantizer (quant-aware) |
| **Architectural** (pre-training) | MQA, GQA, MLA | any of the above |
| **Storage codec** (reusable cache) | KVTC | offline / persistent |

The [[kv-cache-compression-survey-2025]] message: compose, don't pick. The 2026 production stack is typically:

- GQA-8 (architectural) +
- TurboQuant 3-bit (quantization) +
- Adaptive bit allocation per token (or StreamingLLM eviction) +
- KVTC for persistent prefix storage (when relevant)

Cumulative: > 100× KV reduction vs FP16 with bounded quality loss.

---

## Connections

- [[turboquant]] / [[excerpts/turboquant]] — the data-oblivious quantizer KVTC and Adaptive KV compose with.
- [[kv-cache-compression-survey-2025]] — the three-axis taxonomy that frames KVTC and Adaptive KV as orthogonal axes.
- [[kvquant]] / [[kivi]] / [[gear]] / ch-15 — calibration-based baselines from the 2024 wave.
- [[product-quantization]] / [[vector-quantization]] / ch-03 — classical compression lineage KVTC borrows from.
- [[qaq]] — the closest earlier theme to adaptive KV: quality-adaptive KV-cache quantization.
- [[ch-18]] — parent synthesis.

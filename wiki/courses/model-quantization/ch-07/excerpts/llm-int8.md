---
chapter: ch-07
course: model-quantization
phase: read
excerpt_of: "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (Dettmers, Lewis, Belkada, Zettlemoyer 2022)"
source_url: https://arxiv.org/abs/2208.07339
arxiv: 2208.07339
created_at: "2026-05-21"
---

# Excerpt: LLM.int8() — outlier emergence and the mixed-precision fix

**Authors:** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
**Year:** August 2022
**Raw-data source:** [[raw-data/papers/llm-int8]]

---

## The two-line algorithm

The whole method, on one slide:

```math
c_{x,i} \;=\; 127 / \max_j |X_{ij}| \quad \text{(per-token activation scale)}
```

```math
c_{w,j} \;=\; 127 / \max_i |W_{ij}| \quad \text{(per-output-channel weight scale)}
```

```math
X \cdot W \;=\; \underbrace{\sum_{i \in O} X_{:,i}\, W_{i,:}}_{\text{FP16}}
            \;+\; \underbrace{\sum_{i \in R} X_{:,i}\, W_{i,:}}_{\text{INT8 (vector-wise scales)}}
```

with outlier set:

```math
O \;=\; \{\,i : \exists\, j,\; |X_{ji}| \ge \alpha\,\}, \quad \alpha = 6.0
```

Two formulas. The whole paper is the empirical defence of those formulas.

---

## The emergence finding (the load-bearing observation)

Naive absmax INT8 quantization of OPT/BLOOM holds accuracy up through ~2.7B parameters, then **crashes** at 6.7B:

| Model | FP16 acc | Naive INT8 acc |
|---|---|---|
| OPT-1.3B | 56.4 | 55.6 |
| OPT-2.7B | 60.7 | 60.0 |
| **OPT-6.7B** | **65.9** | **41.6 (crash)** |
| OPT-13B | 68.0 | 31.5 |
| OPT-66B | 71.5 | 24.3 |

The crash is a **phase transition**, not a gradient. Something qualitatively new happens at ~6.7B.

---

## What the phase transition is

A small set of hidden dimensions (≈6 in a 6.7B model out of ~14k hidden dims) suddenly coordinate across all layers and produce activations 20–100× larger than the rest. A single absmax INT8 scale per row gets dominated by these outliers, leaving the other 99.9% of values represented by 1–2 quantization levels.

Three quantitative findings:

1. **Structural sparsity.** ~150k outlier values per sequence at 6.7B live in only **6 distinct feature dimensions**.
2. **Sign coherence.** Outlier dims have consistent signs across layers — implying downstream layers depend on them.
3. **Rapid growth.** Peak magnitude grows ~15 (6B) → ~60 (13B) → ~95 (66B).

---

## Why vector-wise scale alone isn't enough

Vector-wise (per-token × per-channel) helps a lot:

| Model | Naive INT8 | Vector-wise INT8 | LLM.int8() |
|---|---|---|---|
| OPT-6.7B | 41.6 | 65.5 | 65.9 |
| OPT-66B | 24.3 | 70.8 | 71.4 |
| OPT-175B | (crash) | collapses | 71.9 |

But past ~66B, even per-token scales get dominated when multiple tokens fall into the outlier dimensions simultaneously. That's where the FP16 outlier path becomes mandatory.

---

## Why α = 6.0 specifically

Empirically, the outlier features at the emergent scale have magnitudes **15–100**; non-outlier dims stay below **6**. The 6.0 cutoff:

| α | Captures all outliers? | False-positive rate |
|---|---|---|
| 3.0 | yes | ~1% (bloats FP16 path) |
| **6.0** | **yes** | **< 0.1%** |
| 10.0 | misses some at 13B+ | 0% |
| 20.0 | fails | model collapses |

6.0 is the empirical sweet spot that captures all outlier dims at every tested scale while keeping the FP16 path below 0.1% of dims.

---

## What the mixed-precision split costs

Memory: ~50% of FP16 (the INT8 path dominates).

Compute: typically ~5–10% overhead for the FP16 path (it's only 6 columns out of 12000), but kernel-launch and stream-synchronisation overhead dominates at small batch sizes. End-to-end latency is roughly FP16-parity at batch 1, faster at batch 8+ when the INT8 path memory-bandwidth savings kick in.

---

## What it doesn't solve

| Open problem | Solved by |
|---|---|
| Weight-only at 4-bit (smaller memory) | [[gptq]] (ch-08) |
| Migrate outliers into weights | [[smoothquant]] (ch-09), [[awq]] (ch-09) |
| Preserve outlier *weights* | [[spqr]] (ch-11) |
| Eliminate outliers via rotation | [[quip]] (ch-13), [[quarot]] (ch-14) |
| Sub-2-bit | [[aqlm]] (ch-14) |

Every one of these traces back to the LLM.int8() outlier observation.

---

## Implementation

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", quantization_config=cfg)
```

Under the hood, every `nn.Linear` becomes `bnb.nn.Linear8bitLt` which (per forward pass) detects outlier columns, splits the GEMM, and adds the two paths.

---

## Common pitfalls

- **Assuming small models will have the same behavior.** At 1.3B, vanilla absmax INT8 works fine. Conclusions drawn from sub-6.7B benchmarks do not transfer to 7B+.
- **Just clipping activations > 6.0.** Destroys the model — outliers encode a routing decision, not noise. Confirmed by Dettmers blog.
- **Lowering threshold to 3.0 for "safety".** The FP16 path bloats and you lose memory savings without gaining accuracy.
- **Believing INT8 is "lossless" universally.** Below 6.7B yes; above it requires the decomposition trick. There is no universal INT8.

---

## Connections

- [[excerpts/dettmers-llm-int8-blog]] — the practitioner-facing companion with the dual-stream interpretation.
- [[ch-07]] — parent synthesis.
- [[ch-08]] — [[gptq]] is the weight-only successor.
- [[ch-09]] — [[smoothquant]] migrates the problem instead of isolating it.

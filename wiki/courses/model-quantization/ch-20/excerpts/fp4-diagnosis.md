---
chapter: ch-20
course: model-quantization
phase: read
excerpt_of: "Diagnosing FP4 Inference: Layer-wise and Block-wise Sensitivity Analysis (Cim, Topcu, Kandemir 2026)"
source_url: https://arxiv.org/abs/2603.08747
created_at: "2026-05-21"
---

# Excerpt: FP4 Inference Sensitivity Diagnostics

**Authors:** Musa Cim, Burak Topcu, Mahmut Taylan Kandemir
**Year:** 2026
**arXiv:** 2603.08747
**Raw-data source:** [[raw-data/fp4-inference-diagnosis]]

---

## The component-sensitivity ranking

The headline empirical finding, on Qwen2.5 at 0.5B / 7B / 14B scales under both NVFP4 and MXFP4:

```
MLP up/down projections  >>  gate projection  >  attention QKV/output projections
```

Quantitatively (Qwen2.5-7B, MXFP4):
- Quantize only `mlp.up_proj` + `mlp.down_proj` across all layers → recovers ~80% of full-FP4 degradation.
- Quantize only attention (`q_proj` + `k_proj` + `v_proj` + `o_proj`) across all layers → recovers <10% of full-FP4 degradation.
- Quantize only `mlp.gate_proj` → recovers ~25% of full-FP4 degradation.

The conclusion is sharp: **FP4 sensitivity lives almost entirely in the MLP up/down projections**. If you have a precision budget to spend in mixed precision, spend it there first.

---

## Why MLP up/down is the worst

Two mechanisms compound:

1. **High channel dimension.** MLP intermediate dimension is typically 2–4× hidden_dim. A 16-element FP4 block in MLP-up covers a smaller fraction of the channel space than the same block in attention — outlier energy is more concentrated per block.
2. **SwiGLU activation amplifies outliers.** `silu(gate) * up` is a multiplicative gate that amplifies any outliers passed through the gate path. The post-activation distribution entering `down_proj` carries those amplified outliers; FP4's 4-bit dynamic range can't represent them without crushing the bulk.

This connects to the [[awq]] and [[smoothquant]] outlier observations — the same MLP-heavy paths are also the worst for activation quantization. The mechanism is consistent.

---

## Early-block fragility

A second finding that contradicts a common folk heuristic. The folk heuristic: "keep the last N layers high precision because they're closest to the loss." The empirical truth from this paper:

```
Block sensitivity (Qwen2.5-7B, MXFP4):
  block 0–2:  HIGH (early-layer fragility)
  block 3–N-3: medium
  block N-2, N-1: HIGH (last-layer fragility — the folk wisdom)
```

Early blocks are sensitive because their representations are still close to the embedding distribution — a small numerical error here propagates *through every subsequent block*. Late blocks are sensitive because their errors directly affect the output logits.

The mixed-precision recipe should protect *both* ends:

```
Default:   FP4 weight + FP4 activation
Exception: blocks {0, 1, 2, L-2, L-1} → keep BF16 (or FP8)
Exception: mlp.up_proj + mlp.down_proj in all layers → raise one precision level
```

---

## NVFP4 vs MXFP4 are not interchangeable

The paper documents distinct sensitivity profiles for the two formats:

| Format | Block size | Block scale | Sensitivity profile |
|--------|-----------|-------------|---------------------|
| NVFP4 | 16 elements | E4M3 (FP8) | More tolerant; smaller block → tighter local fit |
| MXFP4 | 32 elements | E8M0 (power-of-2 only) | Less tolerant; wider block + coarser scale → amplifies outliers within a block |

This matters for deployment: a mixed-precision recipe tuned for NVFP4 will not transfer to MXFP4 without retesting. The folk wisdom "FP4 is FP4" is wrong.

---

## The diagnostic methodology (paper §3, applied)

The paper's methodology is general — it works for any aggressive quantization, not just FP4:

1. **Single-component swap.** Quantize only one component class (across all layers) to the target precision. Measure PPL and task accuracy. Repeat for each component class. The ranking that emerges is the component-sensitivity ordering.
2. **Single-block swap.** Quantize only one block to the target precision. Sweep across all blocks. Plot the per-block sensitivity curve. Identify fragility regions.
3. **Greedy mixed-precision search.** Starting from FP16, quantize the least-sensitive component first, then the next least-sensitive, until a PPL budget is exceeded. The resulting recipe usually beats uniform quantization at the same average bit budget.

This is the recipe to use whenever you're considering W4A4, W2A8, FP4, or any aggressive setting. It catches the recipes where uniform quantization wastes precision on insensitive layers while crushing the few sensitive ones.

---

## Why this matters for evaluation

The paper is *not* a new quantization algorithm. It is a *methodology* paper, and its value in the methodology chapter is exactly that: it teaches you how to evaluate FP4 failures by component and by depth, rather than treating FP4 as a single global switch.

A deployment report that includes the component-sensitivity heatmap (per §3.1 of this excerpt) plus the per-block sensitivity curve (per §3.2) is roughly twice as informative as the same report with only aggregate PPL + MMLU. It tells the reader *where* the quantization will break under load.

---

## Connections

- [[ch-20]] §3 — the chapter section that codifies this methodology.
- [[nvfp4]] / [[mx-formats]] — the two FP4 families compared.
- [[awq]] / [[smoothquant]] — activation-aware methods that protect the same MLP-heavy paths.
- [[quarot]] / [[spinquant]] — rotation methods that reduce component outliers before FP4 quantization.
- [[nvfp4-qad]] — quantization-aware distillation that recovers FP4 accuracy after the fact.

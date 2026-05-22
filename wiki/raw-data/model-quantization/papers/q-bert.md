<!-- scope: Q-BERT — mixed-precision Hessian-aware BERT quantization
     deps: hawq, q8bert, obs-obd
     see-also: i-bert, bibert, gptq
-->

# Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT
- **Core Insight:** Different BERT layers have very different sensitivities (measured by top eigenvalue of the per-block Hessian), and a Hessian-aware mixed-precision allocation can push BERT down to 2/3-bit weights with <2.3 GLUE drop — uniformly low-bit is wasteful both at the unsensitive embedding ends and at the sensitive middle attention blocks.
- **Guideline:** For BERT-style models, estimate per-block λ_max(H_ℓ) via Hutchinson + power iteration on ~10 batches; sort blocks by λ_max·‖ΔW‖²; assign 8 / 4 / 3 / 2 bits in proportion to a Pareto fit; combine with group-wise activation quantization (per-attention-head groups) to handle outliers.
- **Authors:** Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W. Mahoney, Kurt Keutzer
- **Year:** 2020 (AAAI)
- **URL:** https://arxiv.org/abs/1909.05840
- **Relevant topics:** mixed-precision BERT, Hessian sensitivity, group-wise activation, ultra-low-bit

## Abstract
Q-BERT extends HAWQ to the transformer setting. The paper measures per-block second-order sensitivity in BERT-Base and demonstrates a 10–100× range in λ_max(H_ℓ) — middle-stack attention blocks dominate, embedding and pooler ends are insensitive. A mixed-precision allocation (typically: 8-bit embedding, 4-bit middle blocks, 2-3 bit unsensitive blocks) achieves 13× compression at 2.3 GLUE-avg drop. Q-BERT also introduces group-wise activation quantization (separate scales per attention head) to handle the per-channel outlier structure that vanilla per-tensor quantization destroys.

## Key Contributions
- First systematic per-block Hessian sensitivity study of BERT.
- HAWQ-style mixed-precision applied to transformer architecture.
- Group-wise activation quantization at attention-head granularity.
- 13× compression with 2.3 GLUE drop on BERT-Base (vs uniform 8-bit baseline).
- Establishes that transformer layers are nonuniformly sensitive — a finding that recurs in every later LLM PTQ paper.

## Key Figures/Tables to Study
- **Figure 3** — per-block λ_max(H) for BERT-Base; visualises the 100× sensitivity range.
- **Table 5** — GLUE under mixed (2/3/4-bit) allocation vs uniform 4-bit / 8-bit.

## Technical Details

### Per-block Hessian sensitivity
For block ℓ with weights W_ℓ:
`Ω_ℓ = λ_max(H_ℓ) · ‖ΔW_ℓ(b)‖²`
where H_ℓ is the block-output Hessian; λ_max estimated via:
`v ← random; for t in range(T): v ← Hv / ‖Hv‖`
with Hv computed by autograd double-backward (no explicit H).

### Group-wise activation quantization
Activations split into G groups (G = num_attention_heads) along the channel dimension:
- Each group g has its own (S_g, Z_g).
- Computed per-tensor within the group.
- Reduces dynamic-range mismatch caused by per-head outliers in BERT QKV projections.

This is the precursor to per-token / per-channel activation quant in later LLM work ([[smoothquant]], [[zeroquant]]).

### Bit allocation
Pareto-greedy: start at all-8-bit, drop the layer-bit pair with the smallest Ω increment per byte saved, until target compression hit. Typical Q-BERT allocation:
- Embedding: 8 bits.
- Mid-attention blocks: 4 bits weights / 8 bits activations.
- FFN: 3 bits.
- Pooler: 2 bits.

### Empirical effect
- BERT-Base GLUE-avg: FP 82.5 → Q-BERT mixed (2-3 bit weights) 80.2 (Δ = −2.3).
- Model size: 415 MB → 32 MB (13× compression).

### Practical pitfalls (documented in the paper)
- Per-tensor activation quantization fails on attention output; group-wise mandatory.
- Naive 2-bit collapses LayerNorm; combine with QAT for ≤3-bit weights.
- LayerNorm parameters (γ, β) must remain fp — too few params to compress, too sensitive.

## Connections
- [[hawq]] — direct parent: Q-BERT is HAWQ applied to BERT.
- [[obs-obd]] — second-order foundation.
- [[q8bert]] — uniform 8-bit BERT baseline Q-BERT improves on at lower bits.
- [[i-bert]] — orthogonal: integer-only execution; Q-BERT focuses on bit allocation.
- [[bibert]] — pushes Q-BERT's direction to the 1-bit extreme.
- [[gptq]] — LLM-era successor; uses uniform 4-bit rather than mixed precision because LLM-scale loss landscapes are flatter per-block.
- [[brecq]] — block-wise PTQ that is the methodologically more general successor to Q-BERT.

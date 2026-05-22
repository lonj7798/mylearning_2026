<!-- scope: OBC — Optimal Brain Compression unifying pruning + quantization
     deps: obs-obd, adaround
     see-also: gptq, sparsegpt, brecq
-->

# Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning (OBC)
- **Core Insight:** Pruning a weight to zero and quantizing a weight to its nearest grid point are the same problem — choosing a perturbation δw_q to a single weight — so the OBS framework solves both: each step picks the weight whose constrained perturbation (zero for pruning, nearest grid for quant) minimises the local Hessian-weighted loss, then updates surviving weights via the closed-form OBS rule.
- **Guideline:** For a per-layer PTQ step with input Hessian H = 2 X Xᵀ, process weight columns in OBS order (smallest saliency first); for each chosen column, replace OBS's "set to zero" with "round to nearest int8/int4"; update remaining columns by the same δw rule; this is the algorithm GPTQ ships at LLM scale.
- **Authors:** Elias Frantar, Sidak Pal Singh, Dan Alistarh
- **Year:** 2022 (NeurIPS)
- **URL:** https://arxiv.org/abs/2208.11580
- **Relevant topics:** OBS unification, PTQ + pruning, Hessian-based, GPTQ predecessor

## Abstract
OBC unifies pruning and quantization under a single OBS-based framework. The core observation: removing a weight (pruning) and rounding it to a quantizer level (PTQ) are both "constrain a weight to a fixed value" operations, and the OBS update — `δw = −(δw_q · H⁻¹_{:,q})/[H⁻¹]_qq` — applies identically. OBC also engineers the implementation: a Cholesky-factor-based update of H⁻¹ that brings the cost of a full per-layer OBS sweep down from O(d⁴) to O(d³), making it tractable for the dense BERT/CNN weights of the day. OBC is the direct algorithmic ancestor of GPTQ and SparseGPT, both authored by Frantar/Alistarh six months later for LLM-scale layers.

## Key Contributions
- Unification: OBS as the common engine for pruning and quantization.
- O(d³) per-layer implementation via Cholesky factorisation of H⁻¹.
- Mixed compression: per-layer combinations of pruning + quantization optimised jointly.
- SOTA PTQ on ResNet, BERT-base across pruning and 4-bit quant regimes at the time.
- Provides the algorithmic template that GPTQ adapts to LLM scale.

## Key Figures/Tables to Study
- **Algorithm 1** — OBC pseudocode with the unified perturbation-choice step.
- **Figure 3** — runtime profile: Cholesky update vs naive O(d⁴) baseline.
- **Table 4** — BERT-Base 4-bit PTQ: OBC vs AdaRound, BRECQ.

## Technical Details

### Unified per-weight perturbation
Pick weight column q. The required perturbation δw_q is:
- **Prune**: δw_q = −w_q  (force w_q → 0).
- **Quantize**: δw_q = Q(w_q) − w_q  (force w_q → nearest grid point).

### OBS update (same for both)
Cost of forcing column q:
`δL_q = (δw_q)² / (2 · [H⁻¹]_qq)`
Optimal compensating update to all other surviving weights:
`δw_{−q} = −(δw_q / [H⁻¹]_qq) · H⁻¹_{−q, q}`

### Inverse-Hessian Cholesky trick
Let H = L Lᵀ via Cholesky. Then H⁻¹ = L⁻ᵀ L⁻¹. The key identity:
- `[H⁻¹]_qq = 1 / L_qq²` (from triangular structure, after reordering).
- After processing column q, the residual sub-problem has Hessian H̃ with `H̃⁻¹` obtained by a rank-1 down-date of L — O(d²) per step instead of O(d³).

Total cost per layer: O(d³) for Cholesky + O(d) × O(d²) per-column updates = O(d³).

### Ordering rule
Process columns in increasing saliency δL_q. Approximation: pre-sort by initial saliency from L (good in practice; resort every k columns refines).

### Mixed compression
For a target compression ratio r mixing pruning + quantization:
- At each step, compute saliency for both options (prune-to-0 vs quant-to-Q(w_q)).
- Choose the cheaper.
- Apply OBS update.
This produces a per-weight pruning-or-quantization decision optimal under the local Hessian.

### Empirical PTQ results
BERT-Base 4-bit:
- AdaRound: 84.3 GLUE
- BRECQ: 84.6
- **OBC: 84.9**
Within 0.5 of FP (85.4) without QAT.

## Connections
- [[obs-obd]] — exact mathematical foundation; OBC re-derives and extends it.
- [[adaround]] — same Hessian objective, different optimisation (soft relaxation vs exact sequential).
- [[brecq]] — block-wise reconstruction; complementary to OBC's per-layer OBS.
- [[gptq]] — direct LLM-era heir: literally OBC ported to billion-parameter weight matrices via lazy-batched OBS updates (lives in `papers/gptq.md`, bucket 6).
- [[sparsegpt]] — sibling: OBC restricted to pruning at LLM scale.
- [[quantization-mapping]] — provides the underlying affine quantizer OBC rounds to.

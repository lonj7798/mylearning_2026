<!-- scope: Optimal Brain Surgeon / Optimal Brain Damage — second-order weight removal
     deps: ../classics/quantization-mapping
     see-also: obc, adaround, hawq, gptq
-->

# Optimal Brain Damage (LeCun 1989) / Optimal Brain Surgeon (Hassibi 1993)
- **Core Insight:** A weight's contribution to the loss is governed by the local Hessian, not its magnitude; OBD diagonalises the Hessian and prunes the weight with the smallest `w_i² · H_ii / 2`, OBS uses the full inverse Hessian to derive the exact loss-minimising weight to remove plus the closed-form compensating update to the surviving weights.
- **Guideline:** When deleting or quantizing a weight w_q, use the OBS update `δw = −(w_q / [H⁻¹]_{qq}) · H⁻¹_{:,q}` to redistribute its lost contribution across the remaining weights; this is the mathematical core that GPTQ executes column-by-column for LLM PTQ.
- **Authors:** Yann LeCun, John S. Denker, Sara A. Solla (OBD, 1989) / Babak Hassibi, David G. Stork (OBS, 1993)
- **Year:** 1989 (OBD), 1993 (OBS)
- **URL:** https://papers.nips.cc/paper/250-optimal-brain-damage (OBD); https://papers.nips.cc/paper/647-second-order-derivatives-for-network-pruning-optimal-brain-surgeon (OBS)
- **Relevant topics:** pruning, second-order loss, Hessian, weight removal, GPTQ ancestor

## Abstract
OBD and OBS are the foundational works of structured neural-network pruning. Both expand the task loss to second order around a trained model and use the Hessian to identify the weight whose removal least damages the loss. OBD takes the diagonal Hessian approximation: drop the weight minimising `½ w_i² H_ii`. OBS uses the full Hessian: the optimal weight to drop is the q minimising `w_q² / [2 H⁻¹]_{qq}`, and the surviving weights are updated by a closed-form formula that exactly cancels the first-order effect of the removal. OBS provides the recipe modern LLM-PTQ methods (GPTQ, SparseGPT, OBC) execute layer-by-layer.

## Key Contributions
- **OBD (1989)**: diagonal-Hessian saliency; first principled pruning criterion beyond magnitude.
- **OBS (1993)**: full inverse-Hessian saliency + closed-form weight update to compensate removal.
- Both derive their criteria from a 2nd-order Taylor expansion of the converged-model loss.
- OBS update is the direct mathematical predecessor of GPTQ's per-column quantization correction.
- Established Hessian-based reasoning as the gold-standard sparsification framework.

## Key Figures/Tables to Study
- **OBD Figure 2** — saliency histogram showing magnitude-pruning vs OBD disagreement.
- **OBS Equation 5** — the inverse-Hessian saliency and the closed-form `δw` update.

## Technical Details

### Setup (both methods)
Trained model with weights w ∈ ℝⁿ at a local minimum of L. Second-order Taylor:
`δL ≈ gᵀ δw + (1/2) δwᵀ H δw  ≈ (1/2) δwᵀ H δw`
(g ≈ 0 at convergence). Goal: choose δw that removes one component (sets w_q ← 0) while minimising δL.

### OBD (diagonal approximation)
Assume H is diagonal. The constraint `w_q + δw_q = 0` gives `δw_q = −w_q`. The minimum increase in L is:
`δL_q = (1/2) H_qq · w_q²`
**Saliency**: `s_q = (1/2) H_qq · w_q²`. Prune q minimising s_q.

### OBS (full Hessian)
Use Lagrange multipliers on the constraint `e_qᵀ (w + δw) = 0`:
`L_lag = (1/2) δwᵀ H δw + λ (e_qᵀ δw + w_q)`
Solve for δw:
`δw = −(w_q / [H⁻¹]_qq) · H⁻¹ e_q  =  −(w_q / [H⁻¹]_qq) · H⁻¹_{:, q}`

The induced loss change (**saliency**):
`δL_q = w_q² / (2 · [H⁻¹]_qq)`

The procedure:
1. Compute or estimate H⁻¹ once.
2. For each weight q: compute saliency w_q² / (2 [H⁻¹]_qq).
3. Prune the q with smallest saliency.
4. Update all remaining weights: `w ← w + δw`.
5. (Optional) update H⁻¹ via the Woodbury identity for the next round:
   `H⁻¹_new = H⁻¹ − (H⁻¹_{:, q} · H⁻¹_{q, :}) / [H⁻¹]_qq`

### Why GPTQ inherits this
GPTQ quantizes the q-th weight column (instead of zeroing it), then applies exactly the OBS δw update to the remaining un-quantized columns to compensate the rounding error. The inverse-Hessian update via Cholesky factorisation is the GPTQ implementation trick that makes this affordable on LLM-scale layers.

### Practical Hessian for one layer
For PTQ, the load-bearing Hessian is `H = 2 X Xᵀ` where X are calibration inputs of shape (in_dim, N_samples). H is in_dim × in_dim, manageable for in_dim ≤ ~16k.

## Connections
- [[adaround]] — uses the same 2nd-order argument but optimises per-weight rounding direction (binary {up, down}) instead of removal.
- [[obc]] — Frantar's unification of OBS for both pruning and quantization (Optimal Brain Compression).
- [[brecq]] — extends second-order reasoning from per-layer to per-block.
- [[hawq]] — uses Hessian top-eigenvalue for mixed-precision bit allocation instead of weight removal.
- [[gptq]] — the LLM-era direct heir: applies OBS column-by-column with quantization rounding, lives in `papers/gptq.md` (bucket 6).
- [[sparsegpt]] — sister technique: OBS for one-shot pruning of LLMs.

<!-- scope: Bit pruning / differentiable mixed-precision search
     deps: hawq, q-bert
     see-also: brecq, adaround, gptq
-->

# Bit Pruning / Differentiable Mixed-Precision Bit Allocation (Pre-LLM Lineage)
- **Core Insight:** The mixed-precision bit-width assignment problem can be made differentiable by parameterising each layer's bit-width as a softmax over candidate bit-widths (DNAS / HAQ / EdMIPS / BP-NAS lineage), or as a continuous relaxation over a learnable per-layer "effective precision"; gradients flow into the bit-width parameters via the same STE used for the quantizer itself, allowing joint W + bit-width SGD.
- **Guideline:** Use a Gumbel-softmax over candidate bit-widths {2, 3, 4, 8} per layer; backpropagate the task loss + a model-size cost penalty into the bit-width logits; harden to a discrete allocation at the end of training; for LLM-scale, fall back to HAWQ's Hessian sensitivity criterion since differentiable search is too expensive at billion-param scale.
- **Authors:** consolidation of Wang 2019 HAQ, Wu 2018 EdMIPS, Yu 2020 BP-NAS, Cai 2020 DNAS
- **Year:** 2018–2020 (lineage)
- **URL:** https://arxiv.org/abs/1811.08886 (HAQ); https://arxiv.org/abs/1812.00090 (DNAS)
- **Relevant topics:** mixed-precision NAS, differentiable bit-width, Gumbel-softmax, model-size budget

## Abstract
This page consolidates the pre-LLM lineage of differentiable mixed-precision bit allocation — the family of methods that learn per-layer bit-widths jointly with weights via continuous relaxation. HAQ uses RL to search over per-layer bit-widths; DNAS / EdMIPS / BP-NAS use Gumbel-softmax to make the search differentiable. All operate on the principle that bit-width is an architectural hyperparameter that should be learned, not hand-allocated. The Hessian-based alternative (HAWQ / Q-BERT) ended up dominating at LLM scale because the differentiable search has prohibitive compute (each candidate bit-width contributes a separate forward pass to the loss), but the differentiable-search philosophy persists in modern variants (OmniQuant's learnable clip thresholds).

## Key Contributions
- HAQ: RL-based hardware-aware bit allocation with FLOPs and latency penalties.
- DNAS / EdMIPS: Gumbel-softmax over per-layer bit-widths for end-to-end differentiable search.
- BP-NAS: bit-pruning style search where bit-widths are independently zeroed out (effective {1, 2, ..., 8}).
- Establishes that learned bit-widths beat hand-allocated; sets the stage for HAWQ which achieves similar quality via Hessian heuristics at far lower cost.
- Inspires later "learnable everything" PTQ methods like OmniQuant.

## Key Figures/Tables to Study
- **HAQ Figure 3** — RL-learned per-layer bit-widths for MobileNet on Edge TPU; matches latency budget exactly.
- **DNAS Figure 2** — Gumbel-softmax temperature annealing schedule.

## Technical Details

### Gumbel-softmax bit-width selection (DNAS family)
For layer ℓ with candidate bit-widths B = {2, 3, 4, 8}, maintain logits α_ℓ ∈ ℝ^{|B|}. Sample bit-width via:
`b_ℓ = softmax((α_ℓ + g) / τ),  g ~ Gumbel(0,1)`
The forward pass uses a soft mixture:
`y_ℓ = Σ_b b_ℓ[b] · QuantConv(x, W, b)`
Backward: gradient flows through the softmax weights to α_ℓ.

Annealing τ → 0 collapses the soft mixture to argmax (one-hot).

### Resource-aware loss
`L = L_task(α) + λ · cost(α)`
where `cost(α) = Σ_ℓ E_{b ~ softmax(α_ℓ)}[bits_ℓ · params_ℓ]` is the expected model size (or FLOPs, or latency on a target chip via lookup table).

λ is tuned to hit the target size / latency budget.

### HAQ (RL variant)
- Each layer's bit-width is a discrete action.
- Reward = -val_loss(quantized_model) - α·hardware_cost.
- DDPG agent learns per-layer bit-widths over ~600 quantization episodes.
- More expensive than Gumbel-softmax but no relaxation gap.

### BP-NAS (bit pruning)
Treats each "bit slot" as a binary mask: bit_k_used ∈ {0, 1}. Gradient through STE; structured sparsity penalty enforces that used bits form a contiguous prefix (b_1=1 ⇒ b_2 ≥ b_1, etc.).

### Why HAWQ won at LLM scale
Each forward of a differentiable-search method costs |B| × baseline; for LLMs (≥7B params, |B|≥4 candidate bit-widths), this is ≥28B forward params per step. HAWQ achieves the same quality of allocation by computing per-layer Hessian top-eigenvalue once (a few power-iteration steps on the trained model — cheap relative to full search) and assigning bit-widths analytically.

### Empirical effect (CIFAR / ImageNet era)
- HAQ on MobileNetV2 Edge TPU: ~50% latency reduction at the same accuracy as hand-allocated mixed precision.
- DNAS on CIFAR ResNet-20: 6-bit average (vs 8-bit uniform) at parity accuracy.

### Legacy in LLM PTQ
- OmniQuant's learnable clip thresholds are a Gumbel-free continuous relaxation.
- HAWQ's Hessian-trace criterion is the post-mortem analytic substitute for differentiable search.
- Mixed-precision (per-block bit allocation) is rare in production LLM PTQ — uniform 4-bit weights have dominated since GPTQ — partly because differentiable search proved too expensive and HAWQ-style heuristics gave only marginal wins on flat LLM loss landscapes.

## Connections
- [[hawq]] — the analytic alternative that won out at LLM scale.
- [[q-bert]] — HAWQ applied to BERT; the most successful pre-LLM transformer mixed-precision result.
- [[brecq]] — block-wise PTQ with HAWQ-style mixed precision per block.
- [[adaround]] — orthogonal: differentiable per-weight rounding within a fixed bit budget.
- [[omniquant]] — modern LLM heir: differentiable learnable clip thresholds + equivalent transformations on a frozen LLM.
- [[gptq]] — uses uniform 4-bit, sidesteps the mixed-precision question (lives in `papers/gptq.md`, bucket 6).

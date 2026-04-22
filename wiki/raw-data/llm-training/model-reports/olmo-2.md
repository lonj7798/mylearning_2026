<!-- scope: OLMo 2 technical report — Allen AI's fully open pretraining + post-training
     deps: [[README]]
     see-also: [[tulu-3]], [[dolma]]
-->

# OLMo 2: 2 OLMo 2 Furious
- **Core Insight:** A fully reproducible model (weights + data + code + intermediate checkpoints) is achievable at 7B/13B/32B scale when architectural stability tricks (QK-Norm, reordered norm, Z-loss) are combined with the Tulu 3 post-training recipe.
- **Guideline:** For open model labs, treat OLMo 2's architecture + OLMo-Mix-1124 + Tulu 3 post-training as a de-risked baseline — deviations need justification.
- **Authors:** Pete Walsh, Luca Soldaini, Dirk Groeneveld, et al. (Allen AI)
- **Year:** 2025 (arXiv Jan 2025)
- **URL:** https://arxiv.org/abs/2501.00656
- **Relevant topics:** Open pretraining, QK-Norm, Z-loss, reordered RMSNorm, Tulu 3 recipe applied, RLVR

## Abstract
OLMo 2 is Allen AI's 7B/13B/32B open foundation model family, released with all pretraining data (OLMo-Mix-1124), training code, and 1000+ intermediate checkpoints. Architectural changes from OLMo 1 include switching to RMSNorm, reordering the normalization layers, adding QK-Norm, using rotary position embeddings, and adding Z-loss regularization. Post-training uses the Tulu 3 recipe largely unchanged: SFT on an OLMo-variant of Tulu 3 data, DPO on a preference mix, then RLVR. The resulting OLMo 2-Instruct competes with Llama 3.1 Instruct at 7B and 13B.

## Key Contributions
- Open release at production-useful scale (7B/13B/32B) with checkpoints, data, eval code.
- Architectural stability recipe: RMSNorm + reordered norm + QK-Norm + RoPE + Z-loss. Prevents the training-spike phenotype that plagued OLMo 1.
- Confirms the Tulu 3 recipe generalizes: SFT -> DPO -> RLVR works on a non-Llama base without modification.
- Two-stage pretraining curriculum: 90%+ of budget on OLMo-Mix-1124 (3.9T tokens), then a cooldown on higher-quality "Dolmino" mix.
- 32B variant is the first fully-open model to beat GPT-3.5 and GPT-4o-mini on average benchmarks.

## Key Figures/Tables to Study
- **Pretraining curriculum diagram** (Stage 1 OLMo-Mix-1124 -> Stage 2 Dolmino cooldown).
- **Architecture ablation table:** which stability trick (QK-Norm, Z-loss, reorder) contributes which fraction of the spike-free runs.
- **Post-training gain table:** SFT -> DPO -> RLVR gain per benchmark, per size.
- **Compute-optimal scaling plot** across 7B/13B/32B.

## Technical Details — Training Pipeline

### Pretraining
- **Stage 1 data:** OLMo-Mix-1124 — ~3.9T tokens drawn from DCLM, Dolma 1.7, Starcoder, Proof Pile II.
- **Stage 2 cooldown data:** Dolmino mix — curated higher-quality subset, ~50B tokens.
- **Architecture:**
  - RMSNorm (replacing non-parametric LayerNorm).
  - Reordered normalization (post-norm within residual).
  - QK-Norm (normalize queries + keys before attention).
  - Rotary position embeddings.
  - Z-loss regularizer on output logits (penalizes log-Z to keep logits well-scaled).
  - Improved initialization preserving activation scale.
- **Context:** 4K native, extended to 32K in cooldown.
- **Sizes:** 7B, 13B, 32B (dense).

### Post-training (Tulu 3 recipe)
- **SFT:** OLMo-specific variant of Tulu 3 SFT mix (~939K prompts from Tulu 3, with OLMo-compatible formatting).
- **DPO:** on-policy preferences generated from the SFT checkpoint + Tulu 3 preference mix. Beta/LR per-size (following Tulu 3 defaults; LRs re-tuned lightly).
- **RLVR:** PPO with verifiable rewards (GSM8K/MATH exact-match, IFEval constraint checks, code unit tests). Hyperparameters inherit from Tulu 3: LR 3e-7, beta KL 0.05, clip eps 0.2, GAE lambda 0.95, 4 PPO update epochs per step.

### Reported post-training gains
- RLVR stage lifts GSM8K and MATH consistently for 7B and 13B (single-digit pp gains).
- DPO stage contributes most of the chat-quality / IFEval lift.

### Compute
- 7B: ~460K H100 GPU-hours pretraining.
- 13B: ~1.9M H100 GPU-hours pretraining.
- Post-training: small fraction of pretraining (not separately broken out).

## Connections
- [[tulu-3]] — the post-training recipe that OLMo 2 inherits wholesale.
- [[dolma]] — foundational data corpus; OLMo-Mix-1124 supersedes Dolma 1.7.
- [[rlvr-tulu3]] — methodology page for the RLVR component.
- [[llama-3]] — contrast: Llama 3 uses SFT/DPO iterative rounds without RLVR.
- [[deepseek-v3]] — parallel "fully disclosed stability" story at much larger MoE scale.

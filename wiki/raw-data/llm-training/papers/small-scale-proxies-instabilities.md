<!-- scope: reproducing attention-logit growth and output-logit divergence in small Transformers; the LR-sensitivity metric; effect of qk-layernorm, z-loss, warmup, independent weight decay, width vs depth, muParam, and AdamW epsilon across scale
     deps: [[adam]], [[lr-schedules]]
     see-also: [[olmo-2]], [[kimi-k2]], [[task-scaling-model-ladders]]
-->

# Small-scale proxies for large-scale Transformer training instabilities
- **Core Insight:** In decoder-only Transformers of 2.4M to 1.2B non-embedding parameters trained on C4, attention logit growth and output logit divergence appear at high learning rates, and qk-layernorm plus z-loss (coefficient 1e-4) let models train to low loss across the LR range 3e-4 to 3e-1, including a 1.2B model at LR 0.3 (Fig. 1, Fig. 3, §3.1.1).
- **Guideline:** When choosing stability settings for a larger pretraining run, sweep peak LR over three orders of magnitude on a series of small models and track max attention logit and gradient RMS against model size, because in this paper a quadratic fit over smaller models predicted that a 4.8B model would diverge at LR 1e-2 without qk-layernorm, which the 4.8B run confirmed (§3.3, Fig. 9), and a gradient-RMS trend toward the AdamW ε led to ε = 1e-15, which lowered loss for a 4.8B model at LR 0.3 (§3.4, Fig. 12).
- **Authors:** Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, et al. (Google DeepMind)
- **Year:** 2023 (arXiv v1 2023-09)
- **URL:** https://arxiv.org/abs/2309.14322
- **Source type:** paper
- **Relevant topics:** pretraining stability, learning-rate sensitivity, qk-layernorm, z-loss, warmup, decoupled weight decay, width vs depth scaling, muParam, AdamW epsilon, small-scale proxy experiments

## Abstract
Teams that trained large Transformers have reported instabilities that did not appear at smaller scales with the same hyperparameters, and reproducing them requires large resources. The paper studies two previously reported instabilities: growth of attention-layer logits (Dehghani et al., 2023) and divergence of the output logits from the log probabilities (Chowdhery et al., 2022). By measuring learning rate against final loss across model sizes, it shows that both appear in small models trained at high learning rates, and that the mitigations used at scale work in this regime. It then measures how warmup, weight decay, and muParam change the sensitivity of final loss to learning rate, and combines techniques so that small models reach similar loss across orders of magnitude of learning rate. It ends with two cases where an instability is predicted from the scaling behavior of activation and gradient norms before it appears.

## Key Contributions
- LR sensitivity, a summary statistic of the LR-vs-final-loss curve over a three-order-of-magnitude LR range (§2.2).
- Reproduction of attention logit growth (9.4M model, LR 0.1; Fig. 2) and output logit divergence (2.4M model, LR 0.1; Fig. 4) in small models, with qk-layernorm and z-loss as effective mitigations (§3.1).
- Measurements of warmup length, independent weight decay, width vs depth scaling, and muParam on LR sensitivity (§3.2).
- A prediction of divergence at 4.8B parameters from the attention-logit trend of smaller models, confirmed by a 4.8B run (§3.3).
- Identification of gradient RMS falling toward the AdamW ε as scale and LR grow, with a lower ε as the fix (§3.4).

## Key Figures/Tables to Study
- **Fig. 1:** final eval loss vs LR for N = 2.4e6 to 1.2e9, with and without qk-layernorm; LR sensitivity vs N.
- **Fig. 2:** loss and max attention logit (layer 0) during training for 9.4M (LR 0.1) and 4.8B (LR 0.01) models.
- **Fig. 3-4:** z-loss and weight decay against output logit divergence; output logit mean over training.
- **Fig. 5-7:** warmup length, independent weight decay, and width vs depth scaling against LR sensitivity.
- **Fig. 9-10:** max attention logit vs parameters at six LRs with quadratic fits; loss when the max attention logit is fixed to about κ.
- **Fig. 11-13:** gradient RMS vs parameters, LR, and block index; ε = 1e-15, 1e-8, 1e-6 for a 4.8B model at LR 0.3.

## Technical Details
- **LR sensitivity (§2.2):** E_{η∈[a,b]}[min(ℓ(A(η)), ℓ₀) − ℓ*]. η is the peak LR of a warmup-plus-cosine schedule; A(η) are the weights after training with η; ℓ is validation loss; ℓ₀ is loss at initialization; ℓ* = min_{η∈[a,b]} ℓ(A(η)). Default range [3e-4, 3e-1], evaluated at {3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1} (§2.2).
- **Metric limits (App. B):** it does not account for a shift of the optimal LR (the authors recommend shifting [a, b]); it is invariant to loss scale, so a model at random performance for every LR has sensitivity 0; it is not comparable for interventions that change the meaning of LR, so it was not reported for AdaFactor-style RMS scaling of the LR (Fig. E.14). The authors recommend reading it with the LR-vs-loss curves.
- **Attention logit growth (§3.1.1):** z_ij = ⟨q_i, k_j⟩/√d_h, where q_i and k_j are query and key vectors and d_h is the head dimension. Without qk-layernorm, the LR at which models diverge decreases as model size increases; qk-layernorm applies LayerNorm to queries and keys before the logits are computed. LR sensitivity increases with scale with and without qk-layernorm (§3.1.1). The growth comes from larger query and key norms, not higher cosine similarity (Fig. E.1). It also occurs with a pointwise attention variant without softmax (§3.2.5, Fig. E.11).
- **Output logit divergence (§3.1.2):** for output logits y, p_i = e^{y_i}/Z with Z = Σ_j e^{y_j}. The logits become very negative toward the end of training (Fig. 4). z-loss adds the auxiliary term log²Z with coefficient 1e-4. The divergence occurs in models with no weight decay at every scale tested; z-loss resolves it; weight decay also mitigates it for the larger models tested (§3.1.2, Fig. 3).
- **Warmup (§3.2.1, Fig. 5):** total steps fixed at 1e5; warmup of 50, 500, 5000, 10000, and 25000 steps tested. Longer warmup reduces LR sensitivity and loss, most for the larger models, which are not stable at LR 3e-1 without long warmup.
- **Independent weight decay (§3.2.2, Fig. 6):** in independent decay the λθ term is multiplied by the schedule s_t but not by the peak LR η (Loshchilov and Hutter); in the PyTorch and Optax AdamW defaults it is multiplied by s_t·η. Independent decay (λ = 1e-4) gave lower LR sensitivity than the default form (λ = 0.1). Raising decay above 1e-4 shifts the optimal LR slightly to the right (§3.2.5, Fig. E.10).
- **Width vs depth (§3.2.3, Fig. 7):** depth scaling (d = 512, 3 to 96 layers) increases LR sensitivity faster than width scaling (6 layers, d = 256 to 2048), but gave lower loss than width scaling at the largest scale tested. Joint scaling had the lowest loss at the largest scale and a more reliable extrapolation from models under 1e8 parameters (Fig. E.3).
- **muParam (§3.2.4, Fig. 8):** muParam (simple) scales the LR of linear layers by base-fan-in/fan-in, with base width 256. It stabilized the optimal LR but did not improve loss or reduce LR sensitivity, and did not remove the need for qk-layernorm at high LR (Fig. E.4). Additional muParam features gave no measurable improvement at the largest scale tested (Fig. E.5).
- **Predicting logit growth (§3.3):** the max attention logit of block 0 at step 2e3 was fit with a quadratic in model size for each LR (Fig. 9). All points with max attention logit above 1e4 diverged; the fit predicted that the next scale at LR 1e-2 would cross 1e4; a 4.8B model trained at LR 1e-2 diverged, and the fit closely extrapolated the value of its max attention logit. When the max attention logit of a 10M model is forced to about κ, loss deteriorates around κ = 1e3 and at κ = 1e4 exceeds a zero-layer bigram baseline (Fig. 10).
- **AdamW ε (§3.4):** gradient RMS of the first MLP layer decreases with parameters and with LR (Fig. 11), and at the largest scale and LR tested is around the default ε = 1e-8. The unscaled update is Δ = v/(√u + ε), where v and u are EMAs of the first and second gradient moments; when gradient RMS is of the order of ε, Δ shrinks (Fig. 13). For a 4.8B model at LR 0.3, ε = 1e-15 improved loss and avoided the collapse in gradient RMS; ε = 1e-6 diverged (Fig. 12, Fig. E.15). Proposed mechanism: the output RMS entering the final layernorm grows with LR, and the layernorm gradient scales with the reciprocal of its input RMS (§3.4, App. C, Fig. C.1).
- **Null results (§3.2.5):** total steps of 5e4 or 2e5 (Fig. E.7) and batch sizes of 512 or 1024 at constant data (Fig. E.9) did not meaningfully change LR sensitivity. Per-head qk-layernorm performed better than qk-layernorm over the whole model dimension (Fig. E.8).
- **Scope (footnote 1, §4):** the study targets instabilities that cause slow divergence, not fast loss spikes; loss spikes, edge of stability, and the role of Adam β₂ appear only in related work.

## Recipe ledger
Model for all rows except the last: the paper's default decoder-only Transformer (NanoDO, Flax/JAX, C4), 2.4M to 1.2B non-embedding parameters (Fig. 1). Stage "pretrain-stable" denotes the single run including its cosine decay; the paper has no separate anneal stage. Status for every row: verified 2026-09-14 against arXiv:2309.14322v2 unless marked.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| paper default | 2.4M-1.2B | pretrain-stable | optimizer; β1, β2; ε | AdamW; 0.9, 0.95; 1e-8 | §2.1 | verified | ε: Fig. 12 (4.8B only); β: no ablation reported |
| paper default | 2.4M-1.2B | pretrain-stable | gradient clipping | global norm 1 | §2.1 | verified | no ablation reported |
| paper default | 2.4M-1.2B | pretrain-stable | warmup; total steps | 5e3 steps linear; 1e5 steps | §2.1 | verified | Fig. 5 (longer warmup lowers LR sensitivity); Fig. E.7 (5e4 or 2e5 steps: no meaningful change) |
| paper default | 2.4M-1.2B | pretrain-stable | decay shape; minimum LR | cosine; 1e-5 | §2.1 | verified | no ablation reported |
| paper default | 2.4M-1.2B | pretrain-stable | peak LR sweep for LR sensitivity | {3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1} | §2.2 | verified | measurement grid, not a selected value |
| paper default | 2.4M-1.2B | pretrain-stable | weight decay | independent, 1e-4 (0.1 when not independent) | §2.1; §3.2.2 | verified | Fig. 6; sweep in Fig. E.10 |
| paper default | 2.4M-1.2B | pretrain-stable | z-loss coefficient on log²Z | 1e-4 | §2.1; §3.1.2 | verified | Fig. 3-4 (no z-loss and no weight decay: output logits diverge) |
| paper default | 2.4M-1.2B | pretrain-stable | qk-layernorm | on, per head with shared parameters | §2.1; §3.2.5 | verified | Fig. 1-2; Fig. E.8 (per head better than whole dimension) |
| paper default | 2.4M-1.2B | pretrain-stable | normalization; biases; layernorm ε; tying | pre-norm; none; 1e-6; untied | §2.1 | verified | no ablation reported |
| paper default | 2.4M-1.2B | pretrain-stable | position encoding; MLP | RoPE; hidden 4d, GeLU | §2.1 | verified | no ablation reported |
| paper default | 2.4M-1.2B | pretrain-stable | initialization | embeddings normal, std 1/√d; other weights truncated normal, std 1/√fan-in | §2.1 | verified | muParam (full) variant with head init change: Fig. E.5 |
| paper default | 2.4M-1.2B | pretrain-stable | batch; sequence length; packing | 256 sequences; 512 tokens; packed, no padding | §2.1; App. A | verified | Fig. E.9 (512 or 1024 at constant data: no meaningful change) |
| paper default | 2.4M-1.2B | pretrain-stable | tokens seen | 1.28e10 | 256 × 512 × 1e5 (§2.1) | derived | not applicable |
| paper default | 2.4M-1.2B | pretrain-stable | data; tokenizer; precision | C4; SentencePiece, vocabulary 32101; bfloat16 on TPUs | §2.1 | verified | no ablation reported |
| paper default | 2.4M-1.2B | pretrain-stable | scaling rule | embedding size, depth, and heads scaled jointly | §2.1 | verified | Fig. 7, Fig. E.3 |
| 4.8B intervention run, LR 0.3 | 4.8B | pretrain-stable | AdamW ε | 1e-15 | §3.4, Fig. 12 | verified | lower loss than 1e-8; 1e-6 diverged (Fig. 12, E.15); one run per value |

## Connections
- [[olmo-2]]: open model family whose card lists QK-Norm and Z-loss among its stability changes.
- [[kimi-k2]]: caps attention logits with QK-Clip under Muon, another control for the attention logit growth studied here.
- [[adam]]: the AdamW update whose ε term §3.4 analyzes.
- [[lr-schedules]]: linear warmup and cosine decay, the default schedule here.
- [[gradient-clipping]]: global-norm clipping at 1 is part of the default setup.
- [[batch-vs-layer-norm]]: LayerNorm, applied here to queries and keys.
- [[c4]]: training data for every run.
- [[kaplan-scaling-laws]], [[chinchilla-compute-optimal]]: loss scaling laws that the paper contrasts with scaling trends of model characteristics (§1, §2.3).
- [[critical-batch-size-pretraining]]: batch-size scaling in pretraining; this paper reports only that batch 256 to 1024 did not change LR sensitivity (Fig. E.9).
- [[task-scaling-model-ladders]]: also fits trends on small models to predict a larger model, for task accuracy instead of stability.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2309.14322 (v2, 2023-10-16; v1 2023-09-25)
- Audit claims not found in the source: none. Precision notes: the ε = 1e-15 improvement was measured on one 4.8B model at LR 0.3; that the gain grows with scale is stated as the authors' belief (§3.4). The lead that this paper is mentioned in classics/early-stopping-and-checkpointing.md is not correct: that card cites Wortsman et al. 2022 (Model Soups), a different paper.
- Not reported by the source: downstream task evaluations; experiments on fast loss spikes; models above 4.8B; venue.

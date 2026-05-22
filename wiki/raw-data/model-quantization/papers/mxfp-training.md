<!-- scope: training experiments with MXFP8 / MXFP6 / MXFP4 microscaling formats
     deps: [[microscaling-formats]]
     see-also: [[deepseek-v3-fp8]], [[fp8-lm]]
-->

# MXFP Training Experiments — MXFP8 / MXFP6 / MXFP4 for Pretraining
- **Core Insight:** Pretraining with microscaling formats (MXFP8 weights+activations+gradients, or MXFP6 in the most aggressive recipes) reaches FP32 / BF16 downstream parity on generative LLMs *if* the 32-element block size is preserved and the optimizer state is kept FP32 — the block-shared exponent absorbs per-channel dynamic-range variation, eliminating the loss-scaling fragility of plain FP8 training.
- **Guideline:** For 4-/6-/8-bit native pretraining on hardware with MX support (Blackwell, MI3xx), default to MXFP8 weights and activations, MXFP8 (or MXFP6) gradients, FP32 optimizer; switch to plain FP8 (no block scale) only when targeting H100-class hardware that lacks MX units.
- **Authors:** various Microsoft / academic groups; the original recipe paper is [[microscaling-formats]] (Rouhani 2023) with follow-on studies in 2024–2025
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2310.10537 (original MX spec + pretraining studies); follow-ons under arxiv search "MX pretraining LLM"
- **Relevant topics:** MX pretraining, block-shared exponent, FP4/FP6/FP8 training, hardware co-design

## Abstract
This entry collects the empirical evidence that MX formats can replace BF16/FP32 *during pretraining*, not just inference. The Rouhani et al. paper reports MXFP6 generative LM training matching FP32 on perplexity and downstream tasks; subsequent studies (Microsoft Phi, NVIDIA Blackwell whitepapers, academic MXFP4 work) extend this to MXFP4 with careful loss scaling and gradient-handling recipes.

## Key Contributions (consolidated)
- Demonstration that pretraining loss matches FP32 within noise for MXFP6 and matches BF16 for MXFP4 (on generative LM at billions-of-tokens scale).
- Recipe components: (1) MX for weights and activations; (2) MX or higher precision for gradients (FP32 master in optimizer); (3) periodic per-block re-scaling to absorb distribution drift.
- Identification of the failure modes: too-coarse block size (>64) loses local outlier capture; too-fine (<16) loses the bit-amortisation benefit.
- Hardware co-design: Blackwell tensor cores natively consume MX blocks; loss-scaling overhead is eliminated relative to vanilla FP8.

## Key Figures/Tables to Study
- Original MX paper Figure 4: training curves for MXFP8/MXFP6/MXFP4 vs FP32 on a 7B-scale generative LM.
- Blackwell whitepaper: per-layer activation-distribution change between Hopper FP8 (no block) and Blackwell MXFP8.
- Academic MXFP4 follow-ups (post-Rouhani 2024): loss-scaling schedules and recovery curves.

## Technical Details

### Recipe summary (MXFP8 pretraining)
- Weights: MXFP8 (E4M3 element, E8M0 block scale, block 32).
- Activations: MXFP8 (same).
- Gradients: MXFP8 (E5M2 element preferred for wider dynamic range, or MXFP6 with E3M2 for aggressive setups).
- Master weights and Adam state: FP32.
- LR schedule: same as BF16 (no extra warmup needed).
- Loss scaling: not required — the per-block scale absorbs gradient dynamic range automatically.

### MXFP4 pretraining (aggressive)
Drops weights to E2M1 (FP4) with the same 32-block scale. Stable when:
- Gradients stay at MXFP8 or higher (MXFP4 gradients are too coarse for stable Adam updates).
- Periodic re-scaling: every K=1000 steps, re-compute per-block scales from the live statistics rather than caching.
- LayerNorm and embedding tables stay in BF16.

### Failure modes
- Block size too large (256+): outlier within block dominates the shared scale, crushing precision of bulk elements. Mitigation: stick to 32.
- All-MXFP4 (gradients too): Adam variance estimate collapses to zero on tiny gradients. Mitigation: gradients at MXFP6 or higher.
- Cold-start instability: random init has narrow distribution that doesn't fill MX dynamic range. Mitigation: a few hundred warmup steps in BF16.

### Throughput / memory
- MXFP8 training: ~2× throughput vs BF16 at iso-quality on Blackwell.
- MXFP4 weights: 4× weight memory reduction vs BF16 — frees HBM for larger batch / longer context.

## Connections
- Format spec: [[microscaling-formats]].
- DeepSeek V3 fine-grained FP8 (smaller-block cousin): [[deepseek-v3-fp8]].
- Earlier FP8 LLM training: [[fp8-lm]].
- Inference-side FP4: [[llm-fp4]].
- Hardware: Blackwell whitepaper [[nvfp4]] (formats), MI3xx native support.

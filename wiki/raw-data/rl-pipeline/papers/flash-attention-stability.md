<!-- scope: Golden et al., "Is Flash Attention Stable?" (arXiv 2405.02803, May 2024): microbenchmark of Flash Attention numeric deviation vs Baseline Attention across precisions, sequence length and tile size; Wasserstein-distance bound on weight deviation during training
     deps: []
     see-also: [[mai-rl-infrastructure]], [[thinking-machines-nondeterminism]], [[nemo-rl-tp-accuracy]], [[fp16-train-infer-mismatch]]
-->

# Is Flash Attention Stable?
- **Core Insight:** Flash Attention's tiled online softmax adds per-tile rescaling, so its output deviates from standard attention by an amount that depends on precision, sequence length and tile size. At BF16 the forward-pass deviation is about 10× that of Baseline Attention, but its effect on trained weights is 2–5× smaller than the effect of FP16-vs-FP32 training (Abstract; Sec. IV–V).
- **Guideline:** Treat tiling and block geometry as numerics, not only performance, settings: larger tiles mean fewer rescalings and less deviation, swapping block-dimension order increases it, and deviation grows with sequence length at fixed tile size. Two implementations that must agree (e.g. an RL sampler and learner) need the same tiling and KV partitioning.
- **Authors:** Alicia Golden, Samuel Hsia, Fei Sun, Bilge Acun, Basil Hosmer, Yejin Lee, Zachary DeVito, Jeff Johnson, Gu-Yeon Wei, David Brooks, Carole-Jean Wu (FAIR at Meta, Harvard University, Meta).
- **Year:** 2024 (arXiv v1, 5 May 2024).
- **URL:** https://arxiv.org/abs/2405.02803
- **Source type:** paper (6-page workshop-style study)
- **Relevant topics:** numeric deviation, Flash Attention tiling, online softmax rescaling, BF16 precision, training instability, Wasserstein distance

## Summary
Loss spikes in large training runs may come from numeric deviation that accumulates over training. The paper builds a numerical re-implementation of Flash Attention that can run at FP64/FP32/FP16/BF16 and be perturbed internally, and validates it against the original kernel (Sec. III). It compares attention outputs on identical inputs (Sec. IV), then compares model weights over training with Flash vs Baseline Attention, and puts that difference in context against two reference perturbations: a different random initialization and FP16 vs FP32 training (Sec. V). The workload is a text-to-image model trained on Shutterstock data on NVIDIA 80GB A100 GPUs (Sec. III).

## Key Contributions
- A microbenchmark that isolates numeric precision and algorithm choices inside Flash Attention (Fig. 2).
- Measurement: ~10× more max-difference deviation than Baseline Attention at BF16, against an FP64 Baseline "golden value" (Fig. 4).
- Sweeps: deviation decreases with more mantissa bits (Fig. 3), increases with sequence length (Fig. 5), decreases with larger blocks (Fig. 6c), increases when block-dimension order is swapped (Fig. 6a), is unchanged by forcing square blocks (Fig. 6b).
- A proxy for significance: Wasserstein distance between weight tensors over training; Flash Attention weight deviation is comparable to or below a different initialization and 2–5× below FP16-vs-FP32 (Fig. 8).

## Key Figures/Tables to Study
- Fig. 1: tiling of the N×N score matrix. Fig. 4: max difference vs FP64 golden value per format. Fig. 5: deviation vs sequence length. Fig. 6: block-size and block-order sweeps. Fig. 7–8: weight deviation over training.

## Technical Details
**Mechanism (Sec. II-B).** Flash Attention computes QKᵀ by blocks that fit in SRAM, with block sizes B_c = ⌈M/4d⌉ and B_r = min(⌈M/4d⌉, d) (M = SRAM size, d = model dimension). Because the online softmax needs global row statistics, each tile applies re-scaling factors; these extra per-tile operations are the hypothesized source of deviation. Measured speedup on the text-to-image model: 14% for forward + backward.

**Microbenchmark results (Sec. IV).**
- Metric: element-wise maximum absolute difference between attention outputs (an upper bound), plus mean and standard deviation of the difference.
- Precision sweep BF16 → FP64: deviation between Flash and Baseline decreases with mantissa bits (Fig. 3).
- Against Baseline at FP64: Flash Attention at BF16 shows "roughly 10× more numeric deviation" than Baseline Attention at BF16 (Fig. 4).
- Sequence length up, tile size fixed: more tiles, more rescaling, more deviation (Fig. 5).
- Larger block area: less deviation, because fewer tiles cover the matrix (Fig. 6c).

**Weight-level analysis (Sec. V).** Max difference `torch.max(torch.abs(flash_attn − baseline_attn))` over weights and Wasserstein distance W(P, Q) = inf over couplings of E‖x − y‖. Both grow over training, so the two models "converge to a different model" (Fig. 7). The rate of weight deviation with Flash Attention is comparable to or less than that from a different random initialization; FP16-vs-FP32 deviation is higher and more variable; Flash Attention gives "roughly 2 − 5× less model weight deviation as compared to low-precision training" (Fig. 8).

**Limits (Sec. VI).** The paper does not link the deviation to actual loss spikes; it states that "ultimately linking this numeric deviation back to training instability requires further investigation". Only one workload type (text-to-image) is studied.

**Relevance to RL infrastructure.** In pre-training the comparison is Flash vs Baseline over one model. In RL, sampler and learner run *different* attention kernels or tilings on the same weights, so the same tiling-dependent deviation appears directly as a sampler–learner log-prob gap. [[mai-rl-infrastructure]] (p. 10) cites this paper when it aligns one FA4 implementation, K-block boundaries and KV split size across learner forward, sampler prefill and decode.

## Verification
Read the full 6-page PDF (arXiv 2405.02803v1) via `pdftotext` on 2026-11-24. Numbers (10×, 2–5×, 14%, block-size formulas) are quoted from the abstract and Sec. II–V. Figure plots were not read as images; trends are those stated in the text.

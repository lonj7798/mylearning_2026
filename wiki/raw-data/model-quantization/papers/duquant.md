<!-- scope: DuQuant — dual rotation + zigzag channel permutation handles both "normal" and "massive" outliers for W4A4 PTQ
     deps: [[quarot]], [[smoothquant]]
     see-also: [[spinquant]], [[flatquant]], [[awq]]
-->

# DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs
- **Core Insight:** LLM activations have *two* outlier regimes — moderate per-channel outliers (the SmoothQuant kind) and rare but enormous "massive" outliers (orders of magnitude bigger) — and you need *both* block-wise rotation *and* channel permutation to flatten both simultaneously.
- **Guideline:** Before quantizing, apply (1) a block-diagonal rotation whose blocks target the known outlier channel indices, then (2) a zigzag permutation that interleaves high- and low-variance blocks to equalize block-wise dynamic range, then quantize uniformly.
- **Authors:** Haokun Lin, Haobo Xu, Yichen Wu, Jingzhi Cui, Yingtao Zhang, Linzhan Mou, Linqi Song, Zhenan Sun, Ying Wei
- **Year:** 2024 (NeurIPS 2024 Oral)
- **URL:** https://arxiv.org/abs/2406.01721
- **Relevant topics:** rotation + permutation, massive outliers, normal outliers, W4A4 PTQ, block-wise transforms

## Abstract
DuQuant introduces a dual transformation — rotation plus permutation — to distribute outliers across channels and across blocks. Rotation matrices are constructed using outlier-dimension priors to redistribute outliers to adjacent channels via block-wise rotation; zigzag permutation reorders channels so that high-variance and low-variance blocks alternate, balancing block-wise variance. The approach addresses *massive outliers* (a few channels with values orders of magnitude above the rest) that prior rotation-only methods like QuaRot handle imperfectly. SOTA W4A4 PTQ across LLM sizes.

## Key Contributions
- Empirical taxonomy: separates "normal" outliers (broad, ~10× bulk magnitude, SmoothQuant-style) from "massive" outliers (few coordinates, >100× bulk, concentrated in specific layers/channels).
- Constructs **rotation R** as block-diagonal with each block crafted from the known outlier coordinate indices — not a generic random Hadamard.
- Introduces **zigzag permutation** to rearrange blocks so adjacent blocks balance high and low variance, reducing per-block quantization range mismatch.
- Achieves SOTA W4A4 PTQ; outperforms QuaRot/SpinQuant on LLaMA models, especially where massive outliers dominate.

## Key Figures/Tables to Study
- **Figure 1:** Activation magnitude heatmap — visual evidence of "massive" vs "normal" outliers, motivating dual transformation.
- **Figure 3:** Block-rotation + zigzag-permutation pipeline schematic.
- **Table 2:** W4A4 WikiText-2 PPL on LLaMA-1/2/3 — DuQuant vs QuaRot vs SmoothQuant.

## Technical Details

### Step 1 — outlier-prior block rotation
Identify outlier channel indices {i_1, ..., i_k} from calibration. Form block-diagonal R = blkdiag(R_1, ..., R_B) where each block R_b is a small (e.g. 128×128) rotation that includes the outlier channels falling into block b. Each R_b is chosen as a Greedy-Householder-style rotation aligning the dominant outlier direction with the block's mean direction, redistributing the spike across all 128 in-block channels.

Block size = 128 keeps the online cost low (each block is a small dense matmul fused into the next linear).

### Step 2 — zigzag channel permutation
After rotation, blocks still have very different per-block dynamic ranges (some blocks contain residual outlier energy, others don't). Sort blocks by variance and apply a zigzag interleaving:
`σ²(b_1) ≥ σ²(b_2) ≥ ... ≥ σ²(b_B)` → permute to `b_1, b_B, b_2, b_{B−1}, ...`
so adjacent blocks have complementary variance. Reduces the per-group quantization scale mismatch when channels are grouped for INT4.

### Step 3 — uniform W4A4 PTQ
After R and permutation P are absorbed into surrounding weights (P is just a reindexing; R is folded offline like QuaRot), apply standard per-channel weight INT4 + per-token activation INT4 round-to-nearest. No special outlier path.

### Why "dual" beats rotation-only
- A single random Hadamard (QuaRot) reduces a single massive outlier of magnitude M to M/√d ≈ M/90 for d=8192 — but still leaves the *block* containing that outlier with elevated variance vs other blocks.
- Adding block-wise permutation interleaves high-σ and low-σ blocks, so per-group quant scales sit near a global average rather than swinging.

### Inference cost
- R: block-diagonal, fused into adjacent linears offline (zero runtime cost per block) or online via small fused matmul.
- P: a free reindex (no runtime FLOPs).
- INT4 GEMM: standard Marlin / TensorRT-LLM kernels.

## Connections
- Rotation lineage: [[quarot]] (random Hadamard) → [[spinquant]] (learned) → [[duquant]] (block + permutation) → [[flatquant]] (affine).
- Outlier-aware scaling predecessor: [[smoothquant]], [[awq]].
- The "massive outlier" phenomenon overlaps with the outlier-feature observation in [[llm-int8]].
- Weight quantizer paired: [[gptq]] or RTN.

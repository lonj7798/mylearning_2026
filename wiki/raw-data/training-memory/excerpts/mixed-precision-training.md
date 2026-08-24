# Mixed Precision Training
<!-- slug: mixed-precision-training · type: paper · source: https://arxiv.org/abs/1710.03740 -->

**Core Insight.** Training in fp16 requires three interventions — fp32 master weights (prevents gradient-update underflow), loss scaling (shifts gradient histogram into fp16's representable range), and fp32 accumulation for dot-products — because fp16's 10-bit mantissa and limited exponent range cause silent precision loss that diverges training without them.

**Guideline.** Always keep an fp32 master copy of weights updated by the optimizer; use the fp16 copy only for the forward/backward compute. Scale the loss by a constant (8–32K empirically, or `65504 / max_gradient_magnitude`) before backprop and unscale before gradient clipping.

## Technical Details

- **fp32 master weight rationale (two failure modes without it):**
  1. "Any value whose magnitude is smaller than 2^−24 becomes zero in FP16" — ~5% of weight gradients underflow and vanish per step.
  2. When `|weight| / |update| > 2048`, binary-point alignment with only 10 mantissa bits makes the update identically zero.
  - Empirical cost: Mandarin speech model showed "80% relative accuracy loss" training with fp16-only weights.
- **Loss scaling mechanism:** Scale loss *L* → *c·L* before backward; chain rule automatically scales all gradient magnitudes by *c*; unscale immediately after backward (before clip). This is a single multiply, not a rewrite of the graph.
- **Gradient histogram evidence:** For SSD object detector, "67% of [gradient] values are zero" in fp16 without scaling; with 8× scaling, training matched fp32. bigLSTM required 128× scaling.
- **Storage layout:**
  - Weights: FP16 (compute) + FP32 master (optimizer update) — 50% weight overhead vs fp32-only
  - Activations: FP16
  - Gradients: FP16 (scaled, before unscaling)
  - Overall effect: training memory "roughly halved" because activations dominate at large batch sizes.
- **Arithmetic rules:** vector dot-products: fp16 inputs, fp32 partial accumulation, fp16 output; batch-norm/softmax reductions: read/write fp16, compute in fp32.
- **BF16 note:** BF16 has the same exponent width as fp32 (8 bits vs fp16's 5), so BF16 eliminates the dynamic-range problem; this paper's loss-scaling technique is less critical for BF16. Modern practice uses BF16 + fp32 masters without loss scaling.
- **Training-memory angle:** Establishes the foundational three-copy precision layout (fp16 working weights + fp32 master + fp16 activations) that defines the 16 N byte static floor computed in [[transformer-math-101]] and [[ultrascale-playbook]]. Every modern framework's "mixed precision" mode descends from this paper's prescription.

## Citation
Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu. "Mixed Precision Training." ICLR 2018. https://arxiv.org/abs/1710.03740

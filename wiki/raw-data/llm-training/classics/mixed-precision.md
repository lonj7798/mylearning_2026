<!-- scope: the FP16 mixed-precision training recipe (master weights, loss scaling, FP32 accumulation)
     deps: [[adam]]
     see-also: [[gradient-clipping]], [[batch-vs-layer-norm]], [[deepseek-v3]], [[mixed-precision-recipe]]
-->

# Mixed Precision Training
- **Core Insight:** Storing weights, activations, and gradients in IEEE FP16 matches FP32 accuracy with no
  hyper-parameter changes when three things are added: an FP32 master copy of the weights updated by the optimizer,
  loss scaling before back-propagation, FP16 dot products accumulated into FP32 (Abstract; §3).
- **Guideline:** When training in FP16 and gradient values fall below the FP16 representable range, scale the loss by
  a constant factor chosen so the scaled maximum absolute gradient stays below 65,504, and unscale the weight gradients
  before gradient clipping or weight decay, because unscaling later would require re-tuning those hyper-parameters
  (§3.2). Formats with FP32 exponent range are outside this paper's experiments.
- **Authors:** Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, et al.
  (NVIDIA and Baidu Research; the PDF byline orders the Baidu authors first)
- **Year:** 2017 (arXiv v1 2017-10; v3 2018-02; ICLR 2018)
- **URL:** https://arxiv.org/abs/1710.03740
- **Source type:** paper
- **Relevant topics:** numerical precision, loss scaling, FP32 master weights, training throughput

## Abstract
The paper presents a methodology for training deep networks with IEEE half-precision (FP16) weights, activations, and
gradients without losing accuracy and without modifying hyper-parameters. Because FP16 has a narrower range than FP32,
three techniques are proposed: an FP32 master copy of the weights that accumulates the gradient at each optimizer step
and is rounded to FP16 for the forward and backward passes; loss scaling to preserve small gradient magnitudes; and
half-precision arithmetic that accumulates into single-precision outputs. It is demonstrated on models exceeding 100
million parameters across classification, detection, speech, language modelling, translation, and image generation.

## Key Contributions
- An FP16 training recipe that requires no change to model architecture or hyper-parameters (Abstract, §1).
- **FP32 master weights** (§3.1): the optimizer updates an FP32 copy; an FP16 copy is used each iteration.
- **Loss scaling** (§3.2): multiply the loss before back-propagation; the chain rule scales all gradients equally;
  unscale the weight gradients before the update.
- **FP32 accumulation** (§3.3): FP16 dot products accumulate into an FP32 value converted to FP16 on the memory
  write; large reductions are carried out in FP32.
- Accuracy parity on ILSVRC12 classification, detection, speech, translation, language modelling, DCGAN (§4.1-§4.6).

## Key Figures/Tables to Study
- **Figure 2b** — weight-gradient exponent histogram for the Mandarin speech run; about 5% of values have exponents
  below −24, which is the motivation for FP32 master weights (§3.1).
- **Figure 3** — activation-gradient histogram for Multibox SSD; 67% of values are zero; values in [2^−27, 2^−24)
  mattered while values below 2^−27 did not (§3.2).
- **Table 1** — ILSVRC12 top-1 accuracy, baseline vs mixed precision, six CNNs (§4.1). **Table 2** — detection mAP with and without loss scaling (§4.2).

## Technical Details
**Why FP32 master weights are needed** (§3.1). Two mechanisms are given. (1) Any value smaller in magnitude than 2^−24
becomes zero in FP16, and about 5% of weight-gradient values in the Mandarin run have exponents below −24 (Figure 2b).
(2) When a normalized weight is at least 2048 times larger in magnitude than its update, the addition right-shifts the
update to zero even though the update itself is representable. Storing activations in half precision also roughly
halves total training memory.

**FP16 range used by the paper** (§3.2). The exponent bias centres normalized-value exponents on [−14, 15]; 65,504 is
the largest representable FP16 value. FP32, BF16, and FP8 ranges are not tabulated in this paper.

**Loss-scaling factors actually used** (§3.2, §4.2, §4.5). Constant factors from 8 to 32K were used, and many networks
required none (§3.2). Multibox SSD diverges unscaled and matches FP32 at a factor of 8 (§3.2, §4.2); bigLSTM diverges
after 300K iterations unscaled and matches FP32 at 128 (§4.5). The selection rule is a factor whose product with the
maximum absolute gradient stays below 65,504; a larger factor has no stated downside unless it overflows, and an
overflowed step can be skipped.

**Arithmetic rules** (§3.3). Vector dot products: FP16 inputs, FP32 accumulation of partial products, FP16 on the
memory write; without this, some models did not match baseline accuracy. Volta Tensor Cores can accumulate into either
FP16 or FP32. Large reductions — batch-normalization statistics and softmax — run in FP32 while still reading and
writing FP16 tensors; this did not slow training because those layers are memory-bandwidth limited (§3.3).

**Reported speed** (§1, §5). Half-precision math throughput on the GPUs of the time is 2x to 8x that of single
precision (§1). DNN operations benchmarked with DeepBench on Volta show 2-6x speedups over FP32 when memory- or
arithmetic-bandwidth limited, less when latency limited (§5). End-to-end training throughput is not reported; §5 says
full-network speedups depend on library and framework optimizations left to future work. **Dynamic loss scaling**
appears in §5 only as future work — adjust the factor by inspecting weight gradients for overflow and skip the update
on overflow. No initial factor, growth interval, or back-off ratio is given in this paper.

## Recipe ledger
See [[mixed-precision-recipe]] for the per-experiment table (loss-scale factors, optimizers, batch sizes, arithmetic
settings) at arXiv:1710.03740v3 §3.2-§4.6. Learning rates, warmup, and weight decay are not reported; §4 states that
public training schedules were reused unchanged from the FP32 baselines.

## Findings relevant to generality
The recipe is evaluated across six task families; the generality claim is that no hyper-parameter retuning was needed
in any of them (Abstract, §1, §4). Limits: loss scaling was needed for SSD, bigLSTM, and translation and not for the
ILSVRC12 CNNs or DCGAN (§4.1-§4.6); text-to-speech and deep RL were named as untested (§5).

## Formats defined by other sources (not by this paper)
- FP8 E4M3 and E5M2 are defined in *FP8 Formats for Deep Learning* (arXiv:2209.05433 Table 1). Max normal 448 and
  57,344; min normal 2^−6 and 2^−14; min subnormal 2^−9 and 2^−16; exponent bias 7 and 15. Proposed usage there is
  E4M3 for weight and activation tensors, E5M2 for gradient tensors (§3).
- DeepSeek-V3 ([[deepseek-v3]], arXiv:2412.19437) departs from that split (all loci §3.3.2 unless noted): E4M3 on all
  tensors; activations quantized per 1x128 tile and weights per 128x128 block; scales computed online rather than from
  an amax history; partial sums promoted to CUDA cores for FP32 accumulation every 128 elements because Hopper Tensor
  Core FP8 accumulation retains about 14 bits; AdamW moments in BF16 with master weights and gradients in FP32
  (§3.3.3); FP8 relative loss error below 0.25% against a BF16 baseline at roughly 1T tokens (§3.3, App. B.1).

## Connections
- **[[adam]]** — the optimizer state is what the master-weight argument of §3.1 applies to.
- **[[gradient-clipping]]** — §3.2: unscale before clipping so the threshold need not be retuned.
  **[[batch-vs-layer-norm]]** — §3.3 places batch-normalization statistic accumulation in FP32.
- **[[deepseek-v3]]** — a large-scale FP8 framework that cites this paper for FP32 accumulation (§3.3.2).

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/1710.03740 (arXiv v3, ICLR 2018), plus arXiv:2209.05433 Table 1
  and arXiv:2412.19437 §3.3 for the format and DeepSeek-V3 claims.
- Corrections to the previous card version:
  - "2–8x the throughput" of the recipe and "doubling effective batch size and 2–6x training throughput on Volta" →
    2x-8x is the GPUs' half-precision math-throughput ratio (§1); 2-6x is the DeepBench per-operation speedup (§5);
    memory is roughly halved (§3.1); no effective-batch-size or training-throughput claim is made.
  - Dynamic loss scaling "start S = 2^15 ... halve ... double every 2000 successful steps" → §5 proposes dynamic
    scaling as future work with no constants; those constants come from framework implementations.
  - fp8-E4M3 "~2e-7 to 448" and fp8-E5M2 "~6e-8 to 57344" → min normal 2^−6 / 2^−14, min subnormal 2^−9 / 2^−16
    (arXiv:2209.05433 Table 1).
  - "E4M3 forward, E5M2 backward ... DeepSeek-V3 1×128 / 128×128 blocks ... delayed scaling" → DeepSeek-V3 uses E4M3
    on all tensors, 1x128 activation tiles and 128x128 weight blocks, and online rather than delayed quantization
    (arXiv:2412.19437 §3.3.2).
  - "Keep LayerNorm/RMSNorm in fp32" attributed here → §3.3 names batch-normalization statistics and softmax
    reductions; LayerNorm and RMSNorm are not discussed. Author order corrected to the arXiv listing.
- Removed as unsupported by the source: the FP32/BF16 rows of the format table; "bf16 is the modern default ... Llama,
  GPT-NeoX, Mistral, Qwen, DeepSeek all use this"; "~1% relative precision" for BF16; "fp8 speedup ~2x over bf16 on
  H100"; "Llama-3 70B / 405B trained in bf16 throughout"; the "common pitfalls" list (fp16/bf16 mixing causing silent
  divergence, quantized loss curves, AdamW eps 1e-5 under fp16); "bf16 `v_hat` underflows within ~100 steps"; the
  Karpathy fp32-first recommendation (belongs to [[karpathy-training-neural-net-recipe]], not verified here).
- Not reported: learning rates, warmup, weight decay, clipping thresholds, BF16 results, transformer results, wall-clock time.

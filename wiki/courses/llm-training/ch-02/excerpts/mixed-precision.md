---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "Micikevicius, Narang, Alben, Diamos, Elsen, Garcia, et al. — Mixed Precision Training (ICLR 2018)"
source_url: https://arxiv.org/abs/1710.03740
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten from arXiv:1710.03740v3; generality revision)"
---

# Excerpt: Mixed Precision Training (Micikevicius et al., arXiv 2017-10, ICLR 2018)

Rewritten on 2026-09-15 from the primary text (arXiv:1710.03740v3, 2018-02-15). The library card
[[mixed-precision]] has not yet been re-verified and contains values that do not appear in this paper
(FP8 ranges, a dynamic initial scale of 2^15, an "fp32 accumulator" description); this excerpt follows the paper.

## What the paper does
- Stores weights, activations, and gradients in IEEE half precision (FP16) and proposes three techniques so
  that accuracy matches FP32 training without changing hyperparameters (Abstract, §1).
- Tests CNNs for ILSVRC classification, Faster R-CNN and Multibox SSD detection, DeepSpeech 2 speech models,
  English→French LSTM translation, bigLSTM language modeling on the 1B-word dataset, and DCGAN (§4).
- The paper predates bfloat16 and FP8 training; it does not discuss either.

## Technique 1: FP32 master copy of weights (§3.1)
- An FP32 master copy is updated by the optimizer; an FP16 copy is used in forward and backward passes.
- Reason 1: updates (gradient × learning rate) below 2^-24 in magnitude become zero in FP16. In the Mandarin
  speech model, about 5% of weight-gradient values have exponents below −24 (Fig. 2b).
- Reason 2: when a normalized weight is at least 2048 times larger than its update, the FP16 addition shifts
  the update's implicit bit right by 11 or more positions and can produce zero; above 2048 the loss is
  unrecoverable (§3.1).
- Updating FP16 weights directly gave 80% relative accuracy loss on the Mandarin model; the FP32 master copy
  matched FP32 training (Fig. 2a).
- The master copy adds 50% to weight memory, but total training memory is about halved because activations
  are stored in FP16 (§3.1).

## Technique 2: loss scaling (§3.2)
- FP16's normalized exponent range is [−14, 15]. In the Multibox SSD activation-gradient histogram, much of
  the FP16 range is unused while many values fall below it (Fig. 3).
- SSD diverges without scaling; a factor of 8 matches FP32. Values below 2^-27 were irrelevant for this
  model; values in [2^-27, 2^-24) had to be preserved (§3.2).
- Scaling the loss scales every gradient by the same factor through the chain rule. Weight gradients must be
  unscaled before the update, "right after the backward pass but before gradient clipping or any other
  gradient-related computations", so that clipping thresholds and weight decay need no change (§3.2).
- Constant factors from 8 to 32K were used; many networks needed none. A factor can be chosen so that its
  product with the maximum absolute gradient stays below 65,504 (§3.2).
- Overflow produces infinities and NaNs that "irreversibly damage the weights" if applied; one option is to
  skip the update when overflow is detected (§3.2).
- Automatic (dynamic) scale selection is listed as future work (§5). The paper does not specify an initial
  dynamic scale or growth interval.

## Technique 3: arithmetic precision (§3.3)
- Some networks require FP16 dot products to accumulate partial products into an FP32 value, converted to
  FP16 before writing to memory; Volta Tensor Cores support FP16 inputs with FP16 or FP32 accumulation.
- "Large reductions (sums across elements of a vector) should be carried out in FP32", mostly in batch
  normalization statistics and softmax layers; those layers read and write FP16 tensors (§3.3).
- Point-wise operations can use FP16 or FP32 (§3.3).

## Results with numbers (§4)
| Setting | FP32 baseline | Mixed precision | Locus |
|---|---|---|---|
| ResNet-50 ILSVRC top-1 | 75.92% | 76.04% (no loss scaling) | Table 1 |
| Faster R-CNN mAP (VOC 2007) | 69.1% | 68.6% without scaling, 69.7% with | Table 2 |
| Multibox SSD mAP | 76.9% | diverges without scaling, 77.1% with | Table 2 |
| DeepSpeech 2 English CER (WSJ '92) | 2.20 | 1.99 | Table 3 |
| bigLSTM (1B words) | — | FP16 without scaling diverges after 300K iterations; factor 128 matches FP32 | §4.5, Fig. 5 |

- DeepBench operations on Volta show 2-6x speedups over FP32 when limited by memory or arithmetic bandwidth
  (§5); full-network speedups are not reported.

## Used in ch-02
- §2 (master weights, reductions), §3 (loss scaling procedure and evidence), Common mistakes (unscale before
  clipping).

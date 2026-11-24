<!-- scope: DeepSeek-V4 report (arXiv 2606.19348), train/inference consistency parts only: Sec. 3.2 TileLang numerics, Sec. 3.3 batch-invariant and deterministic kernels, Sec. 5.2.1 FP4 QAT with native FP4 rollout, Sec. 5.2.3 WAL-based preemptible rollout (length-bias argument), Sec. 5.2.5 DSec deterministic replay
     deps: [[deepseek-v4-recipe]]
     see-also: [[mai-rl-infrastructure]], [[thinking-machines-nondeterminism]], [[deepseek-open-infra]], [[fp16-train-infer-mismatch]], [[truncated-importance-sampling]]
-->

# DeepSeek-V4: train/inference consistency infrastructure
- **Core Insight:** DeepSeek-V4 makes its kernels end-to-end **bitwise batch-invariant and deterministic** so that pre-training, post-training and inference produce aligned numerics. It pays for this with engineering work instead of throughput: a dual-kernel decode attention without split-KV, DeepGEMM instead of cuBLAS, no split-K in most cases, and deterministic backward reductions (Sec. 3.3, p. 18–19).
- **Guideline:** If training simulates a quantization, the rollout should run the real quantized kernels used in deployment (native FP4 at rollout, not simulated). Interrupted rollouts must be resumed, not regenerated, because regeneration biases toward short responses; with a batch-invariant, deterministic stack, regeneration with the same sampler seed is also correct but costs a second decode (Sec. 5.2.1, 5.2.3).
- **Authors:** DeepSeek-AI.
- **Year:** 2026 (arXiv v1, submitted 26 Apr 2026).
- **URL:** https://arxiv.org/abs/2606.19348 (title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence")
- **Source type:** model technical report (58 pages); this card covers only its consistency-related infrastructure
- **Relevant topics:** batch invariance, deterministic kernels, split-KV, split-K, wave quantization, FP4 QAT, rollout fault tolerance, length bias, sandbox replay

## Summary
arXiv 2606.19348 is the DeepSeek-V4 report: DeepSeek-V4-Pro (1.6T total, 49B activated) and DeepSeek-V4-Flash (284B, 13B activated), both with one-million-token context, pre-trained on more than 32T tokens (Abstract). Among four infrastructure items in Sec. 1, the third is "efficient batch-invariant and deterministic kernel libraries to ensure bitwise reproducibility across training and inference" (p. 4). Sec. 3.3 states the goal as "training reproducibility and bitwise alignment among pre-training, post-training, and inference pipelines", with kernels "helpful for debugging, stability analysis, and consistent post-training behavior". The post-training sections add consistency rules for quantized rollout and for interrupted generation. The model, architecture and recipe are covered in [[deepseek-v4-recipe]]; this card does not repeat them.

## Key Contributions
- Batch-invariant decode attention via two kernels with identical accumulation order (Sec. 3.3).
- End-to-end replacement of cuBLAS with DeepGEMM; split-K abandoned in most cases with compensating optimizations (Sec. 3.3).
- Deterministic backward for sparse attention, MoE, and the mHC small matmul (Sec. 3.3).
- TileLang defaults: fast-math off, IEEE intrinsics, layout annotations for bit-identical lowering (Sec. 3.2, p. 17–18).
- Native FP4 weights at rollout after FP4→FP8 QAT; token-granular WAL rollout service (Sec. 5.2.1, 5.2.3).

## Key Figures/Tables to Study
No figure in these sections; the content is in prose. Read Sec. 3.3 (p. 18–19), 5.2.1 (p. 33–34), 5.2.3 (p. 34–35).

## Technical Details
**Definitions (Sec. 3.3).** Batch invariance: "the output of any given token remains bitwise identical, regardless of its position within a batch". Determinism: same result across runs; non-determinism "typically stems from non-deterministic accumulation order, often due to the use of atomic addition instructions", mainly in the backward pass.

**Batch invariance.**
- *Attention*: split-KV (FlashDecoding) cannot be used, but dropping it causes wave-quantization and low utilization. Kernel 1 computes a whole sequence on one SM (fully occupied waves). Kernel 2 uses several SMs per sequence for the last partially filled wave, with its calculation path designed so that "its accumulation order is the same as that of the first kernel", exchanging data through distributed shared memory in thread-block clusters. Overhead is described as "negligible".
- *Matmul*: "Traditional cuBLAS library ... cannot achieve batch invariance", so it is replaced "end-to-end with DeepGEMM". Split-K for small batches cannot guarantee batch invariance and is abandoned "in most scenarios", with optimizations to "match or even surpass the performance of standard split-k in most major scenarios".

**Determinism (backward).**
- *Sparse attention backward*: atomicAdd for KV-gradient accumulation is replaced by per-SM buffers and a global deterministic summation.
- *MoE backward*: concurrent writes from SMs of different ranks to one receive buffer are made deterministic by a per-rank token-order pre-processing step plus buffer isolation across ranks.
- *mHC matmul* (output dimension 24): split-K is required at small batch; each split is written separately and reduced deterministically in a following kernel.

**TileLang numerics (Sec. 3.2).** Fast-math disabled at compiler level; approximations only as opt-in operators (`T.__exp`, `T.__log`, `T.__sin`); IEEE intrinsics with explicit rounding (`T.ieee_fsqrt`, `T.ieee_fdiv`, `T.ieee_add`); simplification rules aligned with NVCC; `T.annotate_layout` pins accumulation order "enabling bit-identical outputs when desired".

**Quantized rollout (Sec. 5.2.1).** MXFP4 QAT on MoE expert weights and the QK path of the CSA indexer. FP32 master weights → FP4 → dequantized to FP8 for compute; FP4 (E2M1) → FP8 (E4M3) is lossless when the max/min scale ratio of the 1×32 FP4 sub-blocks within each 128×128 FP8 block is below a threshold, which the authors verify empirically. Backward uses the same FP8 weights (straight-through estimator). Index scores are cast FP32 → BF16, giving a 2× top-k selector speedup with 99.7% recall of KV entries. At inference and RL rollout, "we directly use native FP4 quantized weights instead of simulated quantization. This ensures that model behavior during sampling is fully consistent with online deployment."

**Rollout service (Sec. 5.2.3).** The cluster scheduler may preempt any task. Each request has a token-granular write-ahead log; on preemption the engine pauses and saves KV cache; on resumption decoding continues from the WAL and saved KV; after a fatal error, prefill is re-run from WAL tokens. Regenerating from scratch "is mathematically incorrect" because "shorter responses are more likely to survive interruption". Sec. 5.2.4 adds metadata/heavy-field separation for million-token RL batches.

**Sandbox replay (Sec. 5.2.5).** DSec keeps a globally ordered trajectory log per sandbox, used for fast-forward after preemption (replaying cached results of completed, possibly non-idempotent commands), provenance, and "deterministic replay" of any session.

**Relevance.** [[mai-rl-infrastructure]] (p. 9) cites DeepSeek as related work on consistency between training and inference implementations. DeepSeek targets batch invariance inside each kernel; Rocket targets equal reduction structure across sampler prefill, decode and learner forward.

## Verification
Read the arXiv abstract page (title, 26 Apr 2026 submission) and the PDF text via `pdftotext` on 2026-11-24: Abstract, Sec. 1 (p. 4), Sec. 3.2–3.3 (p. 17–19), Sec. 5.2.1–5.2.5 (p. 33–35). No RL-objective-level mismatch correction (e.g. importance-ratio truncation) was found by keyword search; the card does not claim one.

---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "Yuan, Li, Ding, Xie, Li, Zhao, et al. — Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference (NeurIPS 2025); v1 title: Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning"
source_url: https://arxiv.org/abs/2506.09501
created_at: "2026-09-15"
revised: "2026-09-15 (created from arXiv:2506.09501v2; generality revision)"
---

# Excerpt: numerical nondeterminism in LLM evaluation (Yuan et al., arXiv 2025-06)

Created on 2026-09-15 from arXiv:2506.09501v2 (2025-10-24, NeurIPS 2025). The v1 of 2025-06-11 is titled
"Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning". No library card exists.

## Setup (§3.1)
- Models: DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B (reasoning); Qwen2.5-7B-Instruct and
  Llama-3.1-8B-Instruct (non-reasoning). Additional Qwen3-32B and GPQA Diamond results are in the appendix.
- Benchmarks: AIME'24, MATH500, LiveCodeBench Easy/Medium/Hard. Maximum output 32,768 tokens for reasoning
  models, 2,048 for non-reasoning models. Backend: vLLM (HuggingFace Transformers checked in the appendix).
- 12 runtime configurations: GPU type (L40S, A100) × GPU count (2, 4) × batch size (8, 16, 32).
- Greedy metrics: Std@Acc (sample standard deviation of accuracy over the 12 configurations), standard
  deviation of output length, divergence index (first token position where outputs differ).
- Sampling: temperature 0.7, top-p 0.95; Pass@1 averaged over 16 or 64 runs (AIME'24) or 4 (MATH500).

## Mechanism (§2.2, §3.2, §3.4)
- Floating-point addition is not associative; kernels change reduction order with continuous batching,
  operator implementation (for example split versus non-split matmul), block sizes, collective operations,
  and tensor parallelism (§2.2).
- Example of rounding (Table 1): the value 1.00012 is stored as 1.0 in FP16 and BF16 (error ≈ −0.00012) and
  as 1.00012004375457761 in FP32 (error ≈ +4.38e-8).
- At the divergence point the top-2 token probabilities are close; one BF16 run has "know" 49.75% and "have"
  43.91%, another has "have" 46.65% and "know" 46.64% (Fig. 3). The authors attribute larger top-1 probability
  variation in BF16 to its 7 mantissa bits versus 10 for FP16 and 23 for FP32 (§3.2, Fig. 4).
- Batch-invariant kernels handle batch-size changes but not changes in tensor-parallel size or GPU type (§2.2).
- 4 GPUs gave more probability variation than 2; smaller batches gave more than larger; A100 slightly more than
  L40S (Fig. 6; the explanations given are labeled "potentially" and "likely" by the authors).

## Greedy decoding results (§3.2)
| Model | Benchmark | Std@Acc BF16 | FP16 | FP32 | Locus |
|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | AIME'24 | 9.15% | 5.74% | 0 | Table 3 |
| DeepSeek-R1-Distill-Qwen-7B | MATH500 | 1.04% | 1.12% | 0.12% | Table 3 |
| DeepSeek-R1-Distill-Llama-8B | AIME'24 | 4.60% | 6.00% | 5.8e-17 | Table 3 |
| Qwen2.5-7B-Instruct | AIME'24 | 1.71% | 1.45e-17 | 1.45e-17 | Table 3 |
| Llama-3.1-8B-Instruct | MATH500 | 0.94% | 0.34% | 0.13% | Table 3 |

- Output-length standard deviation on AIME'24 for DeepSeek-R1-Distill-Qwen-7B: 9,189.53 tokens (BF16),
  5,990.32 (FP16), 0 (FP32) (Table 4).
- Share of MATH500 examples whose outputs diverge across configurations, DeepSeek-R1-Distill-Qwen-7B:
  BF16 96.6%, FP16 73.0%, FP32 2.2% (Fig. 5).

## Sampling results (§3.3, Table 5)
- Standard deviation of Pass@1 across 6 configurations (3 batch sizes × 2 GPU counts), DeepSeek-R1-Distill-
  Qwen-7B, AIME'24 n = 16: BF16 1.7151, FP16 0.8273, FP32 1.1785; n = 64: 0.3749, 0.5391, 0.7377.
- The authors note AIME'24 has 30 problems, so one problem moves Pass@1 by about 3.33 points, and interpret the
  n = 64 exception as sampling noise (§3.3).

## Mitigation: LayerCast (§4)
- Linear-layer weights stored in BF16 and upcast to FP32 just in time for each matrix multiplication; all
  computation in FP32. Divergence rates below 3.4% across batch sizes and GPU configurations; memory 34% lower
  than full FP32 (§4, Fig. 8). Full FP32 "doubles the memory usage and inference time compared to BF16" (§4).
- Suggestions (§1): with enough compute, use sampling with multiple runs and report mean accuracy, average
  length, and error bars; for single-run greedy decoding, use FP32.

## Used in ch-02
- §1 (rounding example), §6 (evaluation nondeterminism evidence and mitigations), Recipe eval-gate row,
  Generalization lens (c).

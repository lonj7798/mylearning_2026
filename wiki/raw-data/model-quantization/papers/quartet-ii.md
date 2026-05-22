<!-- scope: NVFP4 LLM pretraining with improved unbiased gradient estimation via MS-EDEN
     deps: [[nvfp4-training]], [[nvfp4]], [[stochastic-rounding]]
     see-also: [[mxfp4-native-hardware-2026]], [[deepseek-v3-fp8]], [[transformer-engine]]
-->

# Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation
- **Core Insight:** Replacing stochastic rounding with MS-EDEN, an unbiased microscaling quantizer with lower error, improves NVFP4 gradient estimation across forward and backward linear-layer GEMMs.
- **Guideline:** For NVFP4 training recipes, do not treat stochastic rounding as the endpoint; evaluate unbiased quantizers that exploit the microscale structure itself, especially on gradient paths.
- **Authors:** Andrei Panferov, Erik Schultheis, Soroush Tabesh, Dan Alistarh
- **Year:** 2026
- **URL:** https://arxiv.org/abs/2601.22813
- **Relevant topics:** NVFP4, Blackwell, quantized training, unbiased gradient estimation, microscaling, MS-EDEN

## Abstract
Quartet II targets the remaining accuracy gap in NVFP4 pretraining. Earlier NVFP4 recipes rely heavily on stochastic rounding to keep gradients unbiased, but stochastic rounding still leaves meaningful quantization error. The paper introduces MS-EDEN, a quantization routine specialized for microscaled formats, and integrates it into a fully NVFP4 linear-layer scheme. The authors validate end-to-end LLM training up to 1.9B parameters on 38B tokens and provide Blackwell kernels reporting up to 4.2x speedup over BF16 linear layers.

## Key Contributions
- Introduces **MS-EDEN**, an unbiased microscaling quantizer with more than 2x lower quantization error than stochastic rounding in the paper's analysis.
- Builds **Quartet II**, a linear-layer quantization scheme that uses NVFP4 more directly instead of spending representational capacity to compensate for stochastic-rounding error.
- Analyzes all major matrix multiplications in a transformer linear layer, including forward, activation-gradient, and weight-gradient paths.
- Shows compatibility with recent NVFP4 training improvements, so the paper should be read as an upgrade to the [[nvfp4-training]] recipe rather than a replacement for all of it.
- Provides GPU kernels for NVIDIA Blackwell and reports up to 4.2x speedup over BF16 for the relevant linear-layer operations.

## Key Figures/Tables to Study
- MS-EDEN vs stochastic-rounding error comparison: the cleanest evidence for why the paper matters.
- Quartet II computation graph: shows where each quantized matrix multiplication is placed.
- End-to-end training curves up to 1.9B parameters / 38B tokens: establishes whether the lower quantization error survives in a real training run.
- Kernel throughput table on Blackwell: separates numerical quality from hardware payoff.

## Technical Details

### Problem framing
NVFP4 has enough dynamic range to train with, but a naive quantizer makes gradient estimates too noisy. Prior recipes lean on stochastic rounding because it is unbiased:
```
E[Q_SR(x)] = x
```
Unbiasedness is necessary, but not sufficient. If the variance of `Q(x) - x` is too high, the optimizer still sees excess noise, especially on weight-gradient paths.

### MS-EDEN
MS-EDEN is designed for the microscaled setting where a block of FP4 elements shares a scale. Instead of sampling independently like ordinary stochastic rounding, it uses the block structure to produce an unbiased estimate with lower aggregate error. The practical interpretation for the course is:

- SR is a scalar unbiased estimator.
- MS-EDEN is a block-aware unbiased estimator.
- The block-aware estimator can spend the shared scale more efficiently, so less gradient variance is injected for the same nominal NVFP4 format.

### Quartet II linear-layer scheme
For a linear layer, the paper tracks quantization through:
1. Forward GEMM: `Y = X W^T`
2. Activation-gradient GEMM: `dX = dY W`
3. Weight-gradient GEMM: `dW = dY^T X`

The scheme is built so the same low-precision representation does not silently optimize one path while damaging another. This makes it a useful companion to [[nvfp4-training]], whose 2-D consistent scaling also tries to stop forward/backward quantization noise from disagreeing.

### Empirical scale
| Item | Reported setting |
|------|------------------|
| Format | NVFP4 |
| Main quantizer | MS-EDEN |
| Largest model | 1.9B parameters |
| Tokens | 38B |
| Hardware path | NVIDIA Blackwell kernels |
| Reported speedup | up to 4.2x over BF16 linear layers |

## Connections
- [[nvfp4-training]] — Quartet II is the strongest direct follow-up to the first public NVFP4 pretraining recipe.
- [[stochastic-rounding]] — the baseline unbiased estimator that MS-EDEN improves on.
- [[mxfp4-native-hardware-2026]] — complementary 2026 evidence that gradient-path details dominate FP4 training stability.
- [[deepseek-v3-fp8]] — FP8 training ancestor; Quartet II is part of the push from FP8 to FP4 native training.
- [[marlin-kernel]] / [[machete-kernel]] — inference kernels matter after PTQ; Quartet II shows the same algorithm-kernel coupling for training.

## Notes
This should replace the older "2026 frontier placeholder" in course planning as the concrete 2026 NVFP4 training paper. It is not yet a 10T-token, 10B+ public run; use [[nvfp4-training]] for that scale, and Quartet II for the newer quantizer mechanism.

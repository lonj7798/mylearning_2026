<!-- scope: Native-hardware MXFP4 LLM pretraining study on AMD Instinct MI355X; gradient-path failure analysis
     deps: [[mxfp4-pretraining]], [[mx-formats]], [[stochastic-rounding]]
     see-also: [[quartet-ii]], [[fp4-inference-diagnosis]], [[nvfp4-training]]
-->

# Pretraining Large Language Models with MXFP4 on Native FP4 Hardware
- **Core Insight:** In native MXFP4 training, weight-gradient quantization is the primary convergence failure point; deterministic Hadamard rotations stabilize training where stochastic rounding and randomized rotations do not.
- **Guideline:** When lowering training precision to MXFP4, test Fprop, Dgrad, and Wgrad separately; do not assume a forward-pass-stable FP4 recipe will remain stable once Wgrad is quantized.
- **Authors:** Musa Cim, Poovaiah Palangappa, Miro Hodak, Ravi Dwivedula, Meena Arunachalam, Mahmut Taylan Kandemir
- **Year:** 2026 (submitted 2026-05-11; rev. 2026-05-14)
- **URL:** https://arxiv.org/abs/2605.09825
- **Relevant topics:** MXFP4, native FP4 hardware, AMD Instinct MI355X, FP4 training stability, Wgrad quantization, deterministic Hadamard rotation

## Abstract
This paper revisits MXFP4 pretraining on hardware with native FP4 support rather than software emulation. The authors progressively enable MXFP4 in forward propagation, activation gradients, and weight gradients during Llama 3.1-8B pretraining on C4. The key result is diagnostic: forward and activation-gradient quantization add modest training cost, but quantizing weight gradients is the main source of convergence degradation. Contrary to earlier recipes, stochastic rounding and randomized Hadamard rotations are insufficient once Wgrad is quantized; deterministic Hadamard rotations restore stable optimization.

## Key Contributions
- Provides a controlled native-hardware study of MXFP4 training on AMD Instinct MI355X GPUs.
- Separates FP4 effects across **Fprop**, **Dgrad**, and **Wgrad**, instead of treating "FP4 training" as one knob.
- Identifies Wgrad quantization as the dominant failure mode in full-pipeline FP4 training.
- Reports that stochastic rounding and randomized Hadamard rotations fail to stabilize Wgrad-heavy settings, while deterministic Hadamard rotations do.
- Reframes FP4 training instability as structured microscaling error along sensitive gradient paths, not merely as a lack of stochasticity.

## Key Figures/Tables to Study
- Progressive-enablement table: Fprop only, Fprop + Dgrad, and full Fprop + Dgrad + Wgrad.
- Convergence curves for Wgrad-quantized runs: shows the instability that earlier summaries can miss.
- Intervention comparison: stochastic rounding vs randomized Hadamard vs deterministic Hadamard.
- Native MI355X performance table: useful for contrasting OCP MXFP4 adoption on non-NVIDIA hardware with NVFP4 on Blackwell.

## Technical Details

### Why this updates the 2025 MXFP4 story
The earlier [[mxfp4-pretraining]] page reports the AISTATS 2025 line where RHT + SR made MXFP4 viable at GPT-scale. This 2026 paper does not simply replicate that result; it stress-tests the exact training path on native FP4 hardware and finds that the hardest case is full Wgrad quantization.

### Three GEMM paths
| Path | Meaning | Paper's diagnostic finding |
|------|---------|----------------------------|
| Fprop | forward activation x weight GEMM | relatively stable under MXFP4 |
| Dgrad | activation-gradient GEMM | modest added token cost |
| Wgrad | weight-gradient GEMM | main convergence degradation driver |

### Deterministic Hadamard rotation
The paper's most important practical claim is that the failure is structured. If the error were mostly random scalar-rounding noise, stochastic rounding should help. Instead, deterministic Hadamard rotations help more, suggesting the microscale blocks are aligning poorly with sensitive gradient directions.

### Hardware context
This is valuable because it uses native FP4 support on AMD Instinct MI355X rather than treating FP4 as a simulated data type. It therefore belongs in the hardware/formats track as well as the training-method track.

## Connections
- [[mxfp4-pretraining]] — earlier MXFP4 training result; this paper is the 2026 native-hardware correction and diagnostic follow-up.
- [[mx-formats]] — format family being tested.
- [[quartet-ii]] — NVFP4 counterpart; both papers argue that gradient estimation, not just forward quantization, is the frontier issue.
- [[fp4-inference-diagnosis]] — same first author family; inference sensitivity analysis for MXFP4/NVFP4 complements this training diagnosis.
- [[nvfp4-training]] — contrast NVFP4's 16-element block and FP8 scale with MXFP4's 32-element block and E8M0 scale.

## Notes
Because this paper was revised on 2026-05-14, it should be treated as the current latest MXFP4 training resource in the raw library.

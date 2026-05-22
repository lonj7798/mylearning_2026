<!-- scope: Quantization-aware distillation for recovering NVFP4 inference accuracy on LLMs and VLMs
     deps: [[nvfp4]], [[nvfp4-training]], [[llm-qat]]
     see-also: [[quartet-ii]], [[gpt-oss-mxfp4]], [[tensorrt-llm-quant]]
-->

# Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery
- **Core Insight:** For post-trained LLMs and VLMs, distilling a BF16 teacher into an NVFP4 student with KL divergence can recover near-BF16 accuracy more reliably than ordinary PTQ or task-loss QAT.
- **Guideline:** Use QAD as a single post-training recovery stage for NVFP4 checkpoints when the original model went through SFT, RL, or model merging and recreating the full task-loss pipeline is expensive or unstable.
- **Authors:** Meng Xin, Sweta Priyadarshi, Jingyu Xin, Bilal Kartal, Aditya Vavre, Huizi Mao, Bryan Catanzaro, Song Han, Wei Ping, and NVIDIA collaborators
- **Year:** 2026 (submitted 2026-01-27; rev. 2026-03-03)
- **URL:** https://arxiv.org/abs/2601.20088
- **Relevant topics:** NVFP4, QAD, quantization-aware distillation, post-training recovery, Nemotron, VLM quantization

## Abstract
This technical report presents quantization-aware distillation for NVFP4 inference. The student model runs with NVFP4 quantization, while a frozen full-precision teacher supplies soft targets through a KL-divergence loss over logits. The paper focuses on practical post-trained models, including reasoning LLMs and VLMs, where ordinary QAT is hard to reproduce because model quality depends on multiple stages such as SFT, RL, and model merging. QAD is reported to recover NVFP4 checkpoints close to BF16 accuracy across Nemotron-family models and Llama Nemotron Super v1.

## Key Contributions
- Frames QAD as an accuracy-recovery method specifically for **NVFP4 inference**, not generic low-bit training.
- Avoids re-running the full post-training task-loss pipeline; the quantized student only has to match the BF16 teacher distribution.
- Reports stable recovery across LLMs and VLMs, including AceReason Nemotron, Nemotron 3 Nano, Nemotron Nano V2, Nemotron Nano V2 VL, and Llama Nemotron Super v1.
- Shows robustness to limited or imperfect recovery data, because the teacher distribution carries information beyond hard labels.
- Establishes a practical recipe for production NVFP4 model releases on Blackwell-class hardware.

## Key Figures/Tables to Study
- QAD vs PTQ vs QAT comparison: shows why distillation is positioned as the production recovery method.
- Benchmark table across Nemotron model variants: useful for identifying where NVFP4 needs recovery most.
- Data-coverage ablations: important for practitioners without access to the full training corpus.
- Logit KL loss diagram: the core training signal for the quantized student.

## Technical Details

### Objective
Let `z_T(x)` be teacher logits and `z_S(x)` be NVFP4-student logits. QAD minimizes a KL divergence between softened teacher and student distributions:
```
L_QAD = KL(softmax(z_T / T) || softmax(z_S / T))
```
The key difference from ordinary QAT is that the target is the teacher distribution, not the original SFT/RL task label.

### Why QAD is useful after RL/SFT/model merge
Modern post-training pipelines often combine supervised fine-tuning, preference optimization or RL, distillation, safety data, and model merging. Replaying that stack in quantization-aware mode is expensive and can be unstable. QAD reduces the recovery problem to matching a frozen reference model after quantization is inserted.

### NVFP4 scope
The report targets NVFP4 inference recovery. In NVIDIA's Nemotron description, Nemotron 3 Nano NVFP4 quantizes both weights and activations, unlike weight-only MXFP4 releases such as GPT-OSS. That makes QAD especially relevant to W4A4 deployment.

### Practical recipe
1. Start from a BF16 or high-precision post-trained teacher.
2. Insert NVFP4 quantization into the student.
3. Train on recovery data with KL-to-teacher logits.
4. Validate against BF16 on the same downstream suite, not just calibration loss.
5. Export to the serving stack that supports NVFP4 kernels.

## Connections
- [[nvfp4]] — target format.
- [[nvfp4-training]] and [[quartet-ii]] — training-time NVFP4; QAD is inference recovery after a high-precision model already exists.
- [[llm-qat]] — older data-free QAT baseline; QAD is the 2026 production-oriented distillation variant.
- [[gpt-oss-mxfp4]] — contrast: GPT-OSS is MXFP4 MoE-weight-only; Nemotron QAD is NVFP4 W/A deployment.
- [[tensorrt-llm-quant]] — likely deployment path for NVFP4 checkpoints.

## Notes
This source fills the missing "production NVFP4 reports" gap more concretely than a generic Blackwell placeholder because it reports a repeatable recovery method and named model releases.

<!-- scope: Microsoft Phi family quantization releases (Phi-3 / Phi-3.5 / Phi-4)
     deps: [[awq]], [[gptq]]
     see-also: [[gemma-quant]], [[bitnet-models]]
-->

# Microsoft Phi Quantization
- **Core Insight:** Phi's small-but-strong (3.8B / 14B) design philosophy makes it the natural target for aggressive quantization — Microsoft ships official ONNX INT4 / FP16 / GGUF variants per Phi version for laptop / edge / mobile deployment, and the same team that builds Phi also builds BitNet, so the Phi line is the conventional-quant counterpart to BitNet's native-1-bit line.
- **Guideline:** For Phi-3.5 / Phi-4, default to Microsoft's official ONNX INT4 builds on Windows / DirectML targets, GGUF builds on llama.cpp targets, AWQ on vLLM serving. Use BitNet only when the latency / energy budget actually demands sub-1-bit.
- **Authors:** Microsoft AI / Microsoft Research
- **Year:** 2024-2025 (Phi-3 2024-04, Phi-3.5 2024-08, Phi-4 2024-12, Phi-4-mini 2025-02)
- **URL:** https://huggingface.co/microsoft • https://onnxruntime.ai/blogs/phi3-int4-quantization
- **Relevant topics:** ONNX INT4, GGUF, AWQ, small-model edge deployment, DirectML

## Abstract
The Phi family (Phi-3 3.8B / Phi-3-medium 14B / Phi-3.5 3.8B / Phi-4 14B / Phi-4-mini 3.8B) is Microsoft's small-but-strong instruction-tuned line, designed for laptop / edge deployment. Microsoft ships its own quantized variants in multiple formats: **ONNX INT4** (for Windows / DirectML, the primary Microsoft-stack target), **GGUF** (for llama.cpp / Ollama), **ONNX FP16** (for higher-quality edge), and various community AWQ / GPTQ ports. The Phi line shares its quantization team with the Microsoft Research BitNet team, but Phi itself uses conventional PTQ (not BitLinear) — it's the dense baseline against which BitNet's 1.58-bit results are compared.

## Key Contributions
- **ONNX INT4 official builds**: per-size Phi variants compiled for ONNX Runtime with INT4 weight + INT8/FP16 activation. Primary target for Windows / DirectML deployment.
- **GGUF coverage**: q4_K_M / q5_K_M / q8_0 builds for llama.cpp / Ollama.
- **CPU / NPU optimization**: Phi-3-mini achieves real-time inference on Snapdragon X / Intel Lunar Lake NPUs via the ONNX INT4 build.
- **Phi-4 family** (14B + Phi-4-mini 3.8B + Phi-4-multimodal): full quant matrix per variant.
- **Quant-aware data mix**: Phi's "textbook-quality" data design produces a model that responds well to INT4 PTQ — Microsoft notes < 1 pt MMLU drop typical at AWQ-INT4.

## Key Figures/Tables to Study
- Phi-3 ONNX INT4 throughput table on Snapdragon X / Intel NPU / NVIDIA RTX 4060.
- Quality drop table: Phi-3.5 / Phi-4 at FP16 vs INT4 (AWQ) vs INT4 (ONNX) vs GGUF q4_K_M on MMLU / GSM8K / HumanEval.
- The Phi-vs-BitNet comparison table (in BitNet papers): Phi-3 at INT4 vs BitNet b1.58 at 1.58-bit, same parameter scale.

## Technical Details

### Phi version × quant format matrix
| Model | FP16 | ONNX-INT4 | GGUF | AWQ |
|-------|------|-----------|------|-----|
| Phi-3-mini (3.8B) | yes | yes | community | community |
| Phi-3-medium (14B) | yes | yes | community | community |
| Phi-3.5-mini (3.8B) | yes | yes | community | community |
| Phi-3.5-MoE (16x3.8B) | yes | partial | community | community |
| Phi-4 (14B) | yes | yes | community | community |
| Phi-4-mini (3.8B) | yes | yes | community | community |
| Phi-4-multimodal | yes | partial | partial | partial |

### ONNX INT4 recipe
- Weight: INT4 per-channel symmetric (no zero-point).
- Activation: INT8 / FP16 (depends on backend).
- Backend: ONNX Runtime with DirectML (Windows GPU/NPU) or CUDA-EP (NVIDIA) or CPU-EP.
- Calibration: PTQ on a Microsoft-internal instruct dataset.
- Pre-optimized for Snapdragon X NPU / Intel NPU / NVIDIA RTX consumer GPU.

### GGUF
- q4_K_M is the most common community choice.
- Compatible with llama.cpp / Ollama / LMStudio across CPU / GPU / Apple Silicon.

### Quality typical drops
- AWQ-INT4: < 1 pt MMLU drop vs FP16 for Phi-3 / Phi-3.5.
- ONNX-INT4: similar, slightly larger on coding evals.
- GGUF q4_K_M: ~ 1 pt drop.
- BitNet 1.58-bit (sibling line at similar scale): larger drop in absolute terms but at 4× the compression ratio.

### Phi vs BitNet
- Same Microsoft Research org; same authors (Furu Wei's group runs both).
- Phi = dense BF16 base + PTQ; BitNet = native 1.58-bit training from scratch.
- At 2B scale and 4T tokens, BitNet b1.58 2B-4T (54.19 avg eval) is in striking distance of Phi-3-mini's MMLU 69 → different evals; not directly comparable, but both are positioned as "smallest viable assistant".

### Serving
- Windows / Surface: ONNX Runtime + DirectML.
- llama.cpp / Ollama: GGUF.
- vLLM: AWQ community builds.
- ExecuTorch (mobile): partial Phi support.

## Connections
- [[awq]] / [[gptq]] — algorithms behind the community W4 builds.
- [[gguf-k-quants]] — k-quant ladder for the GGUF releases.
- [[bitnet-models]] — sibling Microsoft line (native 1.58-bit); Phi is the conventional-quant counterpart.
- [[microsoft-bitnet]] (lab page) — the Microsoft Research lab that runs both Phi and BitNet.
- [[llama-3-quantization]] / [[qwen-2.5-quant]] — counterpart model-report pages.
- [[gemma-quant]] — Google's QAT approach; Microsoft uses PTQ for Phi.

<!-- scope: Intel quantization line — Neural Compressor toolkit, BFloat16 + INT8 + FP8 ISA support, AMX instructions
     deps: [[bf16]], [[int8]], [[fp8-e4m3]]
     see-also: [[intel-amx]]
-->

# Intel Quantization — Neural Compressor + AMX + Datacenter CPU/GPU Inference
- **Core Insight:** Intel's quantization work packages standard PTQ/QAT methods around CPU, Gaudi, and cross-vendor format support, with AMX making INT8/BF16/FP8 matmul practical on Xeon.
- **Guideline:** Use this lab track for AMX, Intel Neural Compressor, IPEX, Gaudi FP8, and the cross-vendor FP8/MX standards story.
- **Authors:** Intel AI / Intel Labs / Neural Compressor contributors
- **Year:** 2022–2026
- **URL:** https://github.com/intel/neural-compressor ; https://github.com/intel/intel-extension-for-pytorch
- **Relevant topics:** Intel AMX, Neural Compressor, IPEX, Gaudi, FP8, INT8, BF16

## Summary
Intel's quantization work centers on **CPU-and-Gaudi-first production inference**: the **AMX (Advanced Matrix Extensions)** instructions on Sapphire Rapids / Granite Rapids Xeon CPUs add native INT8 / BF16 (and FP16 / FP8 on newer steppings) tensor-core-style matmul; the **Intel Neural Compressor** toolkit packages PTQ / QAT / SmoothQuant / GPTQ / AWQ as a single API; and the **Gaudi accelerators** (Habana / Intel AI) ship FP8 support on Gaudi 3 with a competing software stack. Intel was a co-author on both the 2022 FP8 spec ([[fp8-formats-paper]]) and the 2023 OCP MX format spec ([[microscaling-formats]]), giving the formats cross-vendor legitimacy.

## Notable Works
- AMX ISA ([[intel-amx]]) — Sapphire Rapids + Granite Rapids; native TMUL tile multiply for INT8 / BF16 / FP16 / FP8.
- Intel Neural Compressor — unified PTQ / QAT toolkit; integrates AWQ, GPTQ, SmoothQuant, RTN; supports CPU, GPU, Gaudi.
- LLM Runtime / IPEX (Intel Extension for PyTorch) — production CPU inference with AMX-accelerated quant kernels.
- [[fp8-formats-paper]] (Micikevicius 2022) — Intel co-author on the joint FP8 spec.
- [[microscaling-formats]] (Rouhani 2023) — Intel co-author on the OCP MX spec.
- Gaudi 3 FP8 reference — Habana Synapse software stack.

## Recurring themes
- **CPU is a first-class quantization target**: Intel pushes the narrative that INT8 / FP8 on Xeon AMX is a real LLM-inference path for cost-sensitive deployment.
- **Format cross-vendor co-authorship**: rather than inventing proprietary formats, Intel anchors the joint NVIDIA / Arm / Intel FP8 spec and the OCP MX consortium, prioritizing interoperability.
- **Toolkit packaging**: Neural Compressor's value is not a new algorithm but a single API across PTQ methods, eliminating glue code.

## Open Resources
- Intel Neural Compressor: https://github.com/intel/neural-compressor
- Intel Extension for PyTorch: https://github.com/intel/intel-extension-for-pytorch
- AMX documentation: https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html
- Habana Synapse (Gaudi): https://docs.habana.ai/

## Connections
- [[nvidia-quantization]] — competitor and co-author on FP8 / MX format specs.
- [[qualcomm-ai-research]] — adjacent on edge-device PTQ; AIMET vs Neural Compressor.
- [[dettmers-group]] / [[han-song-mit]] / [[frantar-alistarh-ist-austria]] — PTQ algorithms that Neural Compressor packages.

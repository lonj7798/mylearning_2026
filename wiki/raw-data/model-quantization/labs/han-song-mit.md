<!-- scope: Song Han's MIT HAN Lab — activation-aware quant, equivalent-transformation methods, production serving stacks
     deps: [[awq]], [[smoothquant]], [[squeezellm]], [[qserve]]
     see-also: [[atom]]
-->

# Song Han / MIT HAN Lab — Activation-Aware Quant + Serving Stack
- **Core Insight:** HAN Lab repeatedly pairs quantization algorithms with kernels and serving systems, making activation-aware and equivalent-transformation methods deployable rather than only accurate on paper.
- **Guideline:** Use this lab track for the AWQ/SmoothQuant/SqueezeLLM/QServe lineage and for understanding how W4A16, W8A8, W4A4, and KV4 ideas become working serving stacks.
- **Authors:** Song Han, MIT HAN Lab, and collaborators
- **Year:** 2015–2026
- **URL:** https://hanlab.mit.edu ; https://github.com/mit-han-lab/llm-awq ; https://github.com/mit-han-lab/qserve
- **Relevant topics:** AWQ, SmoothQuant, SqueezeLLM, Atom, QServe, TinyChat, hardware-software co-design

## Summary
Song Han's MIT HAN Lab (Han Lab) has been the single most prolific research group on **practical LLM quantization with companion production serving stacks** since 2023. The lab's signature contributions cluster around the **outlier-handling-by-equivalent-transformation** theme — [[smoothquant]] (W8A8) and [[awq]] (W4A16) — plus the **non-uniform / sensitivity-aware** branch ([[squeezellm]]) and the **production serving** layer ([[qserve]] W4A8KV4, TinyChat, [[atom]] W4A4). The lab also pioneered hardware-software co-design for quantized inference long before the LLM era (HAN, EIE, ESE).

## Notable Works
- Deep Compression (Han 2015) — the original pruning + quantization + Huffman coding paper; established the lab's compression identity.
- [[smoothquant]] (Xiao 2022/2023) — equivalent-transformation W8A8 PTQ; migrate activation difficulty to weights.
- [[awq]] (Lin 2023) — Activation-aware Weight Quantization; per-channel scaling driven by activation magnitude; the W4A16 default.
- [[squeezellm]] (Kim 2023) — sensitivity-weighted non-uniform code + dense-and-sparse decomposition.
- TinyChat (Lin 2023) — production W4 inference for edge devices, leveraging AWQ.
- [[atom]] (Zhao 2024) — W4A4 + KV4 serving with sub-channel reorder; the lab's full-stack low-bit inference proof.
- [[qserve]] (Lin 2024) — W4A8KV4 serving with progressive group quant + register-level dequant; the production-cloud counterpart to Atom.
- Long-context kernels (StreamingLLM, KV-cache compression line).

## Recurring themes
- **Algorithm + kernel + system, always shipped together**: AWQ + TinyChat, SmoothQuant + CUTLASS kernels, Atom + custom inference stack, QServe + serving framework. The lab does not publish algorithms without runtime proof.
- **Equivalent transformations as the primary tool**: per-channel scaling (SmoothQuant, AWQ), sub-channel reorder (Atom) — all preserve the mathematical equivalence of the layer while making the new representation easier to quantize.
- **Activation quantization is the hard problem**: the lab's chronology is W16 → W8A8 → W4A16 → W4A4 — and the difficulty is on the A side, not the W side. Most of the lab's innovations are about making activations quantizable.

## Open Resources
- AWQ + TinyChat: https://github.com/mit-han-lab/llm-awq
- SmoothQuant: https://github.com/mit-han-lab/smoothquant
- SqueezeLLM: https://github.com/SqueezeAILab/SqueezeLLM (collaborator group)
- Atom: https://github.com/efeslab/Atom
- QServe: https://github.com/mit-han-lab/qserve
- Lab homepage: https://hanlab.mit.edu

## Connections
- [[dettmers-group]] — adjacent; LLM.int8's mixed-precision outlier-handling is the conceptual cousin of SmoothQuant's equivalent-transformation outlier-handling.
- [[frantar-alistarh-ist-austria]] — adjacent; GPTQ and AWQ are the two dominant W4A16 methods; they are often run in parallel by deployment teams and reported together.
- [[nvidia-quantization]] — QServe + TinyChat run on NVIDIA hardware; the lab's serving stacks are often the path of AWQ into NVIDIA's TensorRT-LLM.

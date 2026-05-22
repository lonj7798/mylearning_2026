<!-- scope: Tim Dettmers (UW / now Allen AI / Carnegie Mellon) — outlier-discovery and 4-bit fine-tuning lineage
     deps: [[llm-int8]], [[qlora]], [[spqr]], [[nf4]]
     see-also: [[bitsandbytes-int8]], [[bitsandbytes-nf4]]
-->

# Dettmers Group — Outlier Discovery + 4-bit Fine-Tuning
- **Core Insight:** Dettmers' line made LLM quantization practical by identifying transformer outliers and turning distribution-aware low-bit formats into usable open-source tooling.
- **Guideline:** Read this lab track when studying outlier-aware INT8/NF4 methods and the bitsandbytes runtime path from paper idea to daily fine-tuning workflow.
- **Authors:** Tim Dettmers and collaborators
- **Year:** 2022–2026
- **URL:** https://timdettmers.com ; https://github.com/bitsandbytes-foundation/bitsandbytes
- **Relevant topics:** LLM.int8(), QLoRA, NF4, SpQR, bitsandbytes, outlier features

## Summary
Tim Dettmers (PhD at the University of Washington with Luke Zettlemoyer, postdoc at Allen AI, faculty at Carnegie Mellon as of 2025) leads arguably the single most influential research program on practical LLM quantization. The group's signature output is **discovering and exploiting the outlier-feature problem** in transformer activations — first cataloged in [[llm-int8]] — and the **4-bit fine-tuning revolution** ([[qlora]] / [[nf4]]) that made consumer-GPU fine-tuning of 65B-class models possible. The `bitsandbytes` library, maintained by Dettmers and a small open-source team, is the de facto runtime for both lines.

## Notable Works
- [[llm-int8]] (2022) — mixed-precision INT8 + FP16 outlier path; the first paper to make 6.7B+ LLM INT8 inference work without quality loss.
- [[qlora]] (2023) — NF4 base weights + LoRA adapters + paged optimizer; fine-tune 65B on a single 48GB GPU.
- [[nf4]] (2023, within QLoRA paper) — quantile-based 4-bit code for Gaussian-distributed weights; ~0.5 PPL win over INT4.
- [[spqr]] (2023) — sparse-quantized representation; outlier weights kept in FP16 alongside a quantized base.
- [[bitsandbytes-int8]] (framework) — the production CUDA kernels for LLM.int8().
- [[bitsandbytes-nf4]] (framework) — the production CUDA kernels for NF4 + QLoRA.

## Recurring themes
- **Outliers are not noise, they are signal**: the through-line from LLM.int8 → SpQR → AWQ (downstream) is "find the 0.1-1% extreme weights/activations and treat them differently."
- **Quantize for the actual distribution, not a uniform prior**: NF4 instantiates this for weights; SpQR's sparse-and-dense decomposition instantiates it for the heavy-tail.
- **Lower the bar for who can do LLM research**: every Dettmers release explicitly targets consumer-GPU users; QLoRA's stated motivation was "fine-tune Llama-65B on a single 48GB card".

## Open Resources
- `bitsandbytes`: https://github.com/bitsandbytes-foundation/bitsandbytes (formerly TimDettmers/bitsandbytes)
- Dettmers' blog: https://timdettmers.com (the LLM.int8 explanatory post and the GPU-purchasing guides)
- HF org: https://huggingface.co/timdettmers (research checkpoints + the Guanaco QLoRA-fine-tuned Llama models)

## Connections
- [[han-song-mit]] — adjacent lab; AWQ continues the outlier-handling theme from LLM.int8 with a different mechanism.
- [[frantar-alistarh-ist-austria]] — adjacent lab; GPTQ is the Hessian-based PTQ counterpart to Dettmers' distribution-aware PTQ.
- [[microsoft-bitnet]] — orthogonal direction; BitNet pushes sub-2-bit via training-from-scratch where Dettmers pushes 4-bit via PTQ + LoRA.

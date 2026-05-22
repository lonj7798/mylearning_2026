<!-- scope: OpenAI GPT-OSS model release as a production MXFP4 MoE-weight quantization case study
     deps: [[mx-formats]], [[mxfp4-pretraining]]
     see-also: [[mxfp4-native-hardware-2026]], [[nvfp4-qad]], [[hf-quanto]]
-->

# GPT-OSS 120B/20B — Native MXFP4 MoE Weight Release
- **Core Insight:** GPT-OSS is a major open-weight production case where the released checkpoints are natively MXFP4 for MoE weights, making MXFP4 part of the model artifact rather than an after-the-fact community quant.
- **Guideline:** Treat GPT-OSS as a model-report case for MXFP4 deployment, but distinguish it from full W4A4 FP4 training: the model card states MoE weights are post-trained with MXFP4 quantization, not that every tensor and training path is FP4.
- **Authors:** OpenAI
- **Year:** 2025
- **URL:** https://openai.com/index/introducing-gpt-oss/ ; https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf
- **Relevant topics:** GPT-OSS, MXFP4, MoE quantization, open-weight model release, weight-only quantization, deployment memory

## Summary
OpenAI released GPT-OSS-120B and GPT-OSS-20B as open-weight reasoning models under Apache 2.0. The quantization detail that matters for this course is in the model card: the MoE weights are quantized to MXFP4 at 4.25 bits per parameter, and those MoE weights account for more than 90% of total parameters. This is what lets the 120B checkpoint fit on a single 80GB GPU and the 20B checkpoint fit on systems with about 16GB memory.

## Key Points
- GPT-OSS-120B: 36 layers, 116.8B total parameters, about 5.1B active parameters per token, 128 experts, top-4 active experts, 128k context.
- GPT-OSS-20B: 24 layers, 20.9B total parameters, about 3.6B active parameters per token, 32 experts, top-4 active experts, 128k context.
- Checkpoint sizes in the model card: 60.8 GiB for 120B and 12.8 GiB for 20B.
- Quantization target: MoE weights, not every parameter category.
- Format: MXFP4, 4.25 bits per parameter, aligned with OCP microscaling formats.

## Key Figures/Tables to Study
- Model-card Table 1: parameter counts, active parameters, and checkpoint sizes.
- Model-card quantization section: short but critical distinction that MoE weights are quantized to MXFP4.
- OpenAI release availability section: states the weights are available and natively MXFP4.
- Hugging Face Transformers MXFP4 docs: runtime path and kernel expectations for GPT-OSS.

## Technical Details

### Why MoE weight quantization is enough to matter
The model card reports that MoE weights account for more than 90% of parameters. For a sparse MoE, quantizing the expert MLP weights yields most of the storage win while leaving smaller components, such as attention and embedding/unembedding, outside the primary compression target.

### Memory consequence
| Model | Total params | Active params/token | Checkpoint size | Release quantization |
|-------|--------------|---------------------|-----------------|----------------------|
| GPT-OSS-120B | 116.8B | 5.13B | 60.8 GiB | MoE weights MXFP4 |
| GPT-OSS-20B | 20.91B | 3.61B | 12.8 GiB | MoE weights MXFP4 |

### Important caveat
This is a model-release case study, not proof that the model was fully pretrained in MXFP4. For training-method claims, use [[mxfp4-pretraining]] and [[mxfp4-native-hardware-2026]]. GPT-OSS is best used to teach how low-bit formats enter public model distribution and runtime support.

## Connections
- [[mx-formats]] — GPT-OSS is the most visible model-release example of MXFP4.
- [[mxfp4-pretraining]] / [[mxfp4-native-hardware-2026]] — training-method sources; use them to avoid overclaiming from the model card.
- [[nvfp4-qad]] — contrast with NVFP4 W/A recovery on Nemotron.
- [[gguf-k-quants]] — community release formats; GPT-OSS shows a vendor-native format release instead.
- [[vllm-quant]] / [[hf-quanto]] — runtime integrations need explicit MXFP4 support.

## Notes
The main course warning: many summaries online blur "released in MXFP4" with "trained entirely in MXFP4." The model card supports the narrower, more precise claim: MoE weights were post-trained with MXFP4 quantization.

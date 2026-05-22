<!-- scope: HuggingFace quantization fundamentals blog series
     deps: [[int8]], [[gptq]]
     see-also: [[hf-fp8-deep-dive]], [[bitsandbytes-int8]]
-->

# HuggingFace — Quantization Fundamentals Blog Series
- **Core Insight:** HuggingFace's quantization blogs are the canonical hands-on onboarding for transformers quantization, walking through bitsandbytes INT8, NF4 / QLoRA, GPTQ, and AWQ with runnable code snippets and visual explanations of the underlying number-format mathematics.
- **Guideline:** Read the "Making LLMs lighter with AutoGPTQ and transformers" + "Overview of natively supported quantization schemes in HF Transformers" posts first when starting on HF-quantization integration.
- **Authors:** Younes Belkada, Marc Sun, Pedro Cuenca, Quentin Lhoest, Sourab Mangrulkar (HuggingFace)
- **Year:** 2022-2024 (series)
- **URL:** https://huggingface.co/blog/hf-bitsandbytes-integration ; https://huggingface.co/blog/4bit-transformers-bitsandbytes ; https://huggingface.co/blog/gptq-integration ; https://huggingface.co/blog/overview-quantization-transformers
- **Relevant topics:** bitsandbytes integration, NF4, QLoRA, GPTQ, AWQ, transformers quant API

## Summary
HuggingFace has published a sequence of blog posts walking practitioners through quantization in the Transformers library, covering INT8 (bitsandbytes integration), 4-bit NF4 (QLoRA), GPTQ integration, AWQ integration, and the unified `QuantizationConfig` API. Each post pairs minimal runnable code with conceptual explanations of the underlying technique — usually with one good figure visualizing the quantization-step structure. The series is maintained continuously: it's the standard reference for what the current `bitsandbytes` / `auto-gptq` / `auto-awq` / `quanto` / `torchao` integrations actually do under the hood, which arguments matter, and what perplexity/throughput to expect on standard benchmarks.

## Key Points
- Sequence covers: bitsandbytes INT8 → NF4/QLoRA → GPTQ → AWQ → unified API.
- Every post has runnable HF Transformers code (5-10 lines per example).
- Visual aids: the "intuitive view of float8" diagram has been re-used by many followups.
- Posts are maintained: kwargs are updated when API changes happen.
- The "overview" post is the definitive table of what each method does.

## Technical Details

### Posts in the series (chronological)
| Date | Post | Topic |
|------|------|-------|
| 2022-08 | Making LLMs accessible with bitsandbytes 8-bit | LLM.int8 integration |
| 2023-05 | Making LLMs even more accessible with bitsandbytes 4-bit & QLoRA | NF4 + QLoRA workflow |
| 2023-08 | Making LLMs lighter with AutoGPTQ and transformers | GPTQ |
| 2023-11 | Overview of natively supported quantization schemes | unified API |
| 2024-02 | A gentle introduction to 8-bit matrix multiplication | conceptual |
| 2024-Q3 | Quanto: a PyTorch quantization toolkit | Quanto release |

### Code pattern (HF unified API)
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### What each post adds beyond the original paper
- **bitsandbytes INT8 post**: explains outlier extraction visually; benchmarks vs FP16.
- **NF4 / QLoRA post**: walks through paged optimizer + double quantization; PEFT integration.
- **GPTQ post**: shows the `quantize()` call cost, the Marlin backend speedup.
- **AWQ post**: contrasts AWQ with GPTQ; explains why salient-channel protection works.
- **Overview post**: side-by-side comparison table; "which method should I use" decision tree.

### Decision tree (paraphrased from the overview post)
1. Want minimal accuracy loss + easy setup → bitsandbytes INT8.
2. Want best memory savings for fine-tuning → NF4 + QLoRA.
3. Want fastest inference for already-trained model → GPTQ or AWQ with Marlin kernels.
4. Want full PyTorch-native cross-device → Quanto or torchao.
5. Want server-grade FP8 → use the HF FP8 deep-dive post and TE-based recipes.

## Connections
- [[bitsandbytes-int8]] / [[bitsandbytes-nf4]] — integrations covered in the early posts.
- [[autogptq]] / [[autoawq]] — integrations covered in later posts.
- [[hf-quanto]] — Quanto release post.
- [[hf-fp8-deep-dive]] — sibling post focused on FP8.
- [[qlora]] — paper underlying the NF4 post.

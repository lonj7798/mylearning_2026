<!-- scope: Maxime Labonne's practical gguf and quantization guides
     deps: [[gguf-k-quants]], [[gptq]], [[awq]]
     see-also: [[llama-cpp-ggml]], [[hf-quantization-fundamentals]]
-->

# Maxime Labonne — Practical Quantization & gguf Guides
- **Core Insight:** Labonne's posts on the LLM Course / Towards Data Science / HuggingFace are the practical "I want to ship a quantized model tonight" reference — they walk through GGUF k-quants, AutoGPTQ, AutoAWQ, ExLlamaV2, and bitsandbytes with copy-paste commands and benchmark tables.
- **Guideline:** When you need to pick a quant format under time pressure, read Labonne's "Quantize Llama models with GGUF and llama.cpp" + "4-bit LLM Quantization with GPTQ" first; they answer "which knob, which value" without the algorithm bibliography.
- **Authors:** Maxime Labonne
- **Year:** 2023-2024
- **URL:** https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html ; https://huggingface.co/blog/mlabonne/sft-llama3 ; https://towardsdatascience.com/4-bit-llm-quantization-with-gptq-36b0f4f61a5b
- **Relevant topics:** GGUF, llama.cpp, GPTQ, AWQ, ExLlamaV2, quant variant selection

## Summary
Maxime Labonne maintains the LLM Course on GitHub and a regular blog of practical LLM tutorials. His quantization posts are read-then-execute guides for the most common formats: GGUF (with llama.cpp), GPTQ (with AutoGPTQ + ExLlamaV2), AWQ (with AutoAWQ), and bitsandbytes. Each post walks through the exact commands for converting a HF checkpoint, picking a variant (`q4_k_m`, `q5_k_m`, etc.), running calibration, and uploading the quantized weights to the HF Hub. Labonne's distinctive contribution is the "which variant should I pick" decision rule — a perplexity-vs-size table comparing q3_k_s through q8_0 for the model family in question, so practitioners can make informed accuracy-vs-memory tradeoffs.

## Key Points
- Step-by-step GGUF conversion: HF → safetensors → fp16 gguf → quantized gguf.
- Practical perplexity-vs-bpw table for choosing among q3_k_s, q4_k_m, q5_k_m, q6_k, q8_0.
- AutoGPTQ + ExLlamaV2 workflow for fast W4A16 inference.
- AutoAWQ workflow with the GEMM backend.
- HuggingFace Hub upload conventions for community quants.

## Technical Details

### GGUF conversion pipeline (from "Quantize Llama 2 models using GGUF")
```bash
# 1. Clone llama.cpp + build
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && make

# 2. Convert HF model to fp16 gguf
python convert_hf_to_gguf.py ../llama-2-7b-hf --outfile llama-2-7b-fp16.gguf

# 3. Quantize to k-quant variant
./quantize llama-2-7b-fp16.gguf llama-2-7b-q4_k_m.gguf q4_k_m

# 4. Run inference
./main -m llama-2-7b-q4_k_m.gguf -p "Hello"
```

### Variant decision table (Llama-2-7B, Labonne's reproduction)
| Variant | Size (GB) | PPL (wikitext) | Notes |
|---------|-----------|----------------|-------|
| FP16 baseline | 13.5 | 5.79 | reference |
| q8_0 | 7.16 | 5.79 | safe baseline |
| q6_k | 5.53 | 5.81 | near-lossless |
| q5_k_m | 4.78 | 5.82 | accuracy-leaning |
| q4_k_m | 4.08 | 5.86 | **recommended default** |
| q4_k_s | 3.86 | 5.92 | smaller |
| q3_k_m | 3.30 | 6.18 | edge-device |
| q3_k_s | 2.95 | 6.50 | tight memory |
| q2_k | 2.83 | 7.92 | only if desperate |

### GPTQ workflow (from "4-bit LLM Quantization with GPTQ")
```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4, group_size=128, damp_percent=0.01, desc_act=True
)
model = AutoGPTQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", quantize_config)
model.quantize(calibration_data)
model.save_quantized("llama-2-7b-gptq-4bit")
```

### AWQ workflow
```python
from awq import AutoAWQForCausalLM
quant_config = {"w_bit": 4, "q_group_size": 128, "version": "GEMM"}
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("llama-2-7b-awq")
```

### Hub upload conventions
- `mlabonne/Llama-2-7b-GGUF` — one repo, multiple variants as separate `.gguf` files.
- `mlabonne/Llama-2-7b-GPTQ` — one repo, one variant, `model.safetensors`.
- Model card includes ppl table + quantization recipe.

### When to pick which format
- **GGUF + q4_k_m**: CPU / Apple Silicon / consumer GPU inference.
- **GPTQ-Marlin**: server CUDA inference, batch > 1.
- **AWQ-GEMM**: server CUDA inference with stronger accuracy guarantee.
- **bitsandbytes NF4**: QLoRA fine-tuning on consumer GPU.
- **FP8**: production server with H100/MI300X.

## Connections
- [[gguf-k-quants]] — format Labonne primarily covers.
- [[llama-cpp-ggml]] — engine for GGUF inference.
- [[autogptq]] / [[autoawq]] — libraries Labonne's posts wrap.
- [[hf-quantization-fundamentals]] — HF-official adjacent series.
- [[sebastian-raschka-quant]] — sibling practitioner blog focused on training.

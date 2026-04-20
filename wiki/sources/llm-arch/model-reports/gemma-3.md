<!-- scope: Gemma 3 technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[mistral-7b]], [[jamba]]
-->

# Gemma 3 Technical Report
- **Core Insight:** 5:1 local/global attention interleaving captures most of global attention's benefit at a fraction of cost.
- **Guideline:** Not every layer needs full attention; interleave local and global strategically.

- **Organization:** Google DeepMind
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2503.19786
- **Relevant chapters:** Local vs global attention, KV cache optimization, knowledge distillation, vision-language models, quantization-aware training

## Abstract
We introduce Gemma 3, a multimodal addition to the Gemma family of lightweight open models, ranging in scale from 1 to 27 billion parameters. Gemma 3 introduces vision understanding abilities, wider language coverage, and longer context of at least 128K tokens. The architecture was changed to reduce KV-cache memory by increasing the ratio of local to global attention layers and keeping the span on local attention short. The models are trained with distillation and achieve superior performance to Gemma 2, with the novel post-training recipe significantly improving math, chat, instruction-following, and multilingual abilities, making Gemma3-4B-IT competitive with Gemma2-27B-IT and Gemma3-27B-IT comparable to Gemini-1.5-Pro across benchmarks.

## Architecture Summary

| Component | 1B | 4B | 12B | 27B |
|-----------|-----|-----|------|------|
| Vision Encoder Params | 0 | 417M | 417M | 417M |
| Embedding Params | 302M | 675M | 1,012M | 1,416M |
| Non-Embedding Params | 698M | 3,209M | 10,759M | 25,600M |
| Total Parameters | 1B | 4B | 12B | 27B |

- **Vocabulary size:** 262,144 (256K entries, SentencePiece with split digits and byte-level encodings)
- **Context length:** 128K tokens (1B model: 32K)
- **Attention:** Grouped-Query Attention (GQA) with QK-norm (replacing soft-capping)
- **Local/global attention ratio:** 5 local layers per 1 global layer, starting with local
- **Local attention window:** 1,024 tokens (sliding window)
- **RoPE frequency:** 1M for global layers, 10K for local layers
- **Vision encoder:** SigLIP 400M variant at 896x896 resolution, 256 vision tokens per image (via average pooling)

## Key Architectural Innovations

1. **5:1 local-to-global attention ratio** — dramatically reduces KV cache by using 5 local attention layers (1,024-token sliding window) for every 1 global attention layer (full context). This reduces KV cache overhead from ~60% to less than 15% of model memory, enabling efficient long-context processing.
2. **Pan-and-Scan for vision** — an adaptive windowing algorithm for handling non-square images and high-resolution inputs at inference time. Analyzes image aspect ratio and content to extract optimal crops, improving visual understanding without increasing token count.
3. **Knowledge distillation from larger models** — all Gemma 3 models are trained with distillation, sampling 256 logits per token weighted by teacher probabilities. This enables smaller models to punch above their weight.
4. **QK-norm replacing soft-capping** — uses query-key normalization instead of the logit soft-capping from Gemma 2, improving training stability and simplifying the attention mechanism.
5. **Quantization-Aware Training (QAT)** — fine-tunes each model for ~5,000 steps with quantization simulation, supporting per-channel int4, per-block int4, and switched FP8. The 27B model in int4 uses only 14.1GB (vs. 54GB in bf16).

## Design Decisions and Tradeoffs

- **Local vs global attention ratio (5:1):** Aggressively favors local attention to minimize KV cache. The tradeoff is that most layers cannot attend to the full context — only 1 in 6 layers sees the entire sequence. But empirical results show this is sufficient for strong long-context performance.
- **Different RoPE frequencies:** Global layers use 1M frequency (for long-range dependency), local layers use 10K (optimized for short-range within the 1,024-token window). This dual-frequency approach is novel.
- **Fixed vision encoder (SigLIP 400M):** Same encoder across all model sizes (4B-27B), keeping the vision component lightweight. Tradeoff: the vision encoder may become a bottleneck for larger models, but it keeps the architecture simple and the total parameter count manageable.
- **256 vision tokens per image:** Aggressive compression via average pooling from the SigLIP encoder's output. Fewer tokens = faster processing, but may lose fine-grained visual details.
- **Distillation throughout:** Training with distillation from larger teacher models at all sizes means smaller models are more capable than self-supervised alternatives, but their performance is bounded by the teacher.

## Training Details

| Model | Training Tokens | Infrastructure |
|-------|----------------|----------------|
| 27B | 14T | 6,144 TPUv5p chips |
| 12B | 12T | 6,144 TPUv4 chips |
| 4B | 4T | 2,048 TPUv5e chips |
| 1B | 2T | 512 TPUv5e chips |

**Post-training:**
- SFT + RLHF using improved versions of BOND, WARM, and WARP algorithms
- Reward functions: weight-averaged reward models, code execution feedback, ground-truth math rewards
- Data filtering: remove personal information, unsafe outputs, duplicates; prioritize attribution and hedging

## Performance Highlights

| Benchmark | 4B-IT | 12B-IT | 27B-IT |
|-----------|-------|--------|--------|
| MMLU-Pro | — | — | 67.5% |
| MATH | — | — | 89.0% |
| MMMU (vision) | — | — | 64.9% |
| Global MMLU-Lite | — | — | 75.1% |

- **Chatbot Arena Elo:** Gemma-3-27B-IT scores 1338, ranking 9th overall, ahead of DeepSeek-V3 (1318) and Llama 3.1-405B (1269)
- Gemma3-4B-IT is competitive with Gemma2-27B-IT (a model 7x its size)
- Gemma3-27B-IT is comparable to Gemini-1.5-Pro across benchmarks
- KV cache memory reduced to less than 15% of model memory with the 5:1 local-to-global ratio

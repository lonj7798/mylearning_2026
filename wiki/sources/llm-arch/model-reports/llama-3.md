<!-- scope: Llama 3 technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[llama-1]], [[llama-2]], [[llama-4]]
-->

# The Llama 3 Herd of Models — Technical Report
- **Core Insight:** Scaling-law-driven design with 15T tokens for 405B params proves Chinchilla was right.
- **Guideline:** Invest in training data pipeline proportional to parameter count.

- **Organization:** Meta AI
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2407.21783
- **Relevant chapters:** Scaling laws, long context, vocabulary design, multimodal integration, post-training

## Abstract
Modern artificial intelligence (AI) systems are powered by foundation models. This paper presents a new set of foundation models, called Llama 3. It is a herd of language models that natively support multilinguality, coding, reasoning, and tool usage. Our largest model is a dense Transformer with 405B parameters and a context window of up to 128K tokens. The paper reports that Llama 3 achieves performance comparable to GPT-4 across numerous tasks.

## Architecture Summary

| Component | 8B | 70B | 405B |
|-----------|-----|------|--------|
| Layers | 32 | 80 | 126 |
| Model Dimension | 4,096 | 8,192 | 16,384 |
| FFN Dimension | 14,336 | 28,672 | 53,248 |
| Attention Heads | 32 | 64 | 128 |
| KV Heads (GQA) | 8 | 8 | 8 |
| Peak Learning Rate | 3e-4 | 1.5e-4 | 8e-5 |

- **Vocabulary size:** 128,256 tokens (128K — 4x larger than Llama 2's 32K)
- **Context length:** 128K tokens (32x larger than Llama 2's 4K)
- **Activation function:** SwiGLU
- **Positional encoding:** RoPE with theta=500,000
- **Normalization:** RMSNorm (pre-normalization)
- **Architecture type:** Dense Transformer (no MoE)

## Key Architectural Innovations

1. **4x larger vocabulary (128K tokens)** — expanded from 32K to 128K using a BPE tokenizer trained on multilingual data. Improves encoding efficiency for non-English languages and reduces token count per input.
2. **GQA across all model sizes** — unlike Llama 2 which only used GQA for 34B+, Llama 3 uses 8 KV heads universally, standardizing the architecture and improving inference efficiency at all scales.
3. **128K context window** — massive 32x extension from Llama 2's 4K, enabled by RoPE with theta=500,000 and progressive context extension during training (8K -> 128K).
4. **Dense architecture at 405B scale** — deliberately chose a dense transformer over MoE, arguing that dense models are simpler to train, scale, and serve. This is a notable contrast to models like Mixtral and DeepSeek-V2.
5. **Scaling law-driven design** — used extensive scaling experiments on smaller models to predict optimal architecture and hyperparameters for the 405B model before committing compute.
6. **Compositional multimodal integration** — vision (via image encoder), speech, and tool-use capabilities added post-hoc through adapter-based approaches, keeping the language backbone frozen or lightly fine-tuned.

## Design Decisions and Tradeoffs

- **Dense over MoE:** Chose dense architecture for training stability and simplicity, accepting higher inference cost per token. MoE would have been more inference-efficient but harder to train stably at this scale.
- **Massive vocabulary:** 128K vocabulary improves multilingual efficiency but increases embedding table size. Tradeoff: better compression ratio at the cost of larger embedding parameters.
- **Scaling laws for architecture selection:** Invested significant compute in scaling law experiments (predicting 405B performance from smaller runs), reducing risk of suboptimal hyperparameter choices at massive scale.
- **Progressive context extension:** Trained at 8K initially, then extended to 128K in later stages. This avoids the cost of training on long sequences from the start while still achieving strong long-context performance.
- **Data quality over quantity:** Extensive data filtering and curation pipeline, with significant investment in deduplication, quality filtering, and safety filtering.

## Training Details

- **Pretraining data:** ~15 trillion multilingual tokens
- **Data composition:** ~50% general knowledge, ~25% math/reasoning, ~17% code, ~8% multilingual
- **Training infrastructure:** 16,384 NVIDIA H100 80GB GPUs
- **Training time:** ~54 days for the 405B model (estimated)
- **Optimizer:** AdamW with cosine learning rate schedule
- **Context extension:** Trained at 8K context, then extended to 128K using a 6-stage progressive schedule with increasing RoPE theta

**Post-training pipeline:**
1. Supervised Fine-Tuning (SFT) on high-quality instruction data
2. Direct Preference Optimization (DPO) — replaced PPO from Llama 2
3. Rejection sampling for iterative quality improvement
4. Tool-use training for code execution, web search, and mathematical tools

## Performance Highlights

| Benchmark | 8B | 70B | 405B |
|-----------|-----|------|--------|
| MMLU (5-shot) | 66.6% | 79.3% | 87.3% |
| HumanEval (0-shot) | 62.2% | 80.5% | 89.0% |
| GSM8K (8-shot) | 79.6% | 95.1% | 96.8% |
| ARC Challenge (0-shot) | 79.7% | 93.0% | 96.9% |
| GPQA (0-shot, CoT) | — | — | 51.1% |
| IFEval | — | — | 88.6% |
| MGSM Multilingual (0-shot) | — | — | 91.6% |

- The 405B model performs on par with GPT-4 across a variety of tasks.
- Llama 3 8B matches or exceeds Llama 2 70B on many benchmarks despite being ~9x smaller.
- Strong multilingual capabilities across 8+ languages.

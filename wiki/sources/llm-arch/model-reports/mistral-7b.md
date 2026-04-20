<!-- scope: Mistral 7B technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[mixtral]], [[gemma-3]], [[llama-2]]
-->

# Mistral 7B — Technical Report
- **Core Insight:** Sliding window attention with rolling buffer KV cache enables fixed-memory inference regardless of sequence length.
- **Guideline:** For deployment-constrained models, design attention for bounded memory, not maximum quality.

- **Organization:** Mistral AI
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2310.06825
- **Relevant chapters:** Sliding Window Attention, Grouped-Query Attention, efficient inference, rolling buffer KV cache

## Abstract
We introduce Mistral 7B v0.1, a 7-billion-parameter language model engineered for superior performance and efficiency. Mistral 7B outperforms Llama 2 13B across all evaluated benchmarks, and Llama 1 34B in reasoning, mathematics, and code generation. Our model leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference cost. We also provide a model fine-tuned to follow instructions, Mistral 7B -- Instruct, that surpasses the Llama 2 13B -- Chat model both on human and automated benchmarks.

## Architecture Summary

| Component | Value |
|-----------|-------|
| Parameters | 7.3B |
| Layers | 32 |
| Model Dimension (Hidden) | 4,096 |
| FFN Dimension (Intermediate) | 14,336 |
| Attention Heads | 32 |
| KV Heads (GQA) | 8 |
| Head Dimension | 128 |
| Context Length | 8,192 tokens |
| Vocabulary Size | 32,000 |
| Sliding Window Size | 4,096 tokens |

- **Activation function:** SwiGLU
- **Positional encoding:** RoPE
- **Normalization:** RMSNorm (pre-normalization)

## Key Architectural Innovations

1. **Sliding Window Attention (SWA)** — each attention layer attends to a window of 4,096 tokens from the previous layer. Through recursive stacking across 32 layers, the theoretical effective attention span is approximately W x L = 4,096 x 32 = 131,072 tokens, far exceeding the context length. This provides long-range information flow while keeping per-layer attention cost linear in window size.
2. **Rolling Buffer KV Cache** — uses a fixed-size circular buffer of size W (4,096) for the KV cache, where position i is stored at index i mod W. This caps KV cache memory at a constant regardless of sequence length, achieving 8x reduction in cache memory for 32K-token sequences without quality impact.
3. **Grouped-Query Attention (GQA)** — uses 8 KV heads shared across 32 query heads (4 queries per KV head). Reduces KV cache by 4x and accelerates inference by reducing memory bandwidth requirements during attention computation.
4. **Pre-fill and chunking** — optimizes prompt processing by chunking the input prompt into segments of window size W, allowing efficient prefill with known positions. This separates the prefill phase from generation for better hardware utilization.

## Design Decisions and Tradeoffs

- **SWA window size 4,096:** Chosen to match the hidden dimension for implementation convenience and to provide a good balance between local attention cost and information propagation through layers. Smaller windows would be more efficient but limit per-layer receptive field.
- **GQA (8 KV heads) over MHA (32 KV heads):** 4x KV cache reduction with minimal quality loss. The 4:1 query-to-KV ratio was found to be a sweet spot for 7B-scale models.
- **Fixed sliding window over sparse attention:** SWA is simpler to implement and provides predictable memory usage, unlike sparse attention patterns (e.g., BigBird) which add implementation complexity.
- **No training details released:** Mistral AI deliberately withheld training data composition, optimizer settings, and compute details. This limits reproducibility but protects competitive advantage.
- **Byte-fallback BPE tokenizer:** Uses SentencePiece BPE with byte-fallback for handling unknown characters, matching LLaMA's approach.

## Training Details

Training details were deliberately not disclosed by Mistral AI. The paper focuses on architecture and evaluation rather than training methodology.

- **Tokenizer:** SentencePiece BPE, 32K vocabulary
- **Training data:** Not disclosed
- **Optimizer:** Not disclosed
- **Compute:** Not disclosed
- **License:** Apache 2.0

## Performance Highlights

| Benchmark | Mistral 7B | Llama 2 13B | Llama 1 34B |
|-----------|-----------|-------------|-------------|
| MMLU | 60.1% | 55.6% | 57.8% |
| GSM8K (math) | 52.2% | 34.3% | — |
| HumanEval (code) | 30.5% | 18.9% | — |
| HellaSwag | 81.3% | 80.7% | 83.7% |
| ARC Challenge | 55.5% | 49.4% | 50.4% |

- Outperforms Llama 2 13B across all evaluated benchmarks despite having ~2x fewer parameters.
- Outperforms Llama 1 34B in reasoning, mathematics, and code generation (models ~5x larger).
- Achieves ~3x "equivalent size compression" on reasoning and STEM tasks vs. Llama 2.
- Mistral 7B-Instruct surpasses Llama 2 13B-Chat on both human and automated benchmarks.

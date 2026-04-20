<!-- scope: LLaMA 1 technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[llama-2]], [[llama-3]], [[llama-4]]
-->

# LLaMA: Open and Efficient Foundation Language Models — Technical Report
- **Core Insight:** Public data + existing techniques (RoPE, SwiGLU, pre-norm) properly combined beats proprietary models.
- **Guideline:** Architecture innovation matters less than training data quality and scale when using proven components.

- **Organization:** Meta AI (FAIR)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2302.13971
- **Relevant chapters:** Foundation architectures, training efficiency, scaling laws, open-source LLMs

## Abstract
We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, Chinchilla-70B and PaLM-540B.

## Architecture Summary

| Parameter | 7B | 13B | 33B | 65B |
|-----------|-----|------|------|------|
| Hidden Dimension | 4,096 | 5,120 | 6,656 | 8,192 |
| Attention Heads | 32 | 40 | 52 | 64 |
| Layers | 32 | 40 | 60 | 80 |
| Learning Rate | 3.0e-4 | 3.0e-4 | 1.5e-4 | 1.5e-4 |
| Batch Size (tokens) | 4M | 4M | 4M | 4M |
| Training Tokens | 1.0T | 1.0T | 1.4T | 1.4T |

- **Context length:** 2,048 tokens
- **Vocabulary size:** 32,000 (BPE via SentencePiece)
- **Activation function:** SwiGLU (replaces ReLU in FFN)
- **Positional encoding:** Rotary Position Embeddings (RoPE)
- **Normalization:** RMSNorm (pre-normalization, applied before each sub-layer)

## Key Architectural Innovations

1. **Pre-normalization with RMSNorm** — applies normalization before each transformer sub-layer (instead of after), improving training stability. RMSNorm removes mean centering for 5-15% computational savings per normalization layer.
2. **SwiGLU activation function** — replaces the standard ReLU in the feed-forward network. Uses three linear projections (gate, up, down) but with a reduced intermediate dimension (2/3 of 4d) to maintain the same parameter count. Improves expressiveness.
3. **Rotary Position Embeddings (RoPE)** — replaces absolute positional embeddings. Encodes position through rotation of query and key vectors, naturally producing relative position sensitivity in attention scores with no additional parameters.
4. **Efficient causal multi-head attention** — uses an efficient implementation that does not store attention weights and does not compute masked key/query scores, reducing memory and compute.
5. **Activation checkpointing** — reduces memory usage by recomputing activations during the backward pass rather than storing them.
6. **Training exclusively on public data** — demonstrates that proprietary data is not necessary to achieve state-of-the-art performance, a deliberate design choice for openness.

## Design Decisions and Tradeoffs

- **Public data only:** Chose to train exclusively on publicly available datasets to maximize reproducibility and openness, even though proprietary data could have improved performance further. This was a philosophical stance that influenced the entire Llama lineage.
- **Scaling tokens over parameters:** Followed Chinchilla scaling laws — trained smaller models on more tokens rather than training larger models on fewer tokens. The 7B model was trained on 1T tokens (far beyond the Chinchilla-optimal ~200B for a 7B model), showing that smaller models can match or exceed larger ones when given sufficient data.
- **SwiGLU over ReLU/GELU:** SwiGLU provides better performance at the cost of an additional projection matrix, but the intermediate dimension is reduced from 4d to (2/3)·4d to keep FLOPs constant.
- **RoPE over absolute/learned embeddings:** Better extrapolation to longer sequences and no additional learned parameters, at the cost of slightly more complex attention computation.
- **No dropout:** Omitted dropout during training, relying on the large data volume for regularization.

## Training Details

**Data composition:**

| Dataset | Proportion | Epochs | Disk Size |
|---------|-----------|--------|-----------|
| CommonCrawl | 67.0% | 1.10 | 3.3 TB |
| C4 | 15.0% | 1.06 | 783 GB |
| GitHub | 4.5% | 0.64 | 328 GB |
| Wikipedia | 4.5% | 2.45 | 83 GB |
| Books | 4.5% | 2.23 | 85 GB |
| ArXiv | 2.5% | 1.06 | 92 GB |
| StackExchange | 2.0% | 1.03 | 78 GB |

- **Tokenizer:** BPE via SentencePiece, splitting all numbers into individual digits and using bytes for unknown UTF-8 characters
- **Optimizer:** AdamW (beta1=0.9, beta2=0.95, weight_decay=0.1)
- **LR schedule:** Cosine decay with 2,000 warmup steps
- **Gradient clipping:** 1.0
- **Compute (65B model):** ~1,022,362 A100-GPU-hours on 2,048 A100 80GB GPUs (~380 tokens/sec/GPU), ~21 days, ~449 MWh energy

## Performance Highlights

| Benchmark | LLaMA-7B | LLaMA-13B | LLaMA-65B |
|-----------|----------|-----------|-----------|
| MMLU (5-shot) | 35.1% | 46.9% | 63.4% |
| HumanEval (pass@1) | 10.5% | 15.8% | 23.7% |
| GSM8K | 11.0% | 17.8% | 50.9% |
| MATH | 2.9% | 3.9% | 10.6% |
| HellaSwag | 76.1% | 79.2% | 84.2% |
| NaturalQuestions (64-shot) | 16.1% | 27.1% | 39.9% |

- LLaMA-13B outperforms GPT-3 (175B) on most benchmarks despite being 10x smaller.
- LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B.
- Demonstrated that training smaller models on more tokens is more compute-efficient than training larger models on fewer tokens.

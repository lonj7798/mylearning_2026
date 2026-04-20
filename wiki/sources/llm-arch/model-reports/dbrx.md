<!-- scope: DBRX technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[mixtral]], [[deepseek-v2]], [[llama-4]]
-->

# DBRX: A New State-of-the-Art Open LLM — Technical Report
- **Core Insight:** Fine-grained MoE (16 experts, top-4) gives 65x more expert combinations than coarse MoE (8, top-2).
- **Guideline:** More smaller experts with higher top-k creates more expressive routing.

- **Organization:** Databricks (Mosaic ML)
- **Year:** 2024
- **URL:** https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm
- **Relevant chapters:** Fine-grained Mixture-of-Experts, MoE design choices, data curation, open-source LLMs

## Abstract
DBRX is a general-purpose large language model using a transformer-based decoder-only architecture with a fine-grained mixture-of-experts (MoE) design. It has 132B total parameters with 36B active on any input, and was pretrained on 12T tokens of carefully curated data. DBRX advances the state-of-the-art in efficiency among open models, providing inference up to 2x faster than Llama 2 70B, and surpasses GPT-3.5 on most benchmarks. DBRX is characterized by its use of a larger number of smaller experts — 16 experts choosing 4 — providing 65x more possible combinations of experts than previous MoE designs.

## Architecture Summary

| Component | Value |
|-----------|-------|
| Total Parameters | 132B |
| Active Parameters per Token | 36B |
| Layers | ~40* |
| Number of Experts | 16 per layer |
| Active Experts per Token | 4 |
| Expert Combinations | C(16,4) = 1,820 per layer |
| Context Length | 32K tokens |
| Vocabulary | GPT-4 tokenizer (tiktoken) |

*Exact layer count not disclosed in blog post.

- **Architecture:** Decoder-only Transformer with fine-grained MoE
- **Positional encoding:** Rotary Position Embeddings (RoPE)
- **Activation function:** Gated Linear Units (GLU)
- **Attention:** Grouped-Query Attention (GQA)

## Key Architectural Innovations

1. **Fine-grained MoE with 16 experts, top-4 routing** — uses more, smaller experts (16 choosing 4) compared to Mixtral's 8 choosing 2 or Grok-1's 8 choosing 2. This provides C(16,4)=1,820 possible expert combinations per layer, which is 65x more than C(8,2)=28 combinations. More combinations enable finer-grained specialization and better model quality.
2. **Larger active parameter ratio** — 36B active out of 132B total (27% activation ratio), higher than Mixtral's ~27% (12.9B/46.7B). The top-4 routing means each token uses a larger fraction of total capacity.
3. **Data-centric training philosophy** — Databricks invested heavily in data curation, estimating their data is "at least 2x better token-for-token" than the data used for the MPT model family. Used curriculum learning to adjust the data mix during training.
4. **GPT-4 tokenizer adoption** — uses tiktoken (GPT-4's tokenizer) for superior token efficiency, especially on code and structured text, rather than training a custom tokenizer.

## Design Decisions and Tradeoffs

- **16 experts, top-4 (vs. 8 experts, top-2):** More smaller experts provide exponentially more routing combinations (1,820 vs. 28), improving model quality. However, this increases routing overhead and requires more careful load balancing. Databricks found this tradeoff clearly favored more experts based on "exhaustive evaluation and scaling experiments."
- **132B total / 36B active:** Larger than Mixtral (46.7B/12.9B) but still ~40% the size of Grok-1 in both total and active parameters. Provides more capacity per token than Mixtral while maintaining fast inference.
- **GPT-4 tokenizer over custom:** Leverages a well-tested tokenizer for better efficiency rather than investing in tokenizer training. The tradeoff is dependency on an external tokenizer design.
- **Data quality focus:** Invested heavily in data filtering and curation rather than simply scaling token count. Used Apache Spark for processing and Lilac AI for data exploration and quality assessment.
- **12T tokens (not more):** Focused on data quality over quantity, with careful curation estimated to be 2x more efficient per token than prior approaches.
- **Curriculum learning:** Adjusted data mix during training, though specific schedule details were not disclosed.

## Training Details

- **Pre-training data:** 12 trillion tokens of text and code
- **Data quality:** Estimated "at least 2x better token-for-token" than MPT training data
- **Data management:** Unity Catalog (governance), Apache Spark (processing), Lilac AI (exploration)
- **Infrastructure:** 3,072 NVIDIA H100 GPUs with 3.2 Tbps InfiniBand connectivity
- **Development cycle:** ~3 months main training
- **Training libraries:** MegaBlocks (MoE), LLM Foundry, Composer, Streaming (all Databricks/Mosaic open-source tools)
- **Curriculum learning:** Data mix adjusted during training
- **License:** Open license

**Efficiency metrics:**
- MoE models require ~1.7x fewer FLOPs than dense alternatives for comparable quality
- End-to-end pipeline ~4x more compute-efficient than MPT-7B (May 2023)

## Performance Highlights

| Benchmark | DBRX | Mixtral | Grok-1 | GPT-3.5 |
|-----------|------|---------|--------|---------|
| MMLU | 73.7% | 70.6% | 73.0% | ~70% |
| HumanEval (code) | 70.1% | 54.8% | 63.2% | — |
| GSM8K (5-shot, math) | 66.9% | 61.1% | 62.9% | — |
| HF Open LLM Leaderboard | 74.5% | 72.7% | — | — |
| Databricks Gauntlet | 66.8% | 60.7% | — | — |

**Inference performance:**
- Up to 2x faster than Llama 2 70B
- Up to 150 tokens/sec/user on Databricks Model Serving (8-bit quantization)
- 2-3x higher throughput than equivalent 132B dense models

- Surpasses GPT-3.5 on most benchmarks.
- Competitive with Gemini 1.0 Pro.
- At time of release, highest MMLU score among open-source models.
- Particularly strong on code generation (70.1% HumanEval) and math (66.9% GSM8K).

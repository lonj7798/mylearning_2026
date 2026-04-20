<!-- scope: Mixtral technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[mistral-7b]], [[dbrx]], [[deepseek-v2]]
-->

# Mixtral of Experts — Technical Report
- **Core Insight:** Sparse MoE with 8 experts and top-2 routing gives 6x quality-per-FLOP improvement at inference.
- **Guideline:** If serving cost is your bottleneck, MoE lets you buy quality without proportional inference cost.

- **Organization:** Mistral AI
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2401.04088
- **Relevant chapters:** Sparse Mixture-of-Experts, expert routing, inference efficiency, MoE scaling

## Abstract
We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model. Mixtral has the same architecture as Mistral 7B, with the difference that each layer is composed of 8 feedforward blocks (i.e., experts). For every token, at each layer, a router network selects two of these experts to process the current state and combine their outputs. Mixtral outperforms or matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks, with 6x faster inference than Llama 2 70B.

## Architecture Summary

| Component | Value |
|-----------|-------|
| Total Parameters | 46.7B |
| Active Parameters per Token | 12.9B |
| Layers | 32 |
| Model Dimension (Hidden) | 4,096 |
| FFN Dimension (Intermediate) | 14,336 |
| Attention Heads | 32 |
| KV Heads (GQA) | 8 |
| Head Dimension | 128 |
| Number of Experts | 8 per layer |
| Active Experts per Token | 2 per layer |
| Context Length | 32,768 tokens |
| Vocabulary Size | 32,000 |

- **Base architecture:** Identical to Mistral 7B, but with each FFN block replaced by 8 expert FFN blocks
- **Activation function:** SwiGLU
- **Positional encoding:** RoPE
- **Normalization:** RMSNorm (pre-normalization)
- **Sliding Window Attention:** Inherited from Mistral 7B

## Key Architectural Innovations

1. **Sparse Mixture-of-Experts with top-2 routing** — each transformer layer has 8 expert feedforward blocks, and a learned router network selects 2 experts per token per layer. The outputs are combined additively, weighted by the router's gating values. This means only 12.9B of 46.7B parameters are active per token.
2. **Expert-level parallelism** — each expert is a full Mistral 7B FFN block (14,336 intermediate dim with SwiGLU), providing substantial per-expert capacity. This "coarse-grained" MoE design uses fewer, larger experts compared to fine-grained approaches (like DeepSeek-V2's 160 experts).
3. **Token-level dynamic routing** — different tokens within the same sequence can be routed to different expert pairs, allowing the model to specialize on a per-token basis. Each token at each layer accesses a different pair of experts.
4. **GQA integration with MoE** — combines Grouped-Query Attention (8 KV heads) with MoE FFN blocks, stacking two distinct efficiency techniques in one architecture.

## Design Decisions and Tradeoffs

- **8 experts with top-2 (not more, smaller experts):** Mistral chose fewer, larger experts with moderate top-K routing. This is simpler to implement and communicate but provides only C(8,2)=28 expert combinations per layer, far fewer than fine-grained MoE designs. The tradeoff favors simplicity and inference efficiency over routing diversity.
- **Active parameters ~13B, total ~47B:** The model uses only 27% of its parameters per token, providing a 3.6x efficiency ratio. This allows 47B total knowledge capacity at 13B inference cost.
- **32K context (not 128K+):** Conservative context length compared to later models, keeping training cost manageable while supporting most practical use cases.
- **Identical expert architecture:** All 8 experts share the same architecture (differ only in weights), simplifying implementation and load balancing. Some MoE designs use heterogeneous experts.
- **No auxiliary load balancing loss disclosed:** The paper does not detail its load balancing strategy, unlike DeepSeek-V2/V3 which describe explicit balancing mechanisms.

## Training Details

- **Training data:** Not disclosed in detail; the paper states it was "pretrained with multilingual data using a context size of 32k tokens"
- **Tokenizer:** BPE, 32K vocabulary (same as Mistral 7B)
- **Optimizer:** Not disclosed
- **Compute:** Not disclosed
- **License:** Apache 2.0

## Performance Highlights

| Benchmark | Mixtral 8x7B | Llama 2 70B | GPT-3.5 |
|-----------|-------------|-------------|---------|
| MMLU (5-shot) | 70.6% | 69.8% | ~70% |
| HellaSwag (10-shot) | 86.7% | 85.3% | — |
| ARC Challenge (25-shot) | 66.0% | 64.6% | — |
| GSM8K (math) | 58.4% | 56.8% | 57.1% |
| HumanEval (code) | 34.8% | 29.9% | — |
| MT-Bench | 8.3 | — | ~8.3 |

- Outperforms or matches Llama 2 70B on most benchmarks while using 5x fewer active parameters.
- 6x faster inference than Llama 2 70B.
- Particularly strong on mathematics, code generation, and multilingual tasks (outperforms Llama 2 70B on French, German, Spanish, and Italian).
- Mixtral 8x7B-Instruct surpasses GPT-3.5 Turbo, Claude-2.1, and Gemini Pro on human evaluations.

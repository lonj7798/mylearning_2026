<!-- scope: Qwen3 technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[deepseek-r1]], [[llama-4]]
-->

# Qwen3 Technical Report
- **Core Insight:** A single model can operate in thinking (slow/deliberate) and non-thinking (fast/direct) modes via training pipeline design.
- **Guideline:** Inference-time compute allocation is an architecture decision, not just a prompting strategy.

- **Organization:** Qwen Team, Alibaba Cloud
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2505.09388
- **Relevant chapters:** Thinking/non-thinking unified models, MoE scaling, multilingual expansion, GRPO for reasoning, distillation

## Abstract
Qwen3 is a series of large language models with both dense and Mixture-of-Expert (MoE) architectures, ranging from 0.6B to 235B parameters. A key innovation in Qwen3 is the integration of thinking mode (for complex, multi-step reasoning) and non-thinking mode (for rapid, context-driven responses) into a unified framework, eliminating the need to switch between different models. Compared to Qwen2.5, Qwen3 expands multilingual support from 29 to 119 languages and dialects. All Qwen3 models are publicly accessible under Apache 2.0.

## Architecture Summary

**Dense Models:**

| Model | Params | Layers | Q Heads / KV Heads | Context | Embedding Tie |
|-------|--------|--------|---------------------|---------|---------------|
| Qwen3-0.6B | 0.6B | 28 | 16 / 8 | 32K | Yes |
| Qwen3-1.7B | 1.7B | 28 | 16 / 8 | 32K | Yes |
| Qwen3-4B | 4B | 36 | 32 / 8 | 128K | Yes |
| Qwen3-8B | 8B | 36 | 32 / 8 | 128K | No |
| Qwen3-14B | 14B | 40 | 40 / 8 | 128K | No |
| Qwen3-32B | 32B | 64 | 64 / 8 | 128K | No |

**MoE Models:**

| Model | Total Params | Active Params | Layers | Q Heads / KV Heads | Experts (Total/Active) | Context |
|-------|-------------|---------------|--------|---------------------|------------------------|---------|
| Qwen3-30B-A3B | 30B | 3B | 48 | 32 / 4 | 128 / 8 | 128K |
| Qwen3-235B-A22B | 235B | 22B | 94 | 64 / 4 | 128 / 8 | 128K |

- **Vocabulary size:** 151,669 tokens (byte-level BPE)
- **Activation function:** SwiGLU
- **Positional encoding:** RoPE with RMSNorm pre-normalization
- **Attention:** Grouped-Query Attention (GQA) with QK-Norm
- **MoE routing:** Fine-grained expert segmentation with global-batch load balancing; no shared experts (unlike Qwen2.5-MoE)

## Key Architectural Innovations

1. **Unified thinking/non-thinking modes** — a single model supports both extended chain-of-thought reasoning (thinking mode) and direct responses (non-thinking mode), controlled at inference time. This eliminates the need for separate reasoning and chat models, unlike the DeepSeek-R1 approach which requires a separate model.
2. **Thinking budget mechanism** — users can set token limits for reasoning; if thinking exceeds the budget, the model gracefully transitions to generating a response with incomplete reasoning, enabling adaptive compute allocation.
3. **128 fine-grained MoE experts with top-8 routing** — both MoE models use 128 routed experts with 8 active per token, and no shared experts (a departure from Qwen2.5-MoE and DeepSeek-V2/V3 which use shared experts). Global-batch load balancing replaces per-sequence balancing.
4. **Three-stage pre-training** — General Stage (~30T tokens at 4K), Reasoning Stage (~5T high-quality tokens at 4K), Long Context Stage (hundreds of billions at 32K). This curriculum approach progressively refines capabilities.
5. **Four-stage post-training** — Long-CoT cold start, Reasoning RL (GRPO on 3,995 query-verifier pairs), Thinking mode fusion, General domain RL. This pipeline integrates reasoning capabilities into the unified model.
6. **Strong-to-weak distillation** — smaller models are trained via distillation from flagship models, requiring only 1/10 of the GPU hours of the full four-stage training pipeline while achieving superior pass@1 and pass@64 results.

## Design Decisions and Tradeoffs

- **Unified model over separate reasoning model:** Combining thinking and non-thinking modes in one model is more practical for deployment but requires careful training to prevent mode confusion. The four-stage post-training specifically addresses this integration challenge.
- **No shared experts in MoE:** Unlike DeepSeek-V2/V3 which use shared experts for common knowledge, Qwen3 relies entirely on routed experts. This simplifies the architecture but may require routed experts to redundantly learn common patterns.
- **128 experts (same count for 30B and 235B):** Both MoE models use 128 experts but differ in expert size. This standardization simplifies the routing architecture across scales.
- **Global-batch load balancing:** Balances expert utilization across the global batch rather than per-sequence, which provides better statistical balancing but may cause imbalance within individual sequences.
- **Expanding from 29 to 119 languages:** Massive multilingual expansion requires more diverse training data and may dilute performance on any single language, but dramatically improves global accessibility.
- **GRPO with only 3,995 query-verifier pairs:** Extremely data-efficient RL training, suggesting that high-quality verified queries are more important than quantity for reasoning RL.

## Training Details

- **Total pre-training tokens:** 36 trillion
- **Languages:** 119 languages and dialects (up from 29 in Qwen2.5)
- **Three-stage pre-training:**
  1. **General Stage:** ~30T tokens at 4,096 sequence length
  2. **Reasoning Stage:** ~5T high-quality tokens at 4,096 length
  3. **Long Context Stage:** Hundreds of billions of tokens at 32,768 length

**Data sources:** Coding, STEM, reasoning tasks, books, multilingual texts, and synthetic data

**Four-stage post-training:**
1. Long-CoT Cold Start — chain-of-thought reasoning pattern initialization
2. Reasoning RL — GRPO training on 3,995 query-verifier pairs
3. Thinking Mode Fusion — integration of thinking/non-thinking modes
4. General Domain RL — broad task performance improvement

**Distillation:** Smaller models distilled from flagship models at 1/10 the GPU cost

## Performance Highlights

**Qwen3-235B-A22B (Base):**

| Benchmark | Score |
|-----------|-------|
| AIME'24 | 85.7% |
| AIME'25 | 81.5% |
| LiveCodeBench v5 | 70.7% |
| Codeforces | 2,056 rating |
| MMLU | 87.8% |
| MMLU-Pro | 68.2% |
| BBH | 88.9% |
| GSM8K | 94.4% |
| MATH | 71.8% |

- Qwen3-235B-A22B achieves state-of-the-art results across reasoning, coding, and multilingual benchmarks.
- The unified thinking/non-thinking mode enables a single model deployment for all use cases.
- Qwen3-30B-A3B (3B active parameters) provides strong MoE efficiency for resource-constrained deployments.
- All models released under Apache 2.0.

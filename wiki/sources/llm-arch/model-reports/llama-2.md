<!-- scope: Llama 2 technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[llama-1]], [[llama-3]], [[llama-4]]
-->

# Llama 2: Open Foundation and Fine-Tuned Chat Models — Technical Report
- **Core Insight:** GQA enables scaling to 70B with reasonable serving cost; RLHF quality depends on data curation more than algorithm.
- **Guideline:** Choose attention variant based on target serving hardware, not just training quality.

- **Organization:** Meta AI
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2307.09288
- **Relevant chapters:** RLHF alignment, safety training, scaling context length, Grouped-Query Attention

## Abstract
In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs.

## Architecture Summary

| Parameter | 7B | 13B | 34B | 70B |
|-----------|-----|------|------|------|
| Hidden Dimension | 4,096 | 5,120 | 8,192 | 8,192 |
| Layers | 32 | 40 | 48 | 80 |
| Attention Heads | 32 | 40 | 64 | 64 |
| KV Heads (GQA) | 32 | 40 | 8 | 8 |
| Context Length | 4,096 | 4,096 | 4,096 | 4,096 |

- **Vocabulary size:** 32,000 (same BPE tokenizer as LLaMA 1)
- **Activation function:** SwiGLU
- **Positional encoding:** RoPE
- **Normalization:** RMSNorm (pre-normalization)
- **Grouped-Query Attention (GQA):** Applied to 34B and 70B models for inference efficiency

## Key Architectural Innovations

1. **Grouped-Query Attention (GQA)** — the 34B and 70B models use 8 KV heads shared across query heads, reducing KV cache memory and improving inference throughput without significant quality loss. The 7B and 13B models retain standard multi-head attention.
2. **Doubled context length** — extended from 2,048 (LLaMA 1) to 4,096 tokens.
3. **40% more training data** — trained on 2 trillion tokens vs. 1.0-1.4T for LLaMA 1.
4. **Ghost Attention (GAtt)** — a technique for multi-turn dialogue that concatenates system instructions into user messages across conversation turns, helping the model maintain instruction adherence over extended dialogues.
5. **RLHF alignment pipeline** — comprehensive alignment with Supervised Fine-Tuning (SFT), reward modeling, and iterative RLHF using both Rejection Sampling and Proximal Policy Optimization (PPO).

## Design Decisions and Tradeoffs

- **GQA only for large models:** Applied GQA to 34B and 70B variants where inference cost is critical, while keeping standard MHA for smaller models where the savings are less impactful.
- **Context extension to 4K (not longer):** Conservative extension from 2K to 4K; longer contexts would require more compute and data re-engineering. Llama 3 later pushed to 128K.
- **Two-reward-model approach:** Separate reward models for helpfulness and safety, allowing fine-grained control over the safety-helpfulness tradeoff during RLHF.
- **Rejection Sampling before PPO:** Used rejection sampling fine-tuning as an intermediate step to improve the policy before applying PPO, finding this two-step approach more stable.
- **Safety margin over performance:** Explicitly traded some helpfulness for safety, accepting lower scores on certain benchmarks in exchange for significantly reduced harmful outputs.

## Training Details

- **Pretraining data:** 2 trillion tokens from publicly available sources (no Meta user data)
- **Data mix:** Increased proportion of high-quality, curated data compared to LLaMA 1
- **Optimizer:** AdamW (beta1=0.9, beta2=0.95, weight_decay=0.1)
- **LR schedule:** Cosine decay with 2,000 warmup steps
- **Compute:** Trained on Meta's Research Super Cluster (RSC) and internal production clusters using NVIDIA A100 GPUs

**RLHF Pipeline:**
1. **Supervised Fine-Tuning (SFT):** Public instruction datasets + proprietary annotation data
2. **Reward Modeling:** Two separate reward models (helpfulness and safety) trained on ~1M human preference annotations
3. **Rejection Sampling:** Sample K outputs per prompt, select highest-reward output, retrain
4. **PPO:** Standard proximal policy optimization with KL penalty against the SFT model

## Performance Highlights

| Benchmark | Llama 2 7B | Llama 2 13B | Llama 2 70B |
|-----------|-----------|------------|------------|
| MMLU (5-shot) | 45.3% | 54.8% | 68.9% |
| GSM8K (8-shot) | 14.6% | 28.7% | 56.8% |
| HumanEval (0-shot) | 12.8% | 18.3% | 29.9% |
| HellaSwag (10-shot) | 77.2% | 80.7% | 85.3% |

- Llama 2 70B is competitive with GPT-3.5 on many benchmarks.
- Llama 2-Chat significantly outperforms all open-source chat models on helpfulness and safety human evaluations.
- Safety evaluations show substantial reduction in harmful outputs compared to other open-source models.

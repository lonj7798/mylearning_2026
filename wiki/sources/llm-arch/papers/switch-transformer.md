<!-- scope: Switch Transformer — top-1 MoE routing at scale
     deps: [[attention-is-all-you-need]]
     see-also: [[chinchilla]], [[ultra-scale-playbook]]
-->

# Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
- **Core Insight:** Top-1 expert routing works; you don't need top-k for effective Mixture-of-Experts, and simplicity wins at scale.
- **Guideline:** Start MoE with top-1 routing and a load-balancing auxiliary loss (alpha ~0.01); increase expert count for more capacity at constant FLOPs.
- **Authors:** William Fedus, Barret Zoph, Noam Shazeer
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2101.03961
- **Relevant chapters:** mixture of experts, sparse models, scaling, model architecture

## Abstract
In deep learning, models typically reuse the same parameters for all inputs. Mixture of Experts (MoE) defies this and instead selects different parameters for each incoming example. The result is a sparsely-activated model -- with outrageous numbers of parameters -- but a constant computational cost. However, despite several notable successes of MoE, widespread adoption has been hindered by complexity, communication costs and training instability -- we address these with the Switch Transformer. We simplify the MoE routing algorithm and design intuitive improved models with reduced communication and computational costs. Our proposed training techniques help wrangle the instabilities and we show large sparse models may be trained, for the first time, with lower precision (bfloat16) formats. We design models based off T5-Base and T5-Large to obtain up to 7x increases in pre-training speed with the same computational resources. These improvements extend into multilingual settings where we measure gains over the mT5-Base version across all 101 languages. Finally, we advance the current scale of language models by pre-training up to trillion parameter models on the "Colossal Clean Crawled Corpus" and achieve a 4x speedup over the T5-XXL model.

## Key Contributions
- Simplifies MoE routing to a single-expert selection ("switch" routing), reducing complexity and communication overhead compared to top-k expert selection
- Demonstrates stable training of large sparse models in bfloat16 precision for the first time, enabling practical deployment on modern hardware
- Achieves up to 7x pre-training speedup over dense T5 baselines with the same computational budget (FLOPs)
- Scales to trillion-parameter models while maintaining training stability through carefully designed techniques
- Shows consistent improvements across 101 languages in multilingual settings

## Architecture Details
- **Switch routing:** Each token is routed to exactly one expert (top-1), unlike earlier MoE work that used top-2 or top-k routing. This halves the communication cost compared to top-2 and simplifies the implementation
- **Expert capacity factor:** Each expert has a fixed buffer size = (tokens_per_batch / num_experts) * capacity_factor. Tokens that exceed an expert's capacity are passed through the residual connection (dropped from expert computation). Typical capacity factor is 1.0-1.5
- **Load balancing loss:** An auxiliary loss encourages uniform routing across experts: L_balance = alpha * N * sum(f_i * P_i), where f_i is the fraction of tokens routed to expert i and P_i is the average routing probability to expert i. Alpha is typically 0.01
- **Selective precision:** The router operates in float32 for stability while the expert computations use bfloat16 for speed. This mixed-precision approach was key to stable training
- **Architecture placement:** Switch layers replace the FFN sublayer in every other Transformer layer (or every layer). The attention sublayers remain dense (shared across all tokens)
- **Expert parallelism:** Experts are distributed across devices. Each device holds a subset of experts, and tokens are routed via all-to-all communication. The single-expert routing minimizes this communication
- **Scaling properties:** Increasing the number of experts (and thus parameters) improves quality at constant FLOPs, but with diminishing returns. The paper scales from 8 to 2048 experts
- **Distillation:** Large sparse models can be distilled into smaller dense models, retaining 30-40% of the quality gap between the dense baseline and the sparse model

## Tradeoffs Discussed
- **Complexity vs. simplicity:** Switch routing (top-1) is simpler than top-k but may underperform on tasks where a token benefits from multiple expert perspectives. The paper argues simplicity wins at scale
- **Training instability:** Sparse models are inherently less stable to train than dense models. The paper introduces selective precision and careful initialization, but instability remains a concern at very large scale
- **Expert load imbalance:** Despite the balancing loss, some experts may be overloaded (tokens dropped) while others are underutilized. The capacity factor is a crude mechanism to handle this
- **Communication overhead:** All-to-all routing between devices adds latency, partially offsetting the computational savings from sparsity. This is more pronounced with more experts across more devices
- **Fine-tuning challenges:** Sparse models can be harder to fine-tune than dense models of equivalent quality, and the paper notes that distilled dense models sometimes perform comparably in downstream tasks
- **Memory footprint:** Despite constant FLOPs, the total parameter count is enormous (up to 1.6T), requiring significant memory even though each token only activates a small fraction

<!-- scope: inference-time compute scaling, chain-of-thought, budget forcing, DeepSeek-R1, test-time scaling laws
     parent: [[ch-10]]
-->

# Inference-Time Scaling: The Second Compute Axis

## Training-Time vs. Inference-Time Compute

Traditional scaling laws describe a single axis: training-time compute. You spend $C = 6ND$ FLOPs to produce a model, then serve it with minimal per-query cost (~$2N$ FLOPs per generated token). Inference-time scaling adds a second axis: spending additional compute *per query* to improve the quality of each individual response.

The distinction matters because:
- **Training compute is paid once.** The cost is amortized over all queries.
- **Inference compute is paid per query.** The cost scales with query volume.
- **The two axes have different scaling exponents** and different ceiling effects.

## Mechanisms for Inference-Time Scaling

### Chain-of-Thought (CoT)
The simplest mechanism: generate intermediate reasoning steps before the final answer. Each additional token costs ~$2N$ FLOPs, so a 200-token chain of thought on a 70B model costs $2 \times 70 \times 10^9 \times 200 = 2.8 \times 10^{13}$ FLOPs -- roughly 100x the cost of a single-token generation.

The key insight from Weng ([[weng-why-we-think|blog]]): each generated token can be viewed as adding computation to the problem. From a latent variable perspective:

$$P(y|x) = \sum_z P(z|x) P(y|x,z)$$

where $z$ represents the reasoning trace. Marginalizing over longer, more diverse reasoning traces $z$ enables richer output distributions.

### Best-of-N Sampling
Generate $N$ independent answers, select the best using a reward model or verifier. The compute cost scales linearly with $N$. Effective when the model "knows" the answer but may not produce it on the first try -- increasing $N$ increases the probability of sampling a correct answer.

### Beam Search with Process Reward Models (PRMs)
Maintain $B$ candidate reasoning paths, scoring each intermediate step with a learned verifier. Prune low-scoring branches. This is more compute-efficient than best-of-N because it avoids generating full answers for dead-end reasoning paths.

Weng notes that DeepSeek categorized explicit inference-time scaling methods (MCTS, PRMs) as "unsuccessful attempts" for R1. The challenge: defining per-step correctness rubrics for reasoning is inherently ambiguous, and MCTS faces intractable search spaces for free-form text generation.

### Budget Forcing
Muennighoff & Yang et al. (2025), discussed by Weng, showed that deliberately lengthening reasoning by inserting "wait" tokens shows strong positive correlation between thinking tokens and accuracy. However, simple rejection sampling shows *reversed* scaling -- longer reasoning predicts worse performance. The difference: implicit optimization dynamics during generation (the model choosing to think longer) differ from explicit length constraints (forcing the model to generate more tokens).

## Scaling Laws for Test-Time Compute

Snell et al. (2024) established the foundational scaling relationships:

### 1. Non-Exchangeability
Training-time and inference-time compute are NOT 1:1 substitutes. A 14x smaller model with extensive test-time computation (best-of-N, beam search) roughly matches a base model with greedy decoding -- but only on problems where the smaller model has the *capability* to solve them. The exchange rate degrades rapidly on harder problems.

### 2. Difficulty Dependence
The effectiveness of inference-time compute depends strongly on problem difficulty:

| Difficulty | Benefit of test-time compute | Mechanism |
|------------|------------------------------|-----------|
| Easy | High | Multiple samples increase probability of sampling the correct (already-known) answer |
| Medium | Moderate | Chain-of-thought enables multi-step reasoning that the model can execute but doesn't by default |
| Hard | Low | The model lacks the fundamental capability; no amount of inference compute can compensate |

This has a direct implication for system design: inference-time scaling is most valuable when paired with a **difficulty classifier** that routes easy queries to greedy decoding and allocates more compute to medium-difficulty queries.

### 3. Sequential vs. Parallel Compute
Weng notes a key distinction:
- **Sequential compute** (longer reasoning chains): Benefits easier problems where the model needs to "work through" the steps
- **Parallel compute** (multiple independent attempts): Benefits harder problems where the model may find different valid solution paths
- **Optimal ratio varies by difficulty:** Easier questions favor purely sequential; harder questions favor a mix of sequential and parallel

## DeepSeek-R1: Reasoning from Reward Alone

The most striking result in inference-time scaling comes from DeepSeek-R1-Zero, as documented by Raschka ([[raschka-reasoning-llms|blog]]). Starting from the base DeepSeek-V3 (671B parameters), pure RL training with only accuracy rewards and format rewards (no supervised fine-tuning, no chain-of-thought demonstrations) produced a model that:

- Spontaneously generates reasoning traces
- Exhibits self-correction and backtracking ("aha moments")
- Adjusts reasoning length based on problem difficulty

This is inference-time scaling that *emerges from training*, not from explicit test-time mechanisms. The model learns to allocate variable compute per query by choosing how long to reason before answering.

### Scale-Dependent Strategy Selection

The viability of pure RL for inducing reasoning depends on model scale:
- **671B (DeepSeek-V3):** Pure RL works -- the model has sufficient capacity to discover reasoning strategies through exploration
- **32B (Qwen-32B):** Pure RL underperforms SFT-based approaches. The model lacks capacity for efficient reward-driven exploration
- **<10B:** Distillation from a larger model is most practical. SFT on reasoning traces generated by R1

This scale dependence suggests that inference-time scaling and training-time scaling interact: you need sufficient training-time investment to enable effective inference-time computation.

## Faithfulness Concerns

Weng raises an important caveat: chain-of-thought reasoning may not be *faithful* to the model's actual computation. Evidence includes:

1. **Early answering:** Models sometimes form conclusions before generating reasoning, using the chain-of-thought as post-hoc rationalization
2. **Filler tokens:** In some tasks, replacing reasoning with periods produces no accuracy change
3. **Optimization pressure risks:** When CoT monitors become RL reward signals, models learn to hide reward-hacking behavior within reasoning traces. "Directly optimizing CoT characteristics during RL training proves counterproductive; models adapt by concealing issues rather than resolving them."

These findings do not negate the value of inference-time scaling, but they complicate the interpretation. If reasoning traces are partially post-hoc rationalization, the mechanism by which additional inference compute improves accuracy may be different from what it appears.

## Practical Implications

1. **Inference-time compute is most valuable for medium-difficulty tasks.** Easy tasks do not need it; hard tasks cannot benefit from it. The sweet spot is problems where the model has the capability but needs more computation to reliably exercise it.

2. **Adaptive compute allocation matters more than total compute.** A system that routes queries to different compute tiers based on estimated difficulty will outperform one that applies uniform inference compute to all queries.

3. **Training and inference scaling are complementary, not substitutes.** Training-time scaling builds capability; inference-time scaling helps the model reliably express the capability it has. Neither axis alone is sufficient.

4. **The total cost of a model is training cost + (inference cost per query x query volume).** Inference-time scaling increases the per-query cost. The optimal strategy depends on expected query volume and the value of accuracy improvement.

## References

- [[weng-why-we-think|Weng, "Why We Think" (2025) (blog)]]
- [[raschka-reasoning-llms|Raschka, "Understanding Reasoning LLMs" (2025) (blog)]]

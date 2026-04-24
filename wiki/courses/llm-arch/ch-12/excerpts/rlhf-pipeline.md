# RLHF Pipeline Deep Dive

<!-- scope: detailed mechanics of the InstructGPT RLHF pipeline
     parent: [[ch-12]]
-->

## The Training Dynamics of RLHF

This excerpt expands on the RLHF pipeline described in Section 2 of the main chapter, focusing on the practical details that make RLHF difficult to implement correctly.

---

## Reward Model Architecture

The reward model is typically initialized from the SFT checkpoint. The key modification: remove the language modeling head (the unembedding matrix that maps hidden states to vocabulary logits) and replace it with a **scalar projection head** -- a single linear layer mapping the final hidden state to a scalar reward:

$$r_\phi(x, y) = W_r^\top \, h_L(x, y) + b_r$$

where $h_L(x, y)$ is the hidden state at the last token position of the concatenated (prompt, response) from the final transformer layer. The choice of the *last token* position is important: it sees the entire response through causal attention and can summarize the overall quality.

### Why Initialize from SFT?

The reward model must understand language, instructions, and response quality. Starting from a pre-trained or SFT checkpoint provides this understanding for free. Training a reward model from scratch would require vastly more preference data. The SFT checkpoint also shares the same tokenizer and vocabulary, ensuring compatibility.

### Reward Model Capacity

InstructGPT used a 6B parameter reward model to train a 175B policy. This asymmetry is intentional: the reward model only needs to *judge* quality (a simpler task than *generating* quality), so it can be smaller. However, the reward model must be large enough to capture the nuances of human preferences. Too small, and it produces noisy, exploitable reward signals.

---

## The PPO Training Loop

Each PPO iteration involves:

1. **Sample** a batch of prompts from the training distribution
2. **Generate** responses from the current policy $\pi_\theta$
3. **Score** each (prompt, response) with the frozen reward model $r_\phi$
4. **Compute KL penalty** between $\pi_\theta$ and $\pi^{\text{SFT}}$ for each response
5. **Estimate advantages** using GAE (Generalized Advantage Estimation) with the value function
6. **Update policy** using the clipped PPO objective
7. **Update value function** to reduce value estimation error

### The Four-Model Memory Problem

At peak, RLHF requires four models in GPU memory simultaneously:

| Model | Purpose | Size (typical) | Trainable? |
|-------|---------|---------------|------------|
| Policy $\pi_\theta$ | Generates responses, being optimized | Same as base | Yes |
| Reference $\pi^{\text{SFT}}$ | KL penalty baseline | Same as base | No (frozen) |
| Reward $r_\phi$ | Scores responses | ~6B | No (frozen) |
| Value/Critic $V_\psi$ | Estimates expected future reward | ~6B | Yes |

For a 70B parameter policy, this means roughly 70B + 70B + 6B + 6B = 152B parameters in memory (plus optimizer states for the trainable ones). This is why RLHF often requires 8-16 GPUs even for moderate-sized models.

### KL Penalty Mechanics

The per-token KL divergence is computed as:

$$D_{\text{KL}}^{(t)} = \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi^{\text{SFT}}(y_t | x, y_{<t})}$$

This is accumulated across the response. The reference policy $\pi^{\text{SFT}}$ is frozen -- it never updates. The KL penalty serves two purposes:

1. **Prevents reward hacking:** Without the KL constraint, the policy would find degenerate outputs that score high on the imperfect reward model but are low-quality to humans.
2. **Prevents catastrophic forgetting:** The KL penalty keeps the policy close to the SFT model, preserving the capabilities learned during pre-training.

The coefficient $\beta$ controls the tradeoff. Too low: reward hacking and mode collapse. Too high: the policy barely changes from SFT, wasting the RL signal. InstructGPT used an adaptive $\beta$ that targeted a specific KL budget.

---

## Reward Over-Optimization

As PPO training progresses, a characteristic pattern emerges:

1. **Early training:** Reward and human preference both increase. The policy learns genuine improvements.
2. **Mid training:** Reward continues to increase, but human preference plateaus or starts decreasing. The policy is finding reward model exploits.
3. **Late training:** Reward may still increase, but actual output quality degrades. Classic Goodhart's Law.

This happens because the reward model is a **finite-capacity approximation** of human preferences. Any optimizer powerful enough to be useful is powerful enough to find the boundaries of this approximation and exploit them. Common exploitation patterns:

- Repetitive, confident-sounding text that triggers high reward despite being uninformative
- Excessive hedging and caveats that the reward model interprets as "safe"
- Length gaming: longer responses tend to score higher on many reward models

Mitigations include: aggressive KL penalty, reward model ensembles, periodic reward model retraining, and early stopping based on held-out human evaluations.

---

## InstructGPT's Key Findings

From the paper's evaluation:

- **1.3B InstructGPT preferred over 175B GPT-3** by human labelers (win rate ~70%)
- **Alignment tax is small:** InstructGPT showed minimal regression on standard NLP benchmarks (MMLU, HellaSwag, etc.)
- **Generalization beyond labelers:** The model improved on tasks and prompts not seen during RLHF training
- **Toxicity reduced but not eliminated:** TruthfulQA and toxicity benchmarks improved substantially but imperfectly
- **Public NLP regression was correlated with KL penalty:** Higher $\beta$ (more KL constraint) preserved more benchmark performance but reduced the alignment improvement

These findings established that RLHF is not just a safety filter -- it fundamentally improves the model's utility for users, making it a necessary part of any production LLM pipeline.

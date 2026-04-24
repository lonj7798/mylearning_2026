# Continued Pre-training: Adaptation Without Forgetting

<!-- scope: catastrophic forgetting mechanisms, replay buffers, EWC, relationship to post-training
     parent: [[ch-11]]
-->

## Why Continued Pre-training

The cost of training a frontier model from scratch — tens of millions of dollars and months of GPU time — makes starting over prohibitive for most use cases. Continued pre-training (also called domain-adaptive pre-training or continual pre-training) allows adapting an existing model to new domains, languages, or temporal knowledge without full retraining.

Common use cases:

| Use Case | Distribution Shift | Forgetting Risk |
|----------|-------------------|-----------------|
| Domain adaptation (medical, legal, finance) | Moderate-high | High |
| Language adaptation (English model to Japanese) | Very high | Very high |
| Temporal update (new knowledge after cutoff) | Low-moderate | Low-moderate |
| Context length extension (4K to 128K) | Low (same domain, different structure) | Low |
| Code specialization | Moderate | Moderate |

The forgetting risk correlates directly with the distributional distance between the original training data and the new data. Context length extension has low forgetting risk because the content domain is unchanged — only the sequence structure differs. Language adaptation has very high risk because the token distribution shifts dramatically.

## The Mechanics of Catastrophic Forgetting

### Weight Space View

Neural network parameters occupy a high-dimensional weight space. Pre-training finds a region that performs well on the original data distribution $P_1$. Continued pre-training on $P_2$ moves the parameters toward a region that performs well on $P_2$. If $P_1$ and $P_2$ are far apart, the optimal regions do not overlap, and improving on $P_2$ necessarily degrades performance on $P_1$.

The severity is controlled by three factors:

1. **Learning rate magnitude.** Higher LR = larger steps in weight space = faster forgetting. This is why continued pre-training typically uses 10-50x lower LR than original pre-training.

2. **Number of continued training steps.** More steps = more movement away from the original optimum. Short continued pre-training (a few thousand steps) can adapt the model's surface-level behavior without deeply restructuring representations.

3. **Distributional overlap.** If $P_2$ contains significant content similar to $P_1$, the gradient updates for $P_2$ partially reinforce $P_1$ representations. Complete distribution mismatch causes the fastest forgetting.

### Layer-wise Forgetting Patterns

Not all layers forget equally. Empirical evidence from domain adaptation studies shows:

- **Early layers** (embedding, first few transformer blocks) change least during continued pre-training. They encode low-level linguistic features (syntax, morphology) that are shared across domains.
- **Middle layers** change moderately. They encode semantic representations that partially transfer across domains.
- **Late layers** (final few transformer blocks, language model head) change the most. They encode task-specific and domain-specific patterns.

This suggests a natural mitigation: freeze early layers during continued pre-training. In practice, full fine-tuning with low LR often outperforms selective layer freezing because the parameter interactions across layers are too complex for manual partitioning.

## Replay Buffer Design

The replay buffer approach mixes original training data into continued pre-training batches. Each batch contains a fraction $\alpha$ from $P_1$ (replay) and $(1 - \alpha)$ from $P_2$ (new data):

$$\text{batch} = \alpha \cdot \text{sample}(P_1) + (1 - \alpha) \cdot \text{sample}(P_2)$$

### Choosing the Replay Ratio

The optimal $\alpha$ depends on the forgetting risk:

- **$\alpha = 0$ (no replay):** Maximum adaptation speed, maximum forgetting. Only appropriate when you do not care about original-domain performance (e.g., building a domain-specific model that will never be used for general tasks).
- **$\alpha = 0.05\text{-}0.1$ (light replay):** Sufficient for low-distributional-shift scenarios (temporal updates, context extension). Preserves most original capabilities with minimal slowdown of adaptation.
- **$\alpha = 0.1\text{-}0.2$ (standard replay):** The industry default for domain adaptation. Good balance between adaptation speed and forgetting prevention.
- **$\alpha = 0.3\text{-}0.5$ (heavy replay):** Required for high-distributional-shift scenarios (language adaptation). Significantly slows adaptation but prevents catastrophic forgetting.

### What to Replay

Not all original training data is equally valuable for replay. Strategies:

- **Uniform sampling:** Sample randomly from the original training data. Simple but inefficient — much of the original data is low-value web text.
- **Importance-weighted sampling:** Upweight original data that is most representative of the capabilities you want to preserve. If you care about preserving math ability during domain adaptation, overrepresent math in the replay buffer.
- **Perplexity-based selection:** Replay documents where the continued-pretrained model's perplexity has increased most relative to the original checkpoint. This directly targets the most-forgotten content.

## Elastic Weight Consolidation (EWC)

EWC provides a more principled alternative to replay by explicitly protecting important parameters:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{P_2}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

where:
- $\theta^*$ is the original pre-trained checkpoint
- $F_i$ is the diagonal of the Fisher information matrix, estimating how important parameter $i$ is for $P_1$
- $\lambda$ controls the strength of the consolidation penalty

Parameters with high Fisher information (those that significantly affect $P_1$ performance) are penalized for deviating from their original values. Parameters with low Fisher information are free to adapt to $P_2$.

### Practical Limitations at LLM Scale

Computing the full Fisher information matrix for a model with billions of parameters is prohibitive. Approximations include:

- **Diagonal Fisher:** Only compute the diagonal entries (independent per-parameter importance). Cheap but ignores parameter interactions.
- **Empirical Fisher:** Estimate Fisher from a sample of training data gradients. $F_i \approx \frac{1}{N} \sum_{n=1}^{N} \left(\frac{\partial \mathcal{L}}{\partial \theta_i}\right)^2$. Requires only a forward-backward pass per sample.
- **Block-diagonal Fisher:** Compute per-layer Fisher matrices (small blocks), ignoring cross-layer interactions.

In practice, replay buffers are used far more often than EWC for LLM continued pre-training because:
1. Replay is simpler to implement (just mix data sources)
2. Replay does not require computing or storing the Fisher matrix
3. Replay works well enough for most distributional shifts
4. EWC's diagonal approximation misses critical parameter interactions in Transformers

## Connection to Two-Stage Pre-training

OLMo 2's two-stage approach ([[olmo-2|report]]) can be understood as a special case of continued pre-training where forgetting is a feature, not a bug:

- **Stage 1 → Stage 2 transition:** The distributional shift from OLMo-Mix to Dolmino-Mix is small (Dolmino is a higher-quality subset of similar content, not a different domain). Forgetting risk is minimal.
- **Low learning rate during Stage 2:** The annealing schedule naturally provides the "use lower LR for continued pre-training" safeguard without any special implementation.
- **No explicit replay needed:** Because Dolmino-Mix contains 50% high-quality filtered documents from the same domain as Stage 1, it implicitly replays similar content.

This is why two-stage pre-training is more robust than domain-adaptive continued pre-training: the distributional shift is designed to be small, and the learning rate schedule is designed to make conservative updates.

## Connection to Post-training ([[ch-12]])

Post-training (SFT, RLHF, DPO) is itself a form of continued pre-training with extreme distributional shift: the model transitions from predicting the next token on web text to generating helpful, harmless, and honest responses. The forgetting dynamics are identical — aggressive post-training can degrade the model's pre-trained knowledge and reasoning capabilities.

This is why post-training recipes are careful about:
- **Low learning rates** (typically 10-100x lower than pre-training peak)
- **Short training duration** (a few thousand steps, not millions)
- **KL penalties** in RLHF (an explicit regularizer that penalizes deviation from the pre-trained distribution, analogous to EWC)

The KL penalty in RLHF is particularly revealing: it is mathematically equivalent to a soft version of EWC where the Fisher information is approximated by the pre-trained model's own output distribution. This connection between catastrophic forgetting prevention and alignment training is underexplored but fundamental.

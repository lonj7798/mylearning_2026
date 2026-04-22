<!-- scope: dropout regularization; modern Transformer status; SFT use cases
     deps: []
     see-also: [[label-smoothing]], [[batch-vs-layer-norm]], [[neftune]]
-->

# Dropout: A Simple Way to Prevent Neural Networks from Overfitting
- **Core Insight:** Randomly zeroing a fraction of activations during training acts as an exponential ensemble of subnetworks, dramatically reducing co-adaptation between units.
- **Guideline:** Use **dropout = 0** for LLM pretraining (data is already large enough); use `0.05–0.1` for SFT on small datasets to combat overfitting; never apply dropout in residual streams of frontier-scale models.
- **Authors:** Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov
- **Year:** 2014 (JMLR)
- **URL:** https://jmlr.org/papers/v15/srivastava14a.html
- **Relevant topics:** regularization, generalization, SFT overfitting, NEFTune

## Abstract
Dropout is a regularization technique for neural networks where, during training, individual unit activations are temporarily set to zero with probability `1 - p` (typically `p = 0.5` for hidden units, `p = 0.8` for input units). At test time all units are present but activations are scaled by `p` (or equivalently, training-time activations are scaled by `1/p` — "inverted dropout", the standard implementation). Dropout approximately averages over an exponential ensemble of "thinned" subnetworks, achieves SOTA on multiple vision/speech/NLP benchmarks of its era, and is shown to outperform L2, max-norm, and other regularizers on sufficiently small datasets.

## Key Contributions
- The first systematic treatment of stochastic activation masking as a regularizer (foreshadowed by Hinton's 2012 tech report).
- Empirical demonstration of dramatic test-set gains on MNIST, CIFAR, ImageNet (early CNNs), TIMIT, Reuters — typically 1–3% accuracy.
- Theoretical interpretation as **geometric mean** of an exponential family of thinned networks.
- "Inverted dropout" implementation: scale by `1/p` at training time so test-time forward pass needs no rescaling.
- Variant analyses: input dropout, weight dropout (DropConnect), spatial/channel dropout.

## Key Figures/Tables to Study
- **Figure 7** (filter visualization with vs without dropout): with dropout, learned filters look like meaningful edges; without, they look noisy and co-adapted. Memorable visual.
- **Table 9** (MNIST results): the classic table that established dropout as the default in 2014.
- **Section 7** (effect of dropout rate `p`): a U-shaped curve with optimum near `p = 0.5` for hidden, higher for input.

## Technical Details

**The mechanism** (inverted dropout, training time):
```
mask = bernoulli(p)                      # 1 with prob p, 0 with prob 1-p
y = (x * mask) / p                       # scale up survivors so E[y] = E[x]
```
At inference: pass `x` through unchanged.

**Why it works** (3 interpretations):
1. **Ensemble**: each forward pass uses a random subnetwork; SGD trains all of them with shared weights. Test-time scaling approximates the geometric-mean prediction.
2. **Anti-co-adaptation**: a unit can't rely on any specific other unit being present; representations become more redundant.
3. **Adaptive noise injection**: equivalent to multiplicative Bernoulli noise; closely related to data augmentation.

**Original 2014 defaults** (image/speech CNNs):
- Input dropout: `p = 0.8` (i.e. drop 20%).
- Hidden dropout: `p = 0.5`.
- Max-norm constraint: combine with `||w|| < 3` for best results.

**Transformer-era usage** (Vaswani 2017): dropout = 0.1 applied to (a) sub-layer outputs, (b) attention weights post-softmax, (c) embedding-plus-positional-encoding sum. Standard for translation-scale Transformers.

**LLM-pretraining-era status (2020+)**:
- **GPT-3, Llama-1/2/3, Mistral, Qwen, DeepSeek**: dropout = 0 throughout. Reason: with trillions of training tokens, the model never sees the same example twice; there is nothing to overfit to. Dropout would be pure noise that slows convergence.
- **Some encoder pretrains** (BERT, T5): retain dropout = 0.1, but these are far below LLM scale.

**Where dropout still matters in 2025**:
- **SFT on small datasets** (1k–100k examples): dropout `0.05–0.1` reduces validation loss; common in domain-specific fine-tuning, LoRA training.
- **NEFTune** ([[neftune]]): not strictly dropout, but the same family — adding noise to embeddings during SFT improves generalization, citing dropout's lineage.
- **DPO / preference training**: occasional use of dropout `0.05` on small preference sets.
- **Specialized adapters / LoRA** on small data: dropout `0.05–0.1` on the adapter path.

**Common pitfalls**:
- Applying dropout to residual stream in a 70B pretrain → measurable loss-curve damage; never do it.
- Forgetting to disable dropout at eval (`model.eval()`) → 1–5% spurious validation perplexity hit; very common bug.
- Stacking dropout + weight decay + label smoothing on small SFT datasets → over-regularization; pick at most two.
- Different dropout rates between forward passes in DPO (since two policies are evaluated) → biased pref-loss; use deterministic eval mode for the reference model.

**Variants seen in modern code**:
- **Attention dropout**: `dropout(softmax(QK^T/sqrt(d)))`. Mostly removed in 2025 frontier models.
- **Hidden dropout** (post-MLP): present in many SFT recipes (HF Alignment Handbook keeps `hidden_dropout = 0.0` but exposes the knob).
- **DropPath / Stochastic Depth**: drops entire residual blocks; popular in ViT, rare in LLMs.

## Connections
- **[[label-smoothing]]**: complementary regularizers — both reduce overconfidence; label smoothing on outputs, dropout on intermediates.
- **[[neftune]]**: spiritual successor for the SFT regime; embedding-noise injection that fixes the "tiny SFT dataset" overfitting problem dropout used to handle.
- **[[batch-vs-layer-norm]]**: BatchNorm + dropout interact badly (variance-shift problem, Li 2019); LayerNorm + dropout is fine. Another reason LayerNorm dominates Transformers.
- **[[early-stopping-and-checkpointing]]**: with dropout = 0, early stopping is the only line of defense against SFT overfitting.
- **Karpathy** ([[karpathy-training-neural-net-recipe]]): "regularize: data augmentation, dropout, weight decay, early stopping — in that order of preference."

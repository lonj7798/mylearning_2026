<!-- scope: multi-head attention redundancy analysis, parent: [[ch-02]] -->

# Multi-Head Redundancy Analysis

This excerpt examines a surprising empirical finding: most attention heads in trained Transformers are redundant. We analyze what this means for architecture design and how the field has exploited it to build more efficient models.

---

## 1. The Original Design Intent

Vaswani et al. (2017, [[attention-is-all-you-need|paper]]) motivated multi-head attention as enabling the model to "jointly attend to information from different representation subspaces at different positions." The expectation was that each head would learn a distinct, specialized function.

The paper's Table 2 ablation on WMT EN-DE:

| Heads ($h$) | $d_k$ | BLEU |
|---|---|---|
| 1 | 512 | 25.8 |
| 4 | 128 | 26.3 |
| 8 | 64 | 25.8 |
| 16 | 32 | 25.7 |
| 32 | 16 | 24.7 |

The critical observation: going from 1 head to 4 heads improves BLEU by 0.5 points, but going from 4 to 8 provides no improvement, and 16 or 32 heads actually hurt. **Most of the benefit comes from having a few independent heads, not many.**

---

## 2. What Heads Actually Learn

Clark et al. (2019) and Voita et al. (2019) analyzed the attention patterns of trained Transformers and found that heads cluster into a small number of functional categories:

### Positional heads
Attend to specific relative positions. A "previous token" head puts most weight on position $t-1$; a "next token" head attends to $t+1$ (in encoder self-attention). These heads provide local context without learning complex content-based patterns.

```
Position t's attention weights:
  t-1: 0.85    (previous token dominates)
  t:   0.08    (self)
  t-2: 0.04
  other: ~0
```

### Syntactic heads
Track grammatical dependencies. In English, specific heads learn subject-verb agreement across intervening clauses. The attention weight between a verb and its subject remains high even with many tokens between them.

### Rare-word / high-information heads
Disproportionately attend to infrequent tokens that carry high semantic weight. These heads appear to implement a form of "importance weighting" -- common function words receive low attention, while content words receive high attention.

### Delimiter / structural heads
Attend to punctuation, sentence boundaries, and special tokens ([CLS], [SEP] in BERT; \<s\>, \</s\> in GPT). These heads provide structural scaffolding.

---

## 3. The Pruning Evidence

Voita et al. (2019) systematically pruned heads from a trained 6-layer, 8-head Transformer (48 heads total). Their method:

1. Add a differentiable gate $g_h \in [0, 1]$ to each head's output
2. Train the gates with an $L_0$ sparsity penalty while keeping model weights frozen
3. Observe which heads survive (gate remains near 1.0)

**Results on WMT EN-RU translation:**

| Heads remaining | % of total | BLEU drop |
|---|---|---|
| 48 (all) | 100% | -- |
| 38 | 79% | -0.1 |
| 25 | 52% | -0.3 |
| 15 | 31% | -0.6 |
| 10 | 21% | -1.0 |

**60% of heads can be removed with less than 0.3 BLEU degradation.** The vast majority of heads contribute negligibly to the final output.

### Which heads survive?

The heads that survive pruning are disproportionately:
- **Positional heads** (especially "attend to previous/next token")
- **Syntactic heads** tracking specific grammatical relations
- One or two **global attention** heads per layer that attend broadly

The heads that are pruned are typically:
- Redundant copies of positional heads (multiple heads learning near-identical patterns)
- Heads with near-uniform attention distributions (they attend everywhere, contributing a constant "average" that adds little)
- Heads whose value projections have small norms (they write negligible updates to the residual stream)

---

## 4. Why Redundancy Exists: Optimization vs. Expressiveness

The redundancy is not accidental. Multiple hypotheses explain it:

### Lottery ticket hypothesis (Frankle & Carlin, 2019)
Over-parameterization helps optimization. Having 8 heads when 3 would suffice means the model has more "lottery tickets" -- random initializations that happen to learn useful patterns. The redundant heads are the tickets that did not win but were along for the ride.

### Implicit ensembling
Redundant heads act as an ensemble within each layer. If heads 2 and 5 learn similar attention patterns, their combined output is more robust to noise than either alone. This improves training stability (one head can compensate if the other gets a bad gradient update) at the cost of parameter efficiency.

### Gradient diversity
During training, different heads receive slightly different gradients even when attending to the same positions, because their value projections differ. This gradient diversity prevents heads from collapsing to a single shared pattern during early training, which would be a local minimum. The redundancy emerges because not all of this initial diversity is needed for the final task.

---

## 5. Architectural Implications

The discovery of head redundancy directly motivated three major attention variants:

### Multi-Query Attention (MQA, Shazeer 2019)
If key and value projections are largely redundant across heads, why not share them? MQA uses a single key and value head for all query heads. This is the most aggressive exploitation of redundancy.

$$K = XW_K \text{ (shared)}, \quad V = XW_V \text{ (shared)}, \quad Q_h = XW_Q^h \text{ (per-head)}$$

The KV cache drops by $h\times$ (8x for 8 heads). Quality degrades modestly because query heads can still specialize -- they ask different "questions" -- but they all query the same "database" of keys and values.

### Grouped-Query Attention (GQA, Ainslie et al. 2023)
A compromise: group $h$ query heads into $G$ groups, each sharing one KV pair. With $h = 64$ and $G = 8$: every 8 query heads share one KV head. The insight is that heads within a group tend to be more redundant with each other than heads across groups -- neighboring heads in the parameter space learn more similar patterns.

### Multi-Head Latent Attention (MLA, DeepSeek-V2 2024)
Instead of sharing heads (which limits diversity), compress all KV information into a low-rank latent vector. This preserves head diversity at inference time (each head reconstructs its own KV from the shared latent via a per-head up-projection) while drastically reducing the cache.

---

## 6. The Training vs. Inference Asymmetry

A key insight: redundancy is **beneficial during training** (optimization robustness, gradient diversity) but **wasteful during inference** (unnecessary memory and bandwidth). The optimal architecture differs between the two phases:

| Phase | Optimal head config | Why |
|---|---|---|
| Training | Many heads (MHA) | Optimization stability, gradient diversity |
| Inference | Few KV heads (GQA/MLA) | Cache memory, bandwidth |

This is why the GQA uptraining recipe is so valuable: train with full MHA, then compress to GQA using only 5% additional compute. You get the optimization benefits of redundancy during training and the efficiency benefits of compression during inference.

**Guideline:** When designing a new architecture, choose the training-time head configuration for optimization quality, then plan a separate inference-time compression step (head pruning, GQA conversion, or MLA). These are two different design problems with different optimal solutions.

*Source: [[attention-is-all-you-need|paper]], Voita et al. (2019), Clark et al. (2019)*

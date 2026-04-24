<!-- scope: teacher forcing vs autoregressive sampling deep-dive, parent: [[ch-01]] -->

# Teacher Forcing vs. Autoregressive Sampling

This excerpt examines the train-test mismatch at the heart of every autoregressive LLM: during training, the model sees ground-truth history; during inference, it sees its own predictions. We trace the conventional understanding (exposure bias), the deeper failure mode identified by Bachmann & Nagarajan (2024), and the architectural implications.

---

## 1. Teacher Forcing: The Training Paradigm

In teacher forcing, the model receives the ground-truth prefix at every position during training:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

The key property: $x_1, \ldots, x_{t-1}$ are the **actual** tokens from the training data, not the model's own predictions. This enables massive parallelism -- all $T$ positions are computed simultaneously via causal masking, since the ground-truth prefix for every position is known in advance.

**The parallelism advantage in detail:** Consider a sequence of length 2048. With teacher forcing, the model performs one forward pass through the Transformer, computing all 2048 conditional distributions in parallel. Without teacher forcing (autoregressive training), the model would need 2048 sequential forward passes, each feeding its own previous output as input. The speedup is roughly $T\times$ -- the difference between training a model in days versus months.

```python
# Teacher forcing: one forward pass, all positions in parallel
logits = model(input_ids)  # shape: (batch, seq_len, vocab_size)
# input_ids[t] = ground truth token at position t
# logits[t] = prediction for position t+1, conditioned on ground truth x_{<t}

loss = cross_entropy(logits[:, :-1], input_ids[:, 1:])  # all positions at once
```

---

## 2. Autoregressive Sampling: The Inference Paradigm

At inference time, there is no ground truth. The model generates token by token, feeding each prediction back as input:

$$\hat{x}_t \sim P_\theta(\cdot \mid \hat{x}_1, \ldots, \hat{x}_{t-1})$$

where $\hat{x}_i$ denotes the model's own predictions. The model now conditions on a distribution it has never been explicitly trained on -- its own output distribution.

```python
# Autoregressive inference: sequential, each step depends on previous
generated = [prompt_tokens]
for _ in range(max_new_tokens):
    logits = model(generated)       # forward pass on generated prefix
    next_token = sample(logits[-1]) # sample from model's own distribution
    generated.append(next_token)    # feed back as input
```

---

## 3. Exposure Bias: The Conventional Understanding

The standard critique of teacher forcing is **exposure bias** (Ranzato et al., 2016): the model is never "exposed" to its own errors during training. When it makes a mistake at inference time, all subsequent predictions are conditioned on a prefix the model has never seen -- potentially pushing the distribution far from the training data.

**Error compounding:** If the model makes an error at position $t$, the resulting token $\hat{x}_t$ shifts the input distribution for position $t+1$. If that shift causes another error, the divergence grows. Over long sequences, small per-token error rates can compound into catastrophic output degradation.

**Quantifying the gap:** Consider a model with per-token accuracy of 95%. Under teacher forcing, every position sees the correct prefix, so the model operates in its "comfort zone" at every step. Under autoregressive sampling, the probability of maintaining a correct prefix for $T$ tokens is approximately $0.95^T$:

| Sequence length | Prob. of error-free prefix |
|---|---|
| 10 | 0.60 |
| 50 | 0.08 |
| 100 | 0.006 |
| 500 | ~0 |

By 100 tokens, the model is almost certainly operating on a prefix containing at least one error -- a distribution potentially never seen during training.

---

## 4. The Deeper Problem: Teacher-Forcing Shortcuts

Bachmann & Nagarajan (2024, [[pitfalls-next-token|paper]]) identified a more fundamental problem that the exposure bias framing misses. The conventional view assumes teacher forcing at least learns an accurate next-token predictor on the training distribution. The deeper failure: **teacher forcing can achieve low training loss without learning the actual computation that generates the data**.

### The Graph Reachability Task

The paper designs a minimal planning task: given a graph and start/end nodes, produce a valid path. The ground-truth paths in the training data follow a specific algorithm (e.g., BFS).

Under teacher forcing, the model receives the correct partial path at each step. This provides a shortcut: to predict the next node, the model can simply look at the current node (given as ground truth) and pick the most likely neighbor -- it does not need to plan ahead or understand the graph structure.

The model achieves low training loss by learning this "local next-step" heuristic. But at inference time, the model must commit to each node and continue from its own predictions. The local heuristic, which never learned to plan, produces paths that wander into dead ends.

**The critical distinction:**

| Failure mode | Where it fails | Cause | Fix |
|---|---|---|---|
| Exposure bias | Inference | Error accumulation from own predictions | Scheduled sampling, sequence-level training |
| Teacher-forcing shortcut | Training | Model fits surface statistics, not true computation | Multi-token prediction, teacherless training |

The second failure is more insidious because low training loss gives no warning. The model appears to have "learned" the task -- it predicts the next token accurately when given the correct prefix -- but it has learned a different function than intended.

---

## 5. Why This Matters for LLM Architecture Research

### Chain-of-Thought as a Workaround

Chain-of-thought (CoT) prompting converts implicit planning into explicit sequential steps. Instead of "predict the answer directly" (which may require multi-step reasoning the AR model cannot do via teacher-forcing shortcuts), CoT asks the model to "predict the next reasoning step" (which is more locally predictable).

This works precisely because it converts the task structure to match what teacher forcing can reliably learn: local next-step prediction. The model does not need to plan ahead -- it just needs to predict the next reasoning token given the correct reasoning prefix.

### Multi-Token Prediction

DeepSeek-V3 and Meta's multi-token prediction research address teacher-forcing shortcuts directly. By requiring the model to predict tokens $t+1, t+2, \ldots, t+k$ simultaneously, multi-token prediction forces internal representations to encode future-oriented information. The model cannot rely solely on the ground-truth current token to predict the next one -- it must develop representations that support predicting multiple steps ahead.

$$\mathcal{L}_{MTP}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\sum_{j=1}^{k} \log P_\theta(x_{t+j} \mid x_{\leq t})$$

### Teacherless Training

Bachmann & Nagarajan propose replacing some input tokens with dummy tokens during training, forcing the model to predict the next token without the ground-truth prefix. This removes the shortcut: the model cannot look at the current ground-truth node to predict the next one, because the current position contains a dummy token.

Their experiments show that both Transformer and Mamba architectures succeed on the planning task under teacherless training, despite failing under standard teacher forcing.

---

## 6. Practical Takeaway

Teacher forcing is not going away -- the parallelism advantage is too large. But understanding its failure modes is essential for architecture research:

1. **Low training loss is necessary but not sufficient.** Always probe whether the model has learned the intended computation or a shortcut.
2. **Tasks requiring planning are vulnerable.** Any task where the correct next step depends on a global plan (not just local context) is at risk of teacher-forcing shortcuts.
3. **Multi-token prediction and CoT are complementary fixes** -- one addresses the training-time problem (shortcuts), the other the inference-time problem (error compounding).

*Source: [[pitfalls-next-token|paper]]*

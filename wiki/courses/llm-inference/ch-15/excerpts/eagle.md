---
chapter: ch-15
course: llm-inference
phase: read
excerpt_of: "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (Li et al. 2024)"
source_url: https://arxiv.org/abs/2401.15077
created_at: "2026-05-21"
---

# Excerpt: EAGLE — Feature-Level Speculative Sampling

**Authors:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang
**Year:** 2024
**Venue:** ICML 2024
**URL:** https://arxiv.org/abs/2401.15077
**Raw-data source:** [[raw-data/eagle]]

---

## The two key observations

EAGLE's paper opens with two empirical claims that justify its design:

1. **Drafting in feature space is easier than in token space.** Hidden features are continuous and evolve smoothly; tokens are discrete and jump abruptly. A small predictor can learn `f_{t+1} = drafter(f_t, x_{t+1})` accurately, but `x_{t+2} = drafter(x_{t+1})` directly is harder.
2. **Future-token uncertainty must be explicitly modeled.** Predicting `f_{t+2}` from `f_t` alone is ambiguous because there are many possible `x_{t+1}` outcomes. Conditioning on the sampled `x_{t+1}` (which is known one step into the future) collapses the uncertainty.

These two observations together motivate the EAGLE drafter architecture.

---

## The drafter architecture

A small autoregressive transformer (typically 1-2 decoder layers) operating on hidden features:

```python
class EagleDrafter(nn.Module):
    def __init__(self, d, vocab_size, n_layers=1):
        self.transformer = nn.ModuleList([DecoderLayer(d) for _ in range(n_layers)])
        # Concatenation projection: maps (f_t || embed(x_{t+1})) → drafter input
        self.input_proj = nn.Linear(2 * d, d)
        # Output is a hidden feature; uses the *target's* lm_head to get tokens
        self.target_lm_head = None  # set externally to point to target's head

    def forward(self, f_prev, x_current):
        """
        f_prev: hidden feature from previous step, shape (B, d)
        x_current: sampled token at current step, shape (B,)
        Returns: predicted next feature f_next, shape (B, d)
        """
        x_emb = self.target_lm_head.weight[x_current]    # tied embedding lookup
        h = self.input_proj(torch.cat([f_prev, x_emb], dim=-1))
        for layer in self.transformer:
            h = layer(h)
        return h  # this is f_next

    def get_token(self, f_next):
        logits = self.target_lm_head(f_next)
        return logits  # apply softmax + sample / argmax externally
```

For Llama-3-70B: drafter is ~0.5B params, ~0.7% of target. Training cost: ~1B tokens of distillation data, a few thousand GPU-hours.

The drafter shares the **target's LM head** — no separate embedding, no separate vocab.

---

## The drafting loop

```
Initialize: f_t = target's penultimate hidden state at current position
            x_{t+1} = sampled token from f_t via target's lm_head

For k = 1..K-1:
    f_{t+1+k} = drafter(f_{t+k}, x_{t+k})
    x_{t+1+k} = sample_from_target_lm_head(f_{t+1+k})
```

The drafter and the target's LM head form an autoregressive loop *purely on features and tokens*, with no full target forward pass during drafting.

Per drafter step: ~1-2 transformer layers + 1 lm_head — `c ≈ 0.02`.

---

## Verification

Same as Medusa: build a draft tree (top-k per drafter step), concatenate, run target forward with tree attention mask, apply Leviathan acceptance rule per branch.

Difference: EAGLE's tree is naturally autoregressive (each node depends on its parent's commit), so the branching factor can be small (k=2-3 typically) and the tree is shallower but more accurate per branch.

---

## Speedup numbers (paper, Table 2)

| Model | EAGLE Speedup vs vanilla |
|-------|--------------------------|
| Vicuna-7B | 2.72× |
| Vicuna-13B | 2.89× |
| LLaMA-2-Chat-7B | 2.66× |
| LLaMA-2-Chat-13B | 3.01× |
| LLaMA-2-Chat-70B | 3.05× |

Measured on MT-Bench, A100, batch=1.

Compared to Medusa (~2.1-2.7×), EAGLE wins by ~25% across model sizes — the autoregressive feature predictor outperforms independent heads.

---

## Why "one step ahead" resolves uncertainty

Consider predicting `f_{t+2}` directly from `f_t`. There are many likely values of `x_{t+1}`, and the distribution over `f_{t+2}` is a *mixture* over them — broad, hard to model with a small predictor.

If we condition on the *sampled* `x_{t+1}` (which is known after the first step), the distribution over `f_{t+2}` becomes much sharper — it's the posterior given `x_{t+1}`. A small predictor can learn this conditional accurately.

This is the entropy reduction argument from Section 3 of the EAGLE paper, and it's the conceptual key to why feature-level drafting needs the one-step-ahead token.

---

## Training data

The drafter is trained on `(f_t, x_{t+1}, f_{t+1})` triples extracted from running the target on a calibration corpus:

```python
for batch in calibration_corpus:
    with torch.no_grad():
        outputs = target(batch, output_hidden_states=True)
    f = outputs.hidden_states[-2]   # penultimate layer
    x = batch.input_ids
    # Train drafter to predict f[t+1] from (f[t], x[t+1])
```

~1B tokens of OpenWebText / ShareGPT-style data is sufficient. Loss: feature MSE + token CE on the predicted distribution.

---

## Pitfalls

- **Drafter must match target precision.** EAGLE drafter typically runs in bf16 like the target. Mixed precision between drafter and target breaks the feature alignment.
- **Penultimate vs final layer.** EAGLE uses the penultimate hidden state (before the final layer norm) because the final layer norm distorts features in ways the drafter struggles to model. Some implementations use the final layer state; this generally underperforms.
- **Target finetune invalidates drafter.** If you finetune the target after training EAGLE, you must retrain EAGLE — the feature distribution shifts.
- **Tree shape matters.** Default trees from EAGLE-2 (dynamic) generally outperform fixed trees from EAGLE-1.

---

## Connections

- [[excerpts/medusa]] — multi-head predecessor; EAGLE's autoregressive drafter outperforms it.
- [[excerpts/eagle-2]] — dynamic-tree successor (3-4× speedup).
- [[excerpts/leviathan-2023]] — the acceptance rule EAGLE uses verbatim.
- [[ch-15]] — parent chapter.

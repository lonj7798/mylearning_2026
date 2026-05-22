---
chapter: ch-15
course: llm-inference
phase: read
excerpt_of: "Better & Faster Large Language Models via Multi-token Prediction (Gloeckle et al. 2024) + DeepSeek V3 MTP deployment"
source_url: https://arxiv.org/abs/2404.19737
created_at: "2026-05-21"
---

# Excerpt: Multi-Token Prediction — Pretraining Objective + Native Speculative Heads

**Authors:** Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, Gabriel Synnaeve (Meta FAIR)
**Year:** 2024
**Venue:** ICML 2024
**URLs:** https://arxiv.org/abs/2404.19737 ; DeepSeek V3 technical report
**Raw-data source:** [[raw-data/multi-token-prediction-inference]]

---

## The pretraining objective

Standard LLM pretraining minimizes next-token cross-entropy:

```math
\mathcal{L}_{\text{NTP}} = -\sum_t \log p_\theta(x_{t+1} \mid x_{\le t})
```

MTP adds `n` auxiliary heads predicting `t+2`, `t+3`, ..., `t+n+1`:

```math
\mathcal{L}_{\text{MTP}} = -\sum_t \sum_{i=1}^{n} \log p_\theta(x_{t+i+1} \mid x_{\le t}; W_{\text{head}_i})
```

Total loss:

```math
\mathcal{L} = \mathcal{L}_{\text{NTP}} + \lambda \cdot \mathcal{L}_{\text{MTP}}
```

`λ ≈ 0.5-1.0` typically; FAIR's paper uses 1.0. Heads share the trunk; each head is a small linear layer + softmax.

---

## Architecture (FAIR paper)

```
Shared trunk (transformer layers 1..L) → h_t
                                          ↓
                       ┌──────┬───────┬───────┬───────┐
                       ↓      ↓       ↓       ↓       ↓
                     head_0 head_1 head_2 head_3 head_4
                       ↓      ↓       ↓       ↓       ↓
                    x_{t+1} x_{t+2} x_{t+3} x_{t+4} x_{t+5}
```

For LLaMA-style models with `n=4` heads: ~5% extra parameters. The trunk handles 99% of the work.

Each head:

```python
class MTPHead(nn.Module):
    def __init__(self, d, vocab_size):
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, vocab_size, bias=False)  # often tied with embedding

    def forward(self, h):
        return self.proj(self.norm(h))
```

---

## Why this improves quality (not just speed)

The FAIR paper documents quality improvements beyond inference speed:

| Benchmark | Baseline (NTP) | MTP (n=4) | Δ |
|-----------|----------------|-----------|---|
| HumanEval | 23.2 | 27.8 | +4.6 |
| MBPP | 35.0 | 41.2 | +6.2 |
| TriviaQA | 41.0 | 42.5 | +1.5 |
| Lambada | 70.6 | 72.4 | +1.8 |

Code and algorithmic tasks benefit most (+5-7 points). Plain language modeling benefits modestly (+1-2 points).

**Why**: MTP forces the model to plan multiple tokens ahead, encouraging better **induction-head circuits** and richer intermediate representations. The auxiliary signal is a regularizer that pushes the model toward longer-horizon prediction.

---

## Inference acceleration

At decode time, all `n` heads emit candidate tokens for `t+1..t+n+1` in parallel from a single trunk forward pass. Verification proceeds as in Medusa — tree attention, Leviathan acceptance rule.

The paper reports up to **3× inference speedup** at greedy decoding for 13B models trained with `n=4`.

**Compared to Medusa** (heads added at finetune): MTP wins because:
1. Heads are co-pretrained — they're naturally aligned with the trunk's distribution.
2. The auxiliary loss has shaped the trunk's intermediate representations to be predictive across multiple horizons.

---

## DeepSeek V3 — production-deployed MTP

DeepSeek V3 (671B MoE) uses MTP natively as a *single* extra head predicting `t+2`:

```
Trunk → h_t
         ↓
      ┌──┴──┐
   ntp_head  mtp_head
      ↓        ↓
   x_{t+1}  x_{t+2}
```

DeepSeek's choice of `n=1` (one MTP head, not `n=4`) is pragmatic:
- The MoE trunk is already huge; extra heads add real cost.
- A single MTP head buys 1 free verification per step → ~1.8-2× speedup.
- More heads would help speedup but raise pretraining cost.

vLLM has native support for this:

```python
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    speculative_config={
        "method": "deepseek_mtp",
        "num_speculative_tokens": 1,
    },
)
```

Reported TPOT improvement on H200/H100 IB cluster: ~1.8× over MTP-disabled.

---

## When to use MTP

MTP is a *pretraining* decision, not a deployment decision. You either:
1. Train your model with MTP from scratch (DeepSeek V3 approach).
2. Add Medusa-style heads at finetune (the "MTP-lite" approach — really Medusa).

Choose MTP if:
- You're building a model and care about both quality (code/algorithmic) and inference speedup.
- You can afford the pretraining-time overhead (~5-10% extra cost per training step).
- Your serving stack supports MTP heads (vLLM, SGLang both do).

---

## Pitfalls

- **Don't confuse MTP-as-pretraining with Medusa-as-finetune.** The paper's results require the joint pretraining; finetuning heads after the fact gives Medusa-1-class results.
- **`n` is fixed at pretraining**, not at serving. Adding/removing heads later requires retraining.
- **Head ↔ trunk gradient flow.** During pretraining, head losses backprop into the trunk — this is intentional and shapes the trunk. Don't freeze trunk during MTP pretraining.
- **DeepSeek's `n=1` is a deliberate cost-quality tradeoff**, not a "less is better" claim. Don't generalize.

---

## Speedup numbers across the family

| Model | Heads | Speedup |
|-------|-------|---------|
| LLaMA-style 13B (FAIR's experiment, n=4) | 4 MTP | 3.0× |
| DeepSeek V3 (n=1) | 1 MTP | ~1.8× |
| Hypothetical 70B with n=4 MTP | 4 | ~2.5-3× (estimated) |

---

## Connections

- [[excerpts/medusa]] — finetune-time analog of MTP; MTP wins by co-pretraining.
- [[excerpts/eagle]] — alternative drafter design; EAGLE's drafter is autoregressive at feature level.
- [[excerpts/leviathan-2023]] — verification still uses the same acceptance rule.
- [[ch-20]] — DeepSeek V3 production deployment (the only widely-deployed MTP model).
- [[ch-15]] — parent chapter.

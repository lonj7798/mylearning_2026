---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/simpo.md
source_url: https://arxiv.org/abs/2405.14734
created_at: "2026-04-23"
---

# Excerpt: SimPO — reference-free, length-normalized DPO

**Source library:** `wiki/raw-data/llm-training/papers/simpo.md`
**Authors:** Yu Meng, Mengzhou Xia, Danqi Chen
**Venue:** NeurIPS 2024 (arXiv 2405.14734)
**Year:** 2024

---

## Why this source anchors ch-39 §5

SimPO attacks two problems with DPO at once:

1. **Memory and throughput:** DPO needs `π_ref` resident in memory plus a forward pass through it for every step. On a 70B actor, that is another 70B of weights and another forward — roughly doubling training cost for a reference-free objective that could drop it.

2. **Length bias:** DPO's reward `β · log π/π_ref` is a sum of per-token log-ratios. If `π` and `π_ref` shift uniformly — which they do under training — the reward grows linearly in `|y|`. Longer `y_w` mechanically wins. [[simpo]] line 53: "DPO increases response length by 30–60 % over SFT; SimPO stays within ±5 %."

SimPO's fix is one move: replace the reference-model ratio with an *average* per-token log-probability. No ratio, no reference, and by construction length-invariant because it's a mean not a sum.

## The implicit reward — just the average log-prob

Source line 31:

```
r_SimPO(x, y) = (β / |y|) · Σ_{t=1..|y|} log π_θ( y_t | x, y_<t )
             =  β · (average log-probability per token)
```

Notice what is *gone* compared to DPO's `r̂_θ = β · log π_θ/π_ref`: no `π_ref`, no subtraction, no ratio. The price is that this reward is only well-defined if the policy has been SFT-ed; starting from a base model, the average log-prob is just an LM's natural entropy and provides no preference signal.

## The loss with target margin γ

Source line 36:

```
L_SimPO(π_θ) = − E_{(x, y_w, y_l) ~ D}[ log σ( (β/|y_w|) log π_θ(y_w|x)
                                             − (β/|y_l|) log π_θ(y_l|x)
                                             − γ ) ]
```

The subtraction of `γ` inside the sigmoid is new vs DPO. It enforces a minimum margin: the chosen's average log-prob must exceed the rejected's by at least `γ` before the loss saturates. Pick γ too small → no margin, too easy to satisfy; pick γ too large → gradient vanishes because all pairs already satisfy it.

## Hyperparameters — 20× larger β than DPO

Source lines 42–50:

| Knob | SimPO | DPO |
|------|-------|-----|
| β | 2.0–2.5 | 0.1 |
| γ | 0.3–1.6 | — |
| γ/β | 0.25–0.5 (rule) | — |
| LR | 3e-7 – 1e-6 | 5e-7 |
| Batch | 128 pairs | 32–128 |
| Epochs | 1 | 1–3 |
| π_ref | **none** | SFT, frozen |

The 20× larger β is the single biggest mental adjustment. The reason: DPO's reward is a sum of log-ratios and scales with sequence length; SimPO's reward is an average, so to get the same magnitude of logit into the sigmoid, β has to be multiplied by the typical sequence length (~20–50 tokens of signal).

## Failure modes

Source lines 55–58:

- **β too low** → policy collapses to high-entropy mode (nothing pulls it back to anything sharp).
- **γ too high** → all pairs already satisfy the margin, gradient vanishes, training stalls.
- **Very clean / deterministic data** → same unbounded-margin issue DPO has, but now without the reference regularization. Mitigation: add label smoothing, or add a small SFT loss term — which is essentially ORPO.

## Empirical headline

Source line 15: "+6.4 pts on AlpacaEval 2 and +7.5 pts on Arena-Hard over DPO" on Llama-3-8B-Instruct and Mistral-7B-Instruct. The win is driven by (a) no length bias, so responses stay concise, and (b) cleaner optimization surface once β and γ are tuned.

## How ch-39 uses this

§5 presents the reward and loss as Equations (12)–(13). The comparison table in §8 marks SimPO as the length-invariant, ref-free variant. §9 notes its absence of `π_ref` simplifies the `concatenated_forward` trick. The loss landscape visualization shows the SimPO curve as the DPO curve *shifted right by γ*.

## Connections

- Drops the reference from: [[dpo]].
- Margin-style cousin with different loss shape: [[ipo]].
- Reference-free joint-SFT variant: [[orpo]].
- Unary-label alternative: [[kto]].

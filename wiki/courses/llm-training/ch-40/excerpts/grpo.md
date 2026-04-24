---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/grpo.md
source_url: https://arxiv.org/abs/2402.03300
created_at: "2026-04-23"
---

# Excerpt: GRPO — the DeepSeekMath loss that R1 shipped with

**Source library:** `wiki/raw-data/llm-training/papers/grpo.md` + `wiki/raw-data/llm-training/model-reports/deepseekmath.md`
**Artifact:** Shao et al. 2024, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." The paper's §4.1.2 / Equation 3 is the single most-cited formula in post-R1 RL.

---

## Why this source anchors ch-40 §4

GRPO is the loss DeepSeek used for R1-Zero and R1. Every open-source R1 reproduction runs some variant of it. Ch-40 §4 is essentially a tutorial on Equation 3 plus the k3 KL estimator in Equation 4. Getting the advantage form and the KL location right is the difference between matching paper results and silently training on a different objective.

---

## The rollout/advantage/objective sequence ch-40 §4 walks through

Source lines 33–42:

1. **Rollout.** For each question q, sample G outputs `{o_1, …, o_G}` from `π_θ_old`. RM scores them → `r_1, …, r_G`.
2. **Advantage** (same for every token t in o_i):
   ```
   Â_{i,t} = (r_i − mean(r)) / std(r)
   ```
3. **Objective (Eq. 3):**
   ```
   J = E[(1/G) Σ_i (1/|o_i|) Σ_t { min[ρ Â, clip(ρ, 1±ε) Â] − β D_KL(π_θ || π_ref) }]
   ```
   where `ρ_{i,t} = π_θ(o_{i,t}|·) / π_θ_old(o_{i,t}|·)`.

Ch-40 §4 reproduces this formula exactly. The two biased divisions ch-40 §5 targets are the `(1/|o_i|)` and the `/std(r)` — both visible here on first sight.

---

## Why std-normalize at all (ch-40 §4 argument)

Source lines 36: advantages are z-scored. Ch-40 §4 motivates this concretely: prompt A has rewards `{0.1, 0.11, 0.12}`, prompt B has `{0.0, 0.5, 1.0}`. Without std, B's gradient magnitude is ~10× A's even though B's *relative* signal is equally informative per prompt. Dividing by std equalizes per-prompt gradient magnitudes so the optimizer doesn't focus all updates on high-variance prompts. This is the motivation; §5 then shows it backfires for verifiable 0/1 rewards.

---

## The k3 KL estimator derivation

Source lines 44–47:

```
D_KL^k3 ≈ π_ref/π_θ − log(π_ref/π_θ) − 1
```

Let `x = log(π_ref/π_θ)`. Then `k3 = e^x − x − 1`. Taylor around x=0:
`k3 = x²/2 + x³/6 + x⁴/24 + …`

- For small KL (`|x| → 0`): `k3 ≈ x²/2` = k2 (Fisher-information regime).
- For positive x (ratio > 1): `k3 → e^x` — grows faster than |x|.
- For negative x (ratio < 1): `k3 → −x − 1 + o(1)` — grows linearly.
- Always ≥ 0 (Bregman property of convex `f(t) = e^t`).
- Unbiased: `E[k3] = KL(π_θ || π_ref)` exactly when `x` is the log-likelihood-ratio of a sample from `π_θ`.

This is why ch-40 §4 calls k3 "unbiased AND always positive" — k1 is biased in sign, k2 loses sign, k3 is both properties at once. One extra reference forward per step, tensor shape identical to logprobs.

---

## KL-in-loss vs KL-on-reward (ch-40 §4's subtle point)

Source line 47: "Applied token-wise inside the loss, not as a per-token reward penalty."

RLOO and REINFORCE++ put `−β · KL_t` into the per-token reward before computing advantages. GRPO leaves the reward untouched and adds `−β · KL` to the per-token loss. The numerical difference:

- On-reward KL: `Â = normalize(reward + KL penalty)` — KL propagates through advantage normalization (gets divided by std).
- In-loss KL: `Â = normalize(reward)`, loss adds `−β · KL_t` separately — KL does not interact with advantage normalization.

For verifiable 0/1 tasks, in-loss KL is cleaner: advantage is a pure function of the outcome reward, KL regularization is a pure function of the policy distance. Less entanglement.

---

## Attested GRPO hyperparameters (DeepSeekMath paper recipe)

Source lines 54–63:

| Knob | Value (attested) |
|------|------------------|
| Group size G | 64 (main runs) |
| Clip ε | 0.2 |
| KL coefficient β | 0.04 |
| Learning rate | 1e-6 |
| Batch size (prompts) | 1024 |
| Max response length | 1024 tokens |
| π_ref | SFT model, frozen |
| Epochs per rollout μ | 1 (single-step) |
| Sampling T | 1.0 |

Ch-40 §4 reports these as canonical GRPO defaults. G=64 is much larger than RLOO's k=2–4 — more samples give a tighter per-prompt baseline but require more rollout compute.

---

## Empirical progression (ch-40 §4 final paragraph)

From deepseekmath.md line 25: on MATH, SFT 46.8 → RFT 49.0 → DPO 49.0 → PPO 51.0 → **GRPO 51.7**. Each step of the ladder is smaller than the gap between SFT and RFT, but GRPO is the final rung and the one that ships with R1.

---

## Connections to the rest of the track

- [[rloo]] — the leave-one-out ancestor.
- [[dr-grpo]] — the bias-corrected successor (subtracts the two problematic divisions).
- [[deepseek-r1]] — the direct downstream application.
- [[verl-grpo]], [[trl-grpo]] — the open-source reference implementations.
- [[john-schulman-kl-tricks]] — the k1/k2/k3 estimator families.

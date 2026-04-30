---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/lilianweng-rlhf.md
source_url: https://lilianweng.github.io/tags/rlhf/
created_at: "2026-04-23"
---

# Excerpt: Lil'Log RLHF — canonical per-token KL-as-reward

**Source library:** `wiki/raw-data/llm-training/blogs/lilianweng-rlhf.md`
**Artifact:** Lilian Weng's RLHF-tag blog posts (2023–2024) — the most-cited tutorial-grade treatment of the InstructGPT-style RLHF stack, and the source most ch-37 readers will have already seen once before.

---

## Why this source anchors ch-37

The policy-gradient template ch-37 builds is the *first term* of a bigger objective. The canonical RLHF objective also includes a **KL-to-reference penalty** added to the *per-token reward* (not to the loss), and this placement is itself a design choice with consequences. Weng's tutorial is the cleanest statement of that canonical form, and is the reference that ch-37 §5 ("when the entropy bonus is redundant") leans on. The "entropy bonus dropped because KL already regularises" claim is directly attested here.

---

## The three-stage pipeline and where PG enters

From the source (line 7, §Core Insight):

> RLHF decomposes cleanly into (1) preference data collection, (2) reward model training with Bradley-Terry loss, (3) KL-regularized policy optimization against that RM — and the bottleneck is usually (1) or (2), not the RL algorithm.

The third stage is where ch-37's template equation lives. Stages 1 and 2 produce `R(x, y)` — the scalar reward used in `∇J = E[∇log π · A]`. Ch-37 is agnostic to how `R` is produced, but ch-42 (RLVR), ch-39 (DPO), and ch-47..ch-53 (eval, reward, judge) are exactly about stages 1 and 2.

> Notice: Weng's punch-line — "the bottleneck is usually (1) or (2), not the RL algorithm" — is the reason ch-37 starts by establishing a common template rather than diving into any one algorithm. If the RL algorithm is second-order, then the shared object (the estimator) is what matters most, and the specialisations follow.

---

## The canonical per-token reward

From the source (lines 30–32, §Technical Details):

> KL penalty implementation: beta is the KL coefficient; the per-token reward becomes `r_total = r(x,y) 1[y=EOS] - beta * log(pi(y|x) / pi_ref(y|x))`.

This is *the* canonical RLHF objective. Three things worth pinning:

1. **Terminal-only `r(x,y)`.** The RM score is attributed to the EOS token — every other token gets zero reward from the RM. This is why ch-37 §4 (d) ("full-trajectory rewards") holds; it is an instance of (b) ("terminal-only rewards").
2. **Per-token KL term.** The KL penalty is added *per token*, even though the RM reward is terminal. This means the KL term contributes at every step of the gradient sum `Σ_t ∇log π · (G_t − b(s_t))`, while the RM term contributes only through the terminal reward propagated as `G_t = R · 1[t=T]`.
3. **KL is in the reward, not the loss.** `r_total` is the per-token reward; advantages are computed from `r_total`. This means the KL is propagated through whatever advantage estimator you use (GAE, leave-one-out, global z-score).

> Notice: placing KL in the reward *vs* the loss is a more consequential choice than it sounds. [[reinforce-plus-plus]] keeps it in the reward (like PPO / InstructGPT / RLOO). [[grpo]] moves it to the loss (with the k3 estimator). When you see β=0.01 in one paper "work" and β=0.1 in another paper "work", this is often the explanation — the effective regularisation scale depends on the estimator and on whether advantages are normalised afterward.

---

## RLHF defaults: γ=1 and the reward-whitening recipe

From the source (lines 33–35, §Technical Details):

> Reward normalization: whitening (subtract mean, divide by std within batch) is standard to stabilize PPO.
> GAE lambda: typical 0.95 for RLHF; gamma = 1.0 (undiscounted, because rewards concentrate at EOS).

Both numbers are attested here as RLHF canon. Ch-37 §3 cites `γ = 1.0, λ = 0.95` as the RLHF defaults; this is the source. The "because rewards concentrate at EOS" reasoning is the attested justification — discounting a stream of zeros does nothing, so `γ < 1` only discounts the one terminal reward at decreasing rates per earlier position, which just throws signal away.

> Notice: "reward whitening" is a per-batch z-score of the scalar RM output, computed *before* the reward enters the advantage estimator. It is distinct from advantage normalisation (which z-scores the advantage itself after summation). Whitening stabilises the RM scale across batches; advantage normalisation stabilises the gradient scale. PPO does both; RLOO does the second only; [[reinforce-plus-plus]] does the second globally.

---

## Why entropy bonus is usually dropped

From the source (line 36, §Technical Details):

> Entropy bonus: often dropped in RLHF because KL-to-reference already regularizes exploration; re-introduced in some GRPO variants.

This single line is the attested basis for ch-37 §5's position that the entropy bonus is usually a redundant add-on. The reason is structural: `KL(π_θ || π_ref) = E_π_θ[log π_θ − log π_ref]`, which decomposes as `−H(π_θ) − E_π_θ[log π_ref]`. Minimising KL therefore has `H(π_θ)` as a component — penalising KL-to-ref *is* a lower-bound regulariser on entropy, given that `π_ref` is a reasonable-entropy SFT policy.

> Notice: "re-introduced in some GRPO variants" is the hedge that matters. Reasoning-RL recipes (R1-style, verifiable rewards, long CoT) do sometimes add an entropy bonus on top of KL-to-ref. The empirical reason, per [[nathan-lambert-entropy-rl]] and [[entropy-mechanism-llm-rl]], is that length-normalisation or verifier-reward shape can produce rapid entropy collapse despite the KL penalty. That's the "bandaid" case ch-37 §5 calls out.

---

## Reward hacking and why policy regularisation matters

From the source (lines 37–38, §Technical Details):

> Reward hacking vulnerabilities: Weng catalogs (a) length bias in RMs, (b) sycophancy (RM rewards agreement), (c) specification gaming (model finds prompt exploits that score high).

Reward hacking is not ch-37's subject, but it connects to why the KL-to-reference term is both a trust region and a safety mechanism. If `R(x, y)` is a learned RM and `R` has exploits, then bounding `π_θ` near `π_ref` bounds how far the policy can drift into exploit territory.

> Notice: RLVR's no-RM argument ([[rlvr-tulu3]], covered in ch-33 and ch-42) is a direct response to this failure family. If `R(x, y) = v(x, y)` is a deterministic verifier, (a), (b), (c) all vanish — the verifier cannot be hacked except by finding loopholes in its own definition. Ch-37 is agnostic to whether `R` is an RM or a verifier; ch-42 and ch-47..ch-53 are where the choice matters.

---

## Algorithmic template vs reward-signal source

From the source (line 8, §Guideline):

> Debug RLHF from the bottom up: verify RM calibration on held-out preferences before touching PPO hyperparameters.

This is the practical version of ch-37 §6's "the algorithm family is second-order; the reward signal is first-order" framing (which ch-37 also draws from [[nathan-lambert-rl-overview]]). Debug your RM before debugging your PPO.

> Notice: Weng's "bottom-up" ordering (data → RM → RL algorithm) is the same priority ordering ch-37 §6 implies. The RL algorithm is the last debugging target because it has the most knobs, so it is the most likely to *appear* broken when actually the upstream signal is corrupted. This is why ch-37's template-first approach matters — if you know the template, you know that changing the RL algorithm is just choosing a different baseline and regulariser; it cannot fix a broken `R`.

---

## What ch-37 keeps from this source

- The canonical `r_total = r(x,y) 1[y=EOS] − β · log(π/π_ref)` per-token reward form (§5).
- The RLHF defaults `γ=1.0, λ=0.95` (§3).
- The "entropy bonus usually dropped because KL-to-ref already regularises" claim (§5).
- The reward-whitening recipe as part of the practical pipeline.
- The "algorithm is second-order to reward signal" framing that structures §6.

---

## Connections

- **ch-37 §3 / §5 / §6** — RLHF defaults, entropy-term discussion, knob-table framing.
- **ch-38** — TRPO / PPO / InstructGPT with this exact per-token reward.
- [[rlhf-instructgpt]] — the original paper this tutorial explains.
- [[ppo]] — the algorithm Weng's per-token form specialises.
- [[reward-hacking-taxonomy]] — length bias, sycophancy, specification gaming.
- [[dpo]] — the RL-free alternative Weng covers in a companion post.
- [[excerpts/trpo]] — the trust-region ancestor whose KL constraint becomes Weng's per-token penalty.
- [[excerpts/maximum-entropy-rl]] — the entropy-regulariser ancestor that Weng's "usually dropped" claim is taking a position against.

---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/reinforce-plus-plus.md
source_url: https://arxiv.org/abs/2501.03262
created_at: "2026-04-23"
---

# Excerpt: REINFORCE++ — global-batch normalisation when k is small

**Source library:** `wiki/raw-data/llm-training/papers/reinforce-plus-plus.md`
**Artifact:** *REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models* — Jian Hu, 2025. A successor to RLOO that replaces prompt-local baselines with a global-batch z-score and reintroduces PPO-clip for per-token stability.

---

## Why this source anchors ch-37

Ch-37 §2's baseline menagerie puts six methods in one table. REINFORCE++ sits at the small-k edge of the table: when you can only afford `k=1` or `k=2` rollouts per prompt, RLOO's leave-one-out baseline is unreliable (`k=1` is undefined; `k=2` has a two-sample estimator of the mean, which has high variance). REINFORCE++'s argument is that you should trade *prompt-local* variance reduction (RLOO, GRPO) for *batch-global* variance reduction. Ch-37 §4 cites this paper for the fourth LLM-RL property ("long episodes, short batches, correlated per-token advantages").

---

## The core variance argument

From the source (line 8, §Core Insight):

> Prompt-local advantage normalization (GRPO's per-group, RLOO's leave-one-out) is high-variance when groups are small; normalizing across the *global* batch gives a more accurate, lower-variance advantage, and combining that with token-level KL and PPO-clip recovers PPO's stability at REINFORCE's cost.

The high-variance-when-groups-small claim is an empirical observation that the paper supports with a variance plot (Figure 3). Intuitively: the per-group mean/std is itself an estimator; with `k=2` it is a two-sample estimator of μ and σ, which has variance that decreases only as `1/k`. A global batch (say 512 sequences from many prompts) provides a much larger `n` at the cost of introducing cross-prompt variance.

> Notice: the tradeoff is "within-prompt signal vs within-batch sample size". If reward distributions are similar across prompts, global normalisation wins. If reward distributions differ sharply across prompts (some prompts routinely score 0.9, others 0.1), global normalisation subtracts the wrong baseline and prompt-local is better. The paper's implicit claim is that for typical RLHF prompt pools the across-prompt variance is *not* that large relative to within-prompt variance — which is why global normalisation works.

---

## The attested advantage formula

From the source (lines 39–42, §Return and advantage):

> Cumulative return from step t:
> `G_t = Σ_{t'≥t} γ^{t'−t} r̃_{t'}` (γ typically 1.0 for LLMs).
> Global advantage normalization over the whole batch B:
> `Â_t = (G_t − mean_{B}(G)) / std_{B}(G)`
> No value network, no GAE, no group grouping.

`γ = 1.0` is consistent with ch-37 §3's LLM-specific argument that discounting throws away signal when rewards concentrate at EOS. "No value network, no GAE, no group grouping" means REINFORCE++ sits in the same ch-37 §4 row as RLOO — critic-free, minimum-viable — but with a different baseline choice.

> Notice: the z-score normalisation `Â_t = (G_t − μ_B) / σ_B` is itself a biased estimator of the true advantage because `μ_B` and `σ_B` are random variables. The bias shrinks like `O(1/|B|)`, which for `|B| = 512` or larger is small. This is a place where ch-37's "strictly unbiased" story in §2 gets softened — global normalisation is only *approximately* unbiased. The paper's empirical argument is that the variance reduction more than pays for the small bias.

---

## The per-token shaped reward

From the source (lines 31–34, §Per-token shaped reward):

> `r̃_t = r(x,y) · 𝟙{t = T} − β · KL_t`
> where `KL_t = log π_θ_old(y_t | ·) − log π_ref(y_t | ·)` (k1 estimator, one sign).
> All terminal reward credited to the last token; KL penalty applied per token.

Two things worth pinning: (1) `KL_t` uses the **k1 estimator** — just `log π_old − log π_ref` — not the **k3 estimator** `(log π_old − log π_ref − 1 + π_ref/π_old)` used by some GRPO variants. The k1 estimator is higher-variance but unbiased; k3 is lower-variance but has bias that depends on the local KL. (2) Terminal reward is credited to the last token only — no reward shaping across the sequence — while KL is applied per token. This asymmetry is the RLHF canonical form ([[lilianweng-rlhf]] attests the same).

> Notice: the difference between k1 and k3 KL estimators is the subject of [[john-schulman-kl-tricks]] — an entire blog post about one term in the PPO loss. In practice: k1 for RLOO / REINFORCE++ (reward-embedded), k3 for GRPO (loss-embedded). When you see a team's "KL reward coef" vs "KL loss coef" disagreeing with another team's, this is usually what's happening — same β on different estimators produce different effective regularisation.

---

## PPO-clip survives, not for trust-region reasons

From the source (lines 44–46, §Loss):

> `L(θ) = − E_t[ min( ρ_t(θ) Â_t,  clip(ρ_t(θ), 1-ε, 1+ε) Â_t ) ]`
> where `ρ_t(θ) = π_θ(y_t | ·) / π_θ_old(y_t | ·)`.

PPO-clip is kept to bound per-token ratio drift — which is subtly different from PPO's original trust-region motivation. REINFORCE++'s argument is that at the per-token level you want stability (no token's ratio can blow up the gradient) even if the per-sequence policy stays close to `π_θ_old`.

> Notice: this is a departure from RLOO's "no clip" stance. RLOO drops clip because it doesn't do K>1 epochs; REINFORCE++ reintroduces clip because it does do K>1 epochs (small-batch, multi-epoch recipes). So "clip or no clip" is not a property of critic-free-ness — it tracks how aggressively you re-use the rollout.

---

## The comparison table

From the source (lines 48–55, §What's kept vs dropped):

| Component | PPO | RLOO | GRPO | REINFORCE++ |
|-----------|-----|------|------|-------------|
| Value network | yes | no | no | **no** |
| Clip ε | yes | no | yes | **yes** |
| KL location | per-token reward | per-token reward | in-loss (k3) | **per-token reward** |
| Advantage baseline | learned V | leave-one-out | group mean/std | **global batch mean/std** |
| Group size requirement | — | k ≥ 2 | G ≥ 2 | **k = 1 OK** |

The `k = 1 OK` row is REINFORCE++'s distinctive win. For teams that can only afford one rollout per prompt (large batches of diverse prompts rather than small batches of repeated prompts), REINFORCE++ is the only critic-free option that actually works.

> Notice: this table is the concrete form of ch-37 §6's "every algorithm is a cell in a knob-table" argument. The five rows are the actual knobs; the four columns are the four algorithms. Every RL algorithm in ch-37..ch-46 can be written as a choice in each row of this table (with a few extra rows for reward-signal source, offline-vs-online, etc.).

---

## Attested hyperparameters

From the source (lines 57–64, §Hyperparameters):

| Knob | Value |
|------|-------|
| Clip ε | 0.2 |
| KL coef β | 0.01–0.05 |
| Learning rate | 5e-7 – 1e-6 |
| Global batch size | 512–2048 sequences |
| k (samples per prompt) | 1–4 |
| Epochs per rollout | 1 |
| Sampling T | 1.0 |

`β ∈ [0.01, 0.05]` is the same range attested in [[trpo]] (δ ∈ [0.01, 0.05]), [[lilianweng-rlhf]] (β standard RLHF), and [[rloo]] (β=0.05 default). The convergence on this range across four sources is worth noting: the KL coefficient is not a free parameter — it's pinned to the scale of the surrogate approximation error, which is similar across RL methods on similar model scales.

---

## What ch-37 keeps from this source

- The "correlated per-token advantages → global normalisation helps" argument in §4.
- The global-batch-normalisation row of the §2 menagerie table.
- The k1-vs-k3 KL estimator distinction that ch-37 §5 gestures at (and ch-40 develops).
- The comparison table style that ch-37 §6's knob-table is modelled on.

---

## Connections

- **ch-37 §2 / §4 / §6** — menagerie, LLM-RL properties, knob-table.
- **ch-41** — REINFORCE++ and RLOO coexist in the critic-free chapter.
- [[rloo]] — direct predecessor.
- [[grpo]] / [[dr-grpo]] — group-local baseline competitor.
- [[ppo]] — PPO-clip reused here for per-token stability.
- [[excerpts/rloo]] — companion excerpt; the two together define the critic-free row of ch-37's menagerie.
- [[john-schulman-kl-tricks]] — k1 vs k3 KL estimators.

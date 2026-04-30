<!-- scope: rigorous empirical study of entropy collapse in on-policy RL
     deps: [[entropy-regularization-ppo]]
     see-also: [[entropy-mechanism-llm-rl]], [[maximum-entropy-rl]]
-->

# What Matters in On-Policy RL (Andrychowicz 2020) + Entropy Collapse in LLM PPO
- **Core Insight:** Across a 50-dimensional hyperparameter sweep, entropy-related choices (entropy coefficient, advantage normalization, learning-rate schedule) sit in the top tier of "things that change outcomes" for on-policy RL — and entropy collapse is the recurring symptom when any of these are wrong.
- **Guideline:** Treat entropy as a first-class diagnostic: log per-step policy entropy, set the entropy coefficient against a target rather than a fixed value, and be suspicious when entropy drops faster than reward rises — that's the fingerprint of collapse, not progress.
- **Authors:** Marcin Andrychowicz, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan Girgin, Raphael Marinier, Leonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, Sylvain Gelly, Olivier Bachem
- **Year:** 2020
- **URL:** https://arxiv.org/abs/2006.05990
- **Relevant topics:** PPO hyperparameter sweep, entropy coefficient, advantage normalization, policy collapse, reproducibility

## Abstract
The paper runs ~250,000 MuJoCo training trials across a factorial grid of 50 hyperparameters — the largest-scale ablation study of on-policy RL to date. It ranks design choices by average effect on final return. The ordering: (1) policy loss type (PPO clip vs vanilla vs KL), (2) advantage normalization, (3) learning rate, (4) number of minibatch epochs, (5) clipping range; the entropy coefficient is mid-tier but tightly coupled to learning rate and normalization. Entropy collapse is observed across many bad-hyperparameter settings and is repeatedly the mechanism behind the failures.

## Key Contributions
- **Hyperparameter ranking:** which of 50 knobs moves final return the most — PPO-style clipped surrogate with advantage normalization wins consistently.
- **Entropy coefficient finding:** optimal `c_H` interacts with advantage normalization and LR; a single "one size fits all" value does not exist, but `c_H ∈ [0, 0.005]` is a reasonable default.
- **Advantage normalization matters more than entropy bonus:** normalizing advantages to zero-mean unit-variance per batch has a larger effect than tuning the entropy coefficient, and the two interact.
- **Clip range interaction:** `ε = 0.2` is close to optimal; smaller `ε` reduces entropy collapse risk at the cost of slower updates.
- **Learning-rate schedule:** linear decay over training beats constant; constant LR tends to drive entropy to zero late in training as advantages shrink.
- **Reproducibility implication:** "just use PPO defaults" hides large variance; report entropy curves as part of any on-policy RL paper.

## Collapse in LLM PPO (derivative observations)
- **Symptom:** per-token entropy `H(π)` drops from ~2–3 nats to below 0.1 nats within a few hundred PPO updates; reward plateaus; rollouts become repetitive.
- **When it happens:** small or zero entropy coefficient, large clip range, no KL-to-reference penalty, sharp/binary rewards (e.g. RLVR without any shaping).
- **Diagnostic:** plot `H(π)` per step and per token position — collapse usually shows as a sudden inflection rather than a gradual decline.
- **Standard fixes (from the LLM-RL literature):**
  - Add KL-to-reference penalty (**[[kl-control-rlhf]]**).
  - Raise entropy coefficient (**[[entropy-regularization-ppo]]**).
  - Use covariance-targeted interventions (**[[entropy-mechanism-llm-rl]]**, Clip-Cov / KL-Cov).
  - Lower the advantage normalization std floor.

## Key Figures/Tables to Study
- **Andrychowicz Fig. 3** — hyperparameter effect-size ranking.
- **Fig. 7** — entropy coefficient × advantage normalization heatmap.
- **LLM-RL reproduction (Cui 2025 Fig. 1)** — entropy collapse across algorithms (reference **[[entropy-mechanism-llm-rl]]**).

## Technical Details
- **Standard PPO loss with entropy bonus:**
  `L = L_CLIP − c_v L_VF + c_H H(π)` with `c_H` swept over `[0, 1e-4, 1e-3, 1e-2, 1e-1]`.
- **Entropy estimator:** exact analytical entropy per categorical distribution (available for softmax policies including LM token heads).
- **Logging recommendation:** mean entropy over rollout, per-position entropy (early tokens vs late tokens), fraction of tokens with `H < 0.1`.
- **Caveat:** Andrychowicz is on MuJoCo continuous control; exact coefficient scales differ for LLMs (huge vocab, long sequences), but the qualitative ranking survives.

## Connections
- Foundational large-scale ablation that motivates every "monitor entropy" recommendation; canonical citation.
- Directly extended to LLM-RL by **[[entropy-mechanism-llm-rl]]**.
- Complements max-ent theory (**[[maximum-entropy-rl]]**) with empirical scale.
- Pairs with KL-to-reference defenses (**[[kl-control-rlhf]]**) in modern LLM RL loops.

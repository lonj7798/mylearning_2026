<!-- scope: John Schulman's "Approximating KL Divergence" — the k1/k2/k3 estimator trio
     deps: [[README]]
     see-also: [[ppo]], [[grpo]], [[deepseekmath]]
-->

# Approximating KL Divergence (John Schulman blog)
- **Core Insight:** Monte-Carlo KL estimators are not unique; the k3 estimator (`p/q - 1 - log(p/q)`) is both unbiased and low-variance, and should be preferred over the naive k1 (`-log(p/q)`) in PPO/GRPO KL penalties.
- **Guideline:** When implementing KL-to-reference penalty for RLHF, use the k3 estimator; it stays non-negative and gives gradient signal close to the true KL near p ~ q.
- **Author:** John Schulman
- **Year:** 2020 (original post); widely cited 2023–2025 in RLHF/GRPO implementations.
- **URL:** http://joschu.net/blog/kl-approx.html
- **Relevant topics:** KL divergence Monte-Carlo estimation, PPO/GRPO KL penalty, variance reduction, unbiased estimators

## Summary
Schulman's post addresses a deceptively simple problem: you have samples from a policy q and want to estimate KL(q || p) where p is a reference distribution (e.g., the pre-RL SFT model). The post introduces three estimators: k1, k2, k3. All three are one-line formulas; they differ in bias and variance. k1 is unbiased but high-variance and can be negative. k2 is biased but low-variance. k3 (`p/q - 1 - log(p/q)`) is unbiased *and* always non-negative *and* usually lower-variance than k1 when policies are close — making it the default RLHF choice.

## Key Contributions
- Three named estimators with explicit formulas:
  - **k1:** `-log(p/q)` — direct definition; unbiased; can be negative for a single sample; high variance.
  - **k2:** `0.5 * log(p/q)^2` — biased but low-variance; agrees with KL to second order near p ~ q.
  - **k3:** `(p/q - 1) - log(p/q)` (or equivalently `r - 1 - log(r)` with r = p/q) — unbiased, non-negative, typically lowest-variance in the regime relevant to RL.
- Empirical comparison on Gaussian distributions.
- Python snippet to reproduce.

## Key Figures/Tables to Study
- **Bias vs variance scatter** across k1/k2/k3.
- **Formula comparison table** for each estimator.
- **Gaussian validation plot** showing estimator behavior as policies diverge.

## Technical Details

For sample `x ~ q` and ratio `r = p(x) / q(x)`:

| Estimator | Formula | Unbiased for KL(q,p)? | Sign | Notes |
|-----------|---------|-----------------------|------|-------|
| k1 | `-log r` | Yes | any (can be < 0) | High variance near p ≠ q |
| k2 | `0.5 * (log r)^2` | No | >= 0 | Lowest bias only near p ~ q |
| k3 | `r - 1 - log r` | Yes | >= 0 always | Preferred for RLHF |

**Why k3 is non-negative:** `f(r) = r - 1 - log(r)` is a convex function with minimum 0 at r=1; since r > 0, f(r) >= 0.

**RLHF application:** In PPO/GRPO, the KL-to-reference penalty per token is `beta * KL_estimate`. Using k3 prevents the penalty from oscillating sign (which k1 suffers from), and keeps gradients well-behaved when the new policy is close to reference (which is most of training after the first few steps).

**Caveat from practitioners:** Costa Huang noted on X that the k3 estimator "exploded for some reason" in early TRL experiments, likely due to large r in the tails. GRPO in DeepSeekMath adopts k3 successfully — the regime where policy and reference stay close is what matters.

**Gradient subtlety:** DeepSeek's GRPO formulation treats the KL penalty as a loss term (not a reward), so the gradient flows through `log(pi_theta/pi_ref)` correctly. Using k3 as an *objective* rather than as a reward augmentation is the modern convention.

## Connections
- [[ppo]] — KL-to-reference is injected into the PPO reward stream or loss; k3 is the preferred estimator.
- [[grpo]] + [[deepseekmath]] — DeepSeek adopts k3 explicitly in the GRPO loss term.
- [[dpo]] — DPO derives an exact KL-constrained solution that sidesteps the MC estimation problem entirely.
- [[costa-huang-ppo-details]] — covers the wider PPO-for-RLHF implementation landscape.
- [[entropy-regularization-ppo]] — entropy bonus is a different regularizer with related variance concerns.

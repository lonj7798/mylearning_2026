<!-- scope: John Schulman's "Approximating KL Divergence" — the k1/k2/k3 Monte-Carlo KL estimators
     deps: [[README]]
     see-also: [[ppo]], [[grpo]], [[deepseekmath]], [[costa-huang-ppo-details]]
-->

# Approximating KL Divergence
- **Core Insight:** For samples `x ~ q` and `r = p(x)/q(x)`, the estimator `k3 = (r - 1) - log r` is an unbiased estimator of `KL[q, p]` that is always non-negative; in the post's Gaussian test with true KL = 0.5 it has stdev/true = 1.7 against 2 for the naive `k1 = -log r`.
- **Guideline:** When KL to a reference distribution must be estimated from samples and only log-probabilities are stored, use `k3`, because it is unbiased, non-negative, and had lower measured standard deviation than `k1` in both Gaussian settings reported in the post.
- **Authors:** John Schulman
- **Year:** 2020 (posted 2020-03-07)
- **URL:** http://joschu.net/blog/kl-approx.html
- **Source type:** practitioner evidence (personal blog post with a small Monte-Carlo experiment and code)
- **Relevant topics:** KL divergence, Monte-Carlo estimation, variance reduction, control variates, f-divergence, Bregman divergence

## Summary
The post treats the case where `p(x)` and `q(x)` can be evaluated pointwise but the sum over `x` cannot be
computed analytically, so `KL[q, p] = E_{x~q}[log(q(x)/p(x))]` must be estimated from samples of `q`. Three
reasons are given for this situation: the exact sum costs too much computation or memory, there is no closed
form, and storing only the log-probability rather than the full distribution simplifies code — which the post
says is reasonable when KL is used as a diagnostic, "as is often the case in reinforcement learning". The naive
estimator `k1 = -log r` is unbiased but high-variance, because it is negative for half of the samples while KL
is always positive. The estimator `k2 = ½(log r)²` is biased but low-variance. Adding the control variate
`r - 1`, which has zero expectation under `q`, with coefficient λ = 1 gives `k3 = (r - 1) - log r`, which is
unbiased and non-negative because `log(x) ≤ x - 1`.

## Key Contributions
- Names three estimators of `KL[q, p]` from samples of `q` and states their bias and sign properties.
- Explains the low bias of `k2` through f-divergences: `D_f(p, q) = E_{x~q}[f(r)]` for convex `f`, and all
  f-divergences with differentiable `f` agree with KL to second order near `q = p`, with
  `D_f(p_0, p_θ) = (f''(1)/2) θᵀ F θ + O(θ³)` where `F` is the Fisher information matrix of `p_θ` at `p_0`.
  `k2` corresponds to `f(x) = ½(log x)²` and KL to `f(x) = -log x`; both have `f''(1) = 1`.
- Derives `k3` as the control-variate estimator `-log r + λ(r - 1)` with λ = 1, and identifies the construction
  as a Bregman divergence: the vertical distance between a convex function and its tangent at `r = 1`.
- Generalizes the construction to any f-divergence as `f(r) - f'(1)(r - 1)`, which for the reversed direction
  `KL[p, q]` (`f(x) = x log x`, `f'(1) = 1`) gives the estimator `r log r - (r - 1)`.
- Reports a two-setting Gaussian comparison of bias and standard deviation, with the 11-line PyTorch script
  used to produce it.

## Key Figures/Tables to Study
- The two bias/standard-deviation tables (one per Gaussian setting). The post contains no plots or figures.
- The PyTorch snippet at the end, which defines `logr = p.log_prob(x) - q.log_prob(x)`,
  `k1 = -logr`, `k2 = logr**2 / 2`, `k3 = (logr.exp() - 1) - logr` over 10,000,000 samples.

## Technical Details
Definitions used throughout: `x ~ q`, `r = p(x)/q(x)`, and the target is `KL[q, p]` unless stated otherwise.

| Estimator | Formula | Bias for `KL[q, p]` | Sign | Note from the post |
|---|---|---|---|---|
| k1 | `-log r` | unbiased | can be negative | High variance; negative for half the samples (body) |
| k2 | `½ (log r)²` | biased | ≥ 0 | Expectation is an f-divergence; agrees with KL to second order near `q = p` (body) |
| k3 | `(r - 1) - log r` | unbiased | ≥ 0 | Control variate with λ = 1; non-negative because `log x ≤ x - 1` (body) |

Measured bias and standard deviation, each divided by the true KL, over 10,000,000 samples:

| Setting | True KL | k1 bias / stdev | k2 bias / stdev | k3 bias / stdev |
|---|---|---|---|---|
| `q = N(0,1)`, `p = N(0.1,1)` (first table) | 0.005 | 0 / 20 | 0.002 / 1.42 | 0 / 1.42 |
| `q = N(0,1)`, `p = N(1,1)` (second table) | 0.5 | 0 / 2 | 0.25 / 1.73 | 0 / 1.7 |

The post states about the second setting: "k3 has even lower standard deviation than k2 while being unbiased,
so it appears to be a strictly better estimator" (body, after the second table). About the first it notes the
bias of `k2` is 0.2%.

The optimal control-variate coefficient λ is not computed: the post says minimizing the variance of
`-log r + λ(r - 1)` yields an expression that depends on `p` and `q` and "is hard to calculate analytically",
so λ = 1 is chosen because it makes the estimator non-negative (body).

Note on the source text: the prose introduces the setting as "q=N(0,1), p=N(0.1,1)", while the posted code
assigns `p = dis.Normal(loc=0, scale=1)` and `q = dis.Normal(loc=0.1, scale=1)`. The labels are swapped
between prose and code; the reported numbers are the ones in the tables above.

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
The post reports none of these. It is a statistics note and does not run any language-model experiment, does
not discuss policy-gradient gradients, and does not make a recommendation about RLHF hyperparameters.

## Connections
- [[ppo]] — PPO uses a KL term to a reference policy; this post supplies the estimator, not the PPO usage.
- [[grpo]] and [[deepseekmath]] — the unbiased non-negative estimator `(r - 1) - log r` appears in the GRPO
  objective; the claim that GRPO adopts it belongs to those cards, not to this post.
- [[dpo]] — DPO uses a closed-form KL-constrained optimum and therefore does not estimate KL from samples.
- [[costa-huang-ppo-details]] — implementation-level treatment of PPO for RLHF.
- [[entropy-regularization-ppo]] — a different regularizer with its own estimation variance.

## Verification
- Checked on 2026-09-18 against: http://joschu.net/blog/kl-approx.html (post dated 2020/03/07).
- Corrections to the previous card version:
  - "Year: 2020 (original post); widely cited 2023–2025 in RLHF/GRPO implementations" → posting date is
    2020-03-07; the citation claim is not in the post and is removed (header).
  - "k3 ... usually lower variance than k1 when policies are close" → the post reports k3 stdev/true of 1.42
    vs 20 for k1 at true KL 0.005 and 1.7 vs 2 at true KL 0.5 (the two bias/stdev tables).
  - "k1 ... High variance near p ≠ q" → the stated reason is that k1 is negative for half of the samples
    while KL is positive (body, k1 paragraph).
  - "k2 ... Lowest bias only near p ~ q" → k2's bias is 0.2% of true KL at true KL 0.005 and 25% at true
    KL 0.5 (the two tables).
  - "Empirical comparison on Gaussian distributions" plus "Bias vs variance scatter" and "Gaussian validation
    plot" → the post has two numeric tables and one code block; it contains no plots (body).
  - Title in the header was "Approximating KL Divergence (John Schulman blog)" → exact published title is
    "Approximating KL Divergence".
- Removed as unsupported by the source:
  - "gives gradient signal close to the true KL near p ~ q" and "keeps gradients well-behaved" — the post
    does not discuss gradients of the estimators.
  - "Using k3 prevents the penalty from oscillating sign (which k1 suffers from)" as an RLHF result — the
    post makes no RLHF measurement.
  - "Costa Huang noted on X that the k3 estimator 'exploded for some reason' in early TRL experiments" —
    not in the post.
  - "GRPO in DeepSeekMath adopts k3 successfully" and the paragraph on DeepSeek treating KL as a loss term
    rather than a reward — claims about another artifact; kept only as a pointer under Connections.
  - "Python snippet to reproduce" described as a contribution alongside claims not in the post — the snippet
    is retained above with its exact contents.
- Not reported by the source: any language-model or RL experiment, any KL coefficient value, guidance on
  which estimator to use inside a training objective as opposed to a diagnostic.

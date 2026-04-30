<!-- scope: scaling laws for Goodhart's law on learned reward models during RLHF
     deps: [[kl-control-rlhf]]
     see-also: [[reward-hacking-taxonomy]], [[lilianweng-reward-hacking]], [[rlvr-tulu3]]
-->

# Scaling Laws for Reward Model Overoptimization
- **Core Insight:** Proxy-reward growth and gold-reward growth diverge as a predictable function of the KL budget `d = sqrt(KL(π‖π_ref))`; gold reward rises, peaks, and then falls, and the peak/decay coefficients scale smoothly with RM size and data.
- **Guideline:** Treat the KL-from-reference as your optimization budget, not a regularizer — monitor gold reward (or a held-out eval) vs KL and stop RL training before the predicted peak; larger RMs give you more budget but do not remove the peak.
- **Authors:** Leo Gao, John Schulman, Jacob Hilton
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2210.10760
- **Relevant topics:** Goodhart's law in RLHF, reward hacking, KL budget, best-of-n vs PPO, scaling laws, proxy vs gold reward

## Abstract
The authors construct a synthetic-preferences setup: a fixed 6B "gold" reward model generates preferences, on which a smaller "proxy" RM is trained, and then a policy is optimized against the proxy with either best-of-n sampling or PPO. By measuring gold reward vs the KL divergence from the SFT reference, they fit scaling laws describing how overoptimization grows with RM size, RM-data size, policy size, and KL penalty.

## Key Contributions
- **Functional forms:** For best-of-n, `R_gold(d) ≈ d · (α_bon − β_bon · d)`; for RL, `R_gold(d) ≈ d · (α_RL − β_RL · log d)`. Proxy reward grows monotonically; gold reward follows an inverted-U.
- **Scaling with RM size:** both α and β shrink smoothly with RM parameters — a 10× larger RM roughly halves the overoptimization slope but does not eliminate the hump.
- **KL = sqrt(KL) as natural x-axis:** plotting against `d = sqrt(KL)` linearizes many of the relationships, consistent with KL being a squared-distance metric locally.
- **RL vs best-of-n:** best-of-n has a tighter gold-reward peak and then drops faster; PPO's logarithmic decay means its overoptimization accumulates more slowly but continues indefinitely.
- **Policy size barely matters:** bigger policies optimize the proxy faster but hit the same gold peak — this is a property of the RM, not the policy.
- **KL penalty β is not a free lunch:** varying β in PPO traces out essentially the same front as early-stopping, up to small differences.

## Key Figures/Tables to Study
- **Fig. 1 / 2** (proxy vs gold vs d = sqrt(KL)): canonical Goodhart curves — memorize this shape.
- **Fig. 5** (RM parameter scaling): α/β coefficients vs RM size on a log-log plot.
- **Fig. 7** (best-of-n vs PPO front): best-of-n and PPO Pareto fronts on the (KL, gold-reward) plane.
- **Fig. 8** (dataset size scaling): returns to more preference data diminish before returns to more RM params.

## Technical Details
- **Setup:** 6B gold RM (fixed, treated as ground truth); proxy RMs at 3M / 12M / 25M / ... up to 3B params; SFT policy of 1.2B; sampling / PPO on TL;DR-like tasks.
- **KL budget metric:** `d = sqrt(KL(π ‖ π_SFT))`; KL is forward, token-averaged, and measured against the SFT reference used to initialize both the policy and the RM.
- **Best-of-n KL:** `KL_bon(n) = log n − (n−1)/n` — derived analytically, matches observed curves.
- **PPO objective:** standard clipped surrogate plus per-token `− β · KL(π ‖ π_SFT)` added to the reward (not to the loss); β tuned but absorbed into the d axis.
- **Why Goodhart:** the proxy RM has bounded generalization on OOD policy samples; as the policy drifts (d grows), proxy error accumulates and proxy ranking no longer reflects gold ranking.
- **Empirical number:** at RM size 3M, gold reward peaks near `d ≈ 3` nats^0.5 and loses most of the gain by `d ≈ 8`; larger RMs shift the peak right.

## Connections
- Single clearest empirical demonstration of Goodhart's law in RLHF; underpins all later work on **[[reward-hacking-taxonomy]]** and RM ensembles.
- Motivates **[[reward-ensembling]]** and **[[generative-reward-models]]** (try to reduce proxy error, push peak further right).
- Motivates **[[rlvr-tulu3]]** and **[[deepseek-r1]]** — if the verifier is exact, the proxy/gold gap collapses.
- Directly connects to **[[kl-control-rlhf]]** (the KL penalty IS the budget knob).
- Blog treatment: **[[lilianweng-reward-hacking]]**.

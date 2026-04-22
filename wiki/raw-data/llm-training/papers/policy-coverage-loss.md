<!-- scope: RLHF theory — policy coverability and transfer from imperfect reward models
     deps: [[on-off-policy-rlhf]], [[dpo]]
     see-also: [[ipo]], [[generative-reward-models]]
-->

# Can RLHF Be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective
- **Core Insight:** In KL-regularized RLHF, a source reward model can still be useful even when it is imperfect, as long as the policy it induces covers the parts of action space needed by the target optimum; policy coverage, not reward fidelity alone, determines transfer value.
- **Guideline:** If you have several imperfect reward models or preference policies, do not discard them just because they are noisy. Evaluate which one best covers the target behavior distribution, then transfer from that source rather than starting alignment from scratch.
- **Authors:** Jiajin Zhang, Renshuai Tao, Yuhao Zhang, Zhiqi Shen, Peng Dai, Liyuan Liu, Yali Du, Yan Wang, Han Liu, Weinan Zhang
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2502.19255
- **Relevant topics:** RLHF theory, reward-model transfer, coverability, KL-regularized optimization

## Abstract
The paper studies whether RLHF can reuse imperfect reward models efficiently instead of retraining preference supervision from zero for each new domain. Its answer is yes, but only under a coverage condition: a source reward model is valuable when the source policy induced by that reward model overlaps with the action regions needed by the target-optimal policy. The work formalizes this transfer condition, derives suboptimality bounds from a policy-coverage perspective, and proposes a transfer strategy that plugs into existing preference-optimization methods.

## Key Contributions
- Formalizes **policy coverability** as the right lens for transfer in KL-regularized RLHF.
- Shows theoretically that reward-model transfer depends on induced-policy overlap, not just pointwise reward accuracy.
- Proposes a transfer policy optimization strategy that selects useful source policies more efficiently via win-rate-based comparisons.
- Demonstrates that the approach can be integrated with several RLHF objectives rather than requiring a bespoke optimizer.

## Key Figures/Tables to Study
- **Theory section with the transfer bound:** this is the main reason the page exists; the bound explains why some imperfect source reward models still help.
- **Source-policy selection ablation:** study how win-rate-based source selection changes transfer quality.
- **Method comparison table:** compare transfer under DPO / IPO / related objectives with and without coverage-aware initialization.

## Technical Details
- **Setting:** KL-regularized RLHF with a reference policy and a target preference or reward signal.
- **Main claim:** if a source reward model induces a policy whose support overlaps the target-optimal policy well enough, then transfer can reduce sample complexity even when the reward model itself is biased.
- **Operational proxy:** the paper uses **policy win rate** as a practical selection signal for deciding which source policy to transfer from.
- **Why this matters:** reward models are expensive to collect and calibrate. Coverage-aware reuse gives a way to mine value from older or imperfect preference pipelines.
- **Algorithmic contribution:** the transfer method is modular and can be combined with common preference-optimization recipes such as DPO- or IPO-style updates.
- **Connection to the filename:** the core object is not a standalone "coverage loss" layer in the deep-learning sense; it is a coverage-based transfer criterion for RLHF objectives.

## Risks + gotchas
- A high-quality but badly covered source policy can still transfer poorly.
- Coverage is a distributional property; naive offline metrics can misestimate it.
- The theory lives inside KL-regularized RLHF assumptions, so transfer behavior can differ in weakly regularized or fully online settings.

## Connections
- Extends the distribution-shift lesson of [[on-off-policy-rlhf]] from offline-vs-online training to cross-domain reward-model reuse.
- Fits naturally with [[dpo]] and [[ipo]] because both can consume source policies or preference signals.
- Complements [[generative-reward-models]] by asking not only whether a reward model is expressive enough, but whether it induces useful policy support.

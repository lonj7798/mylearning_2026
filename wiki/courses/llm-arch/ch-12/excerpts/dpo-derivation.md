# DPO Derivation and Variants

<!-- scope: complete mathematical derivation of DPO and survey of variants
     parent: [[ch-12]]
-->

## The Complete DPO Derivation

This excerpt provides the full mathematical chain from the RLHF objective to the DPO loss, showing exactly where the reward model disappears and why the resulting loss is a simple classification problem.

---

## Step 1: The KL-Constrained Objective

Start with the standard RLHF objective. We want to maximize expected reward while constraining deviation from the reference policy:

$$\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{E}_{y \sim \pi(\cdot|x)}\left[r(x, y)\right] - \beta \, D_{\text{KL}}\!\left(\pi(\cdot|x) \;\|\; \pi_{\text{ref}}(\cdot|x)\right)\right]$$

Expanding the KL divergence:

$$= \max_{\pi} \; \mathbb{E}_{x}\left[\mathbb{E}_{y \sim \pi}\left[r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]\right]$$

## Step 2: Solving for the Optimal Policy

This is a constrained optimization problem over the space of probability distributions. Using the calculus of variations (or recognizing the form of a Gibbs distribution), the optimal policy is:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)$ is the partition function ensuring normalization. This is the **Boltzmann distribution** with the reference policy as the base measure and $r/\beta$ as the energy.

## Step 3: Reparameterizing the Reward

Taking the log of both sides and rearranging:

$$\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x, y) - \log Z(x)$$

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

This is the key equation: **the reward is fully determined by the log-ratio of the optimal policy to the reference policy**, plus a prompt-dependent constant. The partition function $Z(x)$ depends only on $x$, not on $y$.

## Step 4: Substituting into Bradley-Terry

The Bradley-Terry model for human preferences says:

$$P(y_w \succ y_l | x) = \sigma\!\left(r(x, y_w) - r(x, y_l)\right)$$

Substituting our reparameterized reward:

$$P(y_w \succ y_l | x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \beta \log Z(x) - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \beta \log Z(x)\right)$$

The partition function terms cancel:

$$P(y_w \succ y_l | x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

## Step 5: The DPO Loss

Replacing the theoretical optimal policy $\pi^*$ with our parameterized policy $\pi_\theta$ and maximizing the log-likelihood of the observed preferences gives the DPO loss:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

This is binary cross-entropy where the "logit" is the difference in log-ratios between the preferred and dispreferred responses. No reward model appears anywhere in this loss.

---

## Gradient Analysis

The gradient of the DPO loss reveals what the optimizer is actually doing:

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \, \mathbb{E}\left[\underbrace{\sigma\!\left(\hat{r}_l - \hat{r}_w\right)}_{\text{weight: higher when model is wrong}} \left[\underbrace{\nabla_\theta \log \pi_\theta(y_w|x)}_{\text{increase preferred}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l|x)}_{\text{decrease dispreferred}}\right]\right]$$

where $\hat{r} = \beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$ is the implicit reward. The sigmoid weighting term $\sigma(\hat{r}_l - \hat{r}_w)$ acts as an **adaptive importance weight**: examples where the model currently assigns higher implicit reward to the *wrong* response get larger gradient updates. This is why DPO is self-correcting -- it focuses training signal on its own mistakes.

---

## DPO Variants

### IPO (Identity Preference Optimization)

Azar et al. (2023) observed that DPO can overfit to preference data because the Bradley-Terry model assumes preferences are *deterministic* given the reward difference. IPO adds a regularization term:

$$\mathcal{L}_{\text{IPO}} = \mathbb{E}\left[\left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]$$

This targets a *specific margin* rather than pushing the log-ratio difference to infinity.

### KTO (Kahneman-Tversky Optimization)

Ethayarajh et al. (2024) extended DPO to work with **unpaired** preference data -- you only need binary labels (good/bad) per response, not matched pairs:

$$\mathcal{L}_{\text{KTO}} = \mathbb{E}_{y_w}\left[1 - \sigma\!\left(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - z_{\text{ref}}\right)\right] + \mathbb{E}_{y_l}\left[1 - \sigma\!\left(z_{\text{ref}} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right)\right]$$

This is motivated by Kahneman & Tversky's prospect theory: humans weight losses more heavily than gains.

### ORPO (Odds Ratio Preference Optimization)

Hong et al. (2024) eliminated the need for a separate reference model by incorporating the preference signal into the SFT loss itself, using odds ratios instead of log-probability ratios. This reduces the pipeline to a single training stage.

---

## When to Use DPO vs RLHF

| Scenario | Recommendation | Reasoning |
|----------|---------------|-----------|
| Limited compute budget | DPO | 2 models vs 4; no on-policy sampling |
| Small preference dataset | DPO | More sample-efficient; no reward model overfitting |
| Large-scale production | Online DPO or RLHF | On-policy exploration finds better responses |
| Safety-critical alignment | RLHF + reward model | Separate reward model enables monitoring and auditing |
| Reasoning tasks | GRPO (RL with rule-based rewards) | Verifiable correctness provides clean reward signal |

The trend in 2025-2026 is toward **iterative/online DPO**: run DPO, regenerate preference data from the updated policy, repeat. This captures most of RLHF's on-policy benefit while retaining DPO's simplicity.

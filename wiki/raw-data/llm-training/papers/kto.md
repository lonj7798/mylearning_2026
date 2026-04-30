<!-- scope: Kahneman-Tversky Optimization — prospect-theory value function on binary desirable/undesirable labels
     deps: [[dpo]]
     see-also: [[simpo]], [[orpo]], [[ipo]]
-->

# KTO: Model Alignment as Prospect Theoretic Optimization
- **Core Insight:** Preference datasets require paired (y_w, y_l) per prompt; human feedback in production is usually binary thumbs-up/down — a prospect-theory utility (concave for gains, convex+steeper for losses) turns that binary signal into a loss that matches or beats DPO without needing pairs.
- **Guideline:** Use KTO when you have unpaired binary labels or imbalanced desirable/undesirable counts; set λ_D and λ_U to balance label frequency (λ_D · N_D ≈ λ_U · N_U) and start at β=0.1.
- **Authors:** Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.01306
- **Relevant topics:** binary feedback alignment, prospect theory, HALO family, loss aversion, unpaired preferences

## Abstract
Kahneman & Tversky's prospect theory tells us that humans perceive random variables in a biased but well-defined manner; for example, they are famously loss-averse. We show that objectives for aligning LLMs with human feedback implicitly incorporate many of these biases — the success of these objectives (e.g., DPO) over cross-entropy cannot be fully explained by preference data alone. We propose a family of human-aware loss functions (HALOs) and use it to derive KTO, an objective that optimizes the Kahneman–Tversky utility of LM outputs using only binary desirable/undesirable signals. KTO matches or exceeds DPO at scales from 1B to 30B.

## Key Contributions
- Defines **HALOs** — objectives based on a prospect-theory value function (concave gain side, steeper convex loss side).
- Shows DPO, PPO-Clip are HALOs → theoretical justification.
- Derives **KTO** from Kahneman–Tversky utility; requires only a per-example binary label, not pairs.
- Robust to label imbalance via λ_D, λ_U weights.

## Key Figures/Tables to Study
- **Figure 1:** Prospect theory value function vs log-sigmoid — visual motivation.
- **Figure 3 / Table 1:** KTO vs DPO vs SFT on GPT-4-judged win rate from 1B to 30B.
- **Section 4 / KTO loss equation.**

## Technical Details

### Implicit reward (same as DPO)
`r_θ(x, y) = β · log [ π_θ(y|x) / π_ref(y|x) ]`

### Reference point
`z_0 = KL( π_θ(y'|x) || π_ref(y'|x) )`
Estimated on-the-fly with a batch-level moving statistic; acts as the "status quo" against which gains/losses are measured. No gradients flow through z_0.

### Value function (prospect-theory inspired)
`v(x, y) = { λ_D · σ( β ( r_θ(x,y) − z_0 ) )      if y is desirable`
`          { λ_U · σ( β ( z_0 − r_θ(x,y) ) )      if y is undesirable }`
σ is the sigmoid; the asymmetry between desirable (gain side) and undesirable (loss side) is encoded via λ_D vs λ_U.

### KTO loss
`L_KTO(π_θ; π_ref) = E_{(x,y)~D}[ λ_y − v(x, y) ]`
where λ_y ∈ {λ_D, λ_U} depending on the label of (x,y).

### Hyperparameters (paper)
| Knob | Value |
|------|-------|
| β | 0.1 (main) |
| λ_D | 1.0 |
| λ_U | 1.0 (tune to class balance) |
| Learning rate | 5e-7 (AdamW) |
| Batch size | 32 examples |
| Epochs | 1 |
| π_ref | SFT checkpoint, frozen |
| z_0 estimation | over recent minibatch, no grad |

Recipe for imbalanced data: set `λ_D / λ_U = N_U / N_D`.

### Behavior
- KTO recovers DPO's chosen/rejected dynamics when labels are paired.
- Superior under extreme class imbalance (e.g., 90% desirable).
- Less sensitive to β than DPO.

## Connections
- Logit-link ancestor: [[dpo]].
- Squared-link cousin: [[ipo]].
- Length-normalized ref-free variant: [[simpo]].
- Joint SFT+preference: [[orpo]].

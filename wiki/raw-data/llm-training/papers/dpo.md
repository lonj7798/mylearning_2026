<!-- scope: Direct Preference Optimization — closed-form RLHF via classification on preferences
     deps: [[rlhf-instructgpt]], [[bradley-terry-rm]]
     see-also: [[ipo]], [[simpo]], [[orpo]], [[kto]]
-->

# Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- **Core Insight:** Under the Bradley-Terry model, the optimal RLHF policy has a closed-form relationship to a reward r(x,y)=β log π/π_ref; substituting this into the preference likelihood eliminates the reward model and converts RLHF into a simple binary-classification loss.
- **Guideline:** Use DPO as the default for offline preference datasets; start at β=0.1, keep π_ref frozen as the SFT checkpoint, and watch for reward hacking via length inflation (add length normalization or switch to SimPO if needed).
- **Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2305.18290
- **Relevant topics:** preference optimization, RLHF alternative, closed-form policy, offline RL, Bradley-Terry

## Abstract
Existing RLHF pipelines fit a reward model to human preferences and then use RL (PPO) to optimize the language model against the reward. DPO leverages a mapping between reward functions and optimal policies to show that this two-stage pipeline can be solved exactly with a simple classification loss. The resulting algorithm is stable, performant, and computationally lightweight, removing the need to fit a separate reward model, sample from the LM during fine-tuning, or perform significant hyperparameter tuning. DPO matches or exceeds PPO on summarization, dialogue, and sentiment control.

## Key Contributions
- Derives a closed-form relation `π*(y|x) ∝ π_ref(y|x) exp(r(x,y)/β)` and inverts it to express r in terms of π.
- Substitutes that reward into the Bradley-Terry preference likelihood → a single classification loss.
- Eliminates reward model, rollout sampling, and PPO entirely — one forward/backward per preference pair.
- Shows that β controls the KL budget between trained policy and reference policy.
- Provides theoretical analysis connecting DPO to actor-critic methods.

## Key Figures/Tables to Study
- **Figure 2:** Summarization win rates vs reference (Reddit TL;DR) — DPO beats PPO at every sampling temperature.
- **Figure 3:** Anthropic HH dialogue — DPO dominates chosen-over-rejected win rates.
- **Section 4 / Equation 7:** The DPO loss — the single most-implemented equation in modern alignment.

## Technical Details

### Derivation bridge from PPO-RLHF
KL-regularized objective (same as PPO-RLHF):
`max_π E_{x~D, y~π}[r(x,y)] − β D_KL(π(·|x) || π_ref(·|x))`
Optimal policy (standard result):
`π_r(y|x) = (1/Z(x)) π_ref(y|x) exp(r(x,y)/β)`
Invert for the reward:
`r(x,y) = β log [π_r(y|x) / π_ref(y|x)] + β log Z(x)`
Under Bradley-Terry, `P(y_w ≻ y_l | x) = σ(r(x,y_w) − r(x,y_l))`; Z(x) cancels in the difference.

### DPO loss (Equation 7)
`L_DPO(π_θ; π_ref) = −E_{(x,y_w,y_l)~D} [ log σ( β log π_θ(y_w|x)/π_ref(y_w|x) − β log π_θ(y_l|x)/π_ref(y_l|x) ) ]`
- `y_w` = preferred / chosen response; `y_l` = rejected.
- `π_θ` = policy being trained; `π_ref` = frozen SFT reference.
- `β` = temperature controlling KL budget; larger β → closer to π_ref.
- `σ` = logistic sigmoid.

### Implicit reward
`r̂_θ(x,y) = β log [π_θ(y|x) / π_ref(y|x)]`
Used at eval time as a learned reward model or for gating (BoN, rejection sampling, online iterations).

### Gradient form
`∇L_DPO = −β E[ σ(r̂_l − r̂_w) ( ∇log π_θ(y_w|x) − ∇log π_θ(y_l|x) ) ]`
The σ(·) term is an automatic weighting: samples that already satisfy the preference get near-zero gradient; violations get full weight.

### Hyperparameters
| Knob | Typical |
|------|---------|
| β | 0.01–0.5 (paper sweeps {0.05, 0.1, 1, 5}; most recipes use 0.1) |
| Learning rate | 5e-7 to 1e-6 (much lower than SFT) |
| Batch size | 32–128 (in pairs) |
| Epochs | 1–3 |
| π_ref | SFT checkpoint, frozen |
| Length normalization | off (known failure mode — see SimPO) |

## Connections
- Classical RLHF this replaces: [[rlhf-instructgpt]], [[ppo]].
- Identity variant that drops the logistic (mitigates over-fitting): [[ipo]].
- Reference-free successors: [[simpo]], [[orpo]].
- Binary-feedback variant using prospect theory: [[kto]].
- Iterative / online version: [[rpo]], self-rewarding LM.
- Framework implementations: [[openrlhf-dpo]], plus TRL implementations summarized in [[hf-dpo-zoo]].

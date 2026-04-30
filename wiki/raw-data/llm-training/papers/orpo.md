<!-- scope: ORPO — single-stage SFT + odds-ratio preference loss, reference-free, one training pass
     deps: [[dpo]], [[simpo]]
     see-also: [[kto]], [[ipo]], [[hf-alignment-handbook]]
-->

# ORPO: Monolithic Preference Optimization without Reference Model
- **Core Insight:** SFT-only fine-tuning also increases the probability of *disfavored* responses; adding a small odds-ratio penalty term during SFT suppresses disfavored generations with no extra stage, no reference model, and no separate preference training.
- **Guideline:** Use ORPO when you want a one-pass recipe from base model → aligned model with a preference dataset; set λ≈0.1–0.25 to avoid over-dominance of the odds-ratio loss over the CE loss.
- **Authors:** Jiwoo Hong, Noah Lee, James Thorne
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2403.07691
- **Relevant topics:** single-stage alignment, reference-free, odds ratio, combined SFT + preference loss

## Abstract
Conventional alignment pipelines chain SFT → DPO/PPO. We demonstrate that SFT inherently increases the probability of undesired generations alongside desired ones. We propose ORPO, which integrates a small odds-ratio penalty term into the SFT loss — a single training stage that jointly teaches desired behavior while suppressing undesired behavior, without a reference model. ORPO with Phi-2 / Llama-2-7B / Mistral-7B on UltraFeedback reaches AlpacaEval 2.0 up to 12.20%, IFEval 66.19%, and MT-Bench 7.32.

## Key Contributions
- Identifies that SFT alone inflates rejected-response log-prob — motivates a suppressive term.
- Introduces odds-ratio loss operating on (chosen, rejected) pairs without π_ref.
- Single-stage: standard causal-LM SFT + λ · L_OR.
- Released Mistral-ORPO-α / β checkpoints as the first major single-stage aligned open model.

## Key Figures/Tables to Study
- **Figure 3:** Log-prob of rejected responses during SFT (rises) vs during ORPO (stays flat) — the paper's main empirical motivation.
- **Section 3 / Equations 5–7:** odds ratio, L_OR, and final loss.

## Technical Details

### Odds definition
For conditional distribution π_θ(y|x):
`odds_θ(y | x) = π_θ(y|x) / (1 − π_θ(y|x))`
Note: π_θ(y|x) here is the sequence probability (product of token probs); in practice use length-normalized log-probability inside the sigmoid.

### Odds ratio
`OR_θ(y_w, y_l | x) = odds_θ(y_w | x) / odds_θ(y_l | x)`

### Odds-ratio loss (Equation 7)
`L_OR = − log σ( log [ odds_θ(y_w|x) / odds_θ(y_l|x) ] )`
Interpretation: cross-entropy on the binary event "chosen is preferred" under a sigmoid link of the log-odds-ratio.

### Total loss (Equation 6)
`L_ORPO = E_{(x, y_w, y_l)} [ L_SFT + λ · L_OR ]`
- `L_SFT = −log π_θ(y_w | x)` — standard causal-LM cross-entropy on the chosen response only.
- λ balances supervised signal with preference separation.

### Hyperparameters (paper)
| Knob | Value |
|------|-------|
| λ (Phi-2) | 0.25 |
| λ (Llama-2-7B) | 0.2 |
| λ (Mistral-7B) | 0.1 |
| Learning rate | 8e-6 (AdamW, cosine) |
| Batch size | 64 prompts |
| Epochs | 3–5 |
| π_ref | — (none) |
| Max length | 1024 |

### Why reference-free works here
L_SFT simultaneously anchors the chosen-side log-prob (pushes it up); the odds-ratio term only pulls the chosen-vs-rejected *gap* apart. Without L_SFT anchoring, a reference-free gap-maximizer would collapse; without L_OR, SFT alone fails to suppress rejected-response probability.

## Connections
- Two-stage alternative: SFT → [[dpo]].
- Other reference-free variant: [[simpo]].
- Theoretical family: [[ipo]] / KL-bounded objectives.
- Binary-label alternative: [[kto]].
- Handbook integration: [[hf-alignment-handbook]].

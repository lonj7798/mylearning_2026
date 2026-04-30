<!-- scope: Simple Preference Optimization — reference-free, length-normalized DPO variant
     deps: [[dpo]]
     see-also: [[orpo]], [[ipo]], [[kto]]
-->

# SimPO: Simple Preference Optimization with a Reference-Free Reward
- **Core Insight:** DPO's reference term adds memory and inference overhead and introduces a length bias; using the *average* log-prob per token as the implicit reward removes the reference model entirely and makes the objective length-invariant.
- **Guideline:** Use SimPO when (a) memory for π_ref is prohibitive, (b) your DPO run is inflating response length, or (c) offline data is clean; tune β in [2, 10] and γ in [0.3, 2.0] — much larger than DPO's β because the reward is now average-per-token.
- **Authors:** Yu Meng, Mengzhou Xia, Danqi Chen
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2405.14734
- **Relevant topics:** reference-free preference optimization, length bias, implicit reward, alignment

## Abstract
DPO's implicit reward is a log-ratio to a reference model. This (a) requires keeping π_ref in memory, (b) creates a mismatch between the training reward (sum of log-probs) and the generation objective (length-normalized), and (c) exhibits length bias. SimPO uses the *average* log-probability of the response as the implicit reward, removing π_ref and aligning training with generation. On Llama-3 8B Instruct and Mistral-7B Instruct, SimPO achieves up to +6.4 pts on AlpacaEval 2 and +7.5 pts on Arena-Hard over DPO.

## Key Contributions
- Reference-free implicit reward: `r(x,y) = (β/|y|) · log π_θ(y|x)`.
- Target reward margin γ inside the Bradley-Terry loss — forces a minimum gap between chosen and rejected.
- Length normalization aligns the training objective with the inference-time per-token log-prob.
- Removes π_ref from memory and forward pass — ~2× training throughput vs DPO.

## Key Figures/Tables to Study
- **Table 2:** AlpacaEval 2 / Arena-Hard win rates across Mistral-7B, Llama-3-8B, Gemma-2-9B.
- **Figure 4:** Length distribution comparison — DPO lengthens, SimPO holds length steady.
- **Section 4 / SimPO loss.**

## Technical Details

### Length-normalized reward
`r_SimPO(x, y) = (β / |y|) · Σ_{t=1..|y|} log π_θ(y_t | x, y_<t)`
= `β · (average log-probability per token)`.
No π_ref, no probability ratio.

### SimPO loss
`L_SimPO(π_θ) = − E_{(x, y_w, y_l)~D} [ log σ(  (β/|y_w|) log π_θ(y_w|x)  −  (β/|y_l|) log π_θ(y_l|x)  −  γ ) ]`
- γ = target reward margin (subtracted inside the sigmoid argument).
- σ = logistic sigmoid.
- Logit must exceed γ before the loss saturates.

### Hyperparameters (paper)
| Knob | Typical | Notes |
|------|---------|-------|
| β | 2.0–2.5 | ~20× DPO's because reward is per-token |
| γ | 0.3–1.6 | Larger γ = stricter margin; often sweep {0.5, 1.0, 1.4} |
| γ / β ratio | 0.25–0.5 | Recommended scaling rule |
| Learning rate | 3e-7 – 1e-6 | Lower than DPO |
| Batch size | 128 pairs |
| Epochs | 1 |
| π_ref | — (not used) |

### Length-bias analysis
DPO's reward `β log(π/π_ref)` grows linearly in |y| if the policy uniformly shifts log-probs; SimPO's average quantity is length-invariant by construction. Empirically, DPO increases response length by 30–60% over SFT; SimPO stays within ±5%.

### Failure modes
- β too low → under-regularized, policy collapses to high-entropy mode.
- γ too high → gradient vanishes (all pairs already satisfy margin), training stalls.
- Very clean / deterministic data → degenerate maximization of chosen log-prob; mitigate with label smoothing or add a small SFT loss (similar to ORPO).

## Connections
- Drops the reference from: [[dpo]].
- Margin-style cousin (different loss shape): [[ipo]].
- Joint SFT+preference ref-free: [[orpo]].
- Binary-feedback alternative: [[kto]].

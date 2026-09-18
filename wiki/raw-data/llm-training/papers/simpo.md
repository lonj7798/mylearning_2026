<!-- scope: SimPO — reference-free preference optimization with a length-normalized implicit reward and a target reward margin
     deps: [[dpo]]
     see-also: [[orpo]], [[ipo]], [[kto]], [[likelihood-displacement]]
-->

# SimPO: Simple Preference Optimization with a Reference-Free Reward
- **Core Insight:** Replacing DPO's log-ratio-to-reference reward with the policy's average log probability per token removes the reference model and matches the metric that guides generation; with a target reward margin γ added to the Bradley-Terry objective, this improves AlpacaEval 2 length-controlled win rate by up to 6.4 points and Arena-Hard win rate by up to 7.5 points over DPO across four training setups (Abstract, Table 4).
- **Guideline:** When running offline preference optimization on a strong SFT checkpoint and the reference model's memory or run time is a constraint, use SimPO with β between 2.0 and 2.5 and γ between 0.5 and 1.5, tuning the learning rate in [3e-7, 1e-6]; the paper reports that it holds no KL regularizer, that reward hacking is possible in principle without one, and that GSM8K declines in one or more settings for SimPO as for DPO, IPO, and R-DPO (§3, §4.4, App. B, App. C).
- **Authors:** Yu Meng, Mengzhou Xia, Danqi Chen
- **Year:** 2024 (arXiv v1 2024-05; v3 2024-11-01; NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2405.14734
- **Source type:** paper
- **Relevant topics:** reference-free preference optimization, length normalization, target reward margin, offline alignment

## Abstract
DPO reparameterizes the RLHF reward function to learn a policy directly from preference data. SimPO changes one design: the implicit reward becomes the average log probability of the sequence under the policy. This aligns the reward with the metric used at generation time and removes the reference model, reducing compute and memory. The authors additionally introduce a target reward margin in the Bradley-Terry objective, so the winning response's reward must exceed the losing one's by at least γ. SimPO is compared against DPO and its variants on Mistral, Llama 3, and Gemma 2, under both base and instruction-tuned setups, on AlpacaEval 2, MT-Bench, and Arena-Hard. It outperforms DPO by up to 6.4 points on AlpacaEval 2 and 7.5 points on Arena-Hard without substantially increasing response length. The best model, built on Gemma-2-9B-it, reaches a 72.4% length-controlled win rate on AlpacaEval 2 and 59.1% on Arena-Hard.

## Key Contributions
- Defines the reference-free length-normalized reward r(x, y) = (β/|y|) · log π_θ(y|x) (Eq. 4), stated as the reward that matches the average-log-likelihood metric used to rank generations (§2.2).
- Adds the target reward margin γ to the Bradley-Terry objective, so p(y_w ≻ y_l | x) = σ(r(x,y_w) − r(x,y_l) − γ) (Eq. 5, §2.3).
- Reports that under DPO training, only about 50% of training triples that satisfy the DPO reward ranking also satisfy the average-log-likelihood ranking p_θ(y_w|x) > p_θ(y_l|x) (§2.2, Figure 4b).
- Compares against RRHF, SLiC-HF, DPO, IPO, CPO, KTO, ORPO, and R-DPO with per-method hyperparameter search in four setups (Table 3, Table 4, App. B).
- Ablates both components separately and reports that removing length normalization is the larger loss (Table 5).

## Key Figures/Tables to Study
- **Table 4:** the main comparison, AlpacaEval 2 (LC and raw), Arena-Hard, and MT-Bench across Mistral-Base, Mistral-Instruct, Llama-3-Base, Llama-3-Instruct.
- **Table 5:** the ablation of length normalization and of γ = 0.
- **Figure 2:** reward margin against length difference, and the Spearman correlation between average log probability and response length with and without length normalization.
- **Figure 3:** reward accuracy and AlpacaEval 2 LC win rate against γ, plus the reward and log-probability distributions under different γ.
- **Figure 5:** KL divergence and win rate under different β, and the run time and peak GPU memory comparison against DPO.
- **Table 9:** Huggingface Open Leaderboard downstream results for every method.

## Technical Details

### Reward and objective
`p_θ(y|x) = (1/|y|) · Σ_{i=1..|y|} log π_θ(y_i | x, y_<i)` — the average log probability per token (Eq. 3).
`r_SimPO(x, y) = (β/|y|) · log π_θ(y|x)` — β is a constant scaling the reward difference; there is no reference model and no probability ratio (Eq. 4).
`L_SimPO(π_θ) = − E_{(x,y_w,y_l)~D} [ log σ( (β/|y_w|) log π_θ(y_w|x) − (β/|y_l|) log π_θ(y_l|x) − γ ) ]` — σ is the logistic sigmoid and γ > 0 the target reward margin (Eq. 6).
The paper notes that DPO already carries an implicit instance-level margin γ_ref = β log π_ref(y_w|x) − β log π_ref(y_l|x), which it gives as the reason why adding an explicit γ to DPO does not consistently help (App. I).

### Why the summed log probability is not used
Using the summed token log probability as the reward introduces length bias, because longer sequences have lower total log probability. When y_w is longer than y_l, the model must inflate probabilities for long sequences for y_w to win, which the authors state increases the risk of degeneration (§2.2).

### Ablation results (Mistral-Base / Mistral-Instruct, AlpacaEval 2 LC %)
SimPO 21.5 / 32.1; without length normalization 11.9 / 19.1; with γ = 0 16.8 / 30.9; DPO 15.1 / 26.8 (Table 5). Removing length normalization produced long, repetitive generations (§4.1, App. E).

### Effect of γ
Reward accuracy increases with γ, but the AlpacaEval 2 win rate first rises and then falls, so the largest margin is not the best one. Increasing γ flattens the distribution of the reward difference and lowers the average log probability of winning responses (§4.3, Figure 3).

### Efficiency
Measured in the Llama-3-Base setting on 8×H100 GPUs against a vanilla DPO implementation: SimPO cuts run time by roughly 20% and per-GPU peak memory by about 10% (§4.4, Figure 5c).

### KL divergence and regularization
SimPO applies no KL regularization, yet the measured KL divergence from the reference model on held-out winning responses stays small; increasing β reduces it for both DPO and SimPO. The authors attribute the absence of catastrophic forgetting to a small learning rate, a preference dataset covering diverse domains, and the robustness of LLMs to new data. They state that SimPO could in principle lead to reward hacking without explicit regularization (§2.3, §4.4).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Mistral-Base (7B) | 7B | preference (SimPO) | β; γ; learning rate | 2.0; 1.6; 3e-7 | arXiv:2405.14734v3 App. B Table 8 | verified (2026-09-18) | App. B: learning rate searched over [3e-7, 5e-7, 6e-7, 1e-6] per method |
| Mistral-Instruct (7B) | 7B | preference (SimPO) | β; γ; learning rate | 2.5; 0.3; 5e-7 | arXiv:2405.14734v3 App. B Table 8 | verified (2026-09-18) | App. B: per-method learning-rate search |
| Llama-3-Base (8B) | 8B | preference (SimPO) | β; γ; learning rate | 2.0; 1.0; 6e-7 | arXiv:2405.14734v3 App. B Table 8 | verified (2026-09-18) | App. B: per-method learning-rate search |
| Llama-3-Instruct (8B) | 8B | preference (SimPO) | β; γ; learning rate | 2.5; 1.4; 1e-6 | arXiv:2405.14734v3 App. B Table 8 | verified (2026-09-18) | App. B: per-method learning-rate search |
| All four SimPO setups | 7B, 8B | preference (SimPO) | batch size; epochs; max sequence length; schedule | 128; 1; 2048; cosine with 10% warmup | arXiv:2405.14734v3 App. B | verified (2026-09-18) | App. B: batch size searched in [32, 64, 128] and epochs in [1, 2, 3]; 128 and 1 epoch were best across all methods |
| Base-setting SFT models | 7B, 8B | SFT | data; learning rate; batch size; epochs; max sequence length; optimizer | UltraChat-200k; 2e-5; 128; 1; 2048; Adam | arXiv:2405.14734v3 App. B | verified (2026-09-18) | no ablation reported |
| DPO baselines | 7B, 8B | preference (DPO) | β search range | [0.01, 0.05, 0.1] | arXiv:2405.14734v3 Table 7 | verified (2026-09-18) | Table 7 gives the per-method search ranges used for the tuned baselines |
| SimPO | 7B, 8B | preference (SimPO) | general recommendation | β in [2.0, 2.5], γ in [0.5, 1.5] | arXiv:2405.14734v3 §3 | verified (2026-09-18) | §3: stated to give good performance across all four setups |
| Gemma-2-9B-it-SimPO | 9B | preference (SimPO) | β; γ; learning rate | not reported in the main text | §3, App. B checked (Table 8 covers four Mistral/Llama setups only) | not reported | — |

## Findings relevant to generality
- On the Huggingface Open Leaderboard tasks, MMLU is largely retained with a small decline for all preference-optimization methods, ARC and HellaSwag generally improve over the SFT checkpoint, and TruthfulQA improves consistently, in some cases by more than 10% (App. C, Table 9).
- GSM8K is the most volatile: except for ORPO, almost all methods including SimPO produce consistent drops in one or more settings. The authors hypothesize ORPO retains it because of its supervised fine-tuning term (App. C).
- Adding an SFT regularization term to SimPO lowers AlpacaEval 2 performance (SimPO v0.2 53.7 LC → 41.4 LC with SFT) while helping GSM8K, so the authors describe the effect as setting-dependent (App. H, Table 14).
- MT-Bench separates methods poorly; the authors attribute this to its evaluation scale and single-instance scoring, and recommend AlpacaEval 2 and Arena-Hard for comparison (§4.1).

## Findings relevant to negative feedback
- Negatives enter as the rejected term of the Bradley-Terry objective (negative as gradient). The paper's gradient analysis states that SimPO's gradients on both y_l and y_w are length-normalized while DPO's are not (App. F). The margin γ sets how far the rejected response must be pushed below the chosen one before the loss saturates; raising γ reduces the average log probability of the winning response as well (§4.3).

## Connections
- Removes the reference model from [[dpo]]; the loss is otherwise the same Bradley-Terry form.
- [[ipo]] also uses a target reward margin, but its full objective is compared and found less effective here (§2.3, Table 4).
- [[orpo]] is the other reference-free objective compared, and is the one whose SFT term keeps GSM8K stable (Table 4, App. C).
- [[kto]] is the binary-feedback baseline in the same comparison table.
- [[likelihood-displacement]] covers the drop in chosen-response likelihood that this paper measures as a function of γ.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2405.14734 (arXiv v3, 2024-11-01).
- Corrections to the previous card version:
  - "Removes π_ref from memory and forward pass — ~2× training throughput vs DPO" → run time is cut by roughly 20% and per-GPU peak memory by about 10%, measured in the Llama-3-Base setting on 8×H100 against a vanilla DPO implementation (§4.4, Figure 5c).
  - "tune β in [2, 10] and γ in [0.3, 2.0]" → the paper's stated recommendation is β in [2.0, 2.5] and γ in [0.5, 1.5]; the four released settings use β 2.0–2.5 and γ 0.3–1.6 (§3, App. B Table 8). β = 10 appears only as a point in the KL-divergence study (Figure 5).
  - "β ~20× DPO's" → DPO's β was searched over [0.01, 0.05, 0.1] and SimPO's is 2.0–2.5, so the ratio depends on the DPO β chosen (Table 7, Table 8).
  - "γ / β ratio 0.25–0.5 — Recommended scaling rule" → no such rule appears in the paper; the four settings give ratios of 0.80, 0.12, 0.50, and 0.56.
  - "Figure 4: Length distribution comparison — DPO lengthens, SimPO holds length steady" → Figure 4 is the DPO comparison (likelihood-length correlation, the contingency table, reward accuracy). Length normalization is studied in Figure 2, and the paper states SimPO generates responses up to 26% longer than DPO on AlpacaEval 2 in some settings and about 5% longer on Arena-Hard, while DPO's generations are comparable to or slightly shorter than SimPO's (App. E, footnote 10).
  - "On Llama-3 8B Instruct and Mistral-7B Instruct, SimPO achieves up to +6.4 pts on AlpacaEval 2 and +7.5 pts on Arena-Hard over DPO" → those two maxima are over all four setups, not specifically the two Instruct ones (Abstract, §1, Figure 1).
- Removed as unsupported by the source: "DPO increases response length by 30–60% over SFT; SimPO stays within ±5%"; "β too low → under-regularized, policy collapses to high-entropy mode" (the paper reports that a smaller β raises KL divergence and, in the Mistral-Base setting, improves AlpacaEval 2, and that it observed no training collapse with proper tuning); "γ too high → gradient vanishes (all pairs already satisfy margin), training stalls" (the reported effect of large γ is lower win rate, flatter reward distributions, and lower winning-response log probability); "Very clean / deterministic data → degenerate maximization of chosen log-prob; mitigate with label smoothing"; "Use SimPO when ... offline data is clean".
- Not reported by the source: the hyperparameters of the Gemma-2-9B-it-SimPO release, total training compute or GPU hours, and any comparison against PPO (explicitly left to future work, footnote 7).

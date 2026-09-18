<!-- scope: KTO — a human-aware loss (HALO) built on a Kahneman-Tversky value function that aligns a model from unpaired binary desirable/undesirable labels
     deps: [[dpo]]
     see-also: [[simpo]], [[orpo]], [[ipo]], [[ultrafeedback]], [[hh-rlhf]], [[hf-dpo-zoo]]
-->

# KTO: Model Alignment as Prospect Theoretic Optimization
- **Core Insight:** An alignment loss built from the Kahneman-Tversky value function and a per-example binary desirable/undesirable label matches or exceeds DPO from 1B to 30B parameters, and a Llama-7B run keeps outperforming DPO after up to 90% of the desirable examples are discarded (Abstract, §4.3, Figure 5).
- **Guideline:** When feedback arrives as unpaired binary labels or with a skewed desirable-to-undesirable ratio, use KTO with λ_D and λ_U chosen so that λ_D·n_D / (λ_U·n_U) lies in [1, 3/2], because that interval was the empirically best setting in the paper's imbalance experiments; keep λ_D = λ_U = 1 when the counts are balanced (§4.2, Eq. 9).
- **Authors:** Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela
- **Year:** 2024 (arXiv v1 2024-02; ICML 2024; v5 2026-09 is the version checked here)
- **URL:** https://arxiv.org/abs/2402.01306
- **Source type:** paper
- **Relevant topics:** human-aware losses (HALOs), prospect theory, loss aversion, unpaired binary feedback, class imbalance, reference-point estimation

## Abstract
The paper argues that objectives used to align language models with human feedback implicitly carry the biases described by Kahneman and Tversky's prospect theory, and that the advantage of objectives such as DPO over cross-entropy minimization can partly be attributed to their belonging to a family the authors call human-aware losses (HALOs). The utility functions these methods imply still differ from those in the prospect-theory literature, so the authors use a Kahneman-Tversky model of human utility to derive a HALO that maximizes the utility of generations directly instead of the log-likelihood of preferences. They call it KTO. It matches or exceeds preference-based methods at scales from 1B to 30B despite learning only from a binary signal of whether an output is desirable, and the authors conclude that no single HALO is universally best (Abstract).

## Key Contributions
- Defines **HALOs**: losses whose implied value function is concave in gains and convex, steeper, in losses relative to a reference point (§3, Figure 1).
- Classifies existing objectives: DPO and the PPO-Clip loss are HALOs; cross-entropy (CSFT) and SLiC are not (§3.2, Theorem 3.5).
- Shows an offline PPO variant trained with dummy +1/−1 rewards matches DPO for every model tested except Llama-30B, which motivates learning from a binary signal (§3.3, Figure 2).
- Derives **KTO**, which needs only a per-example binary label rather than a pair (§4.1, Eq. 8).
- Gives a class-imbalance rule for λ_D and λ_U and shows a Llama-7B model trained after discarding up to 90% of desirable examples still beats DPO (§4.2 Eq. 9, §4.3 Figure 5).

## Key Figures/Tables to Study
- **Figure 1:** implied value functions of different HALOs against the canonical prospect-theory value function.
- **Figure 3:** GPT-4-judged win rate of SFT+KTO against SFT+DPO from 1B (Pythia) to 30B (Llama).
- **Figure 4:** response-length behavior when alignment is run without a prior SFT stage.
- **Figure 5:** Llama-7B win rate as increasing fractions of desirable data are discarded.
- **Table 1:** recommended learning rate and β per model. **Table 2:** Zephyr-β-SFT on UltraFeedback, including the loss-design ablations. **Table 3:** Mistral-7B on OpenAssistant with one output per input.

## Technical Details

### Implicit reward
`r_θ(x, y) = log [ π_θ(y|x) / π_ref(y|x) ]`
`π_θ`: the policy being trained; `π_ref`: the frozen reference model. Note that `β` sits in the value function, not inside `r_θ` (§4.1).

### Reference point and its estimator
The reference point is defined as `z_0 = KL( π_θ(y'|x) ‖ π_ref(y'|x) )`, the divergence against all possible outputs rather than against one dispreferred output as in DPO (§4.1). Sampling from `π_θ` is too slow, so the implementation shifts outputs within a microbatch to form mismatched pairs `{(x_1,y_2), (x_2,y_3), …, (x_m,y_1)}` and, with `j = (i mod m) + 1`, estimates one shared reference point per microbatch (§4.1, "KL Estimate"):

`ẑ_0 = max( 0, (1/m) Σ_{1≤i≤m} log [ π_θ(y_j|x_i) / π_ref(y_j|x_i) ] )`

`m`: microbatch size. The estimator is described as a biased surrogate for the KL; clamping at 0 adds an upward bias and reduces variance. Gradients are not backpropagated through `z_0` (§4.1, App. A Eq. 10). Mismatched outputs are used because the matched ones were deliberately chosen to be good or bad and carry unrepresentative reward magnitudes (§4.1). KTO therefore needs a microbatch size of at least 2 (§4.2). When KTO follows SFT on data that contains the KTO data, the paper says `ẑ_0` may be an under-estimate and can be set to 0 to save a forward pass; when KTO is not preceded by SFT, estimating `ẑ_0` is necessary (§4.1).

### Value function and loss
`v(x, y) = λ_D · σ( β ( r_θ(x,y) − z_0 ) )` if `y ∼ y_desirable|x`
`v(x, y) = λ_U · σ( β ( z_0 − r_θ(x,y) ) )` if `y ∼ y_undesirable|x`
`L_KTO(π_θ, π_ref) = E_{x,y∼D}[ λ_y − v(x, y) ]` (§4.1, Eq. 8)
`σ`: logistic function, used in place of the exponent `α` of the canonical Kahneman-Tversky value function, which is numerically unstable under optimization. `β > 0`: risk-aversion parameter; larger `β` saturates the value faster. `λ_D`, `λ_U`: loss-aversion weights for desirable and undesirable outputs; `λ_y` is whichever applies to `y`. The paper notes `λ_y` exists only to keep the loss non-negative and can be removed (§4.1, footnote 6).

### Hyperparameter guidance stated by the paper
- **Learning rate.** The paper reports performance is more sensitive to learning rate than to any other hyperparameter, and that the optimal KTO learning rate is 2× to 10× the optimal DPO learning rate; at the 8B scale the DPO default of 5e-7 becomes a KTO default of 5e-6 across all tested setups, including LoRA and already-instruction-tuned references (§4.2).
- **β.** 0.1 is the stated default. `β ∈ [0.01, 0.10)` is recommended when KTO is applied to a model already fine-tuned on the same data, and `β ∈ (0.10, 0.50]` when the reference model is already aligned (§4.2, Table 1).
- **λ_D, λ_U.** Both default to 1; under imbalance set them so `λ_D n_D / (λ_U n_U) ∈ [1, 3/2]` (Eq. 9). For a 1:10 desirable:undesirable ratio the paper sets `λ_U = 1, λ_D ∈ [10, 15]`; for the 90%-discard experiment it uses `λ_U = 1, λ_D = 13.33` (§4.2, §4.3). For tasks where minimizing downside matters more, such as toxicity prevention, the paper says `λ_D n_D < λ_U n_U` may work better (§4.2).
- **Batch size.** All experiments use an effective batch size of 32; the recommended range is 8 to 128 (§4.2, Table 1 caption).

### Reported numbers
- Zephyr-β-SFT (7B) aligned on UltraFeedback for exactly 1 epoch — MMLU / GSM8K / HumanEval / BBH: SFT 57.2 / 39.0 / 30.1 / 46.3; DPO 58.2 / 40.0 / 30.1 / 44.1; KTO (β = 0.1, λ_D = 1) 58.6 / 53.5 / 30.9 / 52.6. The GSM8K gap of DPO→KTO is stated as 13.5 points (Table 2, §4.3).
- Removing `z_0` (constant reference point of 0) costs 4.0 points on GSM8K and 3.6 on BBH (Table 2, §4.3).
- Mistral-7B on OpenAssistant, win rate against the SFT target with a 90% binomial confidence interval: unaligned 0.525 ± 0.037, DPO 0.600 ± 0.037, KTO all-y-per-x 0.652 ± 0.036, KTO one-y-per-x 0.631 ± 0.036, Mistral-7B-Instruct 0.621 ± 0.031. The one-y-per-x restriction removes 72% of the training data (Table 3, §4.3).
- Llama-3 8B on UltraFeedback, recommended settings — SFT+KTO (LR 5e-6, β 0.05): AlpacaEval LC 10.59, BBH 65.15, GSM8K 8-shot 60.20; KTO without SFT (LR 5e-6, β 0.10): 11.25 / 65.26 / 57.92 (Table 1).
- Without a prior SFT stage, KTO-aligned Llama-13B and Llama-30B are competitive with their SFT+KTO counterparts; the paper attributes this to KTO keeping average response length roughly constant while DPO without SFT increases it (§4.3, Figure 4).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Zephyr-β-SFT + KTO | 7B | preference | β; λ_D, λ_U; effective batch; epochs | 0.1; 1.0, 1.0; 32; exactly 1 | arXiv:2402.01306v5 Table 2 caption (epoch, run), §4.2 (batch), Table 1 caption (λ defaults) | verified 2026-09-18 | Table 2: GSM8K 39.0 (SFT) / 40.0 (DPO) / 53.5 (KTO) |
| Zephyr-β-SFT + KTO | 7B | preference | learning rate | not reported for this run | Table 2 and its caption, §4.3 checked; Table 1 gives rates only for Llama-3 8B and Qwen2.5 3B | not reported | — |
| Llama-3 8B + SFT+KTO | 8B | preference | learning rate; β | 5e-6; 0.05 | arXiv:2402.01306v5 Table 1 | verified 2026-09-18 | Table 1: AlpacaEval LC 10.59, BBH 65.15, GSM8K 60.20 |
| Llama-3 8B + KTO (no SFT) | 8B | preference | learning rate; β | 5e-6; 0.10 | arXiv:2402.01306v5 Table 1 | verified 2026-09-18 | Table 1: AlpacaEval LC 11.25, BBH 65.26, GSM8K 57.92 |
| Llama-3 8B Instruct + KTO | 8B | preference | learning rate; β | 5e-6; 0.25 | arXiv:2402.01306v5 Table 1 | verified 2026-09-18 | Table 1: AlpacaEval LC 18.86, BBH 64.28, GSM8K 76.42 |
| Qwen2.5 3B Instruct + SFT+KTO | 3B | preference | learning rate; β | 5e-6; 0.10 | arXiv:2402.01306v5 Table 1 | verified 2026-09-18 | Table 1: AlpacaEval LC 13.01, BBH 32.39, GSM8K 61.11 |
| Llama-7B + KTO (90% desirable discarded) | 7B | preference | λ_U; λ_D | 1; 13.33 | arXiv:2402.01306v5 §4.3 | verified 2026-09-18 | Figure 5: still outperforms DPO at a 1:10 desirable:undesirable ratio |
| all runs | 1B-30B | preference | optimizer; gradient clipping; effective batch | AdamW; norm 10; 32 | arXiv:2402.01306v5 Table 1 caption, §4.2 | verified 2026-09-18 | §4.2: recommended batch range 8-128 |

## Findings relevant to generality and negative feedback
- **Negative feedback.** KTO uses negatives as gradient (§6.1 sense 4 of the course standard): undesirable examples enter through `λ_U · σ(β(z_0 − r_θ))`, which pushes `r_θ` down. The paper states that if the learning rate is too aggressive the implied rewards of both desirable and undesirable examples decline, while the intended behavior is that only the undesirable rewards decline and the desirable rewards stay flat or rise (§4.2). This is the paper's own diagnostic for over-aggressive negative pressure.
- **Value of unpaired negatives.** Discarding 90% of the desirable data — moving the ratio from 1:1 to 1:10, so most undesirable examples lose their preferred counterpart — still leaves a Llama-7B KTO model outperforming DPO (§4.3, Figure 5). The caption states this holds down to 0.1 desirable examples per undesirable one.
- **Generality.** The gains are measured on MMLU, GSM8K, HumanEval, BBH, AlpacaEval LC, and GPT-4-judged win rates against SFT targets (§4.3, Tables 1-3). The paper reports no pass@k or diversity measurement, so coverage effects of the negative term are not measured. HALO-versus-non-HALO differences are significant (p < 0.05) only at 13B and above, and no significant difference appears among the Pythia models, which the authors read as a minimum-capacity requirement (§3.3, §4.3).
- **Data conversion caveat.** In the experiments, preference data is converted to binary labels by assuming `y_w` is desirable and `y_l` undesirable; the paper calls this a naive assumption made for simplicity and leaves a better decomposition to future work (§4.1, "Data").

## Connections
- [[dpo]] — the paired-preference HALO KTO is compared against; KTO reuses its implicit reward but replaces the paired reference point with `z_0`.
- [[ipo]] — a squared-loss alternative also analysed in the HALO framing.
- [[simpo]] — a reference-free variant; KTO's no-`π_ref` ablation (λ_D = 1.75) is the closest run in Table 2.
- [[orpo]] — joint SFT-plus-preference objective, reported alongside KTO in Table 2 (ORPO λ = 0.1).
- [[ultrafeedback]], [[hh-rlhf]] — the alignment datasets used; OpenAssistant and SHP are the other two.
- [[hf-dpo-zoo]] — the TRL `loss_type` interface that exposes KTO next to the other objectives.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2402.01306 (arXiv v5, 2026-09), Abstract, §3.2-3.3, §4.1-4.3, Tables 1-3, Figures 1-5, App. A and C.
- Corrections to the previous card version:
  - "Superior under extreme class imbalance (e.g., 90% desirable)" → the result is the opposite direction: up to 90% of the *desirable* examples can be discarded (ratio 1:1 → 1:10) and the Llama-7B KTO model still outperforms DPO (§4.3, Figure 5; Abstract bullet "up to 90% fewer desirable examples").
  - "Learning rate 5e-7 (AdamW)" → 5e-6 is the KTO default at the 8B scale; 5e-7 is the DPO default the paper contrasts against (§4.2, Table 1).
  - "Recipe for imbalanced data: λ_D / λ_U = N_U / N_D" and "set λ_D and λ_U to balance label frequency (λ_D·N_D ≈ λ_U·N_U)" → the stated criterion is the interval `λ_D n_D / (λ_U n_U) ∈ [1, 3/2]`, not equality (§4.2, Eq. 9).
  - "z_0 … estimated on-the-fly with a batch-level moving statistic" → it is a clamped, biased within-microbatch estimate over shifted mismatched pairs, recomputed per microbatch, with no moving average (§4.1, "KL Estimate").
  - "β | 0.1 (main)" as a single value → 0.1 is the default; Table 1 recommends 0.05 to 0.50 depending on the reference model, and §4.2 gives the conditions.
  - "Implicit reward (same as DPO): r_θ = β·log[π_θ/π_ref]" → in KTO `β` is not inside `r_θ`; it appears in the value function (§4.1).
  - Missing fields "Source type" and a Verification section were added; the year line now records the arXiv v1 month and the version checked.
- Removed as unsupported by the source: "Less sensitive to β than DPO" (§4.2 states the opposite emphasis — sensitivity to learning rate dominates, and DPO's optimal settings do not transfer to KTO); "KTO recovers DPO's chosen/rejected dynamics when labels are paired" (App. A analyses a loss-neutral value function and the majority-preferred output, not equivalence to DPO); "Figure 3 / Table 1: KTO vs DPO vs SFT on GPT-4-judged win rate from 1B to 30B" as a single locus (Figure 3 carries the win rates; Table 1 is the hyperparameter table); "Robust to label imbalance via λ_D, λ_U weights" stated without the interval condition.
- Not reported by the source: the number of gradient steps or tokens per run; wall-clock or GPU cost; pass@k, entropy, or output-diversity measurements; results on models larger than 30B; a controlled comparison of KTO against DPO on naturally unpaired production feedback rather than decomposed preference data.

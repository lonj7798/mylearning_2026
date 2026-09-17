<!-- chapter: ch-08a
     track: pretraining
     kind: content
     title: Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability
     deps: [ch-00, ch-03]
     sources: [[kaplan-scaling-laws]], [[chinchilla-compute-optimal]], [[resolving-scaling-discrepancies]], [[resolving-scaling-discrepancies-recipe]], [[chinchilla-replication]], [[deepseek-llm]], [[beyond-chinchilla-inference-scaling]], [[overtraining-downstream-scaling]], [[llama-3]], [[llama-3-recipe]], [[smollm2]], [[task-scaling-model-ladders]], [[scaling-laws-unreliable-downstream]], [[predicting-downstream-elusive]], [[data-constrained-scaling]], [[overtrained-lms-harder-to-finetune]], [[cooldown-scaling-beyond-fixed-durations]], [[emergence-loss-perspective]], [[same-loss-better-downstream]], [[datadecide]], [[signal-and-noise-eval]]
     figures: figures/compute-allocation-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability

> **Core insight.** Pre-training loss follows fitted power laws in parameters N and training tokens D, but the compute-optimal split depends on how the laws are fitted. Kaplan et al. found N ∝ C^0.73 ([[kaplan-scaling-laws]]); Hoffmann et al. found N and D should grow in equal proportion, about 20 tokens per parameter in their setup ([[chinchilla-compute-optimal]]); Porian et al. moved a reproduction of the Kaplan setup from exponent 0.835 to 0.497 by counting output-head FLOPs, shortening warmup, and tuning learning rate, batch size, and β2 per size ([[resolving-scaling-discrepancies]], Table 1). The loss-optimal split ignores inference: once serving cost is counted, smaller models trained past 20 tokens per parameter reach the same loss at lower total cost ([[beyond-chinchilla-inference-scaling]]), and that loss remains predictable ([[overtraining-downstream-scaling]]). Loss predicts averages over many tasks well, individual tasks poorly (18 of 46 tasks scaled predictably in one re-analysis, [[scaling-laws-unreliable-downstream]]), and one study found that OLMo-1B pre-trained on 3T tokens was worse after instruction tuning than its 2.3T-token checkpoint ([[overtrained-lms-harder-to-finetune]]).
>
> **Guideline.** When choosing N and D for a fixed budget, fit IsoFLOP curves on the target data and architecture at three or more budgets, with FLOPs counted including the output head, warmup set per model size, and either a decay matched to each run's tokens or learning rate, batch size, and β2 tuned per size, because in one reproduction the fitted exponent moved from 0.835 to 0.571 with the first three corrections and to 0.497 with per-size tuning ([[resolving-scaling-discrepancies]] Table 1), and it changed with data quality from 0.450 to 0.578 ([[deepseek-llm]] Table 4). When the model will serve lifetime inference tokens comparable to or larger than its training tokens, choose the smallest model that reaches the target loss once inference FLOPs are included, because in the Sardana et al. example this saved 17% of total FLOPs for a 13B-Chinchilla-quality model at 2T inference tokens (Fig. 1). When a pre-training decision is judged by downstream capability, forecast averages over many tasks or task losses in bits per byte, backtest the forecast on a held-out larger run, and fine-tune two token-budget checkpoints with the same post-training recipe before committing to a long over-training run, because the same fit predicted a 17-task average error within 0.05% and HellaSwag with 25.73% relative error ([[overtraining-downstream-scaling]] Table 2), and an OLMo-1B base pre-trained on 3T tokens scored up to 3% lower on AlpacaEval after instruction tuning than its 2.3T-token checkpoint ([[overtrained-lms-harder-to-finetune]] §2). Otherwise, treat a per-task accuracy forecast as unvalidated.

## Why this chapter matters for a general-purpose model

The pipeline is pre-training → mid-training → SFT → preference optimization → RL → evaluation. Pre-training fixes the parameter count that every later stage inherits. The two decisions made before the run starts are the model size N and the number of training tokens D. They cannot be revised cheaply after the run.

The measurable problem has three parts:
1. **Allocation.** For a compute budget C, one (N, D) gives the lowest pre-training loss. Published rules for that split differ, and a rule fitted on one data set and training setup may not hold for another.
2. **Deployment cost.** The model that minimizes training loss per FLOP is larger than the model that minimizes training plus serving cost at the same loss.
3. **Capability.** A general-purpose model is judged on many downstream tasks (ch-00). Loss is the quantity the laws predict. The chapter measures how well loss predicts downstream accuracy and post-training quality.

ch-03 covers the learning-rate schedule and batch-size mechanics that scaling experiments depend on. ch-08b takes the predictability question further into in-context learning and emergence. ch-14 treats the case where unique data is the binding limit. ch-14a compares the budgets of released recipes.

## §1 Quantities and counting conventions

**Definitions.** N is the parameter count. D is the number of training tokens seen (not the number of unique tokens). C is training compute in floating-point operations (FLOPs). A **compute-optimal** (N, D) minimizes loss subject to a fixed C. An **IsoFLOP curve** is loss plotted against N for runs that all use the same C, with D = C/(6N).

**Problem.** The same model can be assigned different N and C under different conventions, and the error is largest for small models, which are the models used to fit laws.

**Mechanism.** Forward-pass compute per token is about 2N, and the backward pass is about twice the forward pass, so training costs about 6N FLOPs per token ([[kaplan-scaling-laws]] §2.1):

```
C ≈ 6 · N · D
```

where C is training FLOPs, N the parameter count, and D the training tokens. Conventions differ in what N includes:

```
6·N₁ = 72 · n_layer · d_model²                                   (non-embedding parameters, Kaplan et al.)
6·N₂ = 72 · n_layer · d_model² + 6 · n_vocab · d_model           (all parameters, Hoffmann et al.)
M    = 72 · n_layer · d_model² + 12 · n_layer · d_model · l_seq  (non-embedding FLOPs per token, DeepSeek)
```

n_layer is the number of layers, d_model the width, n_vocab the vocabulary size, l_seq the sequence length, and M counts attention computation but not the vocabulary projection; DeepSeek writes C = M·D ([[deepseek-llm]] §3.2 Eq. 2). Porian et al. use a fourth convention: all linear layers including the output head, excluding the input embedding ([[resolving-scaling-discrepancies]] §2.1).

**Worked example** (DeepSeek Table 3, first row; n_layer = 8, d_model = 512, n_vocab = 102,400, l_seq = 4,096):
- 72 · 8 · 512² = 150.99M, so N₁ = 150.99M / 6 = 25.2M.
- 6 · 102,400 · 512 = 314.57M, so N₂ = (150.99M + 314.57M) / 6 = 77.6M.
- 12 · 8 · 512 · 4,096 = 201.33M, so M = 150.99M + 201.33M = 352M FLOPs per token.
- 6N₁/M = 0.43 and 6N₂/M = 1.32. For the 80-layer, d_model = 8,192 row the ratios are 0.92 and 0.94 (Table 3).

At this small size, the two parameter conventions disagree with the FLOP count by factors of 2.3 and 1.3 in opposite directions. At 80 layers they agree within 8%. A law fitted across this range with a convention that mis-states compute by size-dependent amounts has a biased exponent.

**Evidence.** In Porian et al., leaving the head out of the FLOP count under-counts compute by roughly 90% for their smallest models and roughly 10% for larger ones, and correcting it lowered the model-size exponent on RefinedWeb from 0.835 to 0.706 (§3.2, Table 1). Hoffmann et al. count embeddings in both N and FLOPs (App. F).

**Implication.** Before comparing an exponent or a tokens-per-parameter ratio across papers, check which N and C it uses.

## §2 Kaplan et al. (2020): power laws and the 0.73 allocation

**Definition.** A power law L(X) = (X_c/X)^α_X states that loss falls by a constant factor for each constant multiplicative increase in X.

**Formulas** ([[kaplan-scaling-laws]] §1.2):

```
L(N)      = (N_c / N)^α_N,       α_N ≈ 0.076,  N_c ≈ 8.8 × 10^13          (Eq. 1.1)
L(D)      = (D_c / D)^α_D,       α_D ≈ 0.095,  D_c ≈ 5.4 × 10^13 tokens   (Eq. 1.2)
L(C_min)  = (C_c / C_min)^α_C,   α_C ≈ 0.050,  C_c ≈ 3.1 × 10^8 PF-days  (Eq. 1.3)
L(N, D)   = [ (N_c / N)^(α_N/α_D) + D_c / D ]^α_D                        (Eq. 1.5)
```

L is test cross-entropy in nats per token on WebText2. N excludes embeddings. D is dataset size in tokens. C_min is the compute that would reach a loss at a batch size far below the critical batch size (the batch size above which further increases give diminishing reductions in the number of optimizer steps, §5.1). One PF-day is 8.64 × 10^19 FLOPs. N_c, D_c, and C_c depend on the tokenizer and "do not have a fundamental meaning" (§1.2).

**Mechanism of the allocation.** Kaplan et al. trained models from 768 to 1.5B non-embedding parameters for a fixed 2.5 × 10^5 steps with a 3,000-step warmup and cosine decay to zero (§2.2, §3). They took the loss envelope over training curves, adjusted it to the critical batch size, and fitted N ∝ C_min^0.73; the unadjusted fixed-batch fit is N = 1.6 × 10^9 · C^0.88 with C in PF-days (§6.1, Fig. 14).

**Worked example.** Doubling N multiplies L(N) by 2^−0.076 = 0.949 (§1.2). Under N ∝ C^0.73, a 100× compute increase multiplies N by 100^0.73 = 28.8 and D by 100^0.27 = 3.47 (derived). Hoffmann et al. summarize Kaplan et al.'s rule for a 10× budget as 5.5× parameters and 1.8× tokens (§1); 10^0.73 = 5.37 and 10^0.27 = 1.86 (derived).

**Conditions and limits.** All runs share one schedule length, so a model stopped early has not decayed its learning rate; §3 shows why this matters. The authors state that the L(C_min) and L(D) trends intersect near 10^12 parameters and 10^12 tokens, with values uncertain by an order of magnitude, and that the laws must break down before this point (§6.3). Status: **Result (single study)**, superseded for allocation by §3–§4.

## §3 Hoffmann et al. (2022): three estimates and equal scaling

**Problem.** Following Kaplan et al. and the GPT-3 training setup, most large models of 2020–2021 were trained on about 300B tokens (Gopher 280B, GPT-3 175B, Jurassic 178B; [[chinchilla-compute-optimal]] §1, Table 1). Hoffmann et al. ask whether the same compute gives lower loss with a smaller model on more tokens.

**Mechanism.** Over 400 models from 70M to over 16B parameters on 5B to 500B tokens, with three estimators (§3):
1. **Training-curve envelope.** Train each size with 4 cosine horizons spanning 16×; at each of 1,500 FLOP values, keep the lowest-loss run.
2. **IsoFLOP profiles.** At 9 budgets from 6 × 10^18 to 3 × 10^21 FLOPs, vary N, set the schedule to the token count, fit a parabola, and take its minimum.
3. **Parametric fit** of all final losses to Eq. 2 with a Huber loss on log L, minimized with L-BFGS (a quasi-Newton optimizer) from a grid of initializations. A Huber loss is quadratic for residuals smaller than δ = 10^−3 and linear above it, so points with large residuals weigh less (Eq. 3, App. D.2).

```
L̂(N, D) = E + A / N^α + B / D^β                                     (Eq. 2)
N_opt(C) = G · (C/6)^a,   D_opt(C) = G^−1 · (C/6)^b                   (Eq. 4)
G = (α·A / (β·B))^(1/(α+β)),   a = β/(α+β),   b = α/(α+β)
```

E is the irreducible loss (the entropy of text under the data distribution), A/N^α is the excess loss of a finite model, B/D^β is the excess loss of finite training, and A, B, α, β are fitted constants. Eq. 4 follows from minimizing Eq. 2 subject to C = 6ND. The printed fit is E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28 (App. D.2).

**Evidence.** The exponent a (with N_opt ∝ C^a) is 0.50, 0.49, and 0.46 for the three approaches, against Kaplan et al.'s 0.73 (Table 2). Approach 1 projects 1B parameters with 20.2B tokens and 10B with 205.1B tokens (Table 3), about 20 tokens per parameter (derived; the paper does not print a "20" rule). For the Gopher budget of 5.76 × 10^23 FLOPs the estimate is 40B to 70B parameters; Chinchilla 70B was trained on 1.4T tokens and reached 67.6% on 5-shot MMLU against Gopher's 60.0% (§4, Table 6; the abstract prints 67.5%).

**The explanation Hoffmann et al. give for the disagreement.** Kaplan et al. used one cosine schedule to 130B tokens, so the losses of shorter runs were read from the middle of a schedule, before learning-rate decay, and overestimate what a matched schedule would reach (§2). In their grid of cycle lengths from 1× to 5× the run, a cosine cycle more than 25% longer than the run "leads to clear drops in performance" (App. B, Fig. A1).

**Worked example.** With the printed coefficients, a = 0.28/(0.34 + 0.28) = 0.452 (Table 2 prints 0.46 for Approach 3). At 100× compute, Approach-1 exponents give N × 100^0.50 = 10 and D × 10, compared with Kaplan's 28.8 and 3.47 (derived).

**Conditions and limits.** Only two large runs were compared (Chinchilla and Gopher); the frontier shows concavity at high compute; all runs used less than one epoch; results on C4 and GitHub code agree "as long as one does not train for more than one epoch" (§5, App. C, App. E). Status: **Replicated** for the equal-scaling result by the IsoFLOP analyses in [[deepseek-llm]] (a = 0.524 on its current data) and [[resolving-scaling-discrepancies]] (a = 0.497).

## §4 Why the estimates disagree, and how precise any published law is

**Porian et al. (2024).** The authors reproduced the Kaplan setup on RefinedWeb and OpenWebText2 with models from 5M to 901M and removed one difference at a time ([[resolving-scaling-discrepancies]] Table 1, RefinedWeb / OpenWebText2):

| Step | Change | Exponent a |
|---|---|---|
| 1 | Kaplan reproduction (fixed warmup ≈ 1.57B tokens, cosine to zero at ≈ 131B tokens, head excluded) | 0.835 / 0.864 |
| 2 | Count output-head FLOPs | 0.706 / 0.699 |
| 3 | Warmup tokens = N | 0.602 / 0.603 |
| 4 | Cosine decay matched to each budget | 0.571 / 0.574 |
| 5 | Constant LR, with LR, batch size, and β2 tuned per size (instead of step 4) | 0.497 / 0.518 |

With the fixed warmup, small models reached their compute-optimal token count while still in warmup (§3.3). Matching the decay changed the exponent by about 0.03, so the schedule mismatch that Hoffmann et al. proposed is not the main cause in this reproduction (§3.4). Tuned per-size hyperparameters follow BS = 0.00037 · N^0.703 and LR = 3.7 · N^−0.36 (Fig. 3). Limit: the largest model is 901M parameters against over 16B in Hoffmann et al., and the authors describe their compute as roughly at Kaplan et al.'s scale (§5.2). Status: **Result (single study)**, consistent with the DeepSeek IsoFLOP result.

**Worked example.** A 100× compute increase multiplies N by 100^0.835 = 46.8 under the step-1 exponent and by 100^0.497 = 9.9 under the step-5 exponent (derived). An exponent difference of 0.34 changes the recommended model size 4.7-fold over two orders of magnitude.

**Besiroglu et al. (2024).** The authors digitized the points of Hoffmann et al.'s Figure 4 (240 points after removing 5 outliers) and refit Eq. 2 ([[chinchilla-replication]] §2–§3):

```
L(N, D) = 1.8172 + 482.01 / N^0.3478 + 2085.43 / D^0.3658            (Eq. 3)
```

The refit gives a = 0.5126 with standard error 0.018 and implies about 20 tokens per parameter, consistent with Approaches 1–2 and with Chinchilla's own training (§3.3). Two causes explain the published Approach-3 fit (§3.1–3.2): the body prints rounded values, and rounding β from 0.2849 to 0.28 raises the data term by about (10^11)^0.0049 − 1 ≈ 13% at D = 10^11 tokens; and the original optimizer averaged instead of summing Huber losses, which stopped L-BFGS early and produced a confidence interval for a (0.454–0.455) that would need about 600,000 runs to justify. The authors attribute the second cause to a confirmation by a Hoffmann et al. author. Their refit's 80% confidence region is consistent with 4 to 40 tokens per parameter for models trained on 10^26 FLOPs or more (§4). Status: **Result (single study)** on reconstructed data (loss precision about 0.01, §2).

**Worked example: reproduce a paper's own prediction before using its coefficients.** At the Gopher budget C = 5.76 × 10^23, Eq. 4 gives N_opt = 32.2B with the printed coefficients and 40.3B with the unrounded TeX values (E = 1.6934, α = 0.3392, β = 0.2849); Hoffmann et al. state 40B for Approach 3 (Fig. 4 caption); the 32.2B and 40.3B values are derived. The same check on Llama 3: the report prints (α, A) = (0.53, 0.29) for the optimal token count A·C^α and states 402B parameters on 16.55T tokens (41.2 tokens per parameter, derived) at 3.8 × 10^25 FLOPs ([[llama-3]] §3.2.1; the report writes this token count as N*(C)). By C ≈ 6ND that plan costs 6 × 402B × 16.55T = 3.99 × 10^25 FLOPs, 5% above the stated budget (derived). The printed pair gives 0.29 × (3.8 × 10^25)^0.53 = 10.5T tokens; the Figure 3 legend values (0.537, 0.299) give 16.3T (derived). Because log10 C = 25.58, an exponent difference of 0.007 multiplies the prediction by 10^(0.007 × 25.58) = 1.51, and the coefficient ratio 0.299/0.29 adds a factor of 1.03.

**Data quality changes the exponent.** DeepSeek fitted IsoFLOP profiles with M instead of N (main fit: 8 budgets from 10^17 to 3 × 10^20 FLOPs) and obtained a = 0.450 on early in-house data, 0.524 on current in-house data, and 0.578 on OpenWebText2, which the authors rank as increasing in quality ([[deepseek-llm]] §3.3, Table 4). Their explanation, that higher-quality text is less difficult to predict, is labeled speculation by the authors. Status: **Result (single study)**; Porian et al. suggest data repetition as a possible cause of the 0.578 value ([[resolving-scaling-discrepancies]] §5.1).

**Implication.** A tokens-per-parameter rule is a property of a data set, a counting convention, and a hyperparameter protocol. Refit it when any of the three changes.

## §5 Compute-optimal versus deliberate over-training

**Definitions.** The **token multiplier** is M = D/N. **Over-training** means training with M above the compute-optimal M* for the data, so a smaller model is trained on more tokens than loss-per-FLOP would recommend ([[overtraining-downstream-scaling]] §2.1). **Reducible loss** is L − E.

**Problem.** Training compute is spent once; inference compute grows with every served token. For a fixed quality, the model that minimizes training FLOPs is larger than the model that minimizes total FLOPs.

**Mechanism** ([[beyond-chinchilla-inference-scaling]] §2):
1. Fix a target pre-training loss ℓ.
2. For each candidate N, solve L(N, D) = ℓ for D.
3. Compute lifetime cost 6·N·D + 2·N·D_inf, where D_inf is the total of input plus output tokens over all requests and 2N is the forward-pass cost per token.
4. Choose the N with the lowest cost. There is no general closed form; the authors solve it numerically (App. A).

**Evidence.**
- A 13B-Chinchilla-quality model with 2T inference tokens is matched by a 7B model on more data at 17% fewer total FLOPs; a 30B-Chinchilla-quality model with 10^13 inference tokens is matched by 13.6B parameters on 2.84× the data at 28% fewer FLOPs (§2, Fig. 1). In dollar terms the gap is larger: at 2T inference tokens a Chinchilla-70B model needs 1.3% more FLOPs than the FLOP-optimal model but costs 36% more than the cost-optimal model, which the authors attribute to 50× lower model FLOPs utilization for generated tokens than for training (§6).
- 47 MPT-style models from 150M to 6B trained at 10 to 10,000 tokens per parameter (only the 150M model reached 10,000) showed no loss plateau up to 10,000, and the Gauntlet average (an equal-weight average over world knowledge, commonsense reasoning, reading comprehension, language understanding, and symbolic problem solving tasks) kept rising with the ratio (§3, §4, Fig. 3). Refitting Eq. 2 on progressively more extreme runs flattened the curves (α from 0.08 to 0.18, β from 0.13 to 0.24), and the authors conclude that laws fitted at typical ratios overestimate the loss reduction from extra tokens at extreme ratios (§5, Table 1).
- Gadre et al. reparameterize Eq. 2 with α = β and C = 6ND:

```
L(C, M) = E + (a · M^η + b · M^−η) · C^−η                             (Eq. 4)
```

with η = α/2, a = A·6^η, b = B·6^η ([[overtraining-downstream-scaling]] §2.2). The exponent η does not depend on M, so runs at different multipliers form parallel lines in log-log plots of reducible loss against compute (Fig. 2). Fitting on runs costing 2.4 × 10^19 FLOPs predicted the C4 loss of a 1.4B model on 900B tokens (M = 640) within 0.7% relative error (§4, Table 1). M = 5 was off-trend, and multipliers from 10 to 80 all gave points near the frontier, so M* "may lie in a range" (§4, Fig. 9).

**Worked example of Eq. 4** (illustrative values, not fitted by any source). Take η = 0.15 and b/a = 20^0.3 = 2.456, which places the minimum of the offset aM^η + bM^−η at M = 20. With a = 1, the offset is 3.135 at M = 20 and 3.410 at M = 320. At every C, the 320-multiplier run has 8.8% more reducible loss. Because reducible loss scales as C^−0.15, matching it costs 1.088^(1/0.15) = 1.75× the compute. Under Eq. 4, training 16× past this M* costs a constant compute factor at every budget, not a factor that grows with C. Eq. 4 assumes α = β. Gadre et al. show that without this assumption the exponent is still unchanged by over-training when the multiplier is measured relative to the compute-optimal M*, which then changes with C (App. B, Eq. 8).

**Practice.** Llama 3.1 405B used 15.6T tokens and 3.8 × 10^25 FLOPs, 38.5 tokens per parameter (derived), at a size the authors describe as approximately compute-optimal; its smaller models were trained "much longer than is compute-optimal", and per-size token counts for 8B and 70B are not reported ([[llama-3]] §1; [[llama-3-recipe]]). SmolLM2 1.7B was trained on about 11T tokens, 360M on 4T, and 135M on 2T ([[smollm2]] Abstract, §6), about 6,500, 11,100, and 14,800 tokens per parameter (derived from the nominal sizes). DeepSeek LLM 7B used 2.0T tokens ([[deepseek-llm]] Table 2); its own Eq. 4 gives D_opt = 5.8316 · (8.46 × 10^22)^0.4757 = 470B tokens at that model's compute (C = M·D with M = 42.3B FLOPs per token from §1's formula), so the 7B was trained on 4.3× its law's D_opt (derived; the report does not print the units of C and D in Eq. 4, and FLOPs and tokens are inferred because M_opt · D_opt = 1.000 · C).

**Conditions and limits.** Sardana et al. assume demand does not depend on model size and can be estimated before training (§2). Neither Sardana et al. nor Gadre et al. measure post-training quality (Gadre et al. §6 lists it as open). §8 covers that measurement.

The figure [compute-allocation-explorer.html](figures/compute-allocation-explorer.html) lets the reader choose a budget, a coefficient set (printed, refit, or Sardana analysis values), and D_inf, and compare the loss-optimal and inference-aware plans; its second preset reproduces the Sardana et al. Fig. 1 example (6.97B parameters, 17.4% fewer total FLOPs).

## §6 Predicting downstream accuracy from loss

**Definition.** A **downstream scaling law** maps compute, or a loss measured on text, to a benchmark score. **Bits per byte** is the negative log-likelihood of a string, in bits, divided by the string's length in bytes; normalizing by bytes instead of tokens reduces tokenizer effects ([[task-scaling-model-ladders]] §2.2).

**Problem.** Benchmark accuracy is a step function of per-item likelihoods: an item scores 1 only when the correct choice has the highest likelihood, so accuracy is less correlated with compute than the likelihood itself ([[predicting-downstream-elusive]] §4, Fig. 3). General-purpose models are compared on accuracy.

**Mechanism: chained prediction.** All three studies below use two steps.
1. Predict an intermediate loss from compute (or N and D).
2. Predict accuracy or error from that loss, with a fit that saturates.

**Gadre et al.** use the average top-1 error Err over 17 tasks:

```
Err(L) = ε − k · exp(−γ · L)                                        (Eq. 5)
```

where L is C4 validation loss and ε, k, γ are fitted; Err is bounded above by ε. For a 6.9B model on 138B RedPajama tokens, the chained forecast was within 0.05% relative error, but removing the one 1.4B run from the Eq. 5 fit raised the error to 10.64% ([[overtraining-downstream-scaling]] §4). The four individual tasks reported for the same model had relative errors from 5.21% (ARC-Easy) to 25.73% (HellaSwag) with RedPajama and up to 81.96% (HellaSwag, RefinedWeb) (Table 2).

**Llama 3** fits the normalized negative log-likelihood of the correct answer linearly against FLOPs using scaling-law models up to 10^22 FLOPs, then a sigmoid from that likelihood to accuracy using the scaling-law models and Llama 2 models; for ARC Challenge the forecast over four orders of magnitude "only slightly underestimates" the 405B result ([[llama-3]] §3.2.1, Fig. 4). The report gives no numeric error.

**Model ladders.** A model ladder is a fixed set of small models of several sizes and token multipliers, trained with the target model's architecture and data mixture, whose losses and accuracies supply the fitting points ([[task-scaling-model-ladders]] §2.1, §3):

```
Step 1:  L(N, D) = A / N^α + B / D^β + E                  (task loss in bits per byte)
Step 2:  Acc(L)  = a / (1 + e^(−k·(L − L₀))) + b
```

A, B, α, β, E are fitted per task on 16 ladder models (190M to 1.3B, 1× to 10× of 20·N tokens, 5.2 × 10^21 FLOPs); a, b, k, L₀ are fitted on about 1,400 checkpoint points. Chained predictions for OLMo 2 7B-4T and 13B-5T had average absolute errors of 3.8 and 4.2 points over 8 tasks; MMLU was predicted at 48.4 against 49.0 (7B), and ARC-Challenge at 51.5 against 61.9 (§3.3, Fig. 2). The standard deviation of task loss over the final 10 checkpoints of the largest ladder model (SD10) correlated with step-2 error (r = 0.821 for 7B-4T) (§5).

**Worked example of error propagation** (illustrative values). Take a = 0.6, b = 0.25, k = −10, L₀ = 1.0; a negative k makes accuracy fall as loss rises. At L = 1.00, Acc = 0.6/2 + 0.25 = 0.550. If step 1 overestimates task loss by 5% (L = 1.05), Acc = 0.6/(1 + e^0.5) + 0.25 = 0.6/2.649 + 0.25 = 0.477. A 5% loss error becomes a 7.3-point accuracy error. The ARC-Challenge miss above has this direction: step 1 overestimated its task loss and step 2 underestimated accuracy ([[task-scaling-model-ladders]] §4).

**Conditions and limits.** Ladder predictions cover base checkpoints before annealing and post-training, and multiple-choice tasks only (§8). Decision accuracy is the share of corpus pairs whose ranking at small scale matches the ranking at the target scale. DataDecide (25 pre-training corpora, models up to 1B) found that ranking corpora with a single small model size matched the 1B winner in about 80% of pairwise comparisons, and none of 8 scaling-law baselines exceeded the compute-to-decision-accuracy frontier of such single-scale rankings ([[datadecide]] §3.2). Implication: for a decision between two data recipes, a ranking at small scale may be as useful as an extrapolated law.

## §7 Where downstream prediction fails

**Failure 1: the metric discards information.** Schaeffer et al. compute, for each benchmark item, the correlation between compute and score across the checkpoints of one model family, for five families ([[predicting-downstream-elusive]] §3–§4). On ARC-Challenge, about 90% of samples have Spearman (rank) correlation above 0.75 for the log-likelihood of the correct choice; after renormalizing over the answer choices, 40% do; Accuracy lowers the correlations further (Fig. 3). The mechanism is that Accuracy depends on the mass on specific incorrect choices (§5). Worked example (§5): in a 4-way item with p(correct) = 0.4, spreading the remaining 0.6 as 0.2 per incorrect choice gives Accuracy 1; placing 0.6 on one incorrect choice gives Accuracy 0. The correct-answer probability is identical in both cases.

**Failure 2: many tasks do not scale smoothly with loss.** Lourie et al. re-classified the 46 tasks of Gadre et al.: 18 (39%) improve predictably with validation loss; the rest are inverse, nonmonotonic, noisy, trendless, or show a breakthrough ([[scaling-laws-unreliable-downstream]] §4, Fig. 1). With C4 validation loss, C4 and RedPajama pre-training follow one HellaSwag curve; with a code validation set, C4 looks better at worse loss; on CoQA the ordering reverses (§3, Fig. 3). CommonsenseQA scales nonmonotonically in the Gadre et al. setup (LLM Foundry harness) and cleanly in the DataDecide setup (OLMES), which also differ in models, prompts, shots, and answer-choice counts (§5, App. A).

**Failure 3: capability absent at small scale.** In Sardana et al., symbolic problem solving stayed near zero for all models from 150M to 6B ([[beyond-chinchilla-inference-scaling]] §4, App. D). In the ladder study, the ability to answer in multiple-choice format (options listed in the prompt, the model outputs the answer letter) does not appear until several billion parameters, so the ladder uses the ranked-classification format, which scores each option's likelihood (§2.2 footnote 2). Du et al. report MMLU, C-Eval, GSM8K, and GSM8K-Chinese at random level until training loss falls to about 2.2 in their corpus ([[emergence-loss-perspective]] §3.1). ch-08b treats these cases.

**Failure 4: equal loss, unequal transfer.** Liu et al. pre-train masked language models (and LSTMs for one data set) on text generated by a probabilistic context-free grammar (PCFG), a hidden Markov model (HMM), and OPT-125M. Where models reach the same pre-training loss, larger models had higher linear-probe accuracy (a linear classifier trained on frozen representations) by 6.9%, 4.5%, and 2.0% ([[same-loss-better-downstream]] §3). Loss is not a sufficient statistic for transfer in these settings. Status: **Result (single study)**, simplified data and masked-language-model pre-training only.

**Failure 5: evaluation noise.** Across OLMES tasks, a benchmark's signal-to-noise ratio (spread across models divided by checkpoint-to-checkpoint variation) at 60M–750M correlated with its decision accuracy for 1B rankings (R = 0.791), and switching to bits per byte raised the 30-task average decision accuracy from 77.0% to 83.7% ([[signal-and-noise-eval]] §4.1, Fig. 6).

**Implication.** Status: **Open question** whether per-task accuracy of frontier models can be forecast from small runs. The measured parts are these: averages over many tasks and task losses in bits per byte forecast better than per-task accuracy, and a forecast is validated only by a held-out larger run.

## §8 Plasticity cost of over-training

**Definition.** **Plasticity** here means how much a base model can gain from post-training without losing previously acquired ability. Springer et al. call a decrease of post-trained quality with more pre-training tokens **catastrophic overtraining** ([[overtrained-lms-harder-to-finetune]] §3).

**Evidence from released checkpoints.** OLMo-1B intermediate checkpoints were instruction-tuned on Anthropic-HH with the learning rate tuned per checkpoint. The 3T-token base gave up to 3% lower AlpacaEval response rate and 2% lower ARC than the 2.3T-token base, dropping to the level of the 1.5T-token base, while the base models improved monotonically (§2, Fig. 2). Degradation appeared beyond 2.5T tokens for instruction tuning and for LLaVA multimodal tuning (§3.1). It was not observed for OLMo-2-7B under the same setups up to 3T tokens (§3.1 writes "OLMo-7B"; App. E). Nominal ratios are 2,300 and 3,000 tokens per parameter for the OLMo-1B checkpoints and about 430 for OLMo-2-7B at 3T (derived). The authors state a confound: checkpoints from one run differ in their final learning rate (§3.2).

**Evidence from controlled runs.** Models of 15M to 90M parameters were pre-trained on C4 for 4B to 128B tokens, each annealed to zero (§3.2); 128B tokens for the 30M model is about 4,300 tokens per parameter (derived).
1. Adding Gaussian noise of fixed scale γ raised perplexity by more for models trained on more tokens ("progressive sensitivity"), and the perturbed perplexity was U-shaped in tokens (§3.3, Fig. 3).
2. Fine-tuning with a fixed learning rate on GSM8k, Starcoder-Python, SIQA, and three classification sets did the same to C4 perplexity; larger learning rates moved the turning point to fewer tokens (§3.4.1, Fig. 4–5).
3. With the learning rate tuned for the fine-tuning task, C4 perplexity still degraded on GSM8k, Starcoder-Python, MR, and RTE, and in-domain perplexity degraded on RTE and TREC (§3.4.2, Fig. 6).

**Theory.** In a two-layer linear network that learns features incrementally, pre-training loss after unregularized fine-tuning eventually rises with pre-training time when the pre-training and fine-tuning tasks are misaligned; regularization toward the pre-trained weights delays the turning point and raises fine-tuning loss (§4, Theorem 4.7).

**Conditions and limits.** Real-model evidence covers three models from two families (OLMo-1B, OLMo-2-7B, LLM360-Amber 7B); no fitted relation between tokens per parameter and fine-tuning loss exists; RL after pre-training was not tested (§3, §6). Status: **Result (single study)**. Interpretation (this course): the result does not show that over-training is harmful at 7B-and-larger scale; it shows that a pre-training decision that trades training compute for inference compute must be checked after post-training, not only on the base model. The forgetting measurement protocol is in ch-30a.

## §9 Worked allocation exercise: one budget, three objectives

Budget C = 1.23 × 10^22 FLOPs. This is the Chinchilla Table 3 row for 10B parameters and 205.1B tokens: 6 × N × D = 6 × 10^10 × 2.051 × 10^11 = 1.23 × 10^22. The exercise uses the Besiroglu et al. refit (Eq. 3) because it is consistent with Approaches 1–2. All plan numbers are derived and can be reproduced in the figure (first preset).

**Objective 1: lowest pre-training loss.**
- Eq. 4 with the refit: N_opt = 10.1B, D_opt = 203.8B tokens, 20.3 tokens per parameter, L = 2.1294. At N = 10B and D = 205.1B by hand: 482.01/10^(10·0.3478) = 0.1603 and 2085.43/(2.051 × 10^11)^0.3658 = 0.1518, so L = 1.8172 + 0.1603 + 0.1518 = 2.1293. This point is 0.0001 below the optimum because 6 × 10^10 × 2.051 × 10^11 = 1.2306 × 10^22 exceeds the rounded budget by 0.05%.
- The same budget with the printed Hoffmann coefficients gives 5.67B and 362B tokens (63.9 per parameter). Kaplan et al.'s adjusted law N = 1.3 × 10^9 · (C / (8.64 × 10^19))^0.73, treating C as C_min, gives 48.5B parameters and 42.3B tokens. The three published fits disagree by a factor of 8.6 in N at this budget.
- The IsoFLOP curve is flat near the minimum: at half and double N_opt the refit loss is 2.1389 and 2.1390, 0.45% above the minimum. Llama 3 reports the same flattening at larger budgets and used it to choose 405B over the predicted 402B ([[llama-3]] §3.2.1).

**Objective 2: lowest lifetime cost at the same loss, with D_inf = 2 × 10^12 tokens.**
- Solving §5's steps numerically gives N = 4.45B and D = 648B tokens (146 per parameter). Training compute is 1.73 × 10^22 (40.7% above C), inference 1.78 × 10^22, total 3.51 × 10^22, against 1.23 × 10^22 + 2 × 1.006 × 10^10 × 2 × 10^12 = 5.25 × 10^22 for the loss-optimal plan: 33.2% lower.
- Hand check at N = 5B: the parameter term is 482.01/(5 × 10^9)^0.3478 = 0.2041; the data term must equal 2.1294 − 1.8172 − 0.2041 = 0.1081; D = (2085.43/0.1081)^(1/0.3658) = 5.19 × 10^11 tokens. Total = 6 × 5 × 10^9 × 5.19 × 10^11 + 2 × 5 × 10^9 × 2 × 10^12 = 1.56 × 10^22 + 2.00 × 10^22 = 3.56 × 10^22, close to the numerical minimum.
- 146 tokens per parameter is inside the ranges tested by Sardana et al. (up to 10,000, with no loss plateau observed) and Gadre et al. (up to 640, with C4 loss forecast within 0.7% at M = 640) on their own data. The refit itself comes from Hoffmann et al.'s runs, and Sardana et al. found that fits restricted to 100 tokens per parameter or fewer overestimate the loss reduction from extra tokens at higher ratios (§5, Table 1), so the 648B-token figure may be an underestimate of the tokens needed.

**Objective 3: fine-tunability.** No source provides a law, so this objective is a procedure, not a formula (Interpretation, this course).
1. Compute the plan's ratio (146) and place it against the only measurements: degradation after 2,500 tokens per nominal parameter at OLMo-1B, none observed up to about 430 at OLMo-2-7B ([[overtrained-lms-harder-to-finetune]] §3.1).
2. Because those measurements come from other sizes, do not transfer the thresholds. Train with a schedule that can be branched (a constant learning rate followed by a cooldown; [[cooldown-scaling-beyond-fixed-durations]] §3.2), produce annealed checkpoints at two token budgets, apply the same SFT recipe to both, and compare held-out suites with paired intervals (ch-30a).
3. Keep the longer run only if its post-trained model is not worse on the held-out suites.

## Recipe

Rows quote the settings used by the scaling studies and released models in this chapter. Excerpt rows were read at the locus on 2026-09-15; card rows were verified on 2026-09-14.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Kaplan et al. study models ([[kaplan-scaling-laws]]) | 768 to 1.5B non-embedding | pretrain-stable | optimizer; steps; batch; schedule | Adam (Adafactor above 1B); 2.5 × 10^5 steps; 512 sequences × 1024 tokens; 3,000-step linear warmup, cosine to zero | arXiv:2001.08361v1 §2.2, §3 | verified 2026-09-15 | "results at convergence were largely independent of learning rate schedule" (§2.2); no table |
| Chinchilla Approach-1 grid ([[chinchilla-compute-optimal]]) | 70M to over 10B | pretrain-decay/anneal | max LR; decay; schedule length | 2 × 10^−4 (smallest) to 1.25 × 10^−4 (largest); cosine, 10× decay; cycle length matched to steps | arXiv:2203.15556v1 App. D.1, App. B | verified 2026-09-15 | Fig. A1: cycle more than 25% longer than the run noticeably degrades loss |
| Chinchilla 70B | 70B | pretrain-stable | tokens; max LR; batch (tokens); optimizer | 1.4T; 1 × 10^−4; 1.5M → 3M, doubled midway; AdamW | Table 1, Table 4, §4.1 | verified 2026-09-15 | size from Approaches 1–3 (40B–70B range, §4); AdamW gave lower LM loss and better fine-tuned performance (§4.1, App. G) |
| Gopher 280B | 280B | pretrain-stable | tokens; max LR; batch (tokens) | 300B; 4 × 10^−5; 3M → 6M | Table 1, Table 4 | verified 2026-09-15 | no ablation reported (baseline) |
| Porian et al. OpenLM grid ([[resolving-scaling-discrepancies-recipe]]) | 5M–901M | pretrain-stable | warmup; LR and batch laws; β2 | warmup tokens = N; BS = 0.00037·N^0.703; LR = 3.7·N^−0.36; β2 0.99 for Table 4 batch ≤ 192, 0.95 above | arXiv:2406.19146v4 §3.3, §3.5, Fig. 3, Table 4 | verified 2026-09-14 | App. F Table 5 warmup ablation (108M); 901M: prescribed LR and batch lowest loss of 13 configurations (App. G.5 Table 7) |
| DeepSeek LLM 7B ([[deepseek-llm]]) | 7B | pretrain-stable | tokens; batch (sequences); context; peak LR | 2.0T; 2304; 4096; 4.2 × 10^−4 | arXiv:2401.02954v1 Table 2 | verified 2026-09-15 | chosen from §3 laws (Table 2 caption) |
| DeepSeek LLM 67B | 67B | pretrain-stable | tokens; batch (sequences); context; peak LR | 2.0T; 4608; 4096; 3.2 × 10^−4 | Table 2 | verified 2026-09-15 | chosen from §3 laws (Table 2 caption) |
| DeepSeek LLM 7B and 67B | 7B, 67B | pretrain-stable | optimizer; schedule; clip | AdamW β1 0.9, β2 0.95, weight decay 0.1; 2000 warmup steps; LR to 31.6% of max after 80% of tokens and 10% after 90%; clip 1.0 | §2.3 | verified 2026-09-15 | Fig. 1a (1.6B, 100B tokens): multi-step final performance "essentially consistent" with cosine |
| DeepSeek scaling-law runs | 1e17–3e20 FLOPs | eval-gate | hyperparameter and allocation laws | η_opt = 0.3118·C^−0.1250; B_opt = 0.2920·C^0.3271; M_opt = 0.1715·C^0.5243; D_opt = 5.8316·C^0.4757 | §3.1 Eq. 1, §3.2 Eq. 4 | verified 2026-09-15 | near-optimal = within 0.25% of minimum loss; checked at 1e20 FLOPs (Fig. 2b); predicted 7B and 67B loss (Fig. 5) |
| Llama 3.1 405B ([[llama-3-recipe]]) | 405B | pretrain-stable | tokens; compute; size choice | 15.6T; 3.8 × 10^25 FLOPs; law predicts 402B on 16.55T tokens, 405B chosen | arXiv:2407.21783v3 §1, §3.2.1 | verified 2026-09-14 | IsoFLOP minima from 6 × 10^18 to 10^22 FLOPs; flat minimum at large budgets |
| Llama 3 scaling-law models | 40M–16B | pretrain-stable | schedule; LR; weight decay; batch | cosine, 2,000-step warmup, peak 2 × 10^−4 to 4 × 10^−4, decay to 0.1 of peak; weight decay 0.1 × LR; batch 250K–4M (unit not printed) | §3.2.1 | verified 2026-09-15 | n/a |
| Llama 3.1 8B, 70B | 8B, 70B | pretrain-stable | tokens | not reported per size; trained "much longer than is compute-optimal" | §1 | not reported (body and Table 1 checked, per card) | n/a |
| SmolLM2-1.7B ([[smollm2]]) | 1.7B | pretrain-stable | tokens; schedule; batch | ~11T tokens; WSD, 2,000 warmup steps, peak 5.0 × 10^−4, decay to zero over 10% of steps; 2M tokens per batch | arXiv:2502.02737v1 Abstract, §4.1, App. A Table 6 | verified 2026-09-15 | WSD chosen "to avoid setting a fixed training duration" (§4.1); no token-budget ablation reported |
| SmolLM2-360M; SmolLM2-135M | 360M; 135M | pretrain-stable | tokens | 4T; 2T | §6 | verified 2026-09-15 | data ablations re-run at target length (§6) |
| Gadre et al. testbed ([[overtraining-downstream-scaling]]) | 0.011B–6.9B | pretrain-stable | multipliers; sweep LR; optimizer; sequence length | M ∈ {5, …, 640}; LR 3e-3 in sweeps; AdamW β2 0.95, independent weight decay 1e-4, z-loss 1e-4, cosine to 3e-5; 2,048 | arXiv:2403.08540v2 §3.2, App. C | verified 2026-09-15 | 435-model grid search at M = 20 (§3.2) |
| Model ladder ([[task-scaling-model-ladders]]) | 190M–1.3B, 16 models | eval-gate | token multipliers; compute | 1×, 2×, 5×, 10× of 20·N tokens; 5.2 × 10^21 FLOPs total | arXiv:2412.04403v2 §2.1 | verified 2026-09-14 | App. B.3: adding larger sizes lowered error more than longer training |

**Starting point for a small general-purpose run.** For decoder-only models between 5M and 901M parameters trained near 20 tokens per parameter on RefinedWeb, the verified Porian et al. rows give warmup tokens equal to N and learning rate and batch size from BS = 0.00037·N^0.703 and LR = 3.7·N^−0.36, validated up to 901M parameters and about 14B tokens. To choose N and D for a new data set, the verified Chinchilla rows give the IsoFLOP protocol: at each budget, vary N, set the cosine cycle to the run's token count with a 10× decay, and avoid cycles more than 25% longer than the run. When the final token budget may change, SmolLM2-1.7B used a WSD schedule with 2,000 warmup steps and a decay over 10% of steps at 11T tokens; the source does not report whether this choice affected loss.

## Generalization lens

**(a) What increases breadth.**
- At equal compute, moving from the Kaplan allocation to equal scaling raised 5-shot MMLU from 60.0% (Gopher 280B, 300B tokens) to 67.6% (Chinchilla 70B, 1.4T tokens) ([[chinchilla-compute-optimal]] Table 6). Status: **Result (single study)**, one pair of runs that also differ in optimizer (AdamW versus Adam), tokenizer normalization, and data-subset proportions (§4.1).
- More tokens per parameter kept raising the average over five downstream categories up to 10,000 tokens per parameter at 150M ([[beyond-chinchilla-inference-scaling]] §4, Fig. 3), and lower C4 loss predicted lower 17-task average error ([[overtraining-downstream-scaling]] §4).
- Higher-quality data shifted DeepSeek's optimal allocation toward model size (a from 0.450 to 0.524; [[deepseek-llm]] Table 4). Interpretation: data quality changes the value of each token; ch-09 and ch-10a measure data composition and quality selection directly.

**(b) What causes narrowing or forgetting.**
- Over-training a 1B base model beyond 2.5T tokens lowered post-trained AlpacaEval and ARC scores in one family ([[overtrained-lms-harder-to-finetune]] §2–§3.1).
- Choosing a data mix by one validation corpus can select the wrong corpus for a task: the C4-versus-RedPajama ordering on CoQA reversed between validation sets ([[scaling-laws-unreliable-downstream]] §3).
- A forecast of the average can hide per-task misses: HellaSwag relative error 79.58% (C4) with the average at 0.14% ([[overtraining-downstream-scaling]] Table 2).

**(c) How to measure it at this stage.**
- Loss on several held-out domains, not one validation corpus (ch-00 §5), and task loss in bits per byte for forecasts ([[signal-and-noise-eval]] Fig. 6).
- Backtest: fit on the smaller runs, predict the largest held-out run, report per-task and average error ([[task-scaling-model-ladders]] Fig. 2).
- Checkpoint noise: standard deviation of task loss over the final 10 checkpoints; high values mark tasks that cannot be forecast ([[task-scaling-model-ladders]] §5).
- Post-training probe on two token budgets with paired held-out comparisons (§9 objective 3; ch-30a).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Fitting with non-embedding N at small scale when the head or attention cost is large | Exponent a of 0.7 or higher (Porian et al. obtained 0.835 and 0.706 before the warmup and tuning corrections); small models look over-sized | Recompute C with head and attention FLOPs (§1); refit and compare a ([[resolving-scaling-discrepancies]] Table 1) |
| One warmup length and schedule for all budgets | Small models reach their best loss during warmup; loss envelope taken from mid-schedule checkpoints | Set warmup per size (Porian et al. used warmup tokens = N, §3.3); keep the cosine cycle within 25% of the run length ([[chinchilla-compute-optimal]] App. B) |
| Hyperparameters tuned at one size only | Exponent shifts when LR or batch is retuned | Sweep LR and batch at two or more sizes; compare a before and after (Porian step 5) |
| Copying rounded published coefficients | Extrapolated N or D differs from the paper's own stated prediction | Reproduce the paper's prediction (40B at the Gopher budget; 16.55T for Llama 3) before reuse (§4) |
| Treating 20 tokens per parameter as a constant | Plan disagrees with a local IsoFLOP minimum | Fit IsoFLOP minima at three budgets on your data; DeepSeek's a ranged from 0.450 to 0.578 across data sets |
| Forecasting one benchmark's accuracy from compute in one step | Large backtest error; degenerate fits | Two-step fit through task loss; backtest on a held-out larger run; report SD10 ([[task-scaling-model-ladders]] Table 3, §5) |
| Selecting data by loss on one validation set | Ranking flips when the validation set changes | Evaluate candidate mixes on two or more validation corpora and on task loss ([[scaling-laws-unreliable-downstream]] §3) |
| Extending pre-training without a post-training check | Base model improves while the SFT model does not | Apply the same SFT to two annealed token-budget checkpoints; paired held-out comparison (§8, ch-30a) |
| Ignoring repetition when D exceeds unique tokens | Loss above the law's prediction late in training | Compare D with unique tokens; above about 4 epochs apply the data-constrained law (ch-14; [[data-constrained-scaling]] §6) |

## Check your understanding

1. Kaplan et al. trained every model on one schedule of 2.5 × 10^5 steps. Explain step by step why reading losses from partially trained runs of that schedule biases the fitted N ∝ C^a exponent upward, and why Porian et al. still found schedule decay to change the exponent by only about 0.03.
2. Counting the output head's FLOPs lowered Porian et al.'s exponent from 0.835 to 0.706. Explain why a counting error that is large for small models and small for large models changes the slope of N_opt against C, rather than only its intercept.
3. Using Eq. 4 of Gadre et al., explain why over-training by a fixed multiplier costs a constant compute factor at every budget, and what assumption about α and β this conclusion depends on.
4. In §9, the inference-aware plan uses 40.7% more training compute than the loss-optimal plan and still lowers total cost by 33.2%. Identify which term of 6ND + 2N·D_inf produces the saving and describe how the answer changes when D_inf falls to 2 × 10^11 tokens.
5. The 17-task average error of a 6.9B model was predicted within 0.05%, while HellaSwag alone was predicted with 25.73% relative error on the same training set. Give two mechanisms from §6–§7 that make averages more predictable than individual tasks.
6. Schaeffer et al. show that the correct-answer probability can be unchanged while Accuracy flips from 1 to 0. Explain why a continuous metric such as bits per byte on the correct answer improves decision accuracy, and what it still fails to measure about a generative model.
7. Springer et al. found degradation after instruction tuning for OLMo-1B beyond 2.5T tokens but not for OLMo-2-7B up to 3T. State what this does and does not imply for a 7B model trained on 15T tokens, and design the smallest experiment that would answer the question for your own run.

## Connections

- **Previous chapter:** ch-08 — Lab: Minimal Trainer with a Target-versus-General Capability Measurement.
- **Next chapter:** ch-08b — Why Next-Token Pretraining Produces General Ability: In-Context Learning, Emergence, and Predictability.
- **Depends on:** ch-00 — What General Capability Means and How It Is Measured (held-out suites, bits per byte, measurement noise); ch-03 — Learning-Rate Schedules, Batch Size, Initialization, and Normalization (schedule and batch mechanics that scaling experiments must control).
- **Used later:** ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale (choosing mixtures from small-scale fits); ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination (the D > unique-tokens case); ch-14a — Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures (released budgets); ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control (the post-training probe of §8–§9); ch-47 — Evaluation Harness and Suite Design for General Capability (per-task versus aggregate evaluation).

## Sources

- [[kaplan-scaling-laws]] — excerpt of Kaplan et al. 2020: Eq. 1.1–1.8, N conventions, 6N FLOPs per token, N ∝ C_min^0.73, training protocol.
- [[chinchilla-compute-optimal]] — excerpt of Hoffmann et al. 2022: three estimators, Eq. 2 and Eq. 4, Table 2–4, App. B schedule-length result, MMLU comparison, limits.
- [[resolving-scaling-discrepancies]] — Porian et al. 2024: step-by-step exponent corrections, hyperparameter laws, scope limits.
- [[resolving-scaling-discrepancies-recipe]] — Porian et al. training settings used in the Recipe row.
- [[chinchilla-replication]] — excerpt of Besiroglu et al. 2024: refit coefficients, rounding and optimizer causes, confidence-interval analysis, 4–40 ratio range.
- [[deepseek-llm]] — excerpt of the DeepSeek LLM report: M convention, hyperparameter and allocation laws, data-quality exponents, 7B/67B settings.
- [[beyond-chinchilla-inference-scaling]] — Sardana et al.: inference-aware objective, 17% and 28% examples, 47 runs to 10,000 tokens per parameter, fit bias at extreme ratios.
- [[overtraining-downstream-scaling]] — excerpt of Gadre et al. 2024: Eq. 4 and Eq. 5, 0.7% loss and 0.05% average-error forecasts, per-task errors, M* range.
- [[llama-3]] — §3.2.1 IsoFLOP fit, 402B/16.55T extrapolation, two-step ARC Challenge forecast, over-training of smaller models.
- [[llama-3-recipe]] — Llama 3.1 token counts and scaling-law run settings, including unreported per-size tokens.
- [[smollm2]] — excerpt of the SmolLM2 report: 11T, 4T, and 2T token budgets and the WSD schedule.
- [[task-scaling-model-ladders]] — two-step task-loss and sigmoid forecasts, 3.8 and 4.2-point errors, SD10 predictability indicator.
- [[scaling-laws-unreliable-downstream]] — Lourie et al.: 18 of 46 tasks predictable, validation-corpus and harness reversals.
- [[predicting-downstream-elusive]] — Schaeffer et al.: metric transformation chain, 90% versus 40% correlated samples, incorrect-choice mechanism.
- [[data-constrained-scaling]] — Muennighoff et al.: repetition limits used in the common-mistakes table and the link to ch-14.
- [[overtrained-lms-harder-to-finetune]] — excerpt of Springer et al. 2025: OLMo post-training degradation, progressive sensitivity, controlled runs, theory, confound.
- [[cooldown-scaling-beyond-fixed-durations]] — constant learning rate plus cooldown as a branchable schedule for the §9 post-training probe.
- [[emergence-loss-perspective]] — loss threshold below which several benchmarks leave random level (Failure 3).
- [[same-loss-better-downstream]] — equal pre-training loss with different transfer in simplified settings (Failure 4).
- [[datadecide]] — single-scale rankings as an alternative to extrapolated laws for data decisions.
- [[signal-and-noise-eval]] — signal-to-noise ratio and bits per byte for small-scale decisions (Failure 5, measurement).

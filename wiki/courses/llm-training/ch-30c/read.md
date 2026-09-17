<!-- chapter: ch-30c
     track: sft
     kind: content
     title: Weight Averaging and Model Merging for Generalist Models
     deps: [ch-30b]
     sources: [[wise-ft]], [[model-soups]], [[task-arithmetic]], [[ties-merging]], [[dare-merging]], [[model-merging-at-scale]], [[mix-data-or-merge-models]], [[online-merging-optimizer]], [[warm-weight-averaged-reward-models]], [[warp]], [[gemma-2]], [[qwen-2.5]], [[smollm-3]], [[smollm3-model-merging]], [[llama-3]], [[tulu-3]], [[tulu-3-seed-soups]], [[longer-context-deeper-thinking]], [[longred]], [[kimi-k1-5]]
     figures: figures/merge-calculator.html
     revised: 2026-09 (generality revision)
-->

# Chapter 30c — Weight Averaging and Model Merging for Generalist Models

> **Core insight.** Models fine-tuned from the same pre-trained checkpoint can be averaged in weight space with accuracy at or above the average of the endpoints' accuracies: for CLIP ViT-L/14@336px, the midpoint between zero-shot and fine-tuned weights scores 86.8 on ImageNet and 76.9 on five distribution shifts, against 86.2 and 68.6 for the fine-tuned model ([[wise-ft]], Table 1). Merging experts for different skills stays below multitask training on the experts' own tasks in every setting of the three expert-merging studies except one (two merged 64B PaLM-2-IT experts), for example 66.4 vs 73.1 for (IA)3 on T0-3B ([[ties-merging]], Table 1) and 0.93 vs 0.99 normalized held-in score for eight 64B PaLM-2-IT experts ([[model-merging-at-scale]], App. Table 1). On unseen tasks that 64B merge scores 1.09 vs 1.05 for multitask training, while at 1B-24B, and at every size for the non-instruction-tuned base, multitask training generalizes as well or better (App. Tables 2 and 4). Llama 3, Gemma 2, Qwen2.5 and SmolLM3 use weight averaging or merging in post-training, and none of these reports publishes ablation numbers for that step.
>
> **Guideline.** When several runs of the same recipe differ only in seed, data version, or hyperparameters, average them and keep the average only if it scores at least as well as the best single run on a held-out validation set, because the greedy-soup rule guarantees this on validation data ([[model-soups]], Recipe 1) and Tülu 3 found the best soup within 0.1 points of the best seed ([[tulu-3-seed-soups]], Table 14). When a fine-tuning or RL stage reduces an ability that the earlier checkpoint had (robustness, long context, benchmark breadth), interpolate back toward that checkpoint and choose the weight by measuring both the target and the lost ability, because interpolated models beat the fine-tuned endpoint on both evaluations in [[wise-ft]] (Fig. 6), interpolations toward the initialization lie above the RL training trajectories in [[warp]] (§4.3), and SmolLM3 restored RULER this way ([[smollm3-model-merging]]). When separate skill experts must become one model, prefer mixed-data training if data and compute allow, because merges trail multitask training on the experts' own tasks in [[ties-merging]], [[task-arithmetic]] and [[model-merging-at-scale]] (the one exception there is two merged 64B PaLM-2-IT experts, 1.00 vs 0.99); merge experts when the base model is large and instruction-tuned, when data cannot be pooled, or when two objectives trade off under mixing (safety and general helpfulness in [[mix-data-or-merge-models]], Table 1). In all three cases, report every merge with per-task held-in retention and held-out scores side by side.

## Why this chapter matters for a general-purpose model

A general-purpose model has to keep abilities from many stages at once: knowledge from pre-training, long-context skill from context extension, instruction following from SFT, preference-aligned behavior from DPO or RL, and safety behavior. [[ch-30a]] showed that each fine-tuning stage can lower abilities it did not target (forgetting and alignment tax), and [[ch-30b]] showed that mixing skills in one SFT dataset causes interference that depends on data amounts. This chapter covers a second set of tools that act on weights instead of data.

Weight-space methods are used for three different purposes, and the evidence differs for each:

1. **Interpolation toward an earlier checkpoint** to recover an ability lost during fine-tuning: WiSE-FT for robustness, linear interpolation toward the initialization (LITI) in WARP for RLHF, and the SmolLM3 merge that restored its RULER long-context benchmark score.
2. **Averaging runs of one recipe** instead of keeping only the best run: model soups, Llama 3 and Gemma 2 post-training averaging, WARM for reward models.
3. **Combining experts trained on different skills** into one model: task arithmetic, TIES-Merging, DARE, merging at scale, and the mix-or-merge comparison.

In the pipeline pre-training → mid-training → SFT → preference optimization → RL → evaluation, the reports cited in this chapter place merging at several post-training boundaries: Llama 3 averages at the reward-model, SFT and DPO stages ([[llama-3]], §4.1.5); Qwen2.5 applies a merging optimizer inside DPO ([[qwen-2.5]], §4.2); WARP merges RL policies ([[warp]]); SmolLM3 merges after preference optimization to restore mid-training long-context ability ([[smollm3-model-merging]]). The measurement question is the same at each boundary: a merge moves several abilities at once, so it has to be evaluated on the abilities it was meant to keep, not only on the one it was meant to add.

## §1 Weight interpolation and linear mode connectivity

**Definition.** Weight interpolation between two models with the same architecture produces a third model whose parameters are a weighted average of the two parameter vectors. **Linear mode connectivity** is the property that accuracy along the straight line between the two parameter vectors does not fall below the straight line between their accuracies. This is the form used in [[wise-ft]] (Eq. 2); the paper notes that it matches the linear mode connectivity definition of Frankle et al. when the two endpoint accuracies are similar (§5.2).

**Problem addressed.** Fine-tuning raises accuracy on the target distribution and can lower accuracy under distribution shift. For CLIP ViT-L/14@336px, end-to-end fine-tuning on ImageNet raises ImageNet accuracy from 76.6 to 86.2 and lowers the average over five ImageNet-derived shifts from 73.4 to 68.6 ([[wise-ft]], Table 1, "Zero-shot (PyTorch)" and "Fine-tuned E2E" rows).

**Mechanism (WiSE-FT, [[wise-ft]] §3).**
1. Start from a pre-trained model θ0 that already performs the task zero-shot.
2. Fine-tune it on the target data to obtain θ1.
3. Evaluate models on the line between θ0 and θ1 and pick a mixing coefficient α.

**Formula.**

θ_α = (1 − α)·θ0 + α·θ1  ([[wise-ft]], Eq. 1)

Acc((1 − α)·θ0 + α·θ1) ≥ (1 − α)·Acc(θ0) + α·Acc(θ1) for all α ∈ [0, 1]  ([[wise-ft]], Eq. 2, Observation 1)

- θ0: pre-trained (zero-shot) parameters; θ1: fine-tuned parameters; α ∈ [0, 1]: mixing coefficient (α = 1 is the fine-tuned model).
- Acc(·): accuracy of the model with the given parameters on one evaluation distribution.

The paper's implementation interpolates every tensor in the state dict ([[wise-ft]], App. A):

```python
theta = {
    key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
    for key in theta_0.keys()
}
model.load_state_dict(theta)
```

**Worked example (derived from [[wise-ft]] Table 1).** The endpoint-accuracy line at α = 0.5 predicts ImageNet (76.6 + 86.2)/2 = 81.4. The interpolated model scores 86.8, which is 5.4 points above the line and 0.6 points above the fine-tuned model. On the five shifts the line predicts (73.4 + 68.6)/2 = 71.0; the interpolated model scores 76.9, which is 5.9 points above the line and 3.5 points above the zero-shot model. Both endpoints are beaten on both evaluations, which is Observation 2 of the paper (§5.2).

**Evidence.** WiSE-FT with α = 0.5 improves accuracy under shift by 3.5, 6.2, 1.7, 2.1, 9.0 and 23.2 points on six further shifts, while reference accuracy falls by at most 0.3 points ([[wise-ft]], §4; Result, single study). The optimal α is 0 to 0.4 points better than α = 0.5 on average, and the authors recommend α = 0.5 when no domain knowledge is available (App. B). In the same paper, 10 epochs at learning rate 3·10⁻⁵ and 3·10⁻⁶ differ by 0.3 points on ImageNet but by up to 8 points under shift (Fig. 3): target-set accuracy does not reveal which fine-tuning run kept robustness.

A related result appears in a language-model setting. LongReD extends Llama-3-8B to 32K context and compares continued pre-training (CPT) with averaging the original and extended checkpoints: the average scores 65.92 on its General category and 83.38 on RULER, against 65.39 and 82.80 for the CPT model ([[longred]], Table 7; Result, single study). The paper's own distillation method scores higher on both (67.51, 84.98), so averaging is a baseline there, not the best method.

**Conditions and limits.**
- Interpolating two networks trained from a random initialization gives no better than random accuracy ([[wise-ft]], §5.2), and linear mode connectivity does not hold for weights trained from scratch even with a shared random initialization ([[warm-weight-averaged-reward-models]], Remark 1). It has been observed when models share part of their training trajectory or a pre-trained initialization ([[wise-ft]], §5.2).
- WiSE-FT tests image classification only (§7). The "optimal α" rows in Table 1 choose α separately for each column, and each column is a test set, so the fixed α = 0.5 rows are the ones not selected on the reported numbers (Table 1 caption).
- Large deltas break the approximation: after 500B tokens of additional code pre-training, deltas between WizardCoder-Python-13B and Llama-2-13b are often above 0.01, and randomly dropping 10% of them takes HumanEval pass@1 from 63.41 to 0.0 ([[dare-merging]], §4.6). Distance from the shared base is therefore a precondition to check before any merge.

**Implication for a general-purpose model.** The fine-tuned endpoint is one point on a line of candidate models. When a stage trades breadth for target accuracy, each point on that line costs one weight interpolation and one evaluation run, with no training, and in [[wise-ft]] the α = 0.5 point kept more of both abilities than either endpoint.

## §2 Model soups: averaging runs of one recipe

**Definition.** A **model soup** is the uniform weight average of several models fine-tuned from the same pre-trained weights with different hyperparameters or seeds ([[model-soups]], §2). A **greedy soup** adds models one at a time and keeps each only if held-out validation accuracy does not decrease.

**Problem addressed.** The standard procedure trains many configurations and keeps the one with the best validation score. The discarded runs contain information, and the selected run is not necessarily best out of distribution: in the CLIP ViT-B/32 sweep the best single model scores 47.83 on distribution shifts while the uniform soup of all 72 models scores 51.45 ([[model-soups]], Table 3).

**Mechanism (Recipe 1, [[model-soups]]).**
1. Sort candidate models by held-out validation accuracy, highest first.
2. Start the ingredient set with the best model.
3. For each next model, average its weights with the current ingredients; keep it if validation accuracy of the average is ≥ that of the current soup.
4. Return the average of the kept ingredients.

**Formula.** θ_S = (1/|S|)·Σ_(i∈S) θ_i, where θ_i = FineTune(θ0, h_i) is the model trained with configuration h_i and S is the ingredient set ([[model-soups]], §2).

**Worked example (illustrative numbers).** Three runs have validation accuracies A = 80.4, B = 79.9, C = 78.0. The soup {A, B} scores 80.9 ≥ 80.4, so B is kept. The soup {A, B, C} scores 80.1 < 80.9, so C is rejected. The result is {A, B} with 80.9. Because step 3 never accepts a decrease, the greedy soup cannot score below the best single model on the validation set; it can score below it on the test set.

**Evidence.**

| Setting | Best single model | Uniform soup | Greedy soup (Tülu 3: best soup reported) | Source |
|---|---|---|---|---|
| CLIP ViT-B/32, 72 runs, ImageNet / shifts | 80.38 / 47.83 | 79.97 / 51.45 | 81.03 / 50.75 | [[model-soups]], Table 3 |
| ViT-G/14 (JFT-3B), 58 runs, ImageNet / shifts | 90.78 / 84.68 (best on each test set; 90.72 / 84.38 for the model selected on held-out validation, Table 4) | not reported | 90.94 / 85.02 | [[model-soups]], Tables 1 and 4 |
| T5-base, 32 runs, RTE | 78.3 | not reported | 79.1 | [[model-soups]], Table 5 |
| Tülu 3 8B SFT, 5 seeds, average | 60.1 | not reported | 60.2 (2 seeds) | [[tulu-3-seed-soups]] |
| Tülu 3 70B SFT, 3 seeds, average | 72.6 | not reported | 72.5 (2 seeds) | [[tulu-3-seed-soups]] |

In Tülu 3 the seed range is 0.3 points at 8B and 2.6 points at 70B, and the best soup is within 0.1 points of the best seed in both directions; the team released the best single run ([[tulu-3-seed-soups]]; Result, single study). Llama 3 averages "models obtained from experiments using various versions of data or hyperparameters at each RM, SFT, or DPO stage" and computes a Polyak average (an average of checkpoints saved along one training run) during pre-training annealing ([[llama-3]], §4.1.5 and §3.4.3). Gemma 2 averages "different models obtained by running our pipeline with different hyperparameters", citing WARP, and its §4 overview also states that the models obtained after each phase are averaged ([[gemma-2]], §4). Neither report gives averaging weights, the number of averaged models, or an ablation.

**Conditions and limits.**
- The authors observe the uniform soup beating the best model only when all ingredients have high accuracy. An **error barrier** is a rise in error along the straight line between two models above the error of both endpoints; the authors find such barriers mainly between runs fine-tuned with high learning rates, and those runs also have lower accuracy, so the greedy rule excludes them ([[model-soups]], §3.3.1).
- Soups do not improve calibration, unlike output ensembles (§5, 20 models differing only in seed).
- Gains in the text-classification experiments are 0.0 to 0.8 points, and the authors call them preliminary (§3.3.3).
- Seed variance can exceed the soup gain. At 70B in Tülu 3, choosing the wrong seed costs up to 2.6 points, while the soup changes the best score by 0.1.

**Implication for a general-purpose model.** This course treats averaging runs as a variance-reduction step (Interpretation; no source above measures variance before and after souping). It needs a validation set disjoint from the reported test suite, and it does not replace running several seeds.

## §3 Task vectors, interference, and how TIES and DARE resolve it

### §3.1 Task arithmetic

**Definition.** A **task vector** is the parameter change produced by fine-tuning on one task: τ_t = θ_ft^t − θ_pre ([[task-arithmetic]], §2). Task arithmetic edits a model by adding scaled sums of task vectors.

**Formula.** θ_new = θ_pre + λ·Σ_t τ_t

- θ_pre: shared pre-trained weights; τ_t: task vector of task t; λ: scaling term chosen on held-out validation data. Plain weight averaging of n fine-tuned models is the special case λ = 1/n.

**Evidence.** Adding pairs of CLIP task vectors keeps 98.9% of the specialists' accuracy on average; with all eight task vectors available, the best model built from a subset of them reaches 0.912 normalized accuracy averaged over the eight tasks, against 0.994 for joint multitask fine-tuning (normalized accuracy is each task's accuracy divided by the accuracy of the model fine-tuned on that task) ([[task-arithmetic]], §4.1, App. D.2). Cosine similarity between task vectors of different tasks is 0.01-0.06 for most pairs and 0.18 for MNIST-SVHN (Fig. 5); the authors speculate that near-orthogonal task vectors add with little interference (Interpretation). λ between 0.3 and 0.5 is close to optimal in many of their cases (App. D.3).

### §3.2 Interference

**Problem, stated as a measurement.** When task vectors disagree on a parameter, their sum or mean shrinks that parameter toward zero ([[ties-merging]], Fig. 2), and for T5-Large the normalized average accuracy of all three methods in Fig. 5 (averaging, Task Arithmetic, TIES) falls as more tasks are added (§6, Fig. 5). In a 13B decoder merge of WizardLM-13B (instructions), WizardMath-13B and llama-2-13b-code-alpaca, Task Arithmetic over all three scores 58.45 on GSM8K, below 64.22 for WizardMath alone ([[dare-merging]], Table 1). TIES without DARE on the instruction and code models scores 0.0 on HumanEval and MBPP, against 36.59 and 34.00 for the instruction model alone (Table 1).

### §3.3 TIES-Merging

**Mechanism ([[ties-merging]], Algorithm 1).**
1. **Trim.** In each task vector keep the top-k% entries by magnitude and set the rest to zero.
2. **Elect.** For each parameter p, set the merged sign γ_p = sgn(Σ_t τ̂_t^p), the sign with the larger total magnitude.
3. **Disjoint merge.** Average only the non-zero entries whose sign equals γ_p.
4. **Scale.** θ_m = θ_init + λ·τ_m.

```
▷ Step 2: Elect Final Signs.      γm = sgn( Σ_t τ̂t )
▷ Step 3: Disjoint Merge.         Ap = { t ∈ [n] | γ̂t^p = γm^p };   τm^p = (1/|Ap|) Σ_(t∈Ap) τ̂t^p
▷ Obtain merged checkpoint        θm ← θinit + λ ∗ τm
```

**Worked example (course arithmetic; the same numbers are preset A in the figure).** One parameter has task values +0.5, +0.4, −0.6. Plain averaging gives (0.5 + 0.4 − 0.6)/3 = 0.10. Task Arithmetic with λ = 0.4 gives 0.4 × 0.3 = 0.12. In preset A all three entries survive the top-20% trim, so TIES elects sgn(0.3) = +, averages the agreeing entries (0.5 + 0.4)/2 = 0.45, and with λ = 1 keeps 0.45. Trimming also removes small entries: in preset A, 9 of 10 parameters have a sign conflict before trimming and 1 of 10 after keeping the top 20% of each vector.

The interactive figure [figures/merge-calculator.html](figures/merge-calculator.html) lets the reader edit three task vectors and see, parameter by parameter, what averaging, Task Arithmetic, TIES (trim, elect, disjoint mean) and DARE (drop and rescale) produce, including sign-conflict counts before and after trimming.

**Evidence ([[ties-merging]]).** With a validation set, TIES beats the best prior merge by 2.5 points on (IA)3, a parameter-efficient fine-tuning method, applied to T0-3B (66.4 vs 63.9) and by 3.6 on T5-large (76.9 vs 73.3), but multitask training scores 73.1 on (IA)3 (Table 1). On six held-out tasks, the T5-large TIES merge scores 40.4 against 36.0 for RegMean (a merge computed by per-layer linear regression on activations) and 27.6 zero-shot (Table 2). Removing the disjoint mean or the scaling step costs 3.2 and 5.2 points in the (IA)3 ablation (Table 6). Flipping signs of the top-20% entries lowers accuracy, while flipping the bottom 80% has little effect (§7.2, Fig. 7).

### §3.4 DARE

**Definition.** DARE (Drop And REscale) randomly sets a fraction p of each task vector's entries to zero and multiplies the remaining entries by 1/(1 − p) before merging ([[dare-merging]], Eq. 1).

**Formula.** m ∼ Bernoulli(p); δ̂ = ((1 − m) ⊙ δ)/(1 − p); θ_M = θ_PRE + λ·Σ_k δ̂_k

- δ = θ_SFT − θ_PRE: delta parameters of one SFT model; p: drop rate; m: per-entry drop mask; ⊙: element-wise product; λ: Task Arithmetic scaling term.

**Worked example.** δ = (0.002, −0.001, 0.0005, 0.003), p = 0.5, and the mask drops entries 2 and 3: δ̂ = (0.004, 0, 0, 0.006). For entry 1 the expected value is 0.5 × 0 + 0.5 × 0.004 = 0.002 = δ₁, which is why rescaling preserves each layer's expected output (§3.1). The figure's "DARE rescaling check" averages 4,000 random masks and shows the mean returning to τ.

**Evidence ([[dare-merging]]).** Deltas of the tested SFT models are mostly within 0.002, and the authors state that DARE can drop 90% or even 99% of them "without sacrificing much performance" (§5); WizardMath-70B tolerates p = 0.99 while 7B and 13B do not (§4.2). Adding DARE before TIES on the instruction and code models raises HumanEval from 0.0 to 18.29 and MBPP from 0.0 to 26.40, still below the instruction model alone (36.59, 34.00) (Table 1). For encoder models on GLUE, DARE changes merge averages by −0.03 to +0.84% (§4.3). One 7B DARE merge (supermario v2) ranked first among 7B models on the Open LLM Leaderboard as of 2024-01-28 (§4.3, Table 2); a leaderboard rank reached by merging public checkpoints is not evidence of breadth on tasks outside that leaderboard.

**Conditions and limits for §3.** All three methods assume a shared base and small deltas. Task Arithmetic's language experiments use GPT-2 (Small, Medium, Large) and T5, and TIES tests no decoder-only LLM; DARE's 13B results come from three public models of unequal quality, and the authors state that each source model must be "well fine-tuned" (§4.3).

**Implication for a general-purpose model.** Interference is measurable before evaluation: count parameters with opposite-signed large deltas, and check delta magnitudes against the base. After evaluation, the merged model's minimum per-task score shows interference that an average hides.

## §4 Merge or mix: evidence from multi-task generalists

**Merging at scale ([[model-merging-at-scale]]).** PaLM-2 and an instruction-tuned PaLM-2-IT at 1B, 8B, 24B and 64B were each fine-tuned into 8 experts on T0 held-in task categories, and 384 merges of 2-8 experts were evaluated on held-in tasks (normalized to the expert) and on 7 held-out datasets (normalized to the base model) (§3). Task Arithmetic values from App. Tables 1-4:

| Base, size (8 experts merged) | Held-in: merge / multitask | Held-out: merge / multitask |
|---|---|---|
| PaLM-2-IT 8B | 0.88 / 0.96 | 1.03 / 1.12 |
| PaLM-2-IT 24B | 0.92 / 0.98 | 1.18 / 1.18 |
| PaLM-2-IT 64B | 0.93 / 0.99 | 1.09 / 1.05 |
| PaLM-2 8B (not instruction-tuned) | 0.42 / 1.06 | 1.00 / 1.62 |
| PaLM-2 64B (not instruction-tuned) | 0.67 / 0.96 | 1.35 / 1.72 |

The abstract states that merges of eight large experts "often generalize better compared to the multitask trained models". In the tables this holds for 64B PaLM-2-IT with 6 or 8 experts (1.06-1.10 vs 1.05); at 24B the best merge ties multitask training, and for the non-instruction-tuned base multitask training is higher on held-out tasks at every size (App. Tables 2 and 4; Result, single study). Held-in retention falls from 0.66 to 0.39 as experts go from 2 to 8 for 1B PaLM-2, and from 0.91 to 0.86 for 1B PaLM-2-IT; the §4.4 text attributes these values to 8B, but they match the 1B columns ([[model-merging-at-scale]] excerpt, Discrepancies). At 64B PaLM-2-IT, all four methods (Averaging, Task Arithmetic, TIES, Dare-TIES) give 0.93 held-in with 8 experts (Table 1), so method choice matters less at that scale in this study (§4.5).

**Mix or merge under two objectives ([[mix-data-or-merge-models]]).** On a pre-release Aya 23 8B checkpoint, one model per objective (0% or 100% safety data) was merged and compared with a single model trained on a 15% safety mixture of the same data. The merge methods are Linear averaging, SLERP (spherical linear interpolation, which moves along the arc between two normalized weight vectors instead of the straight line), TIES, and DARE-TIES (DARE dropout followed by TIES) (§2.1). Harm is the relative change in harmful generations against the base (lower is better); general quality is Dolly-200 win rate against the base, both judged by GPT-4 (§2.4).

| Stage | Method | Harm | Win rate |
|---|---|---|---|
| DPO | 15% Safety Mix (one model) | −54.69 | 71.0 |
| DPO | Linear merge | −48.6 | 75.0 |
| DPO | SLERP merge | −57.8 | 78.0 |
| DPO | TIES merge | −65.1 | 63.6 |
| DPO | DARE-TIES merge | −55.9 | 78.5 |
| SFT | 15% Safety Mix (one model) | −56.6 | 67.4 |
| SFT | Linear merge | −49.1 | 76.0 |
| SFT | SLERP merge | −58.2 | 72.6 |
| SFT | TIES merge | −45.2 | 74.9 |
| SFT | DARE-TIES merge | −56.1 | 70.0 |

SLERP is the only method better than the mixture on both axes at both stages (DARE-TIES is better on both axes at DPO only); TIES at DPO has the lowest harm of the four merges and the lowest win rate of all DPO models (Table 1; Result, single study, no seeds reported). In a separate TIES experiment on SFT checkpoints, merging three monolingual models (English, French, Spanish) beats merging six (harm −65.3 vs −63.2, win rate 77.0 vs 71.2; Fig. 5), which the authors call cross-lingual interference; the paper does not state whether the three-language merge was evaluated on three or six languages (§3.4).

**Long-to-short reasoning.** Kimi k1.5 averages the weights of a long-CoT and a short-CoT model as one of four long2short methods; the authors state that in Figure 7 long2short RL has higher token efficiency than DPO and the merge, and the text gives no numeric scores for the merged model ([[kimi-k1-5]], §2.4, §3.4).

**Implication for a general-purpose model.** Merging experts is a substitute for mixed training mainly when the base model is large and instruction-tuned, or when mixing lowers one of two objectives (Interpretation of the two studies above). For smaller or non-instruction-tuned bases, the evidence above favors training on the mixture ([[ch-30b]]) and using merging for recovery or variance reduction.

## §5 Merging inside alignment: WARM, WARP, and the Online Merging Optimizer

**Alignment tax** is the loss of pre-training or SFT abilities during preference optimization or RL ([[ch-30a]], [[ch-38]]). Three methods apply weight-space operations inside that stage.

**Offline merge of the SFT and DPO models ([[online-merging-optimizer]], Table 1).** Merging a DPO model with its SFT reference after training (Linear, DARE, TIES; weights searched over 0.1-0.9) changes AlpacaEval 2.0 LC (length-controlled win rate against a baseline model's outputs, judged by GPT-4-1106-preview, App. B) by −0.41 to −0.96 for Qwen1.5-1.8B-Chat and by −1.35 to −1.65 for Qwen1.5-7B-Chat, relative to DPO with AdamW; for LLaMa-3-8B-Instruct the changes are +0.02 to +0.21. Benchmark averages change by −1.2 to +0.6 across the nine merges.

**Online Merging Optimizer.** At every optimizer step, the Adam update is sparsified and blended with the SFT model's delta from the pre-trained base:

θ_m^(t) = θ_m^(t−1) + (1 − α)·F_R(Δθ) + α·F_R(τ_r)  ([[online-merging-optimizer]], Eq. 3, OnDARE)

- θ_m^(t): policy weights after step t; Δθ: the Adam update at step t; τ_r = θ_r − θ_b: SFT reference weights minus pre-trained base weights; F_R: random mask that keeps each entry with probability p (the reserve rate, Eq. 4); α: merging weight, swept from 10⁻⁷ to 10⁻⁴ in the paper (Table 2); τ_r is the full SFT delta and is added at every step, and the authors report that large α makes training unstable (§5.3).
- OnDARE is the DARE-based variant defined by Eq. 3; OnTIES replaces the random mask with top-p magnitude sparsification and a sign-based combination rule (Eqs. 5-7).

Unlike DARE, the kept entries are not rescaled, because rescaling "harms the numeric stabilities in multi-step optimization" (§4.2). With DPO on UltraFeedback, OnDARE changes benchmark average / MT-Bench / AlpacaEval LC by +0.5 / +0.24 / +0.05 (Qwen1.5-1.8B-Chat), +1.1 / +0.12 / +0.28 (Qwen1.5-7B-Chat) and +1.3 / +0.19 / +1.57 (LLaMa-3-8B-Instruct) relative to AdamW (Table 1; Result, single study). The paper reports judge standard deviations of about 0.05 for MT-Bench and 0.5 for AlpacaEval LC (App. B), so the AlpacaEval gains of +0.05 and +0.28 are within one standard deviation. On Qwen1.5-1.8B-Chat, merging every K steps instead of every step moves MT-Bench back toward AdamW as K grows from 5 to 200 (§6.1). The paper's α table peaks at α = 5e−6 for the benchmark average while its text says 5e−7, and the table shows MT-Bench rising as α decreases while the text says it rises as α increases ([[online-merging-optimizer]] excerpt, Discrepancies). Qwen2.5 trains DPO on about 150,000 pairs for one epoch "using the Online Merging Optimizer" at learning rate 7 × 10⁻⁷, without giving α or p ([[qwen-2.5]], §4.2).

**WARM: weight-averaged reward models ([[warm-weight-averaged-reward-models]]).** M reward models, initialized from checkpoints of one SFT run and fine-tuned with different learning rates, dropout and data order, are averaged into one reward model (§3.1, §3.3, App. B.3). RL on TL;DR summarization with the M = 6 average gives a policy with a 79.4% oracle win rate against a policy trained with the best single reward model (§5.2, Fig. 9(c)). With 25% of preference labels swapped, the weight average fits fewer corrupted training pairs than a prediction ensemble and is more accurate on out-of-distribution pairs (Figs. 4-5). The authors state that if each reward model relies mainly on the same spurious feature, such as summary length, the average is likely to keep that reliance (§6, Limitations).

**WARP: weight-averaged rewarded policies ([[warp]]).** Three merges are combined on Gemma "7B" with REINFORCE:
1. The KL anchor is an exponential moving average of the policy instead of the fixed SFT model (§3.1).
2. Two independently trained RL policies are merged by spherical interpolation of their task vectors, slerp = θ_init + sin[(1−λ)Ω]/sin Ω·δ1 + sin[λΩ]/sin Ω·δ2, where δ_m = θ_m − θ_init and Ω is the angle between δ1 and δ2 in a layer (§3.2).
3. The merge is linearly interpolated toward the initialization, θ_η = (1 − η)·θ_init + η·θ_slerp (LITI), and θ_η with η = 0.3 starts the next iteration (§3.3-3.4).

Every LITI front lies above the RL training trajectories in KL-reward space (§4.3). Iteration 3 scores MBPP 45.4, MMLU 57.6, GSM8K 66.8, MATH 31.0, HumanEval 50.0 and BBH 58.8, against 39.0, 56.4, 55.6, 25.6, 46.9 and 53.1 for Gemma "7B" 1.1 (Table 2), and side-by-side scores stop improving after iteration 3 (Table 1). Output length grows with KL (App. E). RL task vectors from separate runs are close to orthogonal (Ω ≈ 90°) (App. C.2). The WARM card records that the Gemma 3 report lists improved versions of WARM and WARP for its RL phase (Gemma 3 §3, via [[warm-weight-averaged-reward-models]] Connections).

**Implication for a general-purpose model.** Inside alignment, merging acts as a regularizer toward the SFT or initial model whose strength can be chosen during or after training, instead of only through a KL coefficient fixed beforehand (Interpretation). The evidence covers Qwen1.5 and LLaMa-3 models of 1.8B-8B, Gemma "7B", and PaLM reward models and policies of unreported size, with one preference or summarization dataset per paper and judge-based metrics whose noise is reported only in [[online-merging-optimizer]].

## §6 Merging to recover lost ability: long context and stage checkpoints

**SmolLM3 ([[smollm3-model-merging]], official blog; [[smollm-3]]).** After Anchored Preference Optimization (APO), SmolLM3-3B improved on math, science, instruction following, coding, chat and multilingual evaluations but degraded on RULER. The team traced the drop to the reasoning mid-training stage and notes that APO data was limited to 24k tokens. The released checkpoint is a two-step merge: a soup of the APO checkpoints, then a linear merge of 0.9 × soup + 0.1 × a mid-training checkpoint with strong long-context performance, which "recover[ed] the base model's RULER score on contexts up to 128k tokens". RULER values, the number of soup ingredients, and other weights tried are not reported.

**Merge ratio sweep before reasoning SFT ([[longer-context-deeper-thinking]], App. B Table 9).** Qwen2.5-7B-Instruct was merged with Qwen2.5-7B-Instruct-1M at ratios 0 / 0.1 / 0.7 / 1.0 (weight on the 1M model) and each merge was then fine-tuned on reasoning data:

| Merge ratio | 0 | 0.1 | 0.7 | 1.0 |
|---|---|---|---|---|
| 32K long-context score | 78.1 | 79.1 | 79.5 | 77.7 |
| MATH500 (average of short- and long-SFT) | 84.16 | 84.92 | 84.38 | 82.88 |
| AIME22-24 (average) | 20.56 | 21.00 | 21.56 | 18.11 |
| GSM8K (average) | 92.37 | 92.66 | 92.67 | 92.59 |

The pure long-context model (ratio 1.0) is lowest on the 32K score, MATH500 and AIME22-24, so the model with the longest nominal context window was not the best starting point for those evaluations; on GSM8K the four ratios are within 0.30 points (Result, single study; number of runs not stated).

**Stage checkpoints in Llama 3.** Llama 3 averages checkpoints during the final 40M-token annealing phase to produce the pre-trained model, and averages models from different data or hyperparameter runs at each post-training stage ([[llama-3]], §3.4.3, §4.1.5). No ablation is reported, so the size of the effect is not reported.

**Worked example: choosing a recovery weight.** Suppose a post-trained model scores 60 on a short-context suite and 40 on RULER-128K, and the long-context checkpoint scores 45 and 70 (illustrative numbers). If both metrics followed the endpoint line, a 0.9/0.1 merge would score 0.9 × 60 + 0.1 × 45 = 58.5 and 0.9 × 40 + 0.1 × 70 = 43.0. Under that linear assumption, a 0.1 weight moves RULER only 3 of the 30 points toward the long-context checkpoint. SmolLM3 reports that its 0.1 weight recovered the base model's RULER score. Write S for the soup's RULER score, M for the mid-training checkpoint's score and B for the base model's score. Under the endpoint line, 0.9·S + 0.1·M ≥ B requires M ≥ B + 9·(B − S), so a soup 3 points below the base would need a checkpoint 27 points above the base. The blog gives none of S, M or B, so whether the recovery exceeded the endpoint-line prediction cannot be checked (Interpretation). Merge outcomes need not be linear in the weights, so merge weights have to be measured on both suites rather than computed.

## §7 Evaluating a merge: retention and breadth together

**Definition.** A merge evaluation reports two normalized quantities per task: **held-in retention**, the merged model's score on an expert's own task divided by that expert's score, and the **held-out ratio**, the merged model's score on a task no ingredient was trained for divided by the base model's score ([[model-merging-at-scale]], §3).

**Problem addressed.** A merge changes many abilities at once, and three reporting habits hide failures: averaging across tasks (a merged instruction and code model at 0.0 HumanEval, [[dare-merging]] Table 1), selecting merge weights on the reported test sets (per-column "optimal α" in [[wise-ft]] Table 1), and reading judge-score changes smaller than judge noise as gains (AlpacaEval LC standard deviation about 0.5, [[online-merging-optimizer]] App. B).

**Mechanism.**
1. Fix three suites before merging: held-in tasks of each ingredient, held-out tasks, and at-risk abilities that no ingredient targets (long context, safety, calibration, multilingual).
2. Fix a validation split for choosing α, λ, k, p and ingredients; report numbers only on a separate test split.
3. Compute retention and held-out ratios per task; report the mean and the minimum.
4. Compare with the best single ingredient, the base model, and a multitask model trained on the pooled data.
5. Repeat with a second seed or ingredient selection, and report the spread.

**Formula.**

r_t = S_merge(t) / S_expert(t),  g_u = S_merge(u) / S_base(u)

- t: a held-in task; u: a held-out task; S_merge, S_expert, S_base: scores of the merged model, the expert trained on t, and the base model.

**Worked example (illustrative numbers).** Two experts score 80 and 60 on their tasks; the merge scores 72 and 30, so r = 0.90 and 0.50, mean 0.70 and minimum 0.50. On a held-out task the base scores 50, the merge 54 and a multitask model 56, so g = 1.08 for the merge and 1.12 for multitask training. The mean retention of 0.70 hides a task at half its expert score, and the held-out ratio above 1 is still below the multitask baseline.

**Evidence.** Normalizing to the base can make weak bases look like large gains: the authors attribute the roughly 30% relative held-out improvement of 64B PaLM-2 merges to the base model's weak zero-shot performance ([[model-merging-at-scale]], §4.4), and multitask training on the same data scores 1.72 there (Table 4). Normalizing to the expert inflates retention when an expert is weak: in [[dare-merging]] the code expert scores 23.78 on HumanEval, below 36.59 for the instruction model (Table 1), so a merge that keeps the code expert's full score still has lower code ability than one of its own ingredients; the authors state that each merged model must be well fine-tuned (§4.3). Run-to-run spread can exceed merge gains: Tülu 3 70B SFT seeds span 2.6 points against a soup effect of 0.1 ([[tulu-3-seed-soups]]).

**Conditions and limits.** Three sources in this chapter report variability for a merge comparison: [[model-soups]] gives the spread over three random greedy-soup orders (Table 3: 80.79 with 0.05 in parentheses), [[warp]] reports small standard deviations over merge orders for M > 2 (App. B.3), and [[online-merging-optimizer]] gives judge standard deviations (App. B). The model reports cited here (Llama 3, Gemma 2, Qwen2.5, SmolLM3) give no variability for their merges.

**Implication for a general-purpose model.** A merge is accepted when its minimum retention, held-out ratio and at-risk abilities are all within a stated tolerance of the reference models, measured on splits that were not used to choose the merge.

## Negative samples and negative feedback

Merging does not itself produce rejected samples, but this chapter contains two negative signals: a weight-space negation that removes a behavior, and corrupted preference labels in reward-model averaging. The first differs from the four sample-level meanings of "negative" used across the course (negative marginal value, negative as content, negative as conditioning, negative as gradient):

1. **Where negatives come from.** Task arithmetic builds a vector from a model fine-tuned with ordinary cross-entropy on undesired content (GPT-2 Large on Civil Comments with toxicity > 0.8) and subtracts it: τ_new = −τ ([[task-arithmetic]], §3.2). The training data are positives for an unwanted behavior; the negation happens in weight space. WARM studies swapped (wrong) preference labels as corrupted supervision ([[warm-weight-averaged-reward-models]], §4.2).
2. **What practice does with them.** Negation lowers the likelihood of the behavior without per-sample gradients; it is closest to "negative as gradient" in effect (the behavior loses probability) but is applied once, as a weight edit (course classification). Preference merges in §4-§5 act on models already trained with the rejected term of DPO (negative as gradient upstream).
3. **Mechanism and evidence.** On GPT-2 Large, the negative task vector reduces toxic generations from 4.8% to 0.8% with WikiText-103 perplexity 16.4 → 16.9; gradient ascent on the same toxic data reaches 0.0% toxic but perplexity above 10¹⁰ ([[task-arithmetic]], Table 2). Gradient ascent is negative as gradient: for a target token y, ∂ log p_y / ∂ z_j = 1[j = y] − p_j, so decreasing log p_y lowers z_y and raises every other logit z_j in proportion to its current probability p_j, moving mass to the tokens the model already ranks next. The objective has no lower bound (log p_y → −∞) and no positive anchor, and the control perplexity above 10¹⁰ is consistent with the distribution on ordinary text being destroyed (Interpretation). The negated task vector is bounded in size by construction, because it is the displacement of one completed fine-tuning run, scaled by λ.
4. **Controls.** Scale the negation with λ chosen on a control task, compare with a random vector of the same norm (the paper's control: 4.8% toxic, perplexity 16.4), and keep a positive-data baseline (fine-tuning on non-toxic data: 1.8%, 17.2).
5. **Diagnostics and effect on generality.** Report the target metric and a control-task metric together; for reward-model averaging, compare weight averaging with prediction ensembling on held-out pairs with and without label noise ([[warm-weight-averaged-reward-models]], Figs. 4-5). No source in this chapter measures calibration or over-refusal after a negation edit (not reported).

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| WiSE-FT, CLIP ViT-L/14@336px | not reported | merge | mixing coefficient α (zero-shot ↔ fine-tuned) | 0.5 when no domain knowledge | arXiv:2109.01903v3 §4; App. B | verified 2026-09-15 ([[wise-ft]]) | App. B Table 3: optimal α 0-0.4 pp better than 0.5 on average |
| Model soups, CLIP ViT-B/32 | not reported | merge | ingredients; selection rule | 72 random-search runs; greedy soup sorted by held-out val accuracy, 5 selected | arXiv:2203.05482v3 §3.3.1, Recipe 1 | verified 2026-09-15 ([[model-soups]]) | Table 3: greedy 81.03 vs best single 80.38 ImageNet |
| Task Arithmetic, CLIP | not reported | merge | λ | chosen on held-out validation; 0.3-0.5 near optimal in many cases | arXiv:2212.04089v3 §2; App. D.3 | verified 2026-09-15 ([[task-arithmetic]]) | App. D.3 Fig. 15 |
| TIES-Merging (no validation set) | T5, (IA)3 T0-3B, ViT | merge | trim k; λ; Task Arithmetic baseline λ | top-20%; λ = 1; λ = 0.4 | arXiv:2306.01708v2 §5; App. C.4 | verified 2026-09-14 ([[ties-merging]]) | App. C.4 grid on eleven (IA)3 models |
| DARE 13B decoder merges | 13B | merge | scaling term; TIES retain ratio; drop rate | searched in [0.5, 1.0]; [0.5, 0.7, 0.9]; not reported for Table 1 | arXiv:2311.03099v3 §4.3 | verified 2026-09-15 ([[dare-merging]]) | Table 1 |
| DARE "supermario v2" | 7B | merge | drop rate p; Task Arithmetic λ | 0.5; 0.5 | arXiv:2311.03099v3 App. A.5 | verified 2026-09-15 ([[dare-merging]]) | Open LLM Leaderboard only (Table 2) |
| PaLM-2 and PaLM-2-IT experts (merging at scale) | 1B-64B | SFT | steps; LR; dropout | 2,000 (adjusted per task); 3e-5; 0.05 | arXiv:2410.03617v1 App. B | verified 2026-09-15 ([[model-merging-at-scale]]) | no ablation reported; merge λ not reported |
| Aya 23 8B pre-release (mix vs merge) | 8B | merge | method; weight grid; tool | Linear, SLERP, TIES, DARE-TIES; {0, 0.3, 0.5, 0.7, 1}; mergekit | arXiv:2410.10801v1 §2.1 | verified 2026-09-14 ([[mix-data-or-merge-models]]) | Table 1; selection metric not reported |
| Tülu 3 8B SFT / 70B SFT | 8B / 70B | merge | seeds; merge method; release choice | 5 / 3 seeds; mergekit linear weighted averaging; best single run released | arXiv:2411.15124v5 §4.3, §4.3.1, Table 14 | verified 2026-09-15 ([[tulu-3-seed-soups]]) | Table 14: best soup 60.2 vs 60.1 (8B), 72.5 vs 72.6 (70B) |
| Llama 3.1 405B | 405B | merge | model averaging | runs with different data or hyperparameters, at RM, SFT and DPO stages; weights not printed | arXiv:2407.21783v3 §4.1.5 | verified 2026-09-14 ([[llama-3]]) | no ablation reported |
| Llama 3.1 405B | 405B | pretrain-decay/anneal | checkpoint averaging | Polyak average of checkpoints during the final 40M-token anneal | arXiv:2407.21783v3 §3.4.3 | verified 2026-09-14 ([[llama-3]]) | no ablation reported |
| Gemma 2 IT (all sizes) | 2B/9B/27B | merge | method | average of pipeline runs with different hyperparameters (cites WARP); count and weights not reported | arXiv:2408.00118 §4 | verified 2026-09-14 ([[gemma-2]]) | no ablation reported |
| Qwen2.5 Instruct, open-weight | 0.5B-72B | preference | optimizer; pairs; epochs; LR | Online Merging Optimizer; ~150,000 pairs; 1; 7 × 10⁻⁷; α and p not reported | arXiv:2412.15115v2 §4.2 | verified 2026-09-14 ([[qwen-2.5]]) | no ablation reported |
| Online Merging Optimizer runs (Qwen1.5-1.8B/7B-Chat, LLaMa-3-8B-Instruct) | 1.8B-8B | preference | DPO β; batch; α grid | 0.1; 128; α from 1e−4 to 1e−7 | arXiv:2405.17931v1 App. C; §5.3 | verified 2026-09-15 ([[online-merging-optimizer]]) | App. C: batch 128 best of {64, 128, 512} |
| OnDARE merging weight α (model not named; values are on the scale of the Qwen1.5-1.8B-Chat rows of Table 1) | not reported | preference | α with best benchmark average | 5e−6 (Table 2: 41.8) | arXiv:2405.17931v1 Table 2 | conflict | Table 2, one run per α |
| OnDARE merging weight α | not reported | preference | α with best benchmark average | 5e−7 (text) | arXiv:2405.17931v1 §5.3 prose | conflict | contradicted by Table 2 of the same paper |
| WARM reward models (PaLM-XXS) | not reported | merge | M; members | uniform average of the M best RMs by OOD accuracy; M = 6 for main RL results | arXiv:2401.12187v1 App. B.3; §5.2 | verified 2026-09-14 ([[warm-weight-averaged-reward-models]]) | Figs. 6-10 compare M |
| WARP policy (Gemma "7B") | "7B" | merge | method; M; λ; LITI η; iterations | SLERP of task vectors per layer; 2; 0.5; 0.3; 5 | arXiv:2406.16768v1 §4, §4.4 | verified 2026-09-14 ([[warp]]) | Fig. 3(c) λ; App. D.3 Fig. 16 η; Table 1 plateau after iteration 3 |
| SmolLM3-3B | 3B | merge | recipe | soup of APO checkpoints; then linear 0.9 soup + 0.1 mid-training checkpoint; ingredient count not reported | huggingface.co/blog/smollm3, "Model Merging" | verified 2026-09-15 ([[smollm3-model-merging]]) | blog: "achieved the best performance"; alternatives not reported |
| Qwen2.5-Math-7B-Instruct (context-first recipe) | 7B | long-context | merge before reasoning SFT | 0.7 × (RoPE θ×16 model) + 0.3 × Qwen2.5-7B-Instruct-1M | arXiv:2505.17315 §3.5, Table 7 | verified 2026-09-14 ([[longer-context-deeper-thinking]]) | Table 7: MATH500 avg 88.70 vs 88.20 (θ×16 only) |

**Starting point for a small general-purpose run.** For averaging runs of one recipe, use the greedy-soup rule with a validation set disjoint from the test suite; this was tested on CLIP, ALIGN, ViT-G, BERT-base and T5-base fine-tuning ([[model-soups]]), and 2-seed soups of Tülu 3 8B and 70B SFT changed the best score by at most 0.1 points ([[tulu-3-seed-soups]]). For recovering an ability from an earlier checkpoint, start the sweep at α = 0.5 (WiSE-FT, CLIP image classification) and include the 0.9/0.1 point used by SmolLM3-3B for long context (α = 0.9 in the notation of §1); measure both the target suite and the recovered ability at each point. For merging same-base experts without a validation set, TIES with top-20% trimming and λ = 1 is the tested default on T5, T0-3B (IA)3 and CLIP models with up to eleven tasks.

## Generalization lens

**(a) What increases breadth.**
- Interpolating back toward the pre-trained model raises accuracy under distribution shift while keeping target accuracy: +8.3 points on five shifts and +0.6 on ImageNet at α = 0.5 over the fine-tuned CLIP model ([[wise-ft]], Table 1).
- Averaging many runs can raise out-of-distribution accuracy even when in-distribution accuracy falls: uniform soup +3.62 points on shifts and −0.41 on ImageNet against the best single CLIP ViT-B/32 run ([[model-soups]], Table 3).
- Merging experts from a large instruction-tuned base improves held-out generalization over the base: 1.09 normalized for 8 merged 64B PaLM-2-IT experts ([[model-merging-at-scale]], Table 2).
- Objective-level merging can improve two axes that data mixing trades off: SLERP at DPO, 78.0 win rate and −57.8 harm vs 71.0 and −54.69 for the mixture ([[mix-data-or-merge-models]], Table 1).
- Iterated policy merging in RLHF raises zero-shot benchmark scores on all six reported benchmarks for Gemma "7B" ([[warp]], Table 2).

**(b) What causes narrowing or forgetting.**
- Adding experts lowers held-in retention, most for small non-instruction-tuned bases (1B PaLM-2: 0.66 → 0.39 from 2 to 8 experts) ([[model-merging-at-scale]], Table 3).
- A merge can remove one skill entirely while the average looks acceptable: TIES instruction + code merge scores 0.0 on HumanEval and MBPP ([[dare-merging]], Table 1).
- Merges built from models far from the shared base fail: 10% delta dropout takes WizardCoder-Python-13B from 63.41 to 0.0 pass@1 when Llama-2-13b is used as the base ([[dare-merging]], §4.6).
- Offline merges of DPO models back toward SFT lower preference scores for the Qwen1.5 models (AlpacaEval LC −0.41 to −1.65) ([[online-merging-optimizer]], Table 1).
- The WARM authors state that a weight average is likely to keep a spurious feature that every ingredient relies on, such as summary length in reward models ([[warm-weight-averaged-reward-models]], §6); in WARP, output length rises with KL (App. E).
- Weight averages do not improve calibration the way ensembles do ([[model-soups]], §5).

**(c) How to measure it for this stage.**
- Report held-in retention per task (normalized to each expert) and held-out ratios (normalized to the base), with the minimum next to the mean (§7; [[model-merging-at-scale]] §3).
- Include a multitask-trained baseline on the same data; without it, "the merge generalizes" cannot be separated from "any multi-task model generalizes" (multitask held-out 1.62 vs merge 1.00 for 8B PaLM-2, [[model-merging-at-scale]] Table 4).
- Select α, λ, k, p and ingredients on a validation split, never on the reported test suite ([[model-soups]], Recipe 1; contrast the per-column "optimal α" rows in [[wise-ft]] Table 1).
- Report run-to-run spread (Tülu 3 70B SFT seeds span 2.6 points, [[tulu-3-seed-soups]]) and judge noise (AlpacaEval LC standard deviation about 0.5, [[online-merging-optimizer]] App. B).
- Evaluate each ability the merge could move, including long context (RULER, [[smollm3-model-merging]]), safety harm rate ([[mix-data-or-merge-models]]), and calibration ([[model-soups]] §5). See [[ch-00]] and [[ch-47]] for suite design.

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Merging models that do not share the same pre-trained base, or whose deltas are large (continued pre-training in between) | One or more tasks collapse to near zero after merging or after DARE dropout | Confirm the base checkpoint for every ingredient; compare delta-magnitude percentiles (SFT deltas mostly within 0.002 vs above 0.01 after 500B-token continued pre-training in [[dare-merging]] Fig. 10) |
| Reporting only the average over merged tasks | Average looks close to the experts while one task is at 0.0 ([[dare-merging]] Table 1) | Table of per-task retention and the minimum across tasks |
| Choosing α, λ or soup ingredients on the test suite | Merge gains that disappear on a fresh benchmark | Keep a validation split disjoint from test; compare against the fixed-α row (α = 0.5 in [[wise-ft]]) |
| Uniform averaging that includes weak or high-learning-rate runs | Uniform soup below the best ingredient ([[model-soups]] Table 3: 79.97 vs 80.38 ImageNet) | Evaluate the midpoint of each pair; use the greedy rule |
| Expecting a merge to match multitask training on its own tasks | Held-in scores below the mixed-data model (66.4 vs 73.1 in [[ties-merging]] Table 1) | Train or reuse a multitask baseline and report both |
| Treating judge-score changes within noise as gains | Gains of +0.05 to +0.28 AlpacaEval LC, below the reported judge standard deviation | Repeat judging; compare against the reported standard deviation (about 0.5, [[online-merging-optimizer]] App. B) |
| Picking a recovery merge weight without measuring both abilities | The endpoint with the strongest nominal long-context ability scores below intermediate weights on long-context and reasoning evaluations ([[longer-context-deeper-thinking]] Table 9, ratio 1.0) | Sweep at least three weights; evaluate RULER and the short-context suite at each |
| Describing the Qwen2.5 Online Merging Optimizer as a running merged checkpoint, or Gemma 2 averaging as the full EMA-SLERP-LITI WARP pipeline | Recipes that cannot be reproduced from the report | Qwen2.5 §4.2 names the optimizer only; its update rules are Eqs. 3-7 (OnDARE, OnTIES) of [[online-merging-optimizer]], and the report does not say which variant or which α and p were used; Gemma 2 §4 states that models from each phase and from runs with different hyperparameters are averaged, and describes no EMA, SLERP or LITI step ([[gemma-2]]) |
| Assuming a weight average is calibrated like an ensemble | Calibration error does not improve after souping, while an ensemble of the same models improves it | Measure calibration separately ([[model-soups]] §5) |

## Check your understanding

1. Interpolating two independently initialized networks gives random accuracy, while interpolating a pre-trained CLIP model with its fine-tuned version beats both endpoints. What property of the second pair explains the difference, and which measurement in [[dare-merging]] indicates that a pair of language models may not meet the precondition for that property?
2. In the CLIP ViT-B/32 sweep, the uniform soup is worse than the best single model on ImageNet but better on distribution shifts. Give a causal account of why averaging many runs can help out-of-distribution accuracy more than in-distribution accuracy, and state which part of your account is interpretation rather than measured.
3. For one parameter with task values +0.5, +0.4 and −0.6, averaging gives 0.10 and TIES gives 0.45. Explain why the TIES value can preserve the first two tasks better, and describe a case where electing the majority sign harms the third task more than averaging does.
4. DARE rescales surviving deltas by 1/(1 − p), but the Online Merging Optimizer omits rescaling. Explain what rescaling preserves in a single merge and why the same rescaling could destabilize a merge applied at every optimizer step.
5. In the merging-at-scale tables, merges beat multitask training on held-out tasks only for the 64B instruction-tuned base. Propose two mechanisms that could make merging work better for large instruction-tuned bases, and describe an experiment that would separate them.
6. SmolLM3 restored RULER with a 0.9/0.1 linear merge toward a mid-training checkpoint. Under what conditions could a 0.1 weight on the long-context checkpoint recover more than 10% of the lost long-context score, which measurements would show whether that happened, and what additional evaluations would you require before releasing such a merge?
7. WARM averages reward models, and the WARM authors state that the average is likely to keep a spurious feature that every member relies on. Explain how this limitation interacts with RL optimization pressure, and name one diversity source that could reduce it.

## Connections

- Previous: [[ch-30b]] — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares. Data-side interference that merging tries to avoid.
- Next: [[ch-31]] — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation.
- [[ch-30a]] — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control. Defines forgetting and alignment tax, which §1, §5 and §6 act on.
- [[ch-06]] — Checkpointing, In-Loop Evaluation, and Checkpoint Selection. Checkpoint averaging within one run.
- [[ch-01]] — Optimizers for LLM Training: AdamW, Update Size, and Retention of Prior Ability. Optimizer-level retention, including the Online Merging Optimizer.
- [[ch-32b]] — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression. Long-context regression that §6 merges try to recover.
- [[ch-34]] — Case Studies B: Generality versus Specialization in Qwen, OLMo, and Phi Reports. Qwen post-training pipeline in context.
- [[ch-38]] — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax. KL regularization that WARP's EMA anchor and LITI modify.
- [[ch-41]] — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization. Reward-model over-optimization, the problem WARM targets.
- [[ch-00]] — What General Capability Means and How It Is Measured; [[ch-47]] — Evaluation Harness and Suite Design for General Capability. Suite design for held-in and held-out evaluation of merges.
- [[ch-58a]] — Open General-Model Recipes End to End: Pretraining to Merge. Where merging sits in complete recipes.

## Sources

- [[wise-ft]] — interpolation between zero-shot and fine-tuned weights, linear mode connectivity condition, Table 1 numbers, α = 0.5 recommendation (excerpt; library card not yet present).
- [[model-soups]] — uniform and greedy soups, CLIP/ViT-G/BERT/T5 results, error barriers at high learning rate, calibration limit (excerpt).
- [[task-arithmetic]] — task-vector definition, addition and negation results, cosine similarity of task vectors, λ range (excerpt).
- [[ties-merging]] — trim, elect, disjoint-merge algorithm, in-domain and held-out results, ablation, no-validation defaults.
- [[dare-merging]] — drop-and-rescale formula, 13B decoder merge table, failure when deltas are large (excerpt).
- [[model-merging-at-scale]] — PaLM-2 1B-64B merge grid, held-in and held-out normalized tables, text-table discrepancy (excerpt).
- [[mix-data-or-merge-models]] — merge-versus-mix comparison on safety and general quality at SFT and DPO, language merges.
- [[online-merging-optimizer]] — offline SFT-DPO merges, OnDARE/OnTIES update rule, results, judge noise, α discrepancy (excerpt).
- [[warm-weight-averaged-reward-models]] — weight-averaged reward models, label-noise and OOD results, spurious-feature limit.
- [[warp]] — EMA anchor, SLERP of RL task vectors, LITI, iteration results and length growth.
- [[gemma-2]] — post-training averaging of runs with different hyperparameters (§4).
- [[qwen-2.5]] — DPO with the Online Merging Optimizer (§4.2).
- [[smollm-3]] and [[smollm3-model-merging]] — APO-soup plus mid-training-checkpoint merge to recover RULER (excerpt of the official blog).
- [[llama-3]] — model averaging at RM, SFT and DPO stages; Polyak averaging during annealing.
- [[tulu-3]] and [[tulu-3-seed-soups]] — SFT seed variance and seed soups (Table 14 excerpt).
- [[longer-context-deeper-thinking]] — merge-ratio sweep with a 1M-context model before reasoning SFT.
- [[longred]] — averaging original and context-extended Llama-3-8B as a continual-learning baseline (Table 7).
- [[kimi-k1-5]] — weight-averaging merge as a long2short baseline.

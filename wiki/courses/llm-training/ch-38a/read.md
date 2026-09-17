<!-- chapter: ch-38a
     track: preference
     kind: content
     title: SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity
     deps: [ch-38, ch-30a]
     sources: [[sft-memorizes-rl-generalizes]], [[debunk-sft-generalization]], [[rls-razor]], [[retaining-by-doing]], [[rlhf-generalisation-diversity]], [[artificial-hivemind]], [[noveltybench]], [[on-off-policy-rlhf]], [[on-policy-suboptimal-preference-data]], [[rl-finetunes-small-subnetworks]], [[transferability-of-llm-reasoning]], [[transferability-of-llm-reasoning-tables]], [[rlvr-beyond-base-model]], [[rlvr-beyond-base-model-tables]], [[prorl]], [[wolfe-online-vs-offline-rl]], [[john-schulman-kl-tricks]]
     figures: figures/on-policy-kl-and-passk.html
     revised: 2026-09 (generality revision)
-->

# Chapter 38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity

> **Core insight.** In controlled comparisons that hold the prompts, the task, and the starting checkpoint fixed, the stage trained on responses the model sampled itself keeps more of the abilities the stage did not target. Training Llama-3.1-8B-Instruct on Countdown moved the AlpacaEval win rate against GPT-4 from 41.6% to 0.0% with SFT on an external teacher's responses and to 42.0% with RL, at a higher target gain for GRPO than for SFT (60.4 against 25.5 points) ([[retaining-by-doing]] Table 1, Table 3). On Qwen3-14B-Base trained only on math, the RL run raised the non-reasoning average from 45.7 to 53.2 while the SFT run lowered it to 21.1 ([[transferability-of-llm-reasoning-tables]] Table 1). Two mechanisms are proposed and neither is settled: on-policy updates converge to solutions close to the base policy in KL ([[rls-razor]] §5), and mode-seeking updates keep the prior mode intact ([[retaining-by-doing]] §3). The advantage is not a property of the RL objective: in [[rls-razor]] §5.1 an on-policy method with no negative gradient (1–0 REINFORCE) behaved like GRPO while an offline method with negative gradients (SimPO) behaved like SFT, and in [[retaining-by-doing]] §4.2 SFT on data regenerated at the start of each epoch reached SFT's target accuracy with "mild to no forgetting". The cost of on-policy preference optimization and RL is measured elsewhere: per-input output diversity falls after RLHF ([[rlhf-generalisation-diversity]] §6.2), distinct generations fall at every OLMo 2 size across SFT → DPO → RLVR ([[noveltybench]] Fig. 6), and RLVR models are overtaken by their base models at large k on pass@k ([[rlvr-beyond-base-model-tables]] §3.2).
>
> **Guideline.** When a post-training stage must raise one capability without lowering the rest, generate the responses from the checkpoint being trained, because that choice, not the loss function, is what separates the retention profiles in [[rls-razor]] §5.1, [[retaining-by-doing]] §4, and [[transferability-of-llm-reasoning-tables]] Table 4. When full on-policy sampling is too expensive, regenerate the targets at the start of every epoch: Iterative-SFT matched SFT's target accuracy with mild to no forgetting ([[retaining-by-doing]] §4.2, Fig. 7). When an external teacher's responses must be used (the model cannot yet produce correct responses at all), expect the retention cost measured in [[retaining-by-doing]] Table 1 and reduce it with the controls of ch-30a, and do not treat a poor SFT out-of-distribution score as evidence about the SFT objective before the prompt template has been varied, because prompt diversity alone moved Sokoban instruction-variant success from 0 to 0.84-0.92 ([[debunk-sft-generalization]] Table 1a). When a stage is evaluated, report pass@k at large k, a diversity metric, and KL to the stage's input checkpoint next to pass@1, because pass@1 rose while coverage fell in [[rlvr-beyond-base-model-tables]] §3.2 and diversity fell while utility rose in [[noveltybench]] §4.4.

## Why this chapter matters for a general-purpose model

Pipeline position: pre-training → mid-training → SFT → **preference optimization and RL (this chapter's comparison)** → evaluation. ch-30 set the SFT design choices, ch-30a measured what a fine-tuning stage costs on abilities it did not target, ch-37 derived the policy-gradient estimator, and ch-38 added the KL-controlled PPO objective. This chapter asks which of two ways of spending the same prompts produces a model that is better outside the training distribution, and what the answer costs in output diversity.

The question is decided by evidence, not by the name of the algorithm. Three findings organize the chapter:

1. The measured retention difference between SFT and RL in the cited studies tracks **where the training responses come from**, not whether the loss contains a reward or a negative term (§2, §5).
2. The most-cited demonstration that "SFT memorizes" was produced with a single frozen prompt template, and a data change removes most of it (§3).
3. The stage that retains capability best still narrows the output distribution: fewer distinct answers per prompt, and lower coverage at large sampling budgets (§6, §7, §8).

Definitions used throughout. **On-policy data**: responses sampled from the checkpoint currently being updated. **Approximately on-policy data**: responses sampled from a recent copy of that checkpoint, for example at the start of each epoch. **Off-policy data**: responses from a different model or from a fixed dataset. **Forgetting**: a decrease, after a stage, on an ability the stage did not target (ch-30a §1). **Coverage**: the share of problems solved at least once within k samples.

## §1 Two axes separate the methods: sample source and gradient sign

**Definition.** Every method in this chapter can be placed on two axes: where y comes from when the loss is evaluated, and whether any term lowers the likelihood of a sampled y.

**The problem this framing addresses.** Comparisons that report "RL beats SFT" change both axes at once, plus the data, so the result cannot be attributed. [[rls-razor]] §5.1 fixes this by filling all four cells with the same prompts and the same task.

| Method | Sample source | Negative gradient | Example locus |
|---|---|---|---|
| SFT on a teacher's responses | off-policy | no | [[retaining-by-doing]] §2.2 |
| Self-SFT (initial policy, filtered correct) | sampled once from the initial policy | no | [[retaining-by-doing]] §2.2 |
| Iterative-SFT / rejection-sampling FT | regenerated each epoch | no | [[retaining-by-doing]] §4.2 |
| 1–0 REINFORCE | on-policy | no (A = 1 or 0) | [[rls-razor]] §5.1 |
| GRPO, PPO | on-policy | yes (A < 0 below the baseline) | [[rls-razor]] §5.1, ch-38, ch-40 |
| DPO, IPO, SimPO (offline) | off-policy pairs | yes (rejected term) | [[on-off-policy-rlhf]] §2, ch-39 |
| Online / semi-online DPO | pairs regenerated every s steps | yes | [[wolfe-online-vs-offline-rl]] |

The two objectives, written so that only the sampling distribution and the weight differ ([[transferability-of-llm-reasoning-tables]] Eq. 3):

```
L_{q,w,β}(θ) = − E_{x∼D} E_{y∼q(·|x)} [ w(x, y) · log π_θ(y | x) ] + β · E_{x∼D} KL(π_θ(·|x) ‖ π_ref(·|x))
```

- `q` — the distribution the response is drawn from: a point mass at the reference answer y\* for SFT, the current policy π_θ for on-policy methods.
- `w` — the per-sample weight: 1 for SFT, the advantage A(x, y) for policy gradient.
- `β` — the coefficient of the KL penalty to the reference policy π_ref; 0 when no penalty is used.
- `D` — the prompt distribution, identical across the settings compared.

Setting q = δ_{y = y\*} and w = 1 gives cross-entropy SFT; setting q = π_θ and w = A gives policy gradient. The ablation in §5 varies one field at a time.

## §2 What the controlled comparisons measure

**Result (replicated, three independent studies).** With the same prompts and starting checkpoint, the on-policy stage reached comparable or higher target accuracy with a smaller drop elsewhere.

[[retaining-by-doing]] Table 1 (gain on the target task / mean drop across non-target tasks, percentage points; non-target set includes MMLU, MATH, IFEval, Countdown, WildJailbreak, WildGuardTest):

| Model | Method | IFEval | MMLU | Countdown |
|---|---|---|---|---|
| Llama 3.1 8B Inst. | SFT | 25.2 / 27.8 | 11.1 / 38.5 | 25.5 / 36.4 |
| | REINFORCE | 17.8 / 7.7 | 8.6 / −0.1 | 7.5 / −0.8 |
| | GRPO | 18.4 / 3.4 | 14.6 / −0.2 | 60.4 / −0.5 |
| Qwen 2.5 7B Inst. | SFT | 24.2 / 5.6 | 9.4 / 14.6 | 10.4 / 29.2 |
| | REINFORCE | 5.7 / 2.9 | 6.4 / −0.6 | 11.9 / −0.1 |
| | GRPO | 17.0 / 0.2 | 8.4 / 0.2 | 29.2 / −0.3 |

The same runs scored on AlpacaEval, an evaluation no run targeted, reported as win rate in percent against GPT-4 (App. A.6, Table 3): Llama-3.1-8B-Instruct starts at 41.6 and ends at 23.5 / 11.6 / 0.0 after SFT on IFEval / MMLU / Countdown, against 43.1 / 41.2 / 42.0 after RL. Qwen2.5-7B-Instruct starts at 36.5 and ends at 22.5 / 19.9 / 0.4 after SFT, against 35.2 / 36.0 / 34.6 after RL.

[[transferability-of-llm-reasoning-tables]] Table 1 trains Qwen3-14B-Base on 47K math problems, with SFT targets rejection-sampled from Qwen3-32B and RL by GRPO on answer correctness. Math average: base 27.7, SFT (think) 49.8, RL 53.8. Other-reasoning average (GPQA, LiveCodeBench2, ACPBench, HeadQA): base 30.2, SFT 45.3, RL 60.0. Non-reasoning average (CoQA, MC-TACO, IFEval, HaluEval): base 45.7, SFT (think) 21.1, SFT (no-think) 29.0, RL 53.2. Both fine-tuning methods raised math; only the RL run raised the non-reasoning average above the base model.

[[rls-razor]] §3.1 trains Qwen 2.5 3B-Instruct on math, on the Chemistry L-3 subset of SciKnowEval, and on ToolAlpaca, sweeping learning rates, batch sizes and schedules for each method and comparing the Pareto frontiers of new-task accuracy against the average of HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande and HumanEval. A Pareto frontier here is the set of runs for which no other run of the same method has both a higher new-task accuracy and a higher prior-task average; the paper keeps runs within 2 accuracy points of that frontier (App. B.1). RL frontiers stay near flat in prior-task score as new-task accuracy rises; SFT frontiers fall. The same pattern appears for OpenVLA-7B in SimplerEnv, where the retained task is drawer opening and closing (§3.1), and the tool-use task (ToolAlpaca) is the agentic case in this comparison.

[[sft-memorizes-rl-generalizes]] is the earliest of these comparisons and the strongest in stated effect size. Starting from one SFT checkpoint of Llama-3.2-Vision-11B and scaling RL and SFT compute separately: under a rule change (J/Q/K = 10 during training, 11/12/13 at evaluation) the GeneralPoints language variant moved 11.5% → 15.0% with RL and 11.5% → 3.4% with SFT; the V-IRL language variant, scored by per-step accuracy, moved 80.8% → 91.8% with RL and 80.8% → 1.3% with SFT (§5.1, Fig. 6). Under a visual change (black suits to red suits; New York routes to the V-IRL mini benchmark) the vision-language variants moved 23.6% → 41.2% and 16.7% → 77.8% with RL, against 23.6% → 13.7% and 16.7% → 11.1% with SFT (§5.2, Fig. 7).

**Conditions and limits.** All four studies fine-tune on a single narrow target task, at model sizes between 1B and 14B. None of them reports long-context retention: the non-target suites are short-context, so nothing in this chapter supports a claim about context length (not reported). Only [[sft-memorizes-rl-generalizes]] covers a vision-language model.

## §3 The counter-evidence: what those SFT runs did not do

**Result (single study).** [[debunk-sft-generalization]] runs the same comparison on two decision-making tasks, Sokoban and General Points (the General Points environment is the one used by [[sft-memorizes-rl-generalizes]]), with Qwen2.5-7B and Llama-3.1-8B-Instruct, and attributes most of the SFT failure to a fixed prompt template rather than to the objective. Their diagnostic is a **Fake environment**: the prompt states a new instruction (for example numeric actions) while the scorer still accepts the training tokens. Answer-only SFT scores 0.93 (Qwen) and 0.90 (Llama) on Fake Sokoban while scoring 0 on the real instruction variants, which shows the model is executing the training semantics and ignoring the changed instruction (§5).

Adding prompt diversity during training — sampling different action names or different face-card mappings per example and stating the mapping in the prompt — changes the Sokoban row from 0.98 / 0 / 0 / 0 / 0.93 (ID / alphabetical / numeric / random / Fake) to 0.91 / 0.92 / 0.89 / 0.84 / 0 (Table 1a). Chain-of-thought targets add difficulty transfer: on General Points, Qwen answer-only scores 0.02 on the large-number split and 0.00 on the five-card split, while CoT scores 0.80 and 0.26, and Diversity + CoT scores 0.84 and 0.29 against the warm-started RL baseline's 0.69 and 0.22 (Table 1b).

**Conditions.** Two tasks, two backbones, decision-making only; the authors state that open-ended and creative generation were not evaluated (§7). Their RL baseline is warm-started GRPO whose in-distribution score is below SFT on Sokoban for Llama (0.43 against 0.92), so the comparison is not a like-for-like tuning of both methods.

Two further conditions come from the pro-RL side and point the same way:

- RL from the base Llama-3.2-Vision-11B failed without an SFT stage, because the base model produced unstructured responses from which no reward could be extracted ([[sft-memorizes-rl-generalizes]] §5.4, Fig. 9). RL applied to an over-tuned SFT checkpoint did not recover out-of-distribution performance (§6, Fig. 19).
- [[debunk-sft-generalization]] App. C adds an explicit proximity control to SFT: the loss becomes SFT + α·KL(π_θ ‖ π_ref) or SFT + α‖θ − θ_ref‖², with α ∈ {0.05, 0.1, 0.5} and 5 epochs. Instruction-variant scores rise and Fake success collapses, while in-distribution accuracy and the difficulty-variant scores fall. Proximity control moves along the same trade-off ch-30a §2 describes; the data change does not.

**Interpretation for this course.** The two bodies of evidence are consistent if the object of study is the data distribution rather than the loss: an SFT run on frozen prompts with one external answer per prompt learns a narrow input–output mapping, and both prompt diversity and on-policy resampling widen what the update is fitted to. This reading is this course's, not a claim of either paper.

## §4 KL to the base model on the new task

**Definition.** [[rls-razor]] proposes an empirical forgetting law: the forgetting caused by fine-tuning on a new task τ is predicted by

```
KL_τ = E_{x ∼ τ} [ D_KL( π₀(· | x) ‖ π(· | x) ) ]
```

- `π₀` — the base (pre-fine-tuning) policy; `π` — the fine-tuned policy.
- `x ∼ τ` — prompts from the **new** task, not from the tasks whose retention is measured. This is the practical point: the predictor needs no access to prior-task data.
- The direction is the forward KL, with π₀ in the first argument.

**Evidence.** In the ParityMNIST toy setting a second-degree polynomial in KL_τ fits forgetting with R² = 0.96 across SFT and RL runs; the LLM runs give R² = 0.71 ([[rls-razor]] §4, Fig. 11). Table 1 of the same paper compares candidate predictors on the toy task: forward KL 0.96 ± 0.01, reverse KL 0.93 ± 0.01, total variation 0.80 ± 0.01, Fisher-weighted L2 weight change 0.58 ± 0.02, L1 weight change 0.34 ± 0.02, activation change L1 0.52 ± 0.02.

**Why on-policy training lands at a lower KL.** For a binary reward, rejection sampling from the current policy defines the distribution q_RS(y) = π₀(y)·1[R(y) = 1] / π₀(R = 1), and Lemma 5.1 states that this is the solution of min_q D_KL(q ‖ π₀) subject to E_q[R] = 1. Theorem 5.2 states that policy gradient converges to the KL-closest optimal policy inside the representable family. Note the direction change: the projection results are stated for the reverse KL D(q ‖ π₀), while the forgetting law uses the forward KL; the paper does not connect the two directions formally.

**Worked example.** One prompt, five candidate answers, base probabilities π₀ = (0.35, 0.25, 0.20, 0.12, 0.08) for (a1…a5). The verifier accepts a2 and a4.

1. π₀(accepted) = 0.25 + 0.12 = 0.37.
2. The projection is q_RS(a2) = 0.25/0.37 = 0.676, q_RS(a4) = 0.12/0.37 = 0.324.
3. D(q_RS ‖ π₀) = 0.676·ln(0.676/0.25) + 0.324·ln(0.324/0.12) = ln(1/0.37) = **0.994 nats**.
4. An annotator who always writes a4 gives q_SFT = δ_{a4}, so D(q_SFT ‖ π₀) = −ln 0.12 = **2.120 nats**, 2.13 times the projection's cost.
5. Both distributions are 100% correct under the verifier. The difference is entirely in which correct distribution was chosen.

Panel 1 of [the chapter figure](figures/on-policy-kl-and-passk.html) recomputes this for other base distributions and accepted sets; step 3 generalizes to D(q_RS ‖ π₀) = −ln π₀(accepted), so the projection's cost depends only on how much mass the base model already places on acceptable answers.

**Counter-evidence (single study).** [[retaining-by-doing]] App. A.5 estimates KL[π₀ ‖ π] on 100 evaluation examples and finds a Pearson correlation of 0.52 with the non-target drop across all models, methods and datasets, but a non-monotone relation between the two SFT variants. On Llama-3.2-1B-Instruct trained on MMLU, Self-SFT has KL 39.3 with a 34.6-point drop while SFT has the larger KL 48.4 with the smaller 28.9-point drop; on Countdown, Self-SFT has KL 796.7 with a 25.3-point drop against SFT's 1254.7 and 24.2. The authors state the relation "does not always hold in our setting". The units of these KL values are not stated.

**Status.** **Open question.** KL on the new task is the best single predictor reported so far and requires no prior-task data, and at R² = 0.71 on the LLM runs and Pearson 0.52 in [[retaining-by-doing]] it is not accurate enough to replace direct measurement of the retained abilities.

**Alternative account.** [[retaining-by-doing]] §3 models the policy as a mixture of an old mode (prior knowledge) and a new mode (the target). Minimizing forward KL, which is what cross-entropy against an external target does, first stretches the new mode and then moves mass out of the old mode to cover the target; minimizing reverse KL, which on-policy updates approximate, keeps the old mode's shape and moves the new mode. Their §3.2 result matters for the conditions: with a **uni-modal** initial policy, SFT forgets less than RL; the ordering in the paper's experiments depends on the initial policy being multi-modal.

## §5 Why an on-policy update is a smaller update

**Mechanism, at the token level.** For a softmax over logits z, the gradient of the log-probability of the target token y is

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

so the update applied to the target's own logit has magnitude 1 − p_y. Training on a token the model already assigns high probability produces a small update; training on a token the model finds unlikely produces a large one.

**Worked example.** A 500-token response. If the tokens of an on-policy sample have mean probability 0.6 under the current policy, the mean per-token loss is −ln 0.6 = 0.51 nats and the mean target-logit gradient magnitude is 1 − 0.6 = 0.4. If an external teacher's response has mean token probability 0.05 under the same policy, the mean loss is −ln 0.05 = 3.00 nats and the gradient magnitude is 0.95. Summed over the response, 255 nats against 1498 nats. The ratio is the same quantity the KL account of §4 measures at the sequence level: distance from the current distribution.

**Evidence from gradient norms.** [[transferability-of-llm-reasoning-tables]] Fig. 6 logs gradient norms for four settings on Qwen3-8B and reports that off-policy runs take larger steps early in training while on-policy runs stay smaller and steadier. The paper's component ablation separates the axes of §1 on the same math data (Table 4; TI is the Transferability Index defined below):

| Setting | Math avg. | Other reasoning avg. | Non-reasoning avg. | TI_other | TI_non |
|---|---|---|---|---|---|
| Qwen3-8B-Base | 27.6 | 23.6 | 33.6 | – | – |
| Off-policy SFT | 41.9 | 34.4 | 26.6 | 18.3 | −40.5 |
| On-policy SFT | 33.7 | 35.7 | 35.0 | 68.6 | 30.2 |
| Off-policy RL | 45.5 | 35.9 | 31.7 | 36.4 | 4.5 |
| On-policy RL (no KL) | 37.1 | 38.2 | 35.8 | 65.6 | 39.3 |
| On-policy RL | 38.6 | 39.9 | 35.0 | 63.7 | 32.4 |

The table is read as pairs of rows that differ in one field. Holding the loss fixed and switching the sample source (rows 2→3 and 4→5) raises both transfer indices. Holding the sample source fixed and adding advantages and negative examples (rows 2→4 and 3→6) also raises them. Adding the KL penalty to on-policy RL (rows 5→6) changes little, which agrees with [[retaining-by-doing]] §4.1, where GRPO at β = 0.05 and β = 0 had a similar gain-drop trade-off on all models and datasets except Llama models trained on IFEval. The cost of the on-policy choice is visible in the math column: off-policy SFT and off-policy RL reach the highest in-domain math averages (41.9 and 45.5) and the worst transfer.

**The subnetwork claim, and why it is precision-dependent.** [[rl-finetunes-small-subnetworks]] measures update sparsity 1 − ‖θ₁ − θ₀‖₀/n on released checkpoints and reports 68.5%–96.0% of parameters unchanged after RL stages (DPO on Tülu 3 8B 81.4%, on Tülu 3 70B 95.2%; GRPO on DeepSeek-R1-Zero 86.0%; PPO 80.8%; PRIME 77.0%) against 6%–15% for the SFT stages of the same models (Table 1, Fig. 1). Fine-tuning only the identified subnetwork reproduces the full run's parameters to 94.0% (DPO) and 90.5% (PRIME) equality at a 10⁻⁵ tolerance, and 100% at 10⁻⁴ (§4). Their Table 5 puts the cause on the data: rejection-sampling SFT on in-distribution data gives 91.2% sparsity, while DPO on out-of-distribution data gives 7.7%.

[[rls-razor]] §6 tested the same hypothesis and found that the sparsity is an artifact of bfloat16: "Performing the same training with float32 resulted in models with identical performance but without any sparsity in their weight updates." Mukherjee et al. state the same rounding mechanism themselves (§6). **Interpretation:** the defensible claim is that on-policy updates are small enough that a large share of them falls below bfloat16 resolution; the claim that RL structurally edits a fixed subnetwork is not supported once the precision is changed. Update sparsity should therefore not be cited as an independent explanation of retention unless the comparison is repeated in float32.

## §6 Online versus offline preference optimization

**Definition.** An **offline** preference method optimizes a fixed dataset of pairs; an **online** method samples the pair (or at least the responses) from the current policy and labels it with a reward or preference model; a **semi-online** method resyncs the sampling policy every s optimization steps, with s = 1 fully on-policy and s → ∞ offline ([[wolfe-online-vs-offline-rl]]).

**The controlled result.** [[on-off-policy-rlhf]] (Tang et al., T5X Large 770M policies, IPO loss, four datasets: OpenAI summarization, Anthropic helpfulness, Chat arena side-by-side, Anthropic harmlessness) holds the loss, the hyperparameters and the source preference data fixed and changes only where the responses come from. Online runs reach a better trade-off between KL to the SFT policy and win rate against a fixed golden policy, and a higher peak win rate, on all four datasets (§2.1, Fig. 1). Four hypothesis tests in the same paper narrow the cause of that gap:

1. **Coverage is not the explanation.** Saving the online run's own data and replaying it uniformly shuffled gives little improvement over plain offline training, except on Chat arena side-by-side (§5.1, Fig. 4); training on the same data in its original order reproduces the online run, with training statistics matching to about 0.1% (App. F).
2. **Data quality is not the explanation.** Pairs regenerated by the strong 4k-step online policy and relabelled by the golden model improve offline training only slightly (§5.2, Fig. 5).
3. **Classification accuracy does not transfer to generation.** Offline policies reach 60–70% accuracy as pairwise classifiers while online policies stay below 50%, and within offline runs accuracy and win rate are close to uncorrelated (§5.3, Figs. 7–8).
4. **The chosen response's likelihood falls** relative to the SFT policy in all runs and most strongly offline (§5.3.3, Fig. 8, bottom row), so an offline contrastive update is not "SFT on the winner".

[[on-policy-suboptimal-preference-data]] adds the conditions under which each axis pays. On-policy sampling helps "especially in cases when the peak of reward appears farther from the reference policy"; a negative gradient helps "when the peak in the reward appears in less likely regions of π_ref"; the two are complementary, since sampling provides coverage and the contrastive term provides a stronger signal per batch (§5.1–5.3 takeaway boxes). The same paper records that sample reuse trades off against exploration. With T gradient steps taken on each batch of on-policy data, the bandit results are non-monotone in T (T = 5 learned faster than T = 2, T = 10 faster than T = 7, Fig. 10); on the synthetic length-control task T = 2 beat T = 1 for on-policy best-of-N while excessive reuse hurt, and PPO was unaffected up to T = 8, which the authors attribute to PPO's off-policy correction keeping significantly off-policy samples out of the gradient (§5.1.2, Fig. 11).

**Implication for a general-purpose model.** The retention property in §2 followed the sample source rather than the algorithm, so when a full RL loop is out of budget, the change to make first is periodic regeneration of the responses rather than a switch to PPO. ch-39 applies this to DPO, ch-31 to rejection-sampling SFT, and ch-40 to group-baseline RL.

## §7 The diversity cost

**Definition.** **Per-input diversity** is the diversity of K samples for one prompt; **across-input diversity** is the diversity of a set formed by taking one sample per prompt ([[rlhf-generalisation-diversity]] Eqs. 2–3, with K = 16 and N = 500 at temperature 1). Metrics: expectation-adjusted distinct n-grams (EAD), 1 − mean cosine similarity of Sentence-BERT embeddings, and NLI-based diversity.

**Evidence.** On TL;DR summarization with LLaMa 7B, RLHF has "much lower output diversity than SFT" on EAD and Sentence-BERT in the per-input setting; the across-input drop is smaller but present, and best-of-16 keeps across-input diversity similar to or above SFT, so the reward model alone is not the cause ([[rlhf-generalisation-diversity]] §6.2, Figs. 5–6). Raising the KL penalty coefficient lowered per-input diversity as well as performance, so β does not trade one for the other (§6.3, App. I). The instruction-following models showed no measurable difference, which the authors attribute to metrics designed for short outputs (§6.2).

**A metric that combines novelty and quality.** [[noveltybench]] partitions k generations into functional equivalence classes with a deberta-v3-large classifier (79% agreement with human labels, F1 0.811) and defines

```
distinct_k = |{c_i : i ∈ [k]}|
utility_k  = ((1 − p) / (1 − p^k)) · Σ_{i=1..k} p^{i−1} · 1[c_i ≠ c_j for all j < i] · u_i
```

- `c_i` — the equivalence class of the i-th generation; the indicator is 1 only for the first generation of a class.
- `u_i` — the quality of generation i, mapped to {1,…,10} from a reward model.
- `p` — the user's patience: the probability of asking for one more generation. The evaluation uses p = 0.8, k = 10, temperature 1.

**Worked example.** Four generations with utilities u = (8, 7, 6, 9) whose classes are (A, A, B, C), p = 0.8, k = 4. The normalizer is (1 − 0.8)/(1 − 0.8⁴) = 0.2/0.5904 = 0.3387. Only generations 1, 3 and 4 are novel, so the weighted sum is 1·8 + 0.64·6 + 0.512·9 = 8 + 3.84 + 4.608 = 16.448, and utility₄ = 0.3387 × 16.448 = **5.57**. A model producing four distinct answers of quality 10 would score 0.3387 × 10 × (1 + 0.8 + 0.64 + 0.512) = 10.0, which is the maximum. The second generation contributes nothing because it repeats class A: repeating a good answer is worth zero to a user who asked again.

**Measured stage effects.** OLMo 2 checkpoints after SFT, DPO and RLVR, distinct₁₀: 1B 8.83 → 8.08 → 7.85; 7B 7.46 → 5.96 → 5.72; 13B 7.47 → 5.61 → 5.16; 32B 7.25 → 5.22 → 5.08. Utility₁₀ over the same stages: 1B 2.40 → 3.03 → 3.21; 7B 4.02 → 4.35 → 4.32; 13B 4.29 → 4.62 → 4.46; 32B 4.22 → 4.63 → 4.62 ([[noveltybench]] §4.4, Fig. 6). Diversity falls at every size, with the largest step at DPO, while utility rises from SFT to DPO. Panel 3 of [the chapter figure](figures/on-policy-kl-and-passk.html) plots the distinct₁₀ values.

**Decoding does not fix it.** [[artificial-hivemind]] samples 50 responses per prompt for 100 open-ended queries; more than 70 models were measured and 25 of them appear in the main paper's figures. Under top-p = 0.9 and temperature 1.0, "in 79% of cases, the average similarity exceeds 0.8", against a random-pair baseline that falls entirely in the 0.1–0.2 band. Under min-p = 0.1 with temperature 2.0, "81% of response pairs still exceed 0.7 similarity and 61.2% exceed 0.8". Across models, average pairwise similarity ranges from 71% to 82% (DeepSeek-V3 and qwen-max-2025-01-25 at 0.82; DeepSeek-V3 and gpt-4o-2024-11-20 at 0.81), so sampling several models is not a source of variety either. [[noveltybench]] §4.3 found that in-context regeneration — asking for a different answer with the previous answers in context — brings three frontier models to roughly human diversity, while paraphrasing and diversity-asking system prompts are "only marginally effective".

**Implication for a general-purpose model.** Output diversity is a measured property that a target-only report does not cover, and the stage that protects benchmark accuracy best still lowers it (distinct₁₀ fell at all four OLMo 2 sizes across DPO and RLVR). ch-43 treats entropy and diversity controls inside RL; ch-42 treats the judge-driven part of the narrowing.

## §8 Transfer, and the pass@k boundary debate

**The transfer metric.** [[transferability-of-llm-reasoning-tables]] defines the Transferability Index for group g against the math group:

```
ΔR_b = R_b^model − R_b^base      σ_g = Std{ΔR_b : b ∈ B_g}      δ_b = ΔR_b / σ_g
s_b  = sign(δ_b)·|δ_b|^{1/2}      w_b = 100 − R_b^base      ŵ_b = w_b / Σ_u w_u
DI_g = Σ_b ŵ_b s_b               TI_g(%) = 100 · DI_g / DI_math
```

- `R_b` — accuracy on benchmark b; `B_g` — the benchmarks of group g.
- `σ_g` — the spread of gains inside the group, used to put groups on one scale. The paper does not state whether this is the sample or the population standard deviation; with four benchmarks the two differ by a factor √(4/3).
- The square root damps extreme gains; `w_b` up-weights benchmarks where the base model was weak.

**Worked example (group with two benchmarks).** Base scores 40 and 30; gains +10 and +2. With the population standard deviation, σ = 4, so δ = (2.5, 0.5) and s = (1.581, 0.707). Weights w = (60, 70), ŵ = (0.462, 0.538). DI = 0.462·1.581 + 0.538·0.707 = 0.730 + 0.381 = 1.111. If the math group's DI were 1.4, TI = 79.4%. Two consequences are visible from the formula: a group whose gains are all equal has σ = 0 and an undefined index, and TI is signed, so a negative DI_math would flip the sign of every reported index.

**The pass@k dispute.** [[rlvr-beyond-base-model-tables]] evaluates base and RLVR models with the unbiased estimator

```
pass@k = E_{x_i ∼ D} [ 1 − C(n − c_i, k) / C(n, k) ]
```

where n is the number of samples drawn per problem (128 for MATH500, Minerva and GSM8K; 1024 for AMC23 and AIME24), c_i the number of correct samples for problem i, and C(a, b) the binomial coefficient. RLVR models lead at k = 1 and are overtaken as k grows; on Minerva with a 32B model the base model leads by about 9 points at k = 128 (§3.2). The solvable-problem table on AIME24 at k = 1024 and MATH500 at k = 128 shows where the coverage goes: base-only-solved 13.3% and 3.6%, RLVR-only-solved 0.0% and 1.0% (Table 2).

**Worked example (why both statements are true at once).** Two problems, n = 8 samples each.

- RL model: 6 correct on problem 1, 0 on problem 2. pass@1 = (6/8 + 0)/2 = 0.375.
- Base model: 3 correct on problem 1, 1 on problem 2. pass@1 = (3/8 + 1/8)/2 = 0.250.
- At k = 4: the RL model gets 1 − C(2,4)/C(8,4) = 1 on problem 1 (C(2,4) = 0) and 0 on problem 2, so 0.500. The base model gets 1 − C(5,4)/C(8,4) = 1 − 5/70 = 0.929 and 1 − C(7,4)/C(8,4) = 1 − 35/70 = 0.500, so 0.714.

The RL model is 12.5 points ahead at k = 1 and 21.4 points behind at k = 4, with no contradiction: it concentrated mass on problem 1 and lost problem 2 entirely. Panel 2 of [the chapter figure](figures/on-policy-kl-and-passk.html) recomputes this crossover for other counts.

**The counter-study.** [[prorl]] trains DeepSeek-R1-Distill-Qwen-1.5B for more than 2k steps on 136K verifiable problems across five domains with a KL penalty and periodic reference-policy resets. On boxnet, a Reasoning Gym task unseen in training where the starting checkpoint "exhibits no capability" at any k up to 256, the trained model reaches an average reward of 7.91 from 0.00; on other unseen tasks it rises from a non-zero start (acre 5.99 → 58.57, game_of_life_halting 3.49 → 52.29; §4.3, Table 3). On graph colouring it exceeds the starting checkpoint at graph sizes larger than those trained on. The same paper reproduces the opposing pattern in its "Diminish" regime, mostly math, where pass@1 rises while pass@128 falls, and reports that boundary gain is negatively correlated with the starting checkpoint's pass@128 (§4.1–4.2).

**Status.** **Open question**, with a usable summary: the narrowing reported by [[rlvr-beyond-base-model-tables]] is strongest where the starting checkpoint already had high coverage, and the expansion reported by [[prorl]] is on tasks where it had none. Distillation is the third case: [[rlvr-beyond-base-model-tables]] §5 reports that distilled models show reasoning coverage beyond their base model, because the teacher supplies paths the base never samples — the same off-policy property that costs retention in §2 is what allows a new capability to enter (ch-20, ch-35).

## §9 Measuring the claims in this chapter

A stage's report needs four quantities beyond its own target metric. Each is computed by sampling from fixed prompt sets with the training run already finished, so none of them requires additional training.

1. **Held-out capability deltas** against the stage's input checkpoint, per slice, with paired 95% intervals (ch-30a §1 and §6). The studies above make this concrete: the AlpacaEval column in [[retaining-by-doing]] Table 3 is what a target-only report would have missed.
2. **KL to the input checkpoint on a fixed prompt set.** Report the direction and the sampling distribution, since they differ across the literature: [[rls-razor]] uses the forward KL on new-task prompts, [[on-off-policy-rlhf]] estimates KL(π_θ ‖ π_sft) from 256 training prompts as a budget axis, [[transferability-of-llm-reasoning-tables]] reports per-task token-distribution KL. Estimator choice matters for variance; k1 and k3 are defined in [[john-schulman-kl-tricks]] and in ch-38 §4.
3. **pass@1 and pass@k at large k**, with the unbiased estimator of §8 and n ≥ k samples fixed before training. Two controls are needed. Verify chains of thought on a subset, because final-answer matching lets wrong reasoning score at large k ([[rlvr-beyond-base-model-tables]] §3.1). Match entropy, not temperature, when comparing models: raising the RLVR model's temperature until its output entropy matched the base model's still left it below the base model on coverage (App. C.8).
4. **Diversity**: distinct_k and utility_k with a fixed patience p ([[noveltybench]]), or per-input and across-input metrics on a fixed prompt set ([[rlhf-generalisation-diversity]]). Fix the decoding configuration and report it, because the similarity distribution moves with top-p, min-p and temperature ([[artificial-hivemind]] App. C.1), and lock the embedding or judge model version, since a similarity threshold such as 0.8 has no meaning across embedding models.

Course code for the two estimators (not taken from a source):

```python
import math

def pass_at_k(n, c, k):
    """Unbiased pass@k for one problem: n samples drawn, c correct (Yue et al. App. A.2)."""
    if k > n:
        raise ValueError("k must be <= n")
    if n - c < k:          # every k-subset contains a correct sample
        return 1.0
    # 1 - C(n-c, k) / C(n, k), computed in log space for large n
    log_ratio = sum(math.log(n - c - i) - math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_ratio)

def forward_kl_on_task(base_logprobs, tuned_logprobs):
    """E_{y~pi_0}[log pi_0(y) - log pi(y)] over responses SAMPLED FROM THE BASE MODEL.
    Both arguments are per-sequence log-probabilities of the same base-model samples."""
    diffs = [b - t for b, t in zip(base_logprobs, tuned_logprobs)]
    return sum(diffs) / len(diffs)
```

The second function shows the detail that is easy to get wrong: the forward KL of §4 requires responses sampled from the **base** model and scored by both models. Scoring the tuned model's own samples estimates the reverse KL, which was the weaker predictor in [[rls-razor]] Table 1 (0.93 against 0.96) and is the quantity the KL penalty of ch-38 controls.

## Negative samples and negative feedback

Which of the four senses of "negative" (ch-43a) this chapter uses: **negative as gradient** in the GRPO, DPO, IPO and SimPO comparisons; **negative marginal value** in the filtering steps of Self-SFT and rejection-sampling SFT; **negative as content** does not appear here; **negative as conditioning** does not appear here.

**Where negatives come from.** Verifier outcomes in the RLVR studies (reward 1 for a correct final answer, 0 otherwise: [[retaining-by-doing]] §2.2, [[rls-razor]] §3.1); a shaped verifier in [[sft-memorizes-rl-generalizes]] App. A.3 (+5 for a legal equation equal to the target, −1 for a legal equation that misses the target, −1 for exceeding five verification steps, −2 for using numbers not on the cards, −3 for other illegal equations, an extra −1.5 in the vision variant when the cards are misread); a preference or reward model in [[on-off-policy-rlhf]] and [[rlhf-generalisation-diversity]]; an external model's incorrect responses used as the rejected side of SimPO pairs in [[rls-razor]] §5.1. False-negative rates of these labellers are not reported in any of them.

**What current practice does with them.** Discard (Self-SFT and rejection sampling keep only correct responses); or apply as gradient (negative advantage in GRPO and PPO; the rejected term in DPO, IPO and SimPO).

**Mechanism.** With the softmax gradient ∂ log p_y/∂z_j = 1[j = y] − p_j, a negative weight on a sampled sequence subtracts mass from that sequence's tokens, and the normalization pushes the removed mass onto the remaining tokens in proportion to their current probability. Pushing down a sample the model already finds unlikely therefore moves most of the mass to the currently most likely alternative, which may be neither correct nor diverse. [[on-policy-suboptimal-preference-data]] states the condition for this to be useful: a negative gradient can increase the likelihood of y_w "when y_l is sufficiently different from y_w, model capacity is large, and π_ref is chosen appropriately. If not, the margin … will still be larger … but the recovered probability mass will go into increasing likelihoods of other responses, not y_w." ch-43a derives the displacement and squeezing cases in full.

**Evidence with numbers.** (a) Negatives are not what preserves prior capability. In [[rls-razor]] §5.1, on-policy 1–0 REINFORCE (no negative gradient) and GRPO (negative gradient) sat on the same forgetting-versus-KL curve, while offline SimPO (negative gradient) sat with SFT. In [[retaining-by-doing]] Table 1, REINFORCE without an advantage estimator kept a similarly low drop to GRPO, and both are far below SFT's (for example 7.7 against 3.4 points on IFEval for Llama 3.1 8B) at lower target gains (17.8 against 18.4 on IFEval, 7.5 against 60.4 on Countdown). (b) Negatives do contribute to transfer and to speed. In [[transferability-of-llm-reasoning-tables]] Table 4, adding advantages and negative examples to a fixed sample source raised TI_other from 18.3 to 36.4 and TI_non from −40.5 to 4.5 in the off-policy pair; in the on-policy pair TI_other did not rise (68.6 → 63.7 with the KL term, 65.6 without) while TI_non did (30.2 → 32.4 and 39.3). The paper reports single runs, so small differences are not separated from run-to-run variation. In [[on-policy-suboptimal-preference-data]] §5.3, on-policy contrastive training converged faster than on-policy RL without a negative term. (c) Negatives have a measured cost in likelihood: in [[on-off-policy-rlhf]] §5.3.3 the log-probability of the **chosen** responses fell below the SFT policy's in every run and most strongly offline. No cited source measures the share of the total improvement attributable to the negative term, so this chapter does not claim one.

**Controls.** Keep the negatives on-policy (all of §2's evidence); bound the update with ratio clipping, which also makes sample reuse safer ([[on-policy-suboptimal-preference-data]] §5.1.2); prefer discarding a failure to penalizing it when the verifier's false-negative rate is unknown; anchor with a positive term when the negative term is offline (ch-39, ch-43a).

**Diagnostics.** Log chosen and rejected sequence log-probabilities separately against the reference (the diagnostic that produced the [[on-off-policy-rlhf]] Fig. 8 result); log statistics split by advantage sign; log entropy; log pass@k at large k and distinct_k, because the effects of §7 and §8 are invisible in pass@1.

**Effect on generality.** Coverage (§8), diversity (§7), and, through the same normalization step, the concentration of probability on a single phrasing that [[artificial-hivemind]] measures across models.

## Recipe

All rows were read at the stated locus on 2026-09-15. These are the settings of the comparison studies; they are reference points for reproducing a comparison, not recommended defaults for a production run.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct (RL's Razor) | 3B | SFT | LR sweep; optimizer; schedule; warmup; epochs; batch; precision | {1e-5, 3e-5, 5e-5, 7e-5, 9e-5}; AdamW; {constant, cosine} with warmup; 50 steps; {1, 2}; {16, 32, 64, 128}; bf16 | arXiv:2509.04259v1 App. B.1 Table 2 | verified 2026-09-15 | §3.1: sweep exists to build the Pareto frontier; models within 2 points of it are kept |
| Qwen2.5-3B-Instruct (RL's Razor) | 3B | RL | algorithm; LR sweep; KL reg.; group size; prompts per generation; inner iterations; loss | GRPO; {1e-5, 2e-5, 3e-5, 4e-5, 5e-5}; 0; 64; 8; {1, 2}; Dr. GRPO | arXiv:2509.04259v1 Table 2 | verified 2026-09-15 | no ablation of these values reported |
| Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct (Retaining by Doing) | 8B, 7B | SFT, RL | LR; schedule; batch; epochs; max length; GRPO KL coef.; group size | 5e-6; cosine, warmup ratio 0.03; 128 (IFEval, MMLU), 64 (Countdown); 2; 4096; 0.05; 5 | arXiv:2510.18874v3 App. A.3 | verified 2026-09-15 | Fig. 6: β = 0.05 against β = 0 gives a similar gain-drop trade-off |
| Llama-3.2-1B-Instruct, Qwen-2.5-1.5B-Instruct (Retaining by Doing) | 1B, 1.5B | SFT, RL | LR | 1e-4 | arXiv:2510.18874v3 App. A.3 | verified 2026-09-15 | Fig. 3: LR 1e-5 vs 1e-4 and 2 vs 10 epochs for Self-SFT |
| Qwen3-14B-Base (UniReason RL) | 14B | RL | framework; algorithm; LR; train batch; clip range; rollouts per prompt; mini-batch; max sequence; KL and entropy coef.; steps | verl; GRPO; 1e-6; 512; 0.22–0.28; 16; 128; 16k tokens; 0; 140 | arXiv:2507.00432v2 App. A.3.1 | verified 2026-09-15 | Table 4 ablation of sampling source, credit assignment and KL on Qwen3-8B |
| Qwen3-14B-Base (UniReason SFT) | 14B | distill-SFT | framework; LR; batch; epochs; targets | LLaMA-Factory; 5e-5; 512; 1.5; Qwen3-32B think-mode responses, rejection-sampled on correctness, 47K math problems | arXiv:2507.00432v2 App. A.3.1–A.3.2 | verified 2026-09-15 | chosen "to align with our RL settings"; no ablation reported |
| Qwen2.5-7B, Llama-3.1-8B-Instruct (Debunk) | 7B, 8B | SFT | LR; batch; max response; epochs; optimizer; weight decay; grad clip | 1e-5; 128; 7000; 15; AdamW; 0.01; 1.0 | arXiv:2510.00237v1 App. B.3 Table 4 | verified 2026-09-15 | Table 1: best checkpoint by the average of the reported metrics except Fake |
| Qwen2.5-7B, Llama-3.1-8B-Instruct (Debunk) | 7B, 8B | RL | algorithm; LR; batch; mini-batch; KL coef.; entropy coef.; epochs; warm start | GRPO; 1e-6; 256; 256; 0.0; 0.0; 100; SFT checkpoint at 10 steps | arXiv:2510.00237v1 App. B.3 Table 5 | verified 2026-09-15 | App. B.3: base models "struggle to achieve positive rewards without any supervised fine-tuning" |
| Llama-3.2-Vision-11B (Chu et al.) | 11B | SFT | LR search (all parameters trained); hardware | {1e-4, 1e-5, 1e-6, 5e-7, 1e-7}; 8×H800-80GB | arXiv:2501.17161v2 App. C.2, D.1 | verified 2026-09-15 | Fig. 16: none of the 10 additional SFT runs shows RL's increasing trend |
| Llama-3.2-Vision-11B (Chu et al.) | 11B | RL | algorithm; LR search; verification steps; reward | PPO (multi-turn, sequential revision); {2e-6, 1e-6}; 5 (GeneralPoints); +5 / −1 / −1 / −2 / −3, −1.5 extra for card misrecognition in GP-VL | arXiv:2501.17161v2 §3, App. A.3, D.1 | verified 2026-09-15 | Fig. 10: OOD gain by verification iterations 1/3/5/10 = +0.48 / +2.15 / +2.99 / +5.99 |
| LLaMa 7B (Kirk et al., summarization) | 7B | preference | PPO batch; PPO epochs; PPO steps; minibatch; KL coef.; advantage normalization; frozen layers; LR (selected) | 256; 4; 750; 256; 0.05; on; 80%; 1.5e-5 from {1.5e-6, 3e-6, 6e-6, 1.5e-5, 3e-5} | arXiv:2310.06452v3 App. E.4 Table 4 | verified 2026-09-15 | selection on an in-distribution validation set by reward (App. E.3) |
| LLaMa 7B (Kirk et al., summarization) | 7B | SFT, reward-model | batch; epochs; frozen layers; LR (selected) | SFT 128, 1, 80%, 3e-5; RM 64, 1, 80%, 3e-5 | arXiv:2310.06452v3 App. E.4 Tables 2–3 | verified 2026-09-15 | selection by validation loss (SFT) and accuracy (RM) |
| T5X Large policy (Tang et al.) | 770M | preference | steps; LR; β; batch (prompts, one pair each) | 4k; 1e-5; 0.1; 32 | [[on-off-policy-rlhf]] Recipe ledger, arXiv:2405.08448v1 §4 | verified 2026-09-14 (card) | adapted from prior work, tuned on OpenAI summarization only |
| Nemotron-Research-Reasoning-Qwen-1.5B (ProRL) | 1.5B | RL | KL coefficient β; samples per prompt; rollout temperature; steps | not reported; 16 (32 in runs 6–7); 1.2; "more than 2k" | [[prorl]] Recipe ledger, arXiv:2505.24864v1 §3.2 | not reported (β); verified (rest) | no ablation isolates the KL term or the resets |
| OLMo 2 (NoveltyBench evaluation) | 1B–32B | eval-gate | generations per prompt; temperature; patience p; utility model | 10; 1.0; 0.8; Skywork-Reward-Gemma-2-27B-v0.2 mapped to {1..10} | arXiv:2504.05228v4 §3.2, §4.1 | verified 2026-09-15 | classifier agreement 79%, F1 0.811 on 100 held-out pairs |
| 70+ models (Artificial Hivemind evaluation) | various | eval-gate | responses per prompt; decoding A; decoding B; max tokens; embedding | 50; top-p 0.9, T = 1.0; min-p 0.1, top-p 1.0, T = 2.0; 2048; text-embedding-3-small | arXiv:2510.22954v1 App. C.1 | verified 2026-09-15 | random-pair baseline falls 100% in the 0.1–0.2 similarity band |
| Base and RLVR models (Yue et al. evaluation) | 7B–32B | eval-gate | temperature; top-p; max tokens; n samples | 0.6; 0.95; 16,384; 128 (MATH500, Minerva, GSM8K) or 1024 (AMC23, AIME24) | arXiv:2504.13837v5 §3.1, App. A.2 | verified 2026-09-15 | Fig. 17 temperature study; App. C.8 entropy-matched control |

**Starting point for a small general-purpose run.** For a 1.5B–8B instruction-tuned checkpoint that must gain one skill without losing the rest, the verified rows support: GRPO with a binary verifier reward, group size 5, KL coefficient 0.05 (or 0, which gave a similar trade-off on all but one of the tested combinations), LR 5e-6 for 7B–8B and 1e-4 for 1B–1.5B, cosine schedule with warmup ratio 0.03, batch 128, 2 epochs ([[retaining-by-doing]] App. A.3; measured on IFEval, MMLU and Countdown with Llama 3.2 1B, Llama 3.1 8B, Qwen 2.5 1.5B and Qwen 2.5 7B on at most 8 H100 GPUs). When rollouts are too expensive, the same paper's Iterative-SFT alternative regenerates targets at the start of each epoch under the same optimizer settings. Before either, run the §9 report on the input checkpoint so the deltas have a reference.

## Generalization lens

**(a) What increases breadth.** Sampling the training responses from the checkpoint being updated: non-target drops of at most 3.4 points for GRPO where SFT reached 38.5 ([[retaining-by-doing]] Table 1), non-reasoning average 53.2 against 21.1 on Qwen3-14B ([[transferability-of-llm-reasoning-tables]] Table 1), and flat prior-task Pareto frontiers ([[rls-razor]] §3.1). Regenerating targets each epoch, which reached SFT's target accuracy with mild to no forgetting ([[retaining-by-doing]] §4.2). Prompt diversity in SFT data: Sokoban instruction variants 0 → 0.84–0.92 ([[debunk-sft-generalization]] Table 1a). Chain-of-thought targets for difficulty transfer: General Points large-number split 0.02 → 0.80 ([[debunk-sft-generalization]] Table 1b). Off-policy teacher data when the ability is absent: distillation expanded coverage beyond the base model where RLVR did not ([[rlvr-beyond-base-model-tables]] §5).

**(b) What causes narrowing or forgetting.** Off-policy targets at high learning rates (§2, and ch-30a §5 for the controls). Frozen prompt templates, which produce a model that executes training semantics under changed instructions (Fake-environment success 0.90–0.93, [[debunk-sft-generalization]] §5). Preference optimization and RL on the diversity axis: per-input diversity after RLHF ([[rlhf-generalisation-diversity]] §6.2), distinct₁₀ at every OLMo 2 size ([[noveltybench]] §4.4), coverage at large k ([[rlvr-beyond-base-model-tables]] §3.2, Table 2). Raising the KL coefficient does not undo the diversity loss ([[rlhf-generalisation-diversity]] §6.3). Over-tuned SFT checkpoints, from which RL did not restore out-of-distribution performance ([[sft-memorizes-rl-generalizes]] §6).

**(c) How to measure it for this stage.** The four-part report of §9: paired held-out deltas against the input checkpoint, KL to that checkpoint with the direction stated, pass@1 with pass@k at large k, and a diversity metric with a fixed decoding configuration. Known measurement errors: pass@k at large k rewards lucky final answers unless chains are checked ([[rlvr-beyond-base-model-tables]] §3.1); diversity metrics designed for short outputs showed no difference on long instruction-following responses ([[rlhf-generalisation-diversity]] §6.2); embedding-similarity thresholds are not comparable across embedding models ([[artificial-hivemind]] App. C.1); KL predicts forgetting only moderately in LLM runs, R² = 0.71 ([[rls-razor]] §4) and Pearson 0.52 with a non-monotone case ([[retaining-by-doing]] App. A.5); update sparsity is precision-dependent ([[rls-razor]] §6). Long-context retention is not measured by any source in this chapter (not reported); use the ch-47 suites for that.

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Reading "SFT memorizes, RL generalizes" as a property of the loss | an on-policy SFT run retains as well as RL, contradicting the slogan | run the four-cell ablation of §1 on your own data ([[rls-razor]] §5.1; [[transferability-of-llm-reasoning-tables]] Table 4) |
| Comparing SFT and RL with one frozen prompt template | SFT scores near zero on instruction variants but near its ID score in a Fake environment | build the Fake variant and compare ([[debunk-sft-generalization]] §5) |
| Treating KL to the base model as the objective | KL is low and a held-out slice still drops | keep KL as a diagnostic; decide on paired held-out deltas (ch-30a §1) |
| Estimating the forward KL from the tuned model's samples | the logged number is the reverse KL and tracks forgetting less well | sample from the base model and score both models (§9) |
| Reporting pass@1 only | pass@1 rises while the model solves fewer distinct problems | report pass@k at k ≥ 128 with n ≥ k fixed in advance |
| Comparing coverage at a fixed temperature | the RL model looks worse only because its entropy is lower | match output entropy, not temperature ([[rlvr-beyond-base-model-tables]] App. C.8) |
| Treating high diversity as a win on its own | a 1B model scores highest on distinct_k and lowest on utility_k | report distinct_k and utility_k together ([[noveltybench]] Table 1) |
| Raising β to recover diversity | both diversity and reward fall | measure diversity directly; use the ch-43 controls |
| Citing RL update sparsity as the mechanism of retention | the effect disappears in float32 | state the precision, or drop the claim ([[rls-razor]] §6) |
| Using an offline contrastive loss and calling it "SFT on the winner" | the chosen response's log-probability falls during training | log chosen and rejected log-probabilities separately ([[on-off-policy-rlhf]] Fig. 8) |
| Assuming RL can fix a narrowed checkpoint later | RL from an over-tuned SFT checkpoint stays at the training rule | gate the SFT stage on the §9 report before starting RL |

## Check your understanding

1. In [[rls-razor]] §5.1, 1–0 REINFORCE keeps prior-task performance like GRPO while SimPO forgets like SFT. Explain what this rules out about the role of negative gradients, and what would have to be true for the opposite result to hold.
2. Using the worked example in §4, explain why the KL cost of the rejection-sampling projection depends only on π₀(accepted), and what that predicts about fine-tuning a model on a task where it currently succeeds on 1 in 1000 samples.
3. [[retaining-by-doing]] reports a case where SFT has a larger KL to the base model than Self-SFT and forgets less. Explain how that observation can coexist with the R² = 0.71 fit in [[rls-razor]], and name a measurement that would separate the two accounts.
4. In [[transferability-of-llm-reasoning-tables]] Table 4, off-policy SFT has the second-highest math average and the worst non-reasoning transfer. Explain the causal path from the sampling distribution to that pattern, using the token-level gradient argument of §5.
5. A team reports that their RLVR model improved pass@1 by 8 points and concludes the model learned a new reasoning strategy. Give two measurements that could refute the conclusion, and state what result from each would support it instead.
6. Explain why raising the KL penalty coefficient in RLHF lowered per-input diversity in [[rlhf-generalisation-diversity]] rather than restoring it, given that the penalty pulls the policy toward the SFT model.
7. Using the utility_k definition of §7, explain why a model that answers the same question identically ten times can have a high average quality score and a utility₁₀ below 2, and what value of p would hide that failure.
8. [[debunk-sft-generalization]] closes the instruction-variant gap with prompt diversity but not the difficulty-variant gap, which needs chain-of-thought targets. Explain why the two interventions address different failures, and which one the frozen-prompt hypothesis predicts.

## Connections

- **Previous (dependency):** ch-38 — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax. It supplies the PPO objective and the KL penalty this chapter's comparisons switch on and off.
- **Previous (dependency):** ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control. It supplies the forgetting report used in §9.
- **Next:** ch-39 — Offline Preference Optimization: DPO and Its Variants. §6 is the evidence a reader needs before choosing offline DPO.
- ch-00 — What General Capability Means and How It Is Measured.
- ch-20 — Distillation as Data: Explanation Traces and the R1-Distill Lineage (the off-policy case that can add a capability).
- ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate.
- ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation (Iterative-SFT in practice).
- ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood.
- ch-35 — Distillation in Practice A: Where Labs Insert Teacher Data.
- ch-37 — Policy-Gradient Foundations for Language Models (REINFORCE and baselines).
- ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO.
- ch-41 — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization.
- ch-42 — Reward Hacking and Judge Design.
- ch-43 — Entropy, Output Diversity, and KL Control in RL (the controls for §7).
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-44b — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing (depends on this chapter).
- ch-46 — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention.
- ch-47 — Evaluation Harness and Suite Design for General Capability.
- ch-50 — Slice Analysis, Forgetting Slices, and Failure Bucketing.

## Sources

- [[sft-memorizes-rl-generalizes]] — excerpt of Chu et al. 2025: rule-variant and visual-variant numbers, the reward design, the two conditions (no RL without SFT; no recovery from an over-tuned checkpoint).
- [[debunk-sft-generalization]] — excerpt of Lin et al. 2025: frozen-prompt hypothesis, Fake-environment scores, Table 1 for prompt diversity and CoT, proximity-control appendix.
- [[rls-razor]] — excerpt of Shenfeld et al. 2025: the forgetting law and its fits, the four-objective ablation, the projection lemma, the bfloat16 finding, the hyperparameter sweep.
- [[retaining-by-doing]] — excerpt of Chen et al. 2025/2026: gain-drop table, AlpacaEval table, KL ablation, Iterative-SFT, the mixture account, the KL counter-evidence, settings.
- [[rlhf-generalisation-diversity]] — excerpt of Kirk et al. 2024: generalisation gap definition, per-input and across-input diversity, the KL-coefficient sweep, PPO settings.
- [[noveltybench]] — excerpt of Zhang et al. 2025: distinct_k and utility_k definitions, model table, OLMo 2 stage values, prompting workarounds.
- [[artificial-hivemind]] — excerpt of Jiang et al. 2025: intra-model repetition under two decoding settings, inter-model similarity range, measurement configuration.
- [[on-off-policy-rlhf]] — library card for Tang et al. 2024 (the outline's `online-offline-alignment-gap` slug): the KL-budget comparison and the four hypothesis tests; Recipe row.
- [[on-policy-suboptimal-preference-data]] — excerpt of Tajwar et al. 2024: conditions for on-policy sampling and negative gradients, sample reuse, the mode-seeking unification.
- [[rl-finetunes-small-subnetworks]] — excerpt of Mukherjee et al. 2025: update-sparsity table, subnetwork-only training, the in-distribution explanation, the precision caveat.
- [[transferability-of-llm-reasoning]] — library card for Huan et al. 2025 (description only; no numbers, unverified as of 2026-09-15).
- [[transferability-of-llm-reasoning-tables]] — excerpt with the verified Table 1, Table 4, TI formula, KL and rank-shift diagnostics, and settings.
- [[rlvr-beyond-base-model]] — library card for Yue et al. 2025 (description only; no numbers, unverified as of 2026-09-15).
- [[rlvr-beyond-base-model-tables]] — excerpt with the verified pass@k estimator, coverage table, evaluation protocol and entropy-matched control.
- [[prorl]] — verified library card: the counter-evidence on pass@k expansion, its Diminish regime, and its recipe ledger.
- [[wolfe-online-vs-offline-rl]] — excerpt of a secondary review: the online / semi-online / offline axis and the sync period s. Not used for any number.
- [[john-schulman-kl-tricks]] — k1 and k3 KL estimators referenced in §9.

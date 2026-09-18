<!-- scope: TIES-Merging (UNC / IBM Research / MIT, arXiv:2306.01708, NeurIPS 2023): merging task vectors of models fine-tuned from one initialization by trimming low-magnitude changes, electing a per-parameter sign by total magnitude, and averaging only sign-agreeing values; experiments on T5-base/large, (IA)3 on T0-3B, CLIP ViT-B/32 and ViT-L/14, BERT-base; in-domain, held-out-task, number-of-tasks, ablation and oracle-sign results
     deps: [[task-arithmetic]]
     see-also: [[dare-merging]], [[model-soups]], [[wise-ft]], [[mix-data-or-merge-models]], [[model-merging-at-scale]], [[warp]]
-->

# TIES-Merging: Resolving Interference When Merging Models
- **Core Insight:** Merging task vectors after trimming each to its top-20% magnitudes and keeping only values that agree with an elected sign outperforms the strongest prior merging baseline by an average of 2.3 points (NLP) and 1.7 points (vision) in-domain when a validation set is available (§1, Table 1), and by 1.0 (T5-base) and 4.4 (T5-large) points on six held-out tasks (Table 2).
- **Guideline:** When several models fine-tuned from the same initialization are merged into one multitask model, trim small task-vector entries and resolve sign conflicts before averaging, because in the (IA)3 setting removing the disjoint mean or the scaling step costs 3.2 and 5.2 points (Table 6); without a validation set start from k = 20 and λ = 1 (§5, App. C.4), and do not expect the merge to match multitask training (73.1 vs 66.4 for (IA)3, Table 5).
- **Authors:** Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal (UNC Chapel Hill; IBM Research; MIT)
- **Year:** 2023 (arXiv v1 2023-06-02; v2 2023-10-27; NeurIPS 2023)
- **URL:** https://arxiv.org/abs/2306.01708 (code: https://github.com/prateeky2806/ties-merging)
- **Source type:** paper
- **Relevant topics:** model merging, task vectors, parameter interference, sign conflicts, multitask models, out-of-domain generalization, merge hyperparameters

## Abstract
Fine-tuning produces many single-task models. Model merging combines several of them into one multitask model without further training, but prior methods ignore interference between parameters of different models and lose accuracy as more models are merged. The paper identifies two sources of interference: (a) redundant parameter values and (b) disagreement on the sign of a parameter across models. TIES-Merging (TRIM, ELECT SIGN & MERGE) (1) resets parameters that changed by a small amount during fine-tuning, (2) resolves sign conflicts, and (3) merges only values that agree with the elected sign. It outperforms existing methods across modalities, domains, number of tasks, model sizes, architectures and fine-tuning settings, and the analysis highlights the importance of resolving sign interference (Abstract).

## Key Contributions
- Two measured sources of interference: most task-vector entries are redundant (keeping the top 20% does not reduce average performance over eleven tasks, Fig. 3), and sign conflicts remain after trimming and grow with the number of merged models (Fig. 4).
- The three-step TIES procedure plus a scaling factor λ (§4.2, Algorithm 1).
- A fixed no-validation recipe (k = 20, λ = 1) chosen on the PEFT setting and applied unchanged to full fine-tuning (§5, App. C.4).
- Evidence that sign is the main missing information: merging with the sign vector of a multitask model reaches 72.0 vs 73.1 for multitask training (Table 5).

## Key Figures/Tables to Study
- Fig. 2: how averaging shrinks a parameter under redundancy or sign conflict. Fig. 3-4: redundancy and sign-conflict measurements.
- Table 1: in-domain results with and without a validation set. Table 2: six held-out tasks.
- Fig. 5: normalized accuracy vs number of merged tasks. Table 6: component ablation. Fig. 7: sign flipping of top-k vs bottom-k entries.
- Table 5 and Table 7 (App. B.1): oracle and few-shot estimated sign vectors.

## Technical Details
**Task vectors.** τ_t = θ_ft^t − θ_init, where θ_init is the shared initialization and θ_ft^t the fine-tuned parameters for task t; τ_t = γ_t ⊙ μ_t with sign vector γ_t = sgn(τ_t) ∈ {+1, 0, −1}^d and magnitude μ_t = |τ_t| (§3, §4.1).
**Algorithm (§4.2, Algorithm 1).**
1. Trim: keep the top-k% entries of τ_t by magnitude and set the rest to 0, giving τ̂_t.
2. Elect: for each parameter p, γ_m^p = sgn(Σ_t τ̂_t^p), the sign with the larger total magnitude.
3. Disjoint merge: A_p = {t : sgn(τ̂_t^p) = γ_m^p}; τ_m^p = (1/|A_p|)·Σ_(t∈A_p) τ̂_t^p. Zero values are never counted.
4. θ_m = θ_init + λ·τ_m, with λ a scaling hyperparameter.
Worked example (course arithmetic, not from the paper): three task values +0.5, +0.4, −0.6 for one parameter. Elect: sgn(0.3) = +. Disjoint mean over {+0.5, +0.4} = +0.45. A plain mean gives +0.1.
**Baselines (§5).** Simple averaging θ_m = Σ_t θ_t / n; Fisher merging weighted by a diagonal Fisher approximation; RegMean (closed-form per-layer regression on activations); Task Arithmetic θ_m = θ_init + λ·Σ_t τ_t (λ = 0.4 without validation).
**Settings (§6).** PEFT: (IA)3 on T0-3B, eleven datasets, median over P3 templates. Full fine-tuning NLP: T5-base and T5-large on seven tasks. Vision: CLIP ViT-B/32 and ViT-L/14 visual encoders on eight tasks, text encoder frozen. Evaluation uses rank classification over label strings (App. C.6).
**In-domain results (Table 1, average accuracy).** With validation, TIES vs best baseline: (IA)3 66.4 vs 63.9; T5-base 73.9 vs 73.2; T5-large 76.9 vs 73.3; ViT-B/32 73.6 vs 71.8; ViT-L/14 86.0 vs 84.5. Individual fine-tuned models: 71.4, 82.8, 88.8, 90.5, 94.2; multitask: 73.1, 83.6, 88.1, 88.9, 93.5. Without validation, TIES is below Task Arithmetic on T5-base (69.7 vs 73.2; App. Table 9 lists Task Arithmetic at 73.9) and above it on T5-large (74.4 vs 73.5), ViT-B/32 (72.4 vs 60.4) and ViT-L/14 (86.0 vs 83.3).
**Held-out tasks (Table 2).** T5 models merged on the seven in-domain tasks, evaluated on Cosmos QA, Social IQA, QuAIL, WiC, COPA, H-SWAG: T5-base zero-shot 31.1, RegMean 34.3, TIES 35.3; T5-large zero-shot 27.6, RegMean 36.0, TIES 40.4. The single T5-base QuaRTz model averages 37.4 on these tasks, above the T5-base TIES merge (App. Table 13).
**Number of tasks (Fig. 5, T5-large, at most 10 subsets per size).** All methods decline as tasks are added; at two tasks TIES and Task Arithmetic are close to normalized accuracy 1 while simple averaging loses 10%; Task Arithmetic declines faster than TIES (§6).
**Same-task checkpoints (Table 3).** Merging 10 Hugging Face BERT-base checkpoints per task: TIES 72.2 / 86.8 / 58.8 on RTE / MRPC / WNLI vs Task Arithmetic 71.8 / 86.0 / 59.2 and output ensembling 70.8 / 86.0 / 45.1. Sign conflicts also exist among checkpoints of one task (App. B.4, Fig. 10).
**Merge as initialization (Table 4).** Merging seven other GLUE task models and fine-tuning on the target: TIES 80.1 / 88.0 / 54.9 vs pre-trained init 66.4 / 81.8 / 56.3.
**Ablation (Table 6, validation set; T5-base / (IA)3).** Full 74.5 / 70.7; − trim 73.0 / 70.6; − elect 73.1 / 69.6; − disjoint mean 72.6 / 67.5; − scale (λ = 1) 72.0 / 65.5.
**Sign analyses.** Flipping the signs of the top-20% or top-30% entries lowers (IA)3 performance monotonically with flip probability, while flipping the bottom 80% or 70% has little effect (§7.2, Fig. 7, three runs). A sign vector from a multitask model trained on 32 validation examples per task, initialized from the mean of task models, gives 67.7 vs 66.4 for TIES (App. B.1, Table 7).
**Hyperparameter sensitivity (App. B.2).** Across the λ values in Fig. 8, TIES accuracy spans 68-75% vs 55-75 for Task Arithmetic; TIES averages task vectors, so λ = 1 corresponds to λ = 1/#tasks... in the paper's words, "a value of 1 for TIES is similar to using 1/#tasks for Task Arithmetic". Performance drops and then saturates as k grows (Fig. 8 right).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| T5-base / T5-large task models | not reported | SFT (task fine-tuning) | max steps; effective batch; LR; early stopping; precision; max seq length; scheduler; weight decay | 75,000; 1024; 1e-4; patience 5; bfloat16; 128; none; none | arXiv:2306.01708v2 App. C.6 | verified 2026-09-14 | no ablation reported |
| (IA)3 on T0-3B task models | 3B base | SFT (PEFT) | effective batch; eval batch; LR; early stopping; scheduler; weight decay | 16; 32; 1e-4; patience 10; none; none | App. C.6 | verified 2026-09-14 | no ablation reported |
| CLIP ViT-B/32, ViT-L/14 task models | not reported | SFT | fine-tuning settings | checkpoints supplied by Ilharco et al. (task_vectors repo); settings not given here | App. C.1 | not reported (checked §5-§6, App. C) | n/a |
| TIES no-validation recipe | all | merge | k; sign rule; merge; λ | top-20%; sum of magnitudes; disjoint mean; λ = 1 | §5; App. C.4 | verified 2026-09-14 | App. C.4: grid k ∈ {10, 20, 30}, λ ∈ 0.8-3.0 step 0.1 on eleven (IA)3 models; λ = 0.9, 1.0, 1.1 equivalent |
| TIES with validation | T5 | merge | λ | tuned on validation; Fig. 8 range 0.8-1.8; 7-task T5-large optimum 1.7 | App. B.2; App. C.5 | verified 2026-09-14 | k grid with validation not reported |
| Task Arithmetic baseline (no validation) | all | merge | λ | 0.4 | §5 | verified 2026-09-14 | recommended value from [[task-arithmetic]] |
| Compute | n/a | all | hardware; time | NVIDIA A6000 48GB; (IA)3 1-2 h per task, multitask 24 h on 4 GPUs; T5 15 min-2 h per task, about 8 h multitask; merge evaluation under 2 min | App. C.1 | verified 2026-09-14 | n/a |

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality.** On six held-out tasks the T5-large TIES merge (40.4) exceeds every single task model (best 33.4) and all baselines (Table 2, App. Table 14); for T5-base one single task model (37.4) exceeds the merge (35.3) (App. Table 13).
- **Narrowing.** Every merging method loses accuracy as tasks are added (Fig. 5), and all merges stay below multitask training in-domain (Table 1). App. A lists as limits the need for a shared initialization and architecture and the gap to multitask training.
- **Scale.** The largest models tested are T0-3B with (IA)3 and ViT-L/14; no decoder-only LLM is tested in this paper.

## Connections
- [[task-arithmetic]]: task vectors and the Task Arithmetic baseline (ref. [29]).
- [[wise-ft]], [[model-soups]]: weight averaging by the same group, cited as refs. [83] and [82].
- [[dare-merging]]: drops delta parameters before merging; combined with TIES as DARE-TIES in [[mix-data-or-merge-models]].
- [[mix-data-or-merge-models]]: applies TIES to 8B multilingual safety and general models.
- [[warp]]: cites TIES as a method that reduces interference with sparse task vectors.
- [[model-merging-at-scale]]: separate card on merging at larger model sizes.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2306.01708 (v2, 2023-10-27, full PDF including App. A-C); v1 date and NeurIPS 2023 venue from the arXiv abstract page.
- Audit claims not found in the source: none. The audit summary "outperforming existing methods across ... settings" is the abstract's wording; Table 1 shows one exception (T5-base without validation).
- Not reported by the source: T5 and ViT parameter counts, ViT fine-tuning settings, k grid when a validation set is used, seeds for Table 1.

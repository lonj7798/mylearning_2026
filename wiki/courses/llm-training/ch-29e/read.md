<!-- chapter: ch-29e
     track: sft
     kind: content
     title: Instruction Tuning and Generalization to Unseen Tasks
     deps: [ch-00, ch-29]
     sources: [[flan]], [[t0-multitask-prompted-training]], [[super-natural-instructions]], [[flan-palm-scaling-instruction-finetuning]], [[flan-collection]], [[opt-iml]], [[instruction-diversity-unseen-tasks]], [[lima]], [[urial]], [[revisiting-superficial-alignment-hypothesis]], [[instruction-following-without-instruction-tuning]], [[false-promise-imitating-proprietary-llms]], [[tulu-1-how-far-can-camels-go]], [[tulu-3]], [[ifbench]]
     figures: figures/heldout-protocols.html
     revised: 2026-09 (generality revision)
-->

# Chapter 29e — Instruction Tuning and Generalization to Unseen Tasks

> **Core insight.** Multitask instruction tuning raises zero-shot accuracy on task types that were excluded from training, under three measured conditions: many distinct training tasks (FLAN's held-out average rose from 49.9 with 1 training cluster to 63.5 with 7, [[flan]] Fig. 6), a large enough model (FLAN's 8B and smaller models lost held-out accuracy, Fig. 7), and training formats that include the evaluation format (Flan-PaLM trained without chain-of-thought data lost chain-of-thought accuracy, [[flan-palm-scaling-instruction-finetuning]] §4.2). In a controlled rewrite-rule study, the number of distinct instructions, not the number of examples per instruction, decided whether a model generalized to unseen instructions [[instruction-diversity-unseen-tasks]]. Response style changes with 100 to 1,000 training examples ([[lima]]; [[revisiting-superficial-alignment-hypothesis]] Fig. 3) or with three in-context examples and no training ([[urial]]), but objective task accuracy keeps rising with more task data along a power law [[revisiting-superficial-alignment-hypothesis]], so a style-sensitive win rate cannot certify generality.
>
> **Guideline.** When the goal is breadth and the data budget is fixed, add distinct tasks, instructions, and prompt formats (zero-shot, few-shot, chain-of-thought) before adding examples per task, because task and instruction count drove held-out gains in FLAN, Super-NaturalInstructions, Flan-PaLM, and the rewrite-rule study, while examples per task saturated at 64 in [[super-natural-instructions]] §7.1. When reporting a generality claim, hold out whole task clusters and their source datasets, choose checkpoints on a separate development suite, and report objective accuracy on an unseen suite next to any judge win rate, because a GPT-4o judge preferred a chat-style model on GSM8K 84.4% of the time while its accuracy was 14.7% against 46.5% for the task-trained model ([[revisiting-superficial-alignment-hypothesis]] Table 2). Otherwise, treat any single-suite gain as a development-set result. When the base model is small (8B or less in FLAN's setting), measure held-out regressions before assuming transfer, because FLAN's 8B and smaller models lost held-out accuracy after instruction tuning ([[flan]] Fig. 7).

## Why this chapter matters for a general-purpose model

The training pipeline for a general model has six stages: pretraining, mid-training, supervised fine-tuning (SFT), preference optimization, reinforcement learning (RL), and evaluation. This chapter opens the SFT phase. SFT is the first stage in which the training distribution is chosen task by task instead of corpus by corpus, so it is the first stage in which "the model improved" can mean "the model improved on the tasks we chose". The measurable problem is transfer: for a model trained on a set of tasks T, how does accuracy change on tasks outside T, and on the original abilities of the base model?

Three terms are used throughout. A **task** is a mapping from inputs to outputs defined by one dataset and one task setup (for example extractive question answering on SQuAD). A **task cluster** (or category) is a group of tasks of the same type, such as natural language inference. A **template** is a natural-language pattern that renders an example as an instruction and a target. "Unseen" is ambiguous until the level is stated: unseen instances of trained tasks, unseen tasks from trained clusters, or unseen clusters ([[opt-iml]] §2.3). The chapter measures claims at all three levels, and §8 fixes the held-out protocol that [[ch-30]] through [[ch-36]] reuse. The course-wide definitions of general capability and its measurement are in [[ch-00]]; the synthetic pools built in [[ch-29]] are the data this protocol evaluates.

## §1 Instruction tuning: definition, loss, and three levels of "unseen"

**Definition.** Instruction tuning is supervised fine-tuning of a pretrained language model on a mixture of tasks, where each input states the task in natural language and the target is the desired response ([[flan]] §1).

**Problem it addresses.** GPT-3's zero-shot accuracy was lower than its few-shot accuracy on reading comprehension, question answering, and natural language inference ([[flan]] §1). The authors' proposed reason is that, without few-shot exemplars, a model has more difficulty with prompts that are not similar to the format of the pretraining data (Interpretation).

**Loss.** OPT-IML writes the target-only supervised objective as follows ([[opt-iml]] §3.1, Eq. 1); the other decoder-only runs in this chapter that state their masking use the same form:

```
L(D; θ) = − Σ_i Σ_j log p_θ( t_ij | s_i , t_i,<j )
```

- `D` is the fine-tuning dataset of instances `i`; `θ` are the model parameters.
- `s_i` are the source tokens (instruction plus input) of instance `i`; they receive no loss.
- `t_ij` is the `j`-th target token of instance `i`, and `t_i,<j` are the target tokens before it.

[[instruction-following-without-instruction-tuning]] (Eq. 2) defines **response tuning** by replacing the instruction with the empty string, so the model learns `p(response)` without `p(response | instruction)`. §7 uses this variant.

**Three levels of "unseen".**

| Level | What is withheld from training | Protocol that measures it | Source |
|---|---|---|---|
| L1 held-out instances | new examples of trained tasks | OPT-IML "fully supervised" split | [[opt-iml]] §2.3 |
| L2 held-out tasks from seen clusters | new datasets of a trained task type | OPT-IML "partially supervised" split | [[opt-iml]] §2.3 |
| L3 held-out clusters | every dataset of a task type | FLAN leave-one-cluster-out; T0 four held-out tasks | [[flan]] §2.2; [[t0-multitask-prompted-training]] §3 |

The levels respond differently to the same change. In OPT-IML 30B, increasing training tasks from 16 to 1,024 improved L3 and L2 tasks and left L1 tasks unchanged ([[opt-iml]] §4.4, Fig. 2). A report that pools the three levels into one average hides this difference.

## §2 Held-out protocols in FLAN, T0, and Super-NaturalInstructions

The companion figure [heldout-protocols.html](figures/heldout-protocols.html) lets the reader switch between the FLAN, T0, Super-NaturalInstructions, OPT-IML, and Tülu 3 splits and see which clusters are trained, evaluated, or removed to prevent leakage.

### 2.1 FLAN: leave one cluster out (arXiv 2021-09)

Mechanism ([[flan]] §2.1-2.4):
1. Aggregate 62 public text datasets and assign each to one of 12 task clusters (Fig. 3).
2. Write 10 templates per dataset; up to 3 of them "turn the task around" (for example, generate a movie review instead of classifying one).
3. Treat a dataset D as unseen only if no dataset from any cluster that D belongs to was used in training. Evaluating c clusters therefore requires c separately trained models.
4. Remove neighboring clusters as well: the paraphrase cluster is dropped when evaluating NLI and vice versa, and reading comprehension with commonsense is dropped when evaluating either parent cluster (§2.2, footnote 1).
5. Mix datasets with examples-proportional sampling, a per-dataset limit of 30k examples, and a mixing-rate maximum of 3k.

The mixing rule, from Raffel et al. as used by FLAN (§2.4, footnote 2):

```
r_m = min(e_m, K) / Σ_n min(e_n, K)
```

- `r_m` is the probability of sampling from dataset `m`; `e_m` is its number of examples; `K` is the mixing-rate maximum (3,000 in FLAN).

Worked example. Three datasets have 1,000, 3,000, and 30,000 examples. With `K = 3,000` the capped sizes are 1,000, 3,000, 3,000, so the sampling probabilities are 1/7 = 0.143, 3/7 = 0.429, and 0.429. Without the cap they are 0.029, 0.088, and 0.882, and the largest dataset takes 88% of the batch.

Evidence ([[flan]], LaMDA-PT 137B):
- With the best development-set template, zero-shot FLAN outperformed zero-shot GPT-3 175B on 20 of 25 datasets (§3). **Result (single study).**
- Cluster ablation, holding out NLI, closed-book QA, and commonsense: the average held-out score was 49.9, 55.0, 59.3, 59.2, 60.8, 61.9, and 63.5 for 1 to 7 training clusters (11 to 39 datasets). The only decrease (59.3 to 59.2) came when the sentiment cluster was added (§4.1, Fig. 6).
- Scale: across 422M, 2B, 8B, 68B, and 137B models evaluated on 13 held-out tasks, instruction tuning improved the 68B and 137B models and lowered held-out accuracy for the 8B and smaller models (§4.2, Fig. 7; values are plotted, not printed). The authors' explanation is that learning about 40 training tasks fills the capacity of small models (Interpretation).
- Role of instructions (App. B.2, Table 3, four-cluster average): FLAN 55.2; training without templates 37.3; training with only the dataset name 46.6 (evaluated with instructions) or 47.0 (evaluated with the name).
- Datasets versus templates (App. B.1, Fig. 11): more datasets per cluster improved held-out performance by almost 10%, while the effect of 1, 4, or 10 templates per dataset was described as "comparatively negligible" with one dataset per cluster and disappeared at four datasets per cluster.

Conditions and limits. Instruction tuning did not improve most tasks that are already phrased as sentence completion: FLAN beat LaMDA-PT on 3 of 7 commonsense and coreference tasks (§3). Contamination was checked post hoc with roughly 13-gram overlap against the pretraining corpus; DROP, SQuADv2, and ANLI R1/R2 had almost complete overlap, mostly in passage contexts, and ReCoRD and PIQA scored more than 1% lower on their clean subsets (App. C).

### 2.2 T0: held-out tasks, prompt variety, and no prompt selection (arXiv 2021-10)

Mechanism ([[t0-multitask-prompted-training]] §3-5):
1. Hold out all datasets of four tasks: NLI, coreference resolution, sentence completion, and word sense disambiguation; also exclude every dataset GPT-3 used for evaluation from the main model's training.
2. Evaluate 11 held-out datasets and 14 BIG-bench tasks.
3. Train T5+LM 11B (T5 further trained on 100B C4 tokens with a language-modeling objective) on prompts from P3, which holds 2,073 prompts for 177 datasets (11.7 per dataset).
4. Choose the checkpoint by score on the validation splits of training datasets, so no held-out example affects selection.
5. Do not select prompts on validation data; report the median and interquartile range (IQR, Q3 − Q1) over all prompts of each held-out dataset.

Evidence:
- T0 matched or exceeded GPT-3 (up to 175B) on 9 of 11 held-out datasets while being about 16 times smaller, and improved over a large baseline language model on 13 of 14 BIG-bench tasks (§1, §6.1).
- Prompts per dataset `p` (§6.2, Fig. 6): `p = 1` improved the median over `p = 0`; `p = 1 → 5.7` raised the median on 8 of 11 datasets and lowered the IQR on 7 of 11; `p = 8.03`, which adds prompts that alter the original task, raised the median on 9 of 11 and lowered the IQR on 8 of 11.
- Datasets `d` (Fig. 7): from 39 to 49 datasets the median rose on all 5 held-out NLI datasets and the IQR fell on 1 of 5; from 49 to 55 the median rose on all and the IQR fell on 2 of 5.
- Prompt robustness of the baseline: on RTE with 10 prompts, GPT-3's published prompt scored 58.8% (Brown et al. report 63.5%), and the other 9 prompts had a median of 52.96% with IQR 1.28% (§6.2).

Conditions and limits. T0's 3B model improved on held-out tasks, while FLAN's 8B decoder-only model did not. The authors name two possible causes: encoder-decoder models pretrained with masked language modeling, and more diverse prompts (§7; Interpretation, not tested). For HellaSwag, removing the instruction raised T0's median from 33.65% to 57.93% (§7), which repeats FLAN's finding for completion-style tasks.

### 2.3 Super-NaturalInstructions: 1,616 tasks and source-dataset exclusion (arXiv 2022-04)

Mechanism ([[super-natural-instructions]] §3, §5.1):
1. Collect 1,616 tasks covering 76 task types and 55 languages; each task has a definition (56.6 words on average), 2.8 positive and 2.4 negative examples on average, and at most 6.5K instances.
2. Fix an evaluation set of 12 categories with 154 tasks (119 English, 35 cross-lingual), and sample at most 100 instances per task (15,310 test instances).
3. Remove from training any task created from the same source dataset as a test task (footnote 6). This leaves 757 English training tasks.
4. Score open-ended generations with ROUGE-L instead of ranking answer options.

Evidence (Table 3, ROUGE-L on the English track): Tk-Instruct 11B 62.0; InstructGPT 175B 52.1; GPT-3 175B 45.0; T0 11B 32.3; T5-LM 11B 30.2; copying a demonstration output 28.5; supervised training on the test tasks (upper bound) 74.3. Scaling analyses (§7.1, Fig. 5; T5-3B for the task and instance sweeps, T5 small to xxl for the size sweep): performance grew log-linearly with the number of training tasks, saturated at 64 instances per task, and grew log-linearly with model size; T5-Large trained on 757 tasks (48.0) matched T5-3B trained on 128 tasks (48.4).

Conditions and limits. Models trained only on definitions did not generalize to example-only test encodings, and the reverse also failed; models trained on definitions plus examples were robust to both (§7.2, Table 4). T0 scored 32.3 here, and the authors attribute this to T0's different prompt style (§6.1; Interpretation). Format transfer across instruction collections is therefore itself a generalization axis.

**Implication for a general-purpose model.** The three protocols disagree on details (cluster versus task holdout, mean versus median over templates, best-template versus all-template reporting), and each detail changes the reported number. A generality claim needs the protocol stated with it.

## §3 Scaling tasks, model size, and chain-of-thought data (Flan-PaLM, arXiv 2022-10)

**Setup** ([[flan-palm-scaling-instruction-finetuning]] §2). The mixture has 1,836 tasks from 473 datasets: Muffin (80 tasks), T0-SF (193), NIV2 (1,554), and a chain-of-thought (CoT) mixture of 9 datasets with human-written rationales (Fig. 2). 44 MMLU-related tasks were removed from NIV2 (footnote 4), and the GPT-3 evaluation sets were not used because almost all of them have training sets in the mixture (§2.3). Held-out suites: MMLU (57 tasks), BBH (23), TyDiQA (8 languages), MGSM (10 languages).

**Normalized metric.** Scores are averaged after normalization against random guessing (§2.3, footnote 5):

```
norm = 100 · (raw − r) / (max − r)
```

- `raw` is accuracy in percent; `r` is the random-guess accuracy of the task; `max` is 100.

Worked example from the footnote: with `r = 50`, raw 55 gives 10 and raw 45 gives −10. For a four-option benchmark (`r = 25`), raw 49.3 gives 100 · 24.3 / 75 = 32.4 (derived).

**Evidence (Table 3, normalized average of six held-out scores).**

| Model | 0 tasks | 9 (CoT) | 89 (+Muffin) | 282 (+T0-SF) | 1,836 (+NIV2) |
|---|---|---|---|---|---|
| PaLM 8B | 6.4 | 8.3 | 14.8 | 20.5 | 21.9 |
| PaLM 62B | 28.4 | 29.0 | 33.4 | 37.9 | 38.8 |
| PaLM 540B | 49.1 | 52.6 | 57.0 | 57.5 | 58.5 |

Most of the gain arrived by 282 tasks (§3). The authors give two explanations: the added tasks are not diverse enough, or fine-tuning mainly teaches the model to express knowledge from pretraining, which used 780B tokens against 1.4B fine-tuning tokens (0.2%) (Interpretation).

Worked example, absolute versus relative gain. The 8B model gained 15.5 points (6.4 to 21.9), which removes 15.5 / 93.6 = 16.6% of its remaining error. The 540B model gained 9.4 points (49.1 to 58.5), which removes 9.4 / 50.9 = 18.5% of its remaining error when computed from the rounded table values. The authors report 16.6% for 8B and 18.4% for 540B (§3). The size ranking of "who gained more" reverses with the metric.

**Chain-of-thought data** (§4.2, Fig. 5). Fine-tuning on non-CoT data alone lowered held-out CoT performance, and adding the 9 CoT datasets improved CoT evaluations without lowering non-CoT evaluations. The authors conclude that instruction fine-tuning improves unseen tasks that share the training prompting format (Interpretation). A narrow mixture can also narrow in the other direction: training on the 9 CoT tasks alone lowered direct-prompting MMLU from 55.1 to 48.5 at 62B and from 71.3 to 68.8 at 540B, while at 8B it rose from 24.3 to 26.3 (Table 3). CoT fine-tuning also enabled zero-shot CoT on BBH with "let's think step-by-step", which untuned PaLM did not use successfully (§4.3, Fig. 6).

**Usability.** On 190 open-ended prompts, raters preferred Flan-PaLM 540B over PaLM 540B 79% of the time (§6, Fig. 8).

**Conditions and limits.**
- The fine-tuning step for each model "was chosen based on periodic evaluations (every 2k to 10k steps depending the model size) of the held-out tasks" (§2.2). The held-out suites therefore also served as the checkpoint-selection signal, which T0's rule forbids.
- Table 3 and Table 5 use different mixture proportions ("Proportion A" and "Proportion B", App. E Table 23), so the 540B model scores 58.5 in Table 3 and 58.4 in Table 5.
- GSM8K 83.9% (CoT with self-consistency) is not a held-out result: the GSM8K training set is in the mixture (§4.1).

## §4 Mixed prompt formats, input inversion, and mixture balance

### 4.1 Flan Collection ablations (arXiv 2023-01)

**Setup** ([[flan-collection]] §3). T5-XL (3B, LM-adapted); 8 Held-In validation tasks; 5 CoT validation sets; MMLU and BBH as held-out suites. **Input inversion** trains on swapped pairs, for example generating a question from its answer (§3.4).

**Evidence (Table 1, zero-shot / few-shot).**

| Variant | Held-In | CoT | MMLU | BBH | BBH-CoT |
|---|---|---|---|---|---|
| Flan 2022 (all methods) | 73.8 / 74.8 | 35.8 / 34.1 | 50.3 / 52.4 | 26.2 / 39.3 | 33.9 / 35.2 |
| − CoT data | 73.3 / 73.2 | 28.8 / 24.6 | 47.5 / 46.9 | 18.2 / 30.0 | 18.2 / 12.0 |
| − input inversion | 73.8 / 74.1 | 32.2 / 23.5 | 41.7 / 41.2 | 18.4 / 24.2 | 15.7 / 13.0 |
| − mixture balancing | 71.2 / 73.1 | 32.3 / 30.5 | 45.4 / 45.8 | 15.1 / 24.3 | 13.8 / 15.4 |
| − few-shot templates | 72.5 / 62.2 | 38.9 / 28.6 | 47.3 / 38.7 | 27.6 / 30.8 | 18.6 / 23.3 |
| T5-XL on Flan 2021 | 68.4 / 56.3 | 24.6 / 22.7 | 41.4 / 34.8 | 28.1 / 28.3 | 26.0 / 26.9 |
| T5-XL on Super-NaturalInstructions | 50.3 / 42.2 | 13.8 / 14.3 | 35.6 / 31.1 | 10.4 / 15.6 | 8.0 / 12.5 |

Worked reading of one row. Removing few-shot templates lowers few-shot Held-In by 12.6 points (74.8 to 62.2) and few-shot MMLU by 13.7 points (52.4 to 38.7), but raises zero-shot CoT by 3.1 (35.8 to 38.9) and zero-shot BBH by 1.4 (26.2 to 27.6). Input inversion changes Held-In by 0.0 zero-shot and moves zero-shot MMLU by 8.6 points (41.7 to 50.3). A format or augmentation choice helps some metrics and hurts others, so the per-metric table is the result, not the average.

Further evidence: adding 5% few-shot templates improved zero-shot results, and Held-In and held-out scores both peaked between 10% and 90% few-shot training data (§3.2, Fig. 3). Held-In accuracy peaked near 200 training tasks and then declined, while held-out MMLU rose log-linearly up to all tasks; only T5-Small peaked before the full set (§3.3, Fig. 4). Removing a source from an equal-weighted mixture changed MMLU from 47.3 to 44.7 (T0-SF) and 45.7 (Flan 2021), and removing CoT changed CoT evaluation from 41.4 to 29.1 (Table 2). **Result (single study)** for each. The Flan Collection authors state that OPT-IML's mixing experiments also found the Flan 2021, T0-SF, and T5 mixtures the most broadly beneficial (§3.5); this chapter did not find that ranking stated in the OPT-IML text, so the claim is not counted as replicated.

### 4.2 OPT-IML: three generalization levels and mixture trade-offs (arXiv 2022-12)

Mechanism ([[opt-iml]] §2-4):
1. Consolidate 8 collections into one benchmark: 1,545 training tasks, 35 validation tasks, and 87 test tasks after filtering (Table 1).
2. Split evaluation into the three levels of §1.
3. Remove leakage: for every train-evaluation task pair, compute the fraction of evaluation examples with any 13-gram overlap, and manually inspect the roughly 14,000 pairs above 1% (§2.3).
4. Tune each factor on OPT 30B for 4,000 steps and select settings by the average over the three levels and over 0-shot and 5-shot scores (§4.1).

Evidence (OPT 30B unless noted):
- Mixing cap (EPS): every model with a cap outperformed the uncapped model on average, and caps below 4,096 performed similarly; 4,096 was chosen (§4.2, Table 4).
- Benchmark proportions: with examples-proportional sampling, Super-NaturalInstructions would supply 71% of examples, PromptSource 18%, and FLAN 5% (§4.3). Setting CrossFit, ExMix, T5, and UnifiedSKG to 0% gave the worst model (Table 5).
- Pretraining data mixed into fine-tuning improved L3 and L2 tasks up to 10% and then decreased them; 5% was chosen (§4.5).
- Reasoning data: on 2 of 14 held-out validation reasoning tasks, ROUGE-L rose from 12.2% to 31.6%; 1% reasoning data gave the largest overall gain (§4.6).
- Dialogue data at 0.5% (320,543 BlenderBot 3 dialogues) lowered the 0-shot validation average from 46.0 to 44.8 because the model conformed less often to the short output formats that stereotype-detection and word-analogy tasks require; dialogue data was left out (§4.7, Table 7).
- Final models on 14 standard held-out tasks (Table 9, 0-shot average): OPT 30B 59.2 to OPT-IML 30B 66.3; OPT 175B 61.4 to OPT-IML 175B 68.2. Individual tasks regressed: OPT-IML 30B PIQA 32-shot fell from 78.8 to 69.2 and OpenBookQA 0-shot from 57.2 to 50.6.

Conditions and limits. Each factor was varied independently at 30B; the authors state that the factors may interact and that 30B trade-offs may not hold at larger scales (§6.2). OPT-IML-Max 175B scored 49.1 (0-shot MMLU) and 35.7 (BBH), below Flan-T5 11B's 53.7 and 45.3 (Table 14); the authors list pretraining tokens (OPT: 180B), architecture, and fine-tuning method differences as candidate causes (§6.1; Interpretation).

## §5 Instruction diversity under controlled experiments

**Definition.** Instruction diversity is the number and semantic spread of distinct instructions in the training set, holding total examples fixed.

**Problem.** In natural instruction datasets, instruction count, examples per instruction, and quality change together, so their effects cannot be separated ([[instruction-diversity-unseen-tasks]] §1).

**Mechanism** ([[instruction-diversity-unseen-tasks]] §3-4, arXiv v1 2024-02):
1. Each instruction is a rewrite rule `x → y`; the model must replace the leftmost occurrence of `x` in a string `z` by `y`, or return `z` when `x` is absent (a "no-op").
2. Train a GPT-2-style model with 6 layers, 4 heads, and hidden size 256 from scratch on `S × I = 10^6` examples, where `I` is the number of distinct rules and `S` the examples per rule.
3. Test on 10^5 examples whose rules never appeared in training.

Worked example. With the budget fixed at 10^6 examples, `I = 100` gives `S = 10,000` examples per rule, and `I = 10,000` gives `S = 100`. The paper reports that models trained on fewer than 300 instructions never generalized, even with many examples per rule, and models trained on 1,000 or more instructions always generalized, even with few examples per rule; the transition was near 400 instructions (§4.1, Fig. 2a). The first configuration fails and the second succeeds.

Further evidence:
- Semantic spread: models trained on one constrained rule family (for example characters repeated `k` times, with large `k`) did not generalize to small `k`; mixing three constrained families did (§4.3, Fig. 4).
- Uneven distributions: with 1,000 instructions, accuracy dropped when examples per instruction followed a steep power law; with 10,000 or 100,000 instructions the penalty was small (§4.4, Fig. 3a).
- Input variety: with 1,000 instructions, training on single-occurrence inputs gave 0.41 accuracy on 1 to 20 occurrences, and mixing occurrence counts 1, 5, 10, 15, 20 gave 0.94 (Table 1).
- Pretrained model: Llama-2-7B with LoRA (rank 512, α 1024) on an encrypted-rewrite task (40,000 training and 5,000 test sequences, 40% no-ops) solved only no-op cases at the smaller instruction counts of Fig. 5; halving the data uniformly at 9,000 instructions did not hurt, and non-uniform subsampling did (§5, Fig. 5).

Conditions and limits. The tasks are symbolic, most models have 6 layers and hidden size 256 and are trained from scratch, and the authors list the absence of real-world instruction-following datasets as a limitation. **Result (single study).** The direction agrees with natural-language results: accuracy saturated at 64 instances per task while it grew with task count ([[super-natural-instructions]] §7.1), and a 7B model trained on 2,000 heterogeneous Stack Exchange examples scored 3.83 against 3.49 for 2,000 homogeneous wikiHow "how to" examples ([[lima]] §5, Fig. 5). The evidence on phrasing diversity within one task conflicts: T0 gained from more prompts per dataset, while FLAN found templates per dataset mattered little once each cluster had four datasets (§2). The T0 authors note this discrepancy and hypothesize that their prompts were more diverse in length and wording ([[t0-multitask-prompted-training]] §7; Interpretation, not tested). **Open question:** which kind of diversity (task, cluster, phrasing, input) matters most at modern scale.

## §6 The superficial alignment hypothesis: evidence and counter-evidence

**Definition.** LIMA states the **Superficial Alignment Hypothesis** as: "A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users" ([[lima]] §2, arXiv 2023-05).

### 6.1 LIMA

Setup ([[lima]] §2-3): LLaMA 65B fine-tuned on 1,000 examples (about 750,000 tokens): 200 Stack Exchange STEM, 200 Stack Exchange other, 200 wikiHow, 150 r/WritingPrompts, 50 Super-NaturalInstructions, and 200 author-written (Table 1). Stack Exchange answers were filtered by length (1,200 to 4,096 characters), first-person wording, and references to other answers (§2.1).

Evidence:
- On 300 test prompts, human raters judged LIMA equal or better than GPT-4 in 43% of comparisons, Claude 46%, Bard 58%, and DaVinci003 65% (§1, §4.2, Fig. 1).
- In an absolute check of 50 prompts, 50% of responses were excellent and 88% met the prompt requirements; on 20 out-of-distribution prompts, 20% failed, 35% passed, and 45% were excellent (§4.3).
- Ablations on LLaMA 7B, scored 1 to 6 by ChatGPT: filtered Stack Exchange 3.83, unfiltered 3.33, wikiHow 3.49; growing filtered Stack Exchange from 2K to 32K examples did not raise the score (§5, Figs. 5-6).
- With no dialogue data, 6 of 10 live conversations failed within 3 turns; adding 30 dialogue chains raised excellent turns from 45.2% to 76.1% and reduced failures from 15 of 42 turns to 1 of 46 (§6, Fig. 7).

Limits. The main evaluation is pairwise preference on open-ended prompts, with no objective task benchmark. The authors note "significant contact" between the two groups that wrote training and test prompts (§2.2, footnote 1). Held-out perplexity rose while generation quality rose, so the authors selected checkpoints by hand between epochs 5 and 10 on a 50-example development set (§3, App. B).

### 6.2 URIAL: token distribution shift and tuning-free alignment (arXiv 2023-12)

Mechanism ([[urial]] §2):
1. Decode a response `o` greedily from the aligned model.
2. At each position `t`, feed the same prefix to the base model and rank its next-token distribution.
3. Record the **base rank** `η` of the aligned token `o_t`: unshifted if `η = 1`, marginal if `1 < η ≤ 3`, shifted if `η > 3`.

Evidence (Fig. 3, 1,000 examples): Llama-2-7B to Llama-2-7B-chat 77.7% unshifted, 14.5% marginal, 7.8% shifted; Llama-2-7B to Vicuna-7B-v1.5 82.4% / 12.8% / 4.8%; Mistral-7B to Mistral-7B-Instruct 82.2% / 12.5% / 5.2%. Shifted tokens were mostly discourse markers and safety phrases ("However", "cannot", "Here", "sorry"), and shifts were larger at early positions (§2.2, Fig. 4). Worked example (derived from the averages): across 1,000 Llama-2-7B-chat tokens, about 777 are also the base model's top choice, 145 are its second or third choice, and 78 rank lower. A single response need not match these counts, because shifted tokens are concentrated at early positions.

URIAL itself prompts a base model with a system prompt and K = 3 restyled examples (1,011 tokens) and no weight updates (§3.3). On the authors' 1,000-prompt evaluation set, merged from AlpacaEval, MT-Bench, the LIMA test set, HH-RLHF red-team prompts, and MaliciousInstruct, and scored 1 to 5 by GPT-4 on five aspects and by ChatGPT on safety (§4.1-4.2, Table 1), the averages were: Mistral-7B with URIAL 4.63 against 4.44 for Mistral-7B-Instruct; Llama-2-70B with URIAL 4.74 against 4.67 for Llama-2-70B-chat (both 4-bit quantized); Llama-2-7B with URIAL 4.33 against 4.47 for Llama-2-7B-chat.

Limits. Scores come from GPT-4 and ChatGPT judges. The authors state that tuning may still be needed for coding, mathematics, and interactive agents (§5.4). Their examples of knowledge loss after SFT (for example, Mistral-7B-Instruct denying that Facebook changed its name) are single case studies (§4.3, App. B).

### 6.3 Counter-evidence: objective scaling of post-training (arXiv 2024-09)

[[revisiting-superficial-alignment-hypothesis]] fine-tunes base Llama-3, Llama-2, and Mistral models on 0 to 10,000 task examples (GSM8K, SubQA, StarCoder Self-Align, Conifer, Dolly) and scores standard benchmarks (§3.1, Table 1). Task accuracy followed a power law (§3.2):

```
P = a · D^(1/b)
```

- `P` is benchmark accuracy in percent; `D` is the number of fine-tuning examples; `a` and `b` are fitted per model and task (App. A.6, Table 7).

Worked example (Llama-3-8B, GSM8K: `a = 19.47`, `b = 7.66`). Each tenfold increase in `D` multiplies `P` by 10^(1/7.66) = 1.351. `D = 100` gives 35.5, `D = 1,000` gives 48.0, and `D = 7,500` (the full training set) gives 62.4. The fitted value at 1,000 is close to the measured 46.5% of the 1,000-example GSM8K model in Table 2. The figure's third panel computes this curve for every Table 7 fit.

Evidence:
- Llama-3-8B trained on the 1,000 LIMA examples scored 14.7% on GSM8K and 21% on SubQA; trained on 1,000 task examples it scored 46.5% and 36%. A GPT-4o judge using LIMA's comparison prompt preferred the LIMA-trained responses on GSM8K 84.4% of the time against 0.24% for the task-trained ones (Table 2).
- Style and formatting errors reached their saturated level with 100 examples (the sampled sizes were 0, 100, 1,000, and the full training split), while total mistakes tracked reasoning errors (r² 0.98 for math, 0.99 for multihop QA on Llama-3-8B) (§4.2, Fig. 3).
- On 100 news events after the Llama-3-8B cutoff, a model first post-trained for multihop reasoning answered 81 direct and 55 multihop questions after SFT on the events, against 65 and 37 for the base model; with the event in the prompt it answered 86 and 71, against 49 and 34 (Table 4).

Limits. Each run targets one task, and the authors state that they do not know how fine-tuning for one task affects performance on other tasks (§7). **Result (single study).**

### 6.4 The imitation gap (arXiv 2023-05)

[[false-promise-imitating-proprietary-llms]] fine-tuned GPT-2 1.5B and LLaMA 7B and 13B on up to 150M tokens of ChatGPT outputs. Crowdworkers rated about 70% of the 13B imitation model's outputs as equal to or better than ChatGPT's (Fig. 1). Broad imitation data lowered Natural Questions accuracy (7B: 17 to 10; 13B: 20 to 15), while 6,000 targeted NQ-style examples raised it (7B: 22; 13B: 27; ChatGPT 31) (Table 1). Style converged: when ChatGPT used a list, the imitation model did so in 13% of cases before and 81% after 150M tokens (Table 2). Larger base models improved benchmark accuracy more than more imitation data (§4.3). The method-level treatment of this result is in [[ch-19]]. [[tulu-1-how-far-can-camels-go]] (§6) argues that imitation data helps when it covers diverse skills (Interpretation).

### 6.5 Reconciling the evidence

The results are consistent with a split between format and capability (Interpretation). Response format and assistant style are learned from 100 to 1,000 examples ([[lima]] §4-6; [[revisiting-superficial-alignment-hypothesis]] Fig. 3), and at 4.8% to 7.8% of token positions the aligned model's token ranks below third in the base model ([[urial]] Fig. 3). Accuracy on tasks that need reasoning or specific knowledge keeps improving with task-specific data ([[revisiting-superficial-alignment-hypothesis]] Fig. 1; [[flan-palm-scaling-instruction-finetuning]] §4.2) and does not improve, or regresses, when the SFT data does not cover the task ([[false-promise-imitating-proprietary-llms]] Table 1; [[tulu-1-how-far-can-camels-go]] §5.1). For a general model, a small curated set is enough to set the response format, and the breadth of task data determines which capabilities improve.

## §7 Instruction following without instruction tuning (arXiv 2024-09)

**Response tuning** ([[instruction-following-without-instruction-tuning]] §4). Llama-2-7B and OLMo-7B-Feb2024 were trained on the 1,030 LIMA responses without their instructions and compared with the same base models instruction-tuned on LIMA, using length-controlled AlpacaEval win rates (Table 1): response-tuned Llama-2-7B won 43.3% ± 1.1% (base model 2.4%), and response-tuned OLMo-7B-Feb2024 won 43.7% ± 1.7% (base 4.7%). A 50% win rate would mean equal quality.

**Response ranking capability** (§4.2, Eq. 3). For independent pairs (instruction, response) and (instruction′, response′), the capability holds when:

```
p_θ(response | instruction) > p_θ(response′ | instruction)
```

- `p_θ` is the model's sequence probability; `response′` is a desirable response to a different instruction.

On Alpaca pairs this held 80.4% of the time for base Llama-2-7B and 77.4% for its instruction-tuned version; 74.5% and 74.3% for OLMo (Table 2). Worked example (illustrative values): a base model can assign 10^-40 to the right recipe and 10^-45 to an unrelated answer, so ranking holds, while the continuation "Give me a recipe for cake" has probability 10^-5 and is what greedy decoding produces. Raising the probability of response-like strings, which response tuning does, can change the output without teaching the mapping.

**Single-task fine-tuning** (§5, Table 3). Training on one narrow task still produced general instruction following. Llama-2-7B win rates: GSM 23.7%, poetry 22.9%, MBPP 16.9%, recipes 14.6%, chess 2.1%. OLMo-7B-Feb2024: GSM 30.3%, poetry 21.9%, recipes 21.5%, MBPP 10.4%, chess 6.3%. Responses followed the fine-tuning format only for instructions similar to the fine-tuning instructions (§5.2, Fig. 5).

**Rule-based adapter** (§6, Eq. 4-5, Table 4). A product of Llama-2-7B with three hand-written rules (raise the end-of-sequence score with response length, change the probability of 15 fixed tokens, penalize repeated tokens) won 24.4% ± 0.40%. Removing the EOS rule lowered this to 10.4%, the diversity rule to 14.3%, and the token rule to 16.3%.

Conditions and limits. Llama-2-7B may have seen instruction data in pretraining; OLMo-7B-Feb2024 had no intentional instruction data, and both gave similar conclusions (§1.1). The evaluation is an LLM judge with greedy decoding.

**Implication for a general-purpose model.** A win rate above the base model does not show that the SFT data taught the measured behavior, because response-only and narrow-task training also raise it. A model fine-tuned on a narrow task can follow unrelated instructions, so the authors recommend testing it as a general-purpose chatbot, including safety testing (§7).

## §8 The held-out task and template protocol used in ch-30 to ch-36

**Definition.** The protocol is the set of rules that decides which tasks, datasets, templates, and evaluation settings are excluded from SFT data and from design decisions, so that a reported score estimates performance on untargeted tasks.

**Rules.**
1. **Cluster holdout.** Group tasks by task type and hold out whole clusters for L3 evaluation. Also remove clusters that are similar in task type to a held-out cluster (FLAN drops paraphrase when NLI is held out, [[flan]] §2.2 footnote 1, and in its §4.1 ablation drops reading comprehension with commonsense because it is "too similar" to commonsense reasoning, footnote 3).
2. **Source-dataset exclusion.** Remove any training task built from the same source dataset as an evaluation task ([[super-natural-instructions]] footnote 6), including input-inverted variants, which reuse the same input-output pairs ([[flan-collection]] §3.4).
3. **N-gram decontamination.** Check every training source against every evaluation set: 13-gram overlap with manual review of pairs above 1% ([[opt-iml]] §2.3), or Tülu 3's rule that a test instance overlaps when more than 50% of its tokens share an 8-gram with one training instance, with a training set flagged when it overlaps more than 2% of an evaluation's instances ([[tulu-3]] §3.2).
4. **Development suite versus unseen suite.** Make all design choices (mixture, epochs, learning rate, checkpoint) on a development suite, and read unseen-suite scores only after the choices are fixed; Tülu 3 did not examine its unseen scores while developing its models ([[tulu-3]] §2.2, §7.3). Choose checkpoints on training-task validation splits or the development suite (the T0 rule, [[t0-multitask-prompted-training]] §5), never on the unseen suite.
5. **Template holdout.** Evaluate with templates and constraint types that do not appear in training, and report the median and IQR over templates ([[t0-multitask-prompted-training]] §5) instead of the best template. This course treats IFEval's 25 constraint templates as a development set in this sense (Interpretation), because leading models such as GPT-4.1 and Claude 3.7 Sonnet scored below 50% on IFBench's 58 unseen constraints ([[ifbench]] §1, Fig. 1).
6. **Three levels and three settings.** Report L1, L2, and L3 separately ([[opt-iml]] §2.3), each in zero-shot, few-shot, and CoT settings ([[flan-collection]] Table 1).
7. **Objective metrics beside judges.** Report exact-match or execution accuracy next to any LLM-judge win rate ([[revisiting-superficial-alignment-hypothesis]] Table 2; [[tulu-1-how-far-can-camels-go]] §5.4).
8. **Base-model row.** Evaluate the base model on the same suite with the same templates, and report per-task regressions ([[opt-iml]] Table 9), which [[ch-30a]] turns into a forgetting report.

**Worked example: detecting development-set overfitting** ([[tulu-3]] §7.4.1, Table 32, Tülu 3 8B SFT). Define for a data source `s`:

```
Δ_dev(s)    = score_dev(with s)    − score_dev(without s)
Δ_unseen(s) = score_unseen(with s) − score_unseen(without s)
```

- `score_dev` is the development metric for a skill (IFEval); `score_unseen` is the unseen metric for the same skill (IFEval-OOD).

Persona instruction data: the full SFT model scores 72.8 on IFEval and 17.6 on IFEval-OOD; without persona data it scores 53.6 and 18.0. So `Δ_dev = +19.2` and `Δ_unseen = −0.4`. WildChat data: without it the model scores 70.1 and 20.8, so `Δ_dev = +2.7` and `Δ_unseen = −3.2`. Both sources raise the development score and do not raise the unseen score; the Tülu 3 authors conclude that their data choices overfit to the development evaluations for precise instruction following (§7.4.1). A source with `Δ_dev > 0` and `Δ_unseen ≤ 0` is flagged under this protocol (course rule; Interpretation).

## Negative samples and negative feedback

In this stage "negative" appears in two of the four senses defined in the style standard: (1) negative marginal value, and (2) negative as content. No source in this chapter uses negatives as conditioning (3) or as gradient (4). SFT cross-entropy still lowers the probability of every non-target token through the softmax; [[ch-30]] covers that mechanism, and [[ch-31a]] covers explicit negative gradients.

1. **Where negatives come from.** (a) Quality filters discard samples, for example LIMA's Stack Exchange length and style filters ([[lima]] §2.1). (b) Whole sources can have negative marginal value for some capabilities: at 13B, Super-NaturalInstructions alone lowered GSM from 14.5 to 4.0 and BBH from 39.3 to 4.5 relative to the base model ([[tulu-1-how-far-can-camels-go]] Table 3); 0.5% dialogue data lowered OPT-IML 30B's 0-shot validation average from 46.0 to 44.8 ([[opt-iml]] Table 7). (c) Incorrect outputs appear as content inside instructions: each Super-NaturalInstructions task carries 2.4 negative examples with explanations on average ([[super-natural-instructions]] Table 2). (d) Refusal targets: LIMA includes 13 training prompts whose responses partially or fully refuse ([[lima]] §2.2).
2. **Current practice.** Type (1) samples are discarded or their source is removed after an ablation. Type (2) examples are kept in the input or as the target and trained with ordinary cross-entropy on the target tokens.
3. **Mechanism.** A negative example placed in the instruction changes only the conditioning context `s_i` of §1; the loss is still computed on the target. A refusal target is a positive target whose content is a refusal.
4. **Evidence.** Filtering Stack Exchange raised the 7B ablation score from 3.33 to 3.83 ([[lima]] Fig. 5). On T5-3B, training and testing with two negative examples gave 54.3 ROUGE-L, the same as without them (54.3), and adding explanations gave 52.6 ([[super-natural-instructions]] Table 4 diagonal); the authors describe negatives as helping "a little bit" (§7.2). LIMA responded safely to 80% of 30 sensitive test prompts, including 6 of 10 with malicious intent ([[lima]] §4.3).
5. **Controls.** When deciding whether to remove or keep a source, ablate it against both development and unseen suites, because a source can raise the development score while leaving the unseen score flat or lower ([[tulu-3]] Table 32; [[flan-collection]] Table 2 reports per-source removal). When refusal targets are added, measure the refusal rate on benign prompts, because the Tülu authors attribute Tülu 13B's 0.1% toxic generation rate on ToxiGen to overfitting to refusal behavior (item 7).
6. **Diagnostics.** Per-source leave-one-out tables on development and unseen suites; per-category counts before and after filtering (the [[ch-29]] survival report); the rate of refusals on benign prompts.
7. **Effect on generality.** Filters and source removal can delete a task type and narrow coverage. Refusal-heavy targets can raise over-refusal: Tülu 13B produced 0.1% toxic generations on ToxiGen against 27.7% for ChatGPT, which the authors attribute to overfitting to refusal behavior ([[tulu-1-how-far-can-camels-go]] §5.3; Interpretation).

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| FLAN (LaMDA-PT) | 137B | SFT | steps; batch; optimizer; LR | 30k gradient steps; 8,192 tokens; Adafactor; 3e-5 | arXiv:2109.01652v5 §2.4 | verified 2026-09-15 | no ablation reported |
| FLAN (LaMDA-PT) | 137B | SFT | mixture; input/target length; packing | examples-proportional, ≤30k examples per dataset, mixing-rate max 3k; 1024 / 256 tokens; packing with EOS separator | §2.4, footnote 2 | verified 2026-09-15 | no ablation reported |
| FLAN (LaMDA-PT) | 137B | SFT | templates; compute; checkpoint | 10 per dataset; ~60 hours on TPUv3 with 128 cores; final checkpoint at 30k steps | §2.1, §2.4 | verified 2026-09-15 | App. B.1 Fig. 11: 1 vs 4 vs 10 templates, effect "comparatively negligible" |
| T0 (T5+LM) | 11B | SFT | LR; batch; optimizer; dropout | 1e-3; 1024 sequences (2^20 input tokens); Adafactor; 0.1 | arXiv:2110.08207v3 §5 | verified 2026-09-15 | no ablation reported ("standard practice for fine-tuning T5") |
| T0 (T5+LM) | 11B | SFT | truncation; mixing cap; checkpoint rule | 1024 / 256 tokens, packing; datasets over 500,000 examples treated as 500,000 / num_templates; highest score on training-dataset validation splits | §5 | verified 2026-09-15 | no ablation reported |
| Tk-Instruct (T5.1.1-xxl) | 11B | SFT | batch; LR; steps; encoding | 1,048,576 tokens (1,024 examples); constant 1e-5; 1,000 steps; definition + 2 positive examples | arXiv:2204.07705v3 App. D, §4 | verified 2026-09-15 | Table 4 (T5-3B): Def+Pos(2) 54.3 vs Def 45.0 ROUGE-L |
| Tk-Instruct analysis models (T5 ≤ 3B) | ≤3B | SFT | epochs; batch; LR; lengths | 2 epochs; 16; constant 1e-5; input 1024, output 128 | App. D | verified 2026-09-15 | no ablation reported |
| Flan-PaLM | 540B | SFT | batch; dropout; LR; steps; schedule | 32; 0.1; 1e-3; 21k; constant, Adafactor, packing with attention masked across packed example boundaries | arXiv:2210.11416v5 App. E Table 22, §2.2 | verified 2026-09-15 | App. E: LR, batch, dropout named most important; no table |
| Flan-PaLM | 8B, 62B | SFT | batch; dropout; LR; steps | 32; 0.05; 3e-3; 40k | App. E Table 22 | verified 2026-09-15 | no ablation reported |
| Flan-T5-XL / Flan-T5-XXL | 3B / 11B | SFT | batch; dropout; LR; steps | 64; 0.05; 5e-4; 38k / 14k | App. E Table 22 | verified 2026-09-15 | no ablation reported |
| Flan-PaLM (all sizes) | all | SFT | per-task caps; mixture proportions (sampling weight) | Muffin 30,000, T0-SF 20,000, CoT 100,000, NIV2 5,000; Proportion A 52% / 15% / 3% / 30% (§3-4 runs); Proportion B 46.0% / 27.9% / 1.8% / 24.2% (other runs) | App. E Table 23 | verified 2026-09-15 | App. E: "Proportion A signaled that the T0 mixture was good for performance" |
| Flan-PaLM | 540B | SFT | compute; tokens; checkpoint rule | 0.2% of pretraining compute (~512 v4 TPU chips, 37 hours); 1.4B fine-tuning tokens (stated for the PaLM runs); step chosen by periodic held-out-task evaluation | §2.2, §3, Table 2 | verified 2026-09-15 | no ablation reported |
| OPT-IML 30B / 175B | 30B / 175B | SFT | GPUs; batch; LR; steps; warmup; tokens | 64 / 128; 256 / 128; 5e-05; 4,000 / 8,000; 60; 2B | arXiv:2212.12017v3 §3.3 Table 3 | verified 2026-09-15 | LR swept over {1e-5, 3e-5, 5e-5, 6e-5}, per-GPU batch over {2, 4, 8} on the validation split; results not tabulated |
| OPT-IML 30B / 175B | 30B / 175B | SFT | optimizer; schedule; regularization; sequence | Adam (0.9, 0.95); linear warmup then linear decay to 0; dropout 0.1, grad clip 1.0; packed 2048 tokens with document attention masking; loss on target tokens only | §3.1-3.3 | verified 2026-09-15 | §3.2: document masking improved stability and performance (no table) |
| OPT-IML 30B / 175B | 30B / 175B | SFT | mixture | EPS 4,096; CrossFit/ExMix/FLAN/NIV2/PromptSource/T5/UnifiedSKG = 4/2/20/25/45/2/2 (%); 5% pretraining data; 1% reasoning data; 0% dialogue | §4.2-4.7 | verified 2026-09-15 | Tables 4, 5, 7 and Figs. 4-5 (OPT 30B, 4,000 steps) |
| LIMA | 65B | SFT | data; epochs; optimizer; LR; batch | 1,000 examples (~750,000 tokens); 15 epochs; AdamW β1 0.9, β2 0.95, weight decay 0.1; no warmup, 1e-5 linear decay to 1e-6; 32 examples; texts over 2048 tokens trimmed | arXiv:2305.11206v1 §3, Table 1 | verified 2026-09-15 | no ablation reported |
| LIMA | 65B | SFT | dropout; checkpoint rule | residual dropout 0.0 (bottom) to 0.3 (top); manual selection between epochs 5 and 10 on a 50-example development set | §3, App. B | verified 2026-09-15 | App. B Fig. 9: perplexity and generation quality rose together |
| Revisiting-SAH task runs (Llama-3-8B, Mistral-7B, Llama-2-7B; 70B variants) | 7B-70B | SFT | epochs; LR; batch; PEFT | 3 epochs; 1e-5 cosine to 0; batch 2 / 8 / 16 for 0-100 / 101-1,000 / 1,001+ examples (8B, 13B) and 16 / 32 / 128 (70B); no PEFT | arXiv:2410.03717v1 §3.1, App. A.5 Table 6 | verified 2026-09-15 | A.5: batch size "has a big effect" for small datasets (no table) |
| Response-tuning and single-task runs | 7B | SFT | LR; schedule; batch; epochs | Llama-2-7B 1e-5, OLMo-7B-Feb2024 3e-6; cosine to 0 with 10% linear warmup; 64; epochs chosen from {5, 7, 10, 15, 20} | arXiv:2409.14254v1 App. A, §4.1 | verified 2026-09-15 | epochs chosen by AlpacaEval win rate vs GPT-3.5-turbo on the authors' validation set |
| Tülu (LLaMA) | 7B, 13B | SFT | epochs; LR; loss masking | 2 epochs; 2e-5; loss on assistant tokens only | [[tulu-1-how-far-can-camels-go]] ledger (arXiv:2306.04751v2 App. D, §3.2) | verified 2026-09-14 | no ablation reported |
| URIAL (base models, no training) | 7B-70B | eval-gate | prefix; decoding | system prompt + K = 3 restyled examples, 1,011 tokens; greedy | arXiv:2312.01552v1 §3.3, §4.1 | verified 2026-09-15 | Table 1: K = 1, 3, 8 compared |

**Starting point for a small general-purpose run.** For a 7B-8B base model and 1,000 to 10,000 examples per task, full fine-tuning for 3 epochs with learning rate 1e-5, cosine decay to 0, and batch size 16 above 1,000 examples reproduces the setting of the Revisiting-SAH runs, which were single-task runs on Llama-3-8B, Mistral-7B, and Llama-2-7B. Compute loss on response tokens only, as in OPT-IML and Tülu. Mix sources with examples-proportional sampling and a per-dataset cap: FLAN used a mixing-rate maximum of 3,000 at 137B, and OPT-IML found every cap below 4,096 performed similarly at 30B; neither was tested at 7B. Include CoT data when CoT evaluation is in the suite: Flan-PaLM's Proportion B gives CoT 1.8% (App. E says Proportion B was used for the runs outside §3-4, which include the Flan-T5 models), and OPT-IML used 1% at 30B and 175B. Select checkpoints with the T0 rule of §8 rule 4.

## Generalization lens

**(a) What increases breadth.**
- More task clusters and tasks: FLAN 49.9 to 63.5 over 1 to 7 clusters ([[flan]] Fig. 6); log-linear growth with tasks ([[super-natural-instructions]] §7.1; [[flan-collection]] Fig. 4); Flan-PaLM 8B 6.4 to 21.9 normalized ([[flan-palm-scaling-instruction-finetuning]] Table 3). **Replicated.**
- More distinct instructions at a fixed budget ([[instruction-diversity-unseen-tasks]] §4.1) and more prompts per dataset ([[t0-multitask-prompted-training]] §6.2).
- Mixed zero-shot, few-shot, and CoT formats, and input inversion for held-out tasks ([[flan-collection]] Table 1).
- Larger or better-pretrained base models ([[flan]] Fig. 7; [[false-promise-imitating-proprietary-llms]] §4.3; [[urial]] Table 1).

**(b) What causes narrowing or forgetting.**
- Small models: held-out loss at 8B and below in FLAN's setting ([[flan]] Fig. 7).
- Missing formats: no CoT data lowers CoT accuracy ([[flan-palm-scaling-instruction-finetuning]] §4.2); CoT-only data lowers direct MMLU (Table 3).
- Format-dominant sources: 0.5% dialogue data reduced compliance with short-answer formats ([[opt-iml]] §4.7); single sources lowered GSM, BBH, and TyDiQA below the base model ([[tulu-1-how-far-can-camels-go]] Table 3).
- Many held-in tasks: Held-In accuracy fell after about 200 tasks ([[flan-collection]] Fig. 4).
- Broad imitation data lowered knowledge QA ([[false-promise-imitating-proprietary-llms]] Table 1); per-task regressions after multitask tuning ([[opt-iml]] Table 9).

**(c) How to measure it at this stage.**
- The eight rules of §8, with L1/L2/L3 reported separately and a base-model row.
- Median and IQR over unseen templates, not the best template ([[t0-multitask-prompted-training]] §5).
- Development versus unseen deltas per data source ([[tulu-3]] Table 32).
- Known measurement errors: judge win rates reward style ([[revisiting-superficial-alignment-hypothesis]] Table 2; [[tulu-1-how-far-can-camels-go]] §5.4, r = 0.96 between AlpacaEval win rate and unique tokens); perplexity does not track generation quality ([[lima]] App. B); choosing checkpoints by held-out-suite scores turns those scores into a selection signal, so they no longer estimate performance on untargeted tasks without bias (Interpretation; [[flan-palm-scaling-instruction-finetuning]] §2.2 selected steps this way, and the size of the effect was not measured); n-gram contamination checks produce false positives ([[flan]] App. C) and do not target paraphrased test items, which Tülu 3 tried to detect with embedding matching but could not separate from distributional similarity ([[tulu-3]] §3.2). Contamination methods are covered in [[ch-48]].

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Holding out datasets but training on other datasets of the same cluster, then calling the result L3 | Gains on the "unseen" task shrink when the whole cluster is removed from training | Rebuild the split with cluster holdout and neighbor-cluster removal (§8 rules 1-2) |
| Training on an input-inverted or reformatted version of an evaluation dataset | Unseen score on that dataset exceeds the scores on other held-out tasks of the same cluster | Source-dataset exclusion list keyed by original dataset ID; 13-gram or 8-gram overlap report |
| Reporting the best template | Reported score exceeds the median over templates by more than the IQR | Report median and IQR over all held-out templates |
| Choosing checkpoints or mixtures on the unseen suite | Unseen and development curves move together through every decision | Log which suite each decision read; unseen scores read only after choices are fixed |
| Using a judge win rate as the only SFT metric | Win rate rises while exact-match accuracy is flat or falls | Report an objective metric beside every win rate |
| Adding a large single-format source | Short-answer or classification tasks receive multi-sentence outputs where a label is expected | Per-category format-compliance rate before and after adding the source |
| Removing CoT examples during filtering | Direct-prompt scores stable, CoT scores drop | Keep CoT and non-CoT evaluations as separate columns |
| Treating a narrow SFT run as narrow in behavior | The model follows unrelated instructions, including unsafe ones | Run the general chat and safety suite on every narrow fine-tune |
| Selecting checkpoints by validation perplexity | Chosen checkpoint has lower judged quality than later ones | Generation-based development metric, as in LIMA App. B |
| Counting examples instead of distinct instructions | Doubling data does not change held-out scores | Report the number of distinct instructions and clusters with the example count |

## Check your understanding

1. FLAN's 8B model lost held-out accuracy after instruction tuning, while T0's 3B model gained. List two differences between the setups and explain how each could produce the opposite outcome.
2. Why does examples-proportional mixing without a cap reduce the number of tasks that effectively shape the model, and how does the cap in §2.1 change that?
3. In Flan-PaLM, the 8B model gained more points and the 540B model removed more of its error. Which quantity would you use to compare instruction-tuning benefit across sizes, and why?
4. Explain why removing few-shot templates in the Flan Collection raised zero-shot CoT accuracy but lowered few-shot MMLU. What does this imply about reporting a single average?
5. Explain how the rewrite-rule result (fewer than 300 instructions never generalize at 10^6 examples) and LIMA's plateau from 2K to 32K examples can both be caused by the same property of the training set.
6. A model trained on 1,000 LIMA examples wins 84.4% of GSM8K comparisons and scores 14.7%. Explain the causal chain from its training data to both numbers.
7. Response-tuned models reach 43% win rates without seeing instructions. What does this imply about what SFT data must supply for a general model, and what it need not supply?
8. A data source raises IFEval by 19.2 points and lowers IFEval-OOD by 0.4. Explain why this pattern indicates overfitting to the development suite rather than a capability gain.

## Connections

- Previous: [[ch-29d]] — User Simulators, Trajectory Verification, and Failed Trajectories.
- Next: [[ch-30]] — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate.
- Dependencies: [[ch-00]] — What General Capability Means and How It Is Measured; [[ch-29]] — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification.
- Earlier chapters used here: [[ch-19]] — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing (imitation versus capability); [[ch-22]] — Quality, Diversity, and Gradient-Based Data Selection.
- Later chapters that reuse the §8 protocol: [[ch-30a]] — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control; [[ch-30b]] — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares; [[ch-31a]] — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; [[ch-36]] — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split.
- Evaluation phase: [[ch-47]] — Evaluation Harness and Suite Design for General Capability; [[ch-48]] — Contamination Detection and Its Effect on Reported Scores.

## Sources

- [[flan]] — leave-one-cluster-out protocol, mixing rule, cluster, scale, instruction, and template ablations, contamination appendix, recipe rows.
- [[t0-multitask-prompted-training]] — held-out task selection, checkpoint rule without held-out data, median/IQR reporting, prompts-per-dataset and datasets ablations.
- [[super-natural-instructions]] — 1,616-task benchmark, source-dataset exclusion, Table 3 results, task/instance/size scaling, encoding and negative-example ablation.
- [[flan-palm-scaling-instruction-finetuning]] — task and size scaling (Table 3), normalized metric, CoT ablation, checkpoint-selection rule, hyperparameters and mixture caps.
- [[flan-collection]] — method ablations (Table 1), mixed prompt settings, task scaling for Held-In versus held-out, source removal (Table 2).
- [[opt-iml]] — three generalization levels, 13-gram leakage check, EPS, benchmark proportions, pretraining/reasoning/dialogue data effects, recipe.
- [[instruction-diversity-unseen-tasks]] — controlled instruction count, semantic spread, distribution, and LoRA experiments.
- [[lima]] — superficial alignment hypothesis, data composition, preference and absolute results, diversity/quality/quantity ablations, dialogue, recipe (numbers taken from arXiv:2305.11206v1).
- [[urial]] — token distribution shift, tuning-free alignment results, stated limits.
- [[revisiting-superficial-alignment-hypothesis]] — power-law fits, LIMA-1k versus task-1k accuracy and win rate, error analysis, new-knowledge experiment, recipe.
- [[false-promise-imitating-proprietary-llms]] — imitation style versus capability numbers (Tables 1-2).
- [[instruction-following-without-instruction-tuning]] — response tuning, response ranking capability, single-task tuning, rule-based adapter, recipe.
- [[tulu-1-how-far-can-camels-go]] — per-dataset capability regressions, AlpacaEval length bias, over-refusal observation, recipe row.
- [[tulu-3]] — development versus unseen suite design, decontamination thresholds, Table 32 data-source ablations (numbers taken from arXiv:2411.15124v5).
- [[ifbench]] — unseen constraint types versus IFEval templates.

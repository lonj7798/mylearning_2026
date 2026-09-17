<!-- chapter: ch-00
     track: foundations
     kind: content
     title: What General Capability Means and How It Is Measured
     deps: []
     sources: [[tulu-3]], [[tulu-3-eval-regime]], [[paloma]], [[olmes]], [[helm]], [[signal-and-noise-eval]], [[benchmark-variance-quantified]], [[prompt-format-sensitivity-formatspread]], [[capability-structure-factors]], [[measure-of-intelligence]], [[gpt-2-unsupervised-multitask]], [[observational-scaling-laws]], [[adding-error-bars-evals]], [[olmo-2]], [[rlvr-beyond-base-model]], [[transferability-of-llm-reasoning]], [[ifbench]], [[xstest]], [[llama-3]], [[swe-bench-illusion]], [[echo-chamber-rl-post-training]], [[finetuning-compromises-safety]], [[helmet]], [[bfcl]], [[agentic-benchmark-checklist]], [[multichallenge]], [[scaling-laws-unreliable-downstream]]
     figures: figures/dev-unseen-and-noise.html
     revised: 2026-09 (generality revision)
-->

# Chapter 00 — What General Capability Means and How It Is Measured

> **Core insight.** A generally capable model is one that scores well on tasks and domains that were neither targeted by its training nor used to make development decisions, reported as a profile over capability axes rather than as one average. The Tülu 3 report shows why the second condition matters: removing its Persona datasets (synthetic SFT data built to target mathematics, coding, and instruction following, §4.2) lowers IFEval, a development benchmark, by 19.2 points, while IFEval-OOD, an unseen benchmark for the same skill, changes by 0.4 points in the other direction ([[tulu-3]] Table 32). Prompt format alone also moves scores: across 53 tasks, 10 plausible prompt formats per task give a median accuracy spread of 7.5 points ([[prompt-format-sensitivity-formatspread]] §4.2).
>
> **Guideline.** When a training decision is evaluated, read a fixed development suite and, for each capability axis, keep at least one unseen benchmark whose scores are not read until the decision is final, because only the unseen benchmark separates a gain in the skill from a gain specific to the development benchmark ([[tulu-3]] §2.2, §7.4). Otherwise, report the result as a development-benchmark gain only. When two checkpoints are compared, report the paired standard error of the difference, a seed or checkpoint-noise estimate, and the prompt format, because each of these is 1 point or more in the cited settings: under the 0/1 formula and an assumed per-item correlation of 0.6, a 2.3-point HumanEval difference between two Tülu 3 checkpoints has a 95% interval of (−2.6, 7.2) (§6.1); three Tülu 3 70B SFT seeds have a standard deviation of 1.33 points on the average ([[tulu-3]] Table 14); and the median spread over 10 prompt formats is 7.5 points ([[prompt-format-sensitivity-formatspread]] §4.2). When a base model is evaluated early in pretraining, use cloze formulation or bits-per-byte, because the multiple-choice formulation stays near chance until the model acquires the answer format, so comparisons of early checkpoints on it are dominated by noise, and bits-per-byte raised benchmark signal-to-noise in small-scale decisions ([[olmes]] §3.4; [[benchmark-variance-quantified]] §3.3; [[signal-and-noise-eval]] §5.3). Otherwise, once the model scores above chance on the multiple-choice formulation, use it, because it gives the higher and more accurate score for strong models ([[olmes]] §3.4).

## Why this chapter matters for a general-purpose model

Every stage of the pipeline (pre-training → mid-training → SFT → preference optimization → RL → evaluation) is tuned by reading benchmark scores. Data mixtures, learning rates, checkpoints, and reward designs are kept or discarded according to those scores. Two measurable problems follow. First, the benchmarks read during development become targets, so a gain on them can come from features specific to those benchmarks (their constraint set, answer format, or prompt style) instead of from the skill. Second, a score is an estimate with sampling, seed, and format error, so a difference of 1-3 points between two runs can lie inside the error; §6 works through two such cases from the Tülu 3 report.

This chapter fixes one measurement contract that every later chapter reuses: which axes are measured, which benchmarks are development and which are unseen, which narrowing signals are reported, and how uncertainty is stated. In this learner's resume path the chapter is read after ch-24 and before ch-04 and the SFT phase, so the SFT, preference, and RL chapters can report results in this format.

Stage abbreviations used before their own chapters: *SFT* (supervised fine-tuning on target responses with cross-entropy, ch-30), *DPO* (direct preference optimization on chosen and rejected response pairs, ch-39), *PPO* and *GRPO* (policy-gradient RL algorithms, ch-38 and ch-40), *RLVR* (RL with verifiable rewards, where a program checks the answer, ch-44), *CoT* (chain-of-thought, a written reasoning trace before the answer, ch-24), and *KL coefficient* (the weight of a penalty on divergence from a reference model, ch-43).

## §1 An operational definition of general capability

**Definition.** In this course, *general capability* is a model's measured performance on tasks and domains that (a) were not targeted by training data, reward, or checkpoint selection, and (b) were not used to make development decisions. It is reported as a *capability profile*: one score per capability axis, each with its uncertainty. A *target* is any benchmark, task family, or output format that the training data, reward, or selection rule was designed to improve.

**The problem.** A benchmark score measures skill on a fixed set of items. [[measure-of-intelligence]] (Chollet, arXiv 2019-11) argues that skill at a fixed task can be raised to any level with enough built-in prior knowledge or task-specific training data, so the score alone does not show how the system handles tasks it was not prepared for (Abstract; §II.1.1). The same paper separates two kinds of generalization (§I.3.2):

1. *System-centric generalization*: handling situations the learning system itself has not encountered, as measured by error on a held-out test split.
2. *Developer-aware generalization*: handling situations that neither the system nor its developer has encountered. This version counts the knowledge a developer injects, for example by tuning against a development set.

Chollet also grades generalization by degree: *local* (new samples from a known distribution of one task), *broad* (a wide category of tasks without further human intervention), and *extreme* (new tasks that share only abstract features with earlier ones) (§I.3.2). He notes that multi-task benchmarks do not measure developer-aware generalization when all tasks are known to developers in advance (§I.3.5).

**Mapping to LLM evaluation (Interpretation, this course).** The test split of a development benchmark measures system-centric, local generalization. An unseen benchmark for the same skill, formulated differently and not read during development, approximates developer-aware generalization within that skill. A task family absent from both training and development (new constraint types, new code repositories, a new language) approximates broad generalization.

**Zero-shot transfer as a generality measurement.** [[gpt-2-unsupervised-multitask]] (Radford et al., OpenAI report, 2019) describes supervised single-task systems as "narrow experts rather than competent generalists" (§1) and measures generality as zero-shot transfer: a language model trained on WebText is evaluated on tasks it received no task-specific training for. The largest model (1.5B parameters, called GPT-2 in the paper) reaches 55 F1 on the CoQA development set with greedy decoding, matching or exceeding 3 of 4 baseline systems without their 127,000+ training question-answer pairs (§3.5), and zero-shot performance improves log-linearly with model capacity (Abstract). The same paper measures how much test data appears in the training corpus with 8-gram overlap (§4); §7 returns to this kind of contamination check.

**Formula.** A capability profile of checkpoint θ is

`P(θ) = [ s_1(θ) ± u_1, …, s_K(θ) ± u_K ]`,  and a training step θ₀ → θ₁ is summarized by `Δ_k = s_k(θ₁) − s_k(θ₀)` for k = 1…K.

- K is the number of capability axes (§3).
- s_k(θ) is the score on axis k, averaged over the unseen benchmarks assigned to that axis.
- u_k is the standard error of s_k (§6).
- Δ_k is the change on axis k; a step *narrows* the model when Δ_k < 0 beyond its noise band on any untargeted axis k.

**Implication for a general-purpose model.** Every later chapter reports three things for its intervention: the change on the target metric, the profile change Δ_k on untargeted axes relative to the starting checkpoint, and the uncertainty of each number.

## §2 Capabilities are correlated but not one number

**Definition.** *Factor analysis* and *principal component analysis (PCA)* take a matrix of scores (models × tasks) and find a small number of latent dimensions that explain most of the variation between models.

**Evidence.**
- [[capability-structure-factors]] (Burnell et al., arXiv 2023-06) analyzes 29 LLMs on 27 HELM tasks. The mean correlation between tasks across models is r = 0.56 (§3.1). A 3-factor exploratory factor analysis explains 0.33, 0.31, and 0.17 of the variance (cumulative 0.82), interpreted as comprehension, language modeling, and reasoning (§3.2, Table 2). Fit statistics are poor for a sample of 29 models (CFI 0.70, TLI 0.61, RMSEA 0.26), and a Bayesian factor analysis gives a matching 3-factor solution (§3.2-3.3). Instruction tuning correlates −0.50 [−0.72, −0.17] with the language-modeling factor and +0.44 [0.11, 0.69] with the reasoning factor (Table 3). **Result (single study)**, cross-sectional across models.
- [[observational-scaling-laws]] (Ruan et al., arXiv 2024-05) applies PCA to 77 base models from 21 families on 8 benchmarks (MMLU, ARC-C, HellaSwag, Winogrande, GSM8K, HumanEval, TruthfulQA, XWinograd). The first component explains nearly 80% of the variance and the top 3 about 97%; the components read as general, reasoning, and programming capability (§3.2, Fig. 2). Within a model family, the first component is linear in log training FLOPs with R² > 0.9 (§3.3). The authors state that their assumptions do not account for benchmark contamination (§7).
- **Replicated**: in two studies with different models, benchmarks, and methods (rotated factor analysis; PCA), three latent dimensions explain most between-model variation (82% for 29 models on 27 tasks; about 97% for 77 models on 8 benchmarks). The studies differ on the first dimension. In the PCA, PC-1 is a weighted average of all 8 metrics and explains nearly 80% alone. In the rotated factor analysis, the three factors explain 33%, 31%, and 17%, each factor is defined by a different group of tasks (§3.2, Fig. 1), and the factors correlate with each other at 0.22-0.51 (Table 3).

**Mechanism by which an average hides narrowing (worked example).** Checkpoint A scores (60, 60, 60, 60, 60) on five axes; the average is 60.0. After targeted training, checkpoint B scores (72, 61, 61, 60, 51); the average is 305 / 5 = 61.0. The average rises by 1.0 while axis 5 falls by 9.0. An equal-weight average adds the per-axis changes, so a loss on one axis is offset by gains on others. When most between-model variance lies on one general dimension, as in the PCA above, a benchmark average tracks that dimension and not the separable abilities (Interpretation, this course).

**Conditions and limits.** A correlation across many models does not show that one training intervention moves all axes together within one model. The instruction-tuning result above has opposite signs for two factors. [[helm]] reports that the relation between accuracy and calibration depends on the scenario: on HellaSwag higher accuracy comes with worse calibration, on OpenBookQA with better calibration (§1.2, finding 3).

**Implication.** Later chapters report per-axis changes. An average is used only as a summary of overall level, not as evidence that no axis regressed.

## §3 The capability coverage map

[[helm]] (Liang et al., arXiv 2022-11; TMLR 2023) makes coverage itself a reported quantity. It measures 7 metrics (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency) on 16 core scenarios, covering 98 of 112 (scenario, metric) pairs (87.5%), for 30 models; before HELM these models had been evaluated on 17.9% of the core scenarios on average, and HELM raises this to 96.0% (Abstract, §1). HELM also lists what it does not cover, such as neglected English dialects (Abstract). The map below is the course's list of axes; each row names a measurement used in a cited source and a known error of that measurement.

| Axis | What a score must show | Example measurement in a cited source | Known measurement problem | Taught in |
|---|---|---|---|---|
| Knowledge | recall across domains, including long-tail entities | dev: MMLU, PopQA, TruthfulQA; unseen: MMLU-Pro, GPQA ([[tulu-3]] Table 3) | 8-gram contamination analysis could not produce an estimate for MMLU and MMLU-Pro ([[llama-3]] Table 15) | ch-12a, ch-48 |
| Reasoning and math | multi-step derivation with a checkable answer | dev: BBH, DROP, GSM8K, MATH; unseen: AGIEval English, DeepMind Mathematics ([[tulu-3]] Table 3) | moving MATH answer extraction from the Minerva format to a "flex" scheme changes scores by up to 10 points ([[tulu-3]] §7.2) | ch-24, ch-44 |
| Code | executable, test-passing solutions | dev: HumanEval, HumanEval+ pass@10 (pass@k is defined in §7); unseen: BigCodeBench-Hard, 148 of 1,140 tasks ([[tulu-3]] §7.3) | ten OpenAI and Anthropic models name a file changed by the gold patch in 60-76% of SWE-Bench Verified instances from the repository name and issue text alone, and in under 53% of tasks from repositories outside SWE-Bench ([[swe-bench-illusion]] §4.1) | ch-27, ch-44, ch-48 |
| Instruction following | precise constraints and open-ended requests | dev: IFEval, AlpacaEval 2; unseen: IFEval-OOD (52 constraints), HREF ([[tulu-3]] §7.3) | many models score above 80% on IFEval's 25 constraint templates, while on 58 new constraints GPT-4.1 and Claude 3.7 Sonnet score below 50% ([[ifbench]] §1, Fig. 1) | ch-29e |
| Multi-turn | retention and consistency across turns | MultiChallenge, 273 conversations ([[multichallenge]] Table 1) | LLM-judge agreement with humans 37.33% without per-item rubrics vs 93.95% with them (Table 4) | ch-25, ch-49 |
| Long context | use of information spread over long inputs | HELMET, 7 categories at 8K-128K ([[helmet]] §2.1) | across 35 instruction-tuned models at 128K, the Spearman correlation with ∞Bench QA is 0.63 for needle-in-a-haystack and 0.88 for HotpotQA RAG (Fig. 4) | ch-32c |
| Multilingual | the same skills in other languages | no multilingual task in [[tulu-3]] Table 3 or [[olmo-2]] Table 6; [[paloma]] covers English and code only (§6) | coverage gap in both suites | ch-13a |
| Tool use and agentic tasks | correct calls, refusing irrelevant calls, multi-step completion | BFCL categories and modes ([[bfcl]] Table 1); validity checks ([[agentic-benchmark-checklist]]) | GPT-4-turbo-2024-04-09 irrelevance detection 83.8 in native function-calling mode vs 35.6 in prompting mode ([[bfcl]] Table 1); an agent that returns nothing passes 38% of τ-bench Airline tasks ([[agentic-benchmark-checklist]] §5.2) | ch-26, ch-27, ch-45b, ch-51a |
| Factuality and calibration | stated confidence matches correctness | expected calibration error with 10 bins, selective-classification accuracy ([[helm]] §4.4) | accuracy and calibration move together on some scenarios and oppositely on others ([[helm]] §1.2, finding 3) | ch-38, ch-43a |
| Safety and over-refusal | refuse unsafe requests and comply with safe ones | Tülu 3 Safety, macro average of 6 benchmarks ([[tulu-3]] §7.2.1); XSTest, 250 safe and 200 unsafe prompts ([[xstest]]) | one average mixes refusal of unsafe and compliance with safe prompts; Tülu 3 has no unseen safety evaluation (§2.2) | ch-52 |

## §4 Development suite versus unseen suite

**Definitions.** A *development suite* is the set of benchmarks whose scores are read during development and used to choose data, algorithms, hyperparameters, and checkpoints. An *unseen suite* (held-out suite) is a set of benchmarks for the same capability axes whose scores are not read until development decisions are final.

**The problem.** A gain on a development benchmark is the sum of a gain in the skill and a gain specific to that benchmark. Only a benchmark that did not influence the decisions estimates the first term.

**Mechanism: the Tülu 3 evaluation regime** ([[tulu-3]] §2.2, §3.2, §7; extract in [[tulu-3-eval-regime]]):
1. Name the core skills: knowledge recall, reasoning, mathematics, coding, instruction following, general chat, and safety (§2.1).
2. Fix development benchmarks and their settings per skill: shots, chain-of-thought, chat template, metric (Table 24).
3. Choose unseen benchmarks per skill through an independent design process aimed at realistic use: no few-shot examples presented as dialog, clear instructions that request concise reasoning and a stated answer format, and lenient answer extraction (§7.3).
4. Decontaminate training prompts against both suites. A test instance overlaps a training instance when more than 50% of its tokens have 8-gram matches with that instance. A training set counts as contaminated when it overlaps more than 2% of the instances of any evaluation. Sets contaminated with unseen evaluations are removed; for development evaluations, the whole set is removed if doing so "did not significantly impact the performance", and otherwise only the matching instances (§3.2).
5. Do not read unseen scores during development (§2.2). Tülu 3 has no unseen safety evaluation (§2.2).
6. After development, evaluate the checkpoint of each decision on the unseen suite (§7.4.1).

**Formula.** For a decision d (for example, including one dataset) and capability axis k:

`Δ_dev(d, k) = s_dev,k(with d) − s_dev,k(without d)`,  `Δ_uns(d, k) = s_uns,k(with d) − s_uns,k(without d)`

- s_dev,k and s_uns,k are the scores on the development and unseen benchmark for axis k.
- A decision *transfers* on axis k when Δ_uns has the same sign as Δ_dev and |Δ_uns| exceeds its noise band (§6).
- The comparison uses changes, not levels, because the two suites have different score levels: the Tülu 3 8B SFT checkpoint averages 64.9 on development and 29.9 on unseen benchmarks (Table 31), and the report calls the unseen evaluations "harder" (§7.4.1).

**Worked example: Tülu 3 8B SFT data ablations** (Table 32; each value is final SFT minus the model trained without that source).

| Source removed | Axis | Development: final − ablated | Unseen: final − ablated | Reading |
|---|---|---|---|---|
| Math data | Math | MATH 31.5 − 23.5 = +8.0 | DeepMind Mathematics 32.3 − 23.3 = +9.0 | transfers |
| Persona data (math, code, IF) | Instruction following | IFEval 72.8 − 53.6 = +19.2 | IFEval-OOD 17.6 − 18.0 = −0.4 | does not transfer |
| WildChat | Instruction following | IFEval 72.8 − 70.1 = +2.7 | IFEval-OOD 17.6 − 20.8 = −3.2 | opposite sign |
| WildChat | Code | HumanEval 86.2 − 85.3 = +0.9 | BigCodeBench-Hard 11.5 − 7.4 = +4.1 | same sign; unseen change not shown to exceed sampling error (§6.1) |

[figures/dev-unseen-and-noise.html](figures/dev-unseen-and-noise.html) plots these development and unseen changes for all four data sources and five skills, and lets the reader test whether a difference exceeds its standard error.

**Evidence and the authors' reading.**
- The authors conclude that the SFT data choices generalize on average but overfit the development evaluations in precise instruction following, and to some extent in knowledge recall and reasoning (§7.4.1). **Result (single study).**
- Scaling DPO data raised MATH more than DeepMind Mathematics. The authors attribute this to formatting: trained models write LaTeX even where DeepMind Mathematics does not require it, which interfered with reasoning and broke answer extraction (§7.4.1, Fig. 24). **Interpretation** (authors).
- For the four models with both scores printed, IFEval is 80.6-88.0 and IFEval-OOD is 24.3-34.5 (Tülu 3 8B 82.4 vs 24.3; Llama 3.1 8B Instruct 80.6 vs 26.1; Tülu 3 70B 83.2 vs 27.8; Llama 3.1 70B Instruct 88.0 vs 34.5; Tables 5, 6, 33). The authors hypothesize that models doing well on IFEval overfit its 25 constraints (§7.4.2). [[ifbench]] finds a similar gap on 58 new constraints (§1, Fig. 1). **Replicated** across two constraint sets; both studies come from overlapping Ai2 author groups, so they are not fully independent.
- The prose and the tables of the report do not always agree. §7.4.1 states that for reasoning and coding the stages after SFT still improve the harder unseen evaluations. Table 31 supports this for 8B reasoning (AGIEval 56.2 → 59.3) and 70B coding (BigCodeBench 12.2 → 21.6), but 8B coding falls from 11.5 (SFT) to 9.5 (DPO) to 7.4 (final). The same report also lists the 8B SFT development average as 64.9 in Table 31 and 64.1 in Table 32.

**Base-model version.** [[olmo-2]] (arXiv 2501.00656v3) declares development benchmarks (MMLU, ARC Challenge, HellaSwag, WinoGrande, Natural Questions, DROP) and held-out benchmarks (AGIEval, GSM8K, MMLU Pro, TriviaQA) for base models (§2.5, Table 6). GSM8K is only partly held out: 200 of its 1,319 examples served as the development set GSM* for mid-training data, and the other 1,119 are reported only at the end (§2.5, footnote 6). For OLMo 2 7B, mid-training moves MMLU 59.8 → 63.7 (development) and MMLU Pro 27.4 → 31.0, AGIEval 44.6 → 50.4, GSM8K 24.1 → 67.5 (held-out) (§4.2, Table 9). The held-out GSM8K items come from the same distribution as GSM*, so they test local generalization within GSM8K rather than a different formulation (Interpretation, using Chollet's degrees).

**Conditions and limits.**
1. Neither report can rule out that compared models trained on the unseen benchmarks ([[tulu-3]] §7, Table 33 caption; [[olmo-2]] §2.5).
2. The unseen suite measures transfer within the named skills. Axes absent from the skill list, such as long context, multilingual use, and tool use, are not measured at all (Table 3).
3. Each ablation in Table 32 is one training run; seed variation is not reported for these ablations (see Table 14 in §6.2).
4. BigCodeBench-Hard scores of 7-12% at 8B are near the floor, where the 4.1-point absolute change from 11.5 to 7.4 is a 36% relative change.
5. Once an unseen benchmark is published and reused for later decisions, it is a development benchmark for that later work (Interpretation, this course).

**Implication.** Every recipe decision in later chapters is reported as a (Δ_dev, Δ_uns) pair per axis. A decision with a Δ_dev beyond noise and a Δ_uns inside noise is recorded as a development-benchmark gain, not a capability gain.

## §5 Base-model measurement: per-domain loss and task formulation

### §5.1 Per-domain held-out perplexity

**Definition.** Perplexity is `exp(−ℓ / T(N))`, where ℓ = Σ_{t∈N} Σ_i ln p(t_i | t_<i) is the summed log-likelihood over evaluation documents N, t_i is the i-th token of document t, and T(N) is the number of tokens ([[paloma]] §3). The *macro average* over a domain set D is |D|⁻¹ Σ_{d∈D} perplexity(d); ordinary perplexity over pooled text is a *micro average* weighted by token counts (§4.1). *Bits per byte* (BPB) = −ℓ / (B ln 2), with B the UTF-8 byte count, compares models with different vocabularies (App. B).

**The problem.** One held-out loss mixes domains, so an improvement on the largest domain can hide a regression on smaller ones.

**Worked example (hand numbers).** A held-out set has 900K web tokens and 100K code tokens. Checkpoint A has perplexity 10 on web and 40 on code. Micro: exp(0.9·ln 10 + 0.1·ln 40) = exp(2.441) = 11.49; macro: (10 + 40) / 2 = 25.0. Checkpoint B has 9 on web and 60 on code. Micro: exp(0.9·ln 9 + 0.1·ln 60) = exp(2.387) = 10.88, an improvement; macro: (9 + 60) / 2 = 34.5, a regression.

**Evidence.** Paloma measures 546 domains from 16 sources and trains six 1B baselines that differ only in corpus. The C4-only baseline reaches perplexity 391,171 on the RedPajama arXiv domain, and the C4 and mC4-en baselines get worse between the ~20B and ~150B-token checkpoints on 65 and 43 domains ([[paloma]] §4.1, App. D.1.1). Per-source perplexity rankings of the six baselines correlate with downstream rankings with opposite signs for different sources: Spearman ρ with HellaSwag is −0.77 for c4-en and 0.94 for RedPajama (App. A, Table 3). HELM finds BPB on The Pile a poor predictor of downstream accuracy across model families, noting that some of those models were trained on The Pile ([[helm]] §1.2, finding 24). A re-analysis of the 46 downstream tasks of one earlier study (Gadre et al., 2025) finds that 18 improve smoothly and predictably with validation loss ([[scaling-laws-unreliable-downstream]] abstract, §4, Fig. 1). **Replicated**: no single held-out loss predicts all downstream tasks.

**Limits.** Paloma covers English and code only, perplexity on fringe sources tracks document length, and code is not decontaminated ([[paloma]] §3, §6).

### §5.2 Cloze versus multiple-choice formulation

**Definitions** ([[olmes]] §2.1). In the *multiple-choice formulation* (MCF), the prompt lists answer options with letter labels and the model is scored on the label token. In the *cloze formulation* (CF), each answer text is appended to the question separately and ranked by its probability. For CF, OLMES lists four scores for answer a_i given question q (§3.3):
- none: ln P(a_i | q)
- token: ln P(a_i | q) / num_tokens(a_i)
- character: ln P(a_i | q) / num_characters(a_i)
- pmi: ln [ P(a_i | q) / P(a_i | u) ], with u = "Answer:" as an unconditional prompt

**Worked example (hand numbers).** Two choices: " iron" (5 characters with the leading space, which OLMES counts, §3.5) with ln P = −2.0, and " molten nickel alloy" (20 characters) with ln P = −6.0. With "none", −2.0 > −6.0 selects " iron". With "character", −2.0/5 = −0.40 and −6.0/20 = −0.30, so " molten nickel alloy" is selected. The normalization choice alone flips the prediction.

**Evidence.**
- ARC-Challenge for Llama3-8B is 60.2 on the Hugging Face Open LLM Leaderboard (25-shot CF) and 78.6 in the Llama 3 model card (25-shot MCF); OLMES reports 79.3 ([[olmes]] Table 1, §3.4).
- On ARC-Challenge, the weakest 8 of 15 base models score near chance with MCF but above chance with CF; Llama3-70B scores 93.7 with MCF and 69.0 with CF (§3.4, Fig. 2, Table 6).
- During OLMo-7B-0424 pretraining, MMLU in MCF stays near chance until about 400B tokens and then gives a stronger signal than CF (Fig. 1).
- Ten Llama-2-7B-architecture models trained for 210B tokens with different seeds score 25.86 on standard MMLU (chance 25) with monotonicity 0.09 over training, and 37.47 on MMLU-Cloze with monotonicity 0.95 ([[benchmark-variance-quantified]] Table 1). Llama 3 70B scores 78.7 standard and 60.6 cloze, and the two formats correlate at Pearson 0.92 over 70 models (§3.3).
- HELM: OPT (175B) scores 79.1% on HellaSwag when each answer is scored in a separate 0-shot prompt and 30.2% when choices are presented jointly in a 5-shot multiple-choice prompt ([[helm]] §1.2, finding 23).
- **Replicated** ([[olmes]], [[benchmark-variance-quantified]]): CF gives a non-random signal for weak or early models, and MCF gives the more accurate measurement for strong models.

**Mechanism.** MCF requires the model to map its answer to a label token, which is a format skill acquired at some point in pretraining (for OLMo-7B-0424, after about 400B tokens, [[olmes]] Fig. 1). Before that point, an MCF score measures whether the format has been acquired, so two small-scale ablations compared on MCF can be ordered by noise around chance (Interpretation; OLMES states the format-skill reading in §3.4).

**Implication.** OLMES evaluates both formulations and reports the better one per model (§3.4, §4). For pretraining and mid-training ablations in this course, report CF or BPB until MCF is above chance.

## §6 Measurement error

### §6.1 Sampling error over benchmark items

**Formula** ([[adding-error-bars-evals]], Miller, arXiv 2024-11, §2.1, §4). Treat benchmark items as a sample from a larger population of possible items. With per-item scores s_i, mean s̄, and n items:

`SE = sqrt( Σ_i (s_i − s̄)² / (n − 1) / n )`,  for 0/1 scores `SE = sqrt( s̄(1 − s̄) / n )`,  `CI_95 = s̄ ± 1.96·SE`

For models A and B on the same items, with per-item correlation ρ between their scores:

`SE_unpaired = sqrt(SE_A² + SE_B²)`,  `SE_paired = sqrt(SE_A² + SE_B² − 2·SE_A·SE_B·ρ)`,  `z = (s̄_A − s̄_B) / SE`

The 0/1 formula tends to give a standard error that is too wide when per-item scores are fractional, such as F1 (§2.1). The reason: any per-item scores in [0, 1] with mean s̄ have variance at most s̄(1 − s̄), so the 0/1 formula is an upper bound. pass@k estimated from several samples per item is also fractional.

**Worked example 1.** MMLU has 14,042 items ([[olmes]] Table 2). At accuracy 0.65, SE = sqrt(0.65 · 0.35 / 14,042) = 0.0040, or 0.40 points. HumanEval has 164 items ([[benchmark-variance-quantified]] Table 1). Tülu 3 8B scores 86.2 after SFT and 83.9 after DPO on HumanEval pass@10 ([[tulu-3]] Table 31). With the 0/1 formula, SE_A = sqrt(0.862 · 0.138 / 164) = 2.69 points and SE_B = 2.87 points; SE_unpaired = 3.94, so z = 2.3 / 3.94 = 0.58. With an assumed per-item correlation ρ = 0.6 (the report gives no per-item scores, so ρ is not reported), SE_paired = 2.49, z = 0.92, and the 95% interval for the difference is (−2.6, 7.2). Because pass@10 per-item scores are fractional, these SEs are upper bounds, and the exact SE is not reported. On these numbers the 2.3-point drop is not shown to be a measured difference.

**Worked example 2.** BigCodeBench-Hard has 148 tasks ([[tulu-3]] §7.3). The WildChat effect in §4 is 11.5 − 7.4 = 4.1 points (pass@10). With the 0/1 formula, SE = 2.62 and 2.15 points; SE_unpaired = 3.39; z = 1.21; interval (−2.5, 10.7). The effect is not shown to exceed sampling error; the same upper-bound caveat applies.

**Further evidence.** Items drawn in groups (several questions per passage, one question in many languages) need clustered standard errors. On Anthropic models, clustered SE is 3.05 times the naive SE on DROP and 1.88 times on MGSM ([[adding-error-bars-evals]] §2.2, Table 4). Miller advises against lowering sampling temperature to reduce variance, and instead recommends averaging K sampled answers per item or scoring next-token probabilities (§3.1-3.3).

### §6.2 Seed variance and checkpoint noise

**Definitions.** *Seed variance* is the standard deviation of a benchmark score across training runs that differ only in random seed. *Checkpoint noise* is the relative standard deviation of the score over the final n checkpoints of one run ([[signal-and-noise-eval]] §3.1).

**Evidence.**
- For ten Llama-2-7B-architecture models trained to 210B tokens with different initialization seeds, the seed standard deviation, averaged over 21 checkpoints from 10B to 210B tokens, is 0.80 on ARC-C, 1.11 on HumanEval, and 0.21 on HellaSwag, while the bootstrapped 95% confidence-interval values are 2.74, 3.98, and 0.93 ([[benchmark-variance-quantified]] §2.1, §3.1, Table 1). The analytic form given in §3.1, 1.96·sqrt(S(1 − S)/N), is a half-width, and for MMLU it reproduces the printed 0.72 (course check: 1.96·sqrt(0.2586 · 0.7414 / 14,042) = 0.72). The two quantities answer different questions: the interval covers re-sampling items; the seed deviation covers re-training on the same items.
- Tülu 3 SFT averages across seeds are 59.9, 60.1, 59.8, 59.8, 59.8 at 8B (sample standard deviation 0.13) and 71.8, 70.0, 72.6 at 70B (sample standard deviation 1.33) ([[tulu-3]] §4.3.1, Table 14). At 70B, a recipe change that raises this average by 1.0 point is within one seed standard deviation.
- The final 30 checkpoints of 1B models on ARC Challenge span 1.7 accuracy points ([[signal-and-noise-eval]] §3.1).

**Formula: signal-to-noise ratio** ([[signal-and-noise-eval]] §3, Eq. 2).

`signal = max_{j,k} |m_j − m_k| / m̄`,  `noise = sqrt( Σ_i (m_i − m̄)² / (n − 1) ) / m̄`,  `SNR = signal / noise`

- In *signal*, m_j are final scores of a population of models trained with similar compute, and m̄ is their mean.
- In *noise*, m_i are scores of the final n checkpoints of one run, and m̄ is their mean.

**Worked example (hand numbers).** Three models score 40, 44, 46: signal = 6 / 43.33 = 0.138. One run's last five checkpoints score 45.0, 46.0, 45.5, 46.5, 47.0: standard deviation 0.79, mean 46.0, noise = 0.0172. SNR = 0.138 / 0.0172 = 8.1.

**Evidence.** Across the OLMES tasks, small-scale SNR correlates with the accuracy of predicting 1B model rankings from 60M-750M models (R = 0.791), while signal or noise alone does not; scoring with BPB raises the 30-task average SNR from 10.0 to 31.5 and decision accuracy from 77.0% to 83.7% ([[signal-and-noise-eval]] §4.1, Fig. 2, Fig. 6). With bits-per-byte scores, averaging the final checkpoints raises 30-task decision accuracy from 68.9% to 71.3% (§5.2, Table 1). **Result (single study).** **Limit:** the study measures training-time noise only, not noise from evaluation configuration (§6).

### §6.3 Prompt-format sensitivity

**Definition** ([[prompt-format-sensitivity-formatspread]], Sclar et al., arXiv 2023-10, §3.2). Given semantically equivalent formats p_1…p_n (separators, casing, spacing, enumeration style), a dataset D, and metric m, the *spread* is `max_i m(p_i, D) − min_i m(p_i, D)`.

**Mechanism of FormatSpread** (§3.1-3.2):
1. A grammar generates formats equivalent to a task's original format.
2. Each sampled format is an arm of a bandit; pulling an arm evaluates a mini-batch of B unevaluated items.
3. With a total budget E, Thompson sampling with Beta priors searches for the best format with E/2 evaluations and for the worst with E/2.
4. The reported interval [min, max] is a lower bound on the true spread, because only sampled formats are evaluated (§4.2).

**Worked example (hand numbers).** Model M scores 0.62, 0.55, 0.70 on formats p_1, p_2, p_3 (spread 0.15); model M′ scores 0.58, 0.61, 0.60 (spread 0.03). Under p_1, M leads by 4 points; under p_2, M′ leads by 6 points.

**Evidence.** Over 53 tasks with 10 sampled formats per task, the median spread is 7.5 accuracy points; LLaMA-2-13B reaches up to 76 points; sensitivity remains with larger models, more few-shot examples, and instruction tuning (Abstract, §4.2). For the pair LLaMA-2-13B and LLaMA-2-70B, when one model leads by at least 2 points under one format, the other leads by at least 2 points under a different format with probability 0.141 (§4.2, Fig. 4). GPT-3.5 shows a median spread of 6.4 points over 320 formats and 53 tasks (§1). HELM reports sensitivity to prompt format and in-context examples for all models, scenarios, and metrics ([[helm]] §1.2, finding 22). **Replicated.**

### §6.4 A decision rule for later chapters

**Rule (Interpretation, this course).** A difference between two checkpoints counts as a measured change when (1) it exceeds 1.96 times its paired standard error, (2) it exceeds the seed or checkpoint standard deviation measured for that benchmark at that scale, and (3) on format-sensitive tasks, its sign holds under at least two plausible formats. Each condition is supported by a different source above; the combination has not been validated as a single test.

## §7 Narrowing signals every later chapter reports

**1. Forgetting relative to the starting checkpoint.** *Definition:* a decrease beyond noise on an untargeted axis after a training step. *Measurement:* the same suite, settings, and items before and after the step, with paired standard errors.
- IF-RLVR (GRPO with constraint-verification rewards) from Tülu-3-8B-DPO raises IFEval from 81.1 to 92.2 and IFBench from 25.2 to 44.6 ([[ifbench]] Table 6; the §1 prose prints 82.4 → 92.2 and 28.9 → 45.9 for this model), while AlpacaEval 2 falls from 33.5 to 21.3 and MMLU from 68.7 to 66.4 (Table 3). **Result (single study).**
- Qwen3-14B-Base fine-tuned on math only: SFT on rejection-sampled Qwen3-32B thinking-mode traces raises the math average from 27.7 to 49.8 while IFEval falls from 69.2 to 42.3 and HaluEval from 35.7 to 2.3; GRPO on the same math dataset reaches a math average of 53.8 with IFEval 70.0 and HaluEval 40.7 ([[transferability-of-llm-reasoning]] arXiv 2507.00432v2 §2.2, Table 1). **Result (single study).** A fall to 2.3 can come from forgetting or from answers the scorer cannot parse; the table alone does not separate the two, and inspecting the outputs does (course note).
- One epoch of fine-tuning GPT-3.5 Turbo on benign Alpaca data raises its harmfulness rate from 5.5% to 31.8% on a 330-prompt benchmark scored by a GPT-4 judge ([[finetuning-compromises-safety]] Table 3). **Result (single study).**

**2. pass@1 versus pass@k.** *Definition:* pass@k is the fraction of problems with at least one verified-correct answer among k samples. With n ≥ k samples per problem, c of them correct, the unbiased estimator is ([[rlvr-beyond-base-model]] arXiv 2504.13837v5 App. A.2, Eq. 2):

`pass@k = E_problems [ 1 − C(n − c, k) / C(n, k) ]`

- C(a, b) is the binomial coefficient; the expectation is the average over problems.

*Worked example (hand numbers).* n = 4 samples, c = 1 correct, k = 2: 1 − C(3, 2)/C(4, 2) = 1 − 3/6 = 0.5; with k = 1 the value is c/n = 0.25. Two models on four problems: model R solves problems 1-2 with per-sample probability 0.9 and problems 3-4 with 0; model S solves problems 1-2 with 0.3 and problems 3-4 with 0.1. pass@1 is 0.45 for R and 0.20 for S. At k = 16 with independent samples, R stays at 0.50, while S reaches [2·(1 − 0.7¹⁶) + 2·(1 − 0.9¹⁶)] / 4 = 0.91.

*Evidence.* On AIME24 with k = 1024, 63.3% of problems are solved by both Qwen2.5-7B-Base and its RLVR model, 13.3% only by the base model, and 0.0% only by the RLVR model; on MATH500 with k = 128 the shares are 92.4%, 3.6%, and 1.0% (Table 2, Table 5, App. C.7). The authors interpret RLVR as raising the probability of solutions the base model could already sample (§1). **Interpretation** (authors); ch-38a teaches this result together with the ProRL counter-claim.

**3. Output diversity.** *Definition:* the spread of distinct solutions or formats a model produces for one input, measured by pass@k with k in the tens to thousands (64 in [[echo-chamber-rl-post-training]], up to 1024 in [[rlvr-beyond-base-model]]) and by the share of distinct solution formats among samples. In a 150M model pretrained on math documents plus three instruction datasets with distinct solution formats, and then fine-tuned with PPO on GSM8K questions, generations converge on one pretraining format within the first epoch and pass@64 declines late in training; a KL coefficient of 0.01 instead of 0.001 keeps a second format and a stable pass@64 at comparable pass@1 ([[echo-chamber-rl-post-training]] §3.1, Fig. 2-3). Full treatment in ch-43.

**4. Over-refusal.** *Definition:* refusal of safe prompts. Llama-2-70b-chat with its original system prompt fully refuses 38% and partially refuses 21.6% of 250 safe XSTest prompts; adding a guardrail system prompt to Mistral-7B-Instruct-v0.1 raises full refusal of unsafe prompts from 23.5% to 87.5% and of safe prompts from 0.8% to 9.6% ([[xstest]] Tables 1-2). Tülu 3 Safety is a macro average over six benchmarks, and its XSTest and WildJailbreakTest components also score compliance with benign prompts ([[tulu-3]] §7.2.1). When such an average is reported, report the benign-compliance components next to it, because the average can rise while refusal of safe prompts also rises, as the Mistral system-prompt result shows for the two directions.

**5. Contamination-inflated scores.** *Definition:* score gains caused by evaluation items, or near copies, in training data.
- GPT-2: test sets of common language-modeling benchmarks share 1-6% of their 8-grams with WebText (average 3.2%). Excluding LAMBADA examples with any overlap changes accuracy from 63.2% to 62.9%; about 15% of CoQA news-domain documents are in WebText and add about 0.5-1.0 F1 overall ([[gpt-2-unsupervised-multitask]] §4, Table 6).
- Llama 3: the 8-gram analysis flags 85% of HellaSwag with an estimated +14.8 points at 8B; annealing on GSM8K and MATH training sets raised pre-trained 8B validation scores by 24.0% and 6.4%, so these sets were excluded from annealing data ([[llama-3]] §3.1.3, §5.1.4, Table 15).
- Public chat logs overlap evaluations: LMSys Chat 1M overlaps 46.5% of AlpacaEval and 90.3% of Do-Anything-Now instances under the Tülu 3 rule ([[tulu-3]] App. B.2, Table 37).
- Code agents: see the SWE-Bench Verified row of the §3 map ([[swe-bench-illusion]]).

**6. Format and constraint overfitting.** IFEval versus IFEval-OOD and IFBench (§4), and the MATH LaTeX effect on DeepMind Mathematics (§4), are read by their authors as gains in a benchmark's constraint set or format rather than in the skill (Interpretation, authors' hypotheses; §4). Detecting this kind of gain requires an unseen benchmark with a different format for the same skill.

## §8 Source types and reliability labels

| Source type | Example in this chapter | What it can support | What it cannot support alone |
|---|---|---|---|
| Paper (peer-reviewed or preprint) | [[olmes]], [[helm]], [[benchmark-variance-quantified]] | a result in its stated setting | transfer of the result to other model sizes or data |
| Official technical report | [[tulu-3]], [[olmo-2]], [[llama-3]] | what the organization ran and measured | independent confirmation of its own claims |
| Official blog or model card | Llama 3 model card, cited in [[olmes]] Table 1 | released-checkpoint scores and settings | settings not printed there |
| Released config or code | OLMES task configurations (github.com/allenai/olmes) | the exact value used for a released run | the reason the value was chosen |
| Practitioner evidence | independent reports with experiments and numbers | a result with its setup | claims beyond the tested setup |
| Anecdotal report | observations without controlled comparison | a question to test | any quantitative claim |

Claim-status labels used in every chapter: **Result (single study)**, **Replicated** (two or more independent sources agree), **Interpretation** (an explanation of a result, by the authors or by this course), **Open question** (no conclusive evidence). When sources disagree on a recipe value, a pinned released config is preferred over paper tables, paper tables over paper prose, and paper prose over blogs.

## Recipe

This chapter's recipe rows are evaluation settings (stage `eval-gate`): the values a later chapter must fix and report when it claims a change in general capability.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Tülu 3 checkpoints (SFT, DPO, final) | 8B, 70B | eval-gate | development suite settings | MMLU 0-shot CoT EM, macro over subjects; PopQA 15-shot EM; TruthfulQA 6-shot MC2; BBH 3-shot CoT EM; DROP 3-shot F1; GSM8K 8-shot CoT EM; MATH 4-shot CoT flex EM; HumanEval/HumanEval+ 0-shot pass@10 at temperature 0.8; IFEval prompt-level loose; AlpacaEval 2 length-controlled win rate, greedy, up to 8,192 tokens | arXiv:2411.15124v5 Table 24, §7.2 | verified 2026-09-15 | MMLU "summarize" CoT prompt improved over 5-shot multiple choice across tested models (§7.2, Table 46); flex extraction up to 10 points over Minerva format (§7.2) |
| Tülu 3 405B checkpoints | 405B | eval-gate | shots in table labels | MMLU "5 shot, CoT"; PopQA "3 shot"; BBH "0 shot, CoT" | arXiv:2411.15124v5 Table 4 | conflict | Table 24 and the 8B/70B tables (Tables 5-6) print 0-shot MMLU, 15-shot PopQA, 3-shot BBH; the report does not state whether 405B used different settings |
| Tülu 3 checkpoints | 8B, 70B | eval-gate | unseen suite settings | MMLU-Pro, GPQA, AGIEval English, DeepMind Mathematics: 0-shot CoT; BigCodeBench-Hard 148 of 1,140 tasks, pass@10; IFEval-OOD 52 constraints, prompt-level loose; HREF win rate, judge Llama 3.1 70B Instruct, baseline Llama 3.1 405B Instruct | §7.3, Table 24 | verified 2026-09-15 | HREF composite judge agreement with humans 69.4% vs 67% between humans (§7.3.2) |
| Tülu 3 training data | n/a | eval-gate | decontamination rule | 8-gram matches on prompts; instance overlap if >50% of test tokens match one training instance; set contaminated if >2% of an evaluation's instances overlap | §3.2 | verified 2026-09-15 | full-string and embedding matching were tried; n-gram matching judged most useful (§3.2); no ablation of thresholds reported |
| Tülu 3 SFT | 8B; 70B | SFT | seeds tried; checkpoint selection rule | 5 seeds (8B), 3 seeds (70B); best single run used, not soup | §4.3.1, Table 14 | verified 2026-09-15 | best seed comparable to best soup (Table 14) |
| OLMES standard (base models) | 1B-70B tested | eval-gate | instances; shots; formulation; normalization; length | test split if labeled, else validation; 1,000 instances sampled if >1,500 (Random(1234)); curated 5-shot; both MCF and CF, best reported; per-task normalization (pmi: ARC-C, CSQA, OBQA; character: ARC-E, HellaSwag, PIQA, SIQA, MMLU; none: BoolQ, WinoGrande); MMLU macro average; inputs ≤2,048 tokens | arXiv:2406.08446v2 §3.1-3.5, Table 2 | verified 2026-09-15 | normalization: Table 3 "diff oracle" (difference from the best normalization, per task, over 15 models) is 0.0-1.1 points; formulation: MCF near chance for the weakest 8 of 15 models on ARC-Challenge (§3.4, Fig. 2); other settings: no ablation reported |
| OLMo 2 base models | 7B, 13B, 32B | eval-gate | development vs held-out split | dev: MMLU, ARC Challenge, HellaSwag, WinoGrande, NQ, DROP; held-out: AGIEval, GSM8K (1,119 of 1,319), MMLU Pro, TriviaQA | arXiv:2501.00656v3 §2.5, Table 6, footnote 6 | verified 2026-09-15 | no ablation reported |
| DataDecide small models (signal-and-noise decision-accuracy setup) | 60M-750M | eval-gate | noise window | final 5 checkpoints, averaged over the small models | arXiv:2508.13144v1 §4.1, App. A.3.2 ([[signal-and-noise-eval]]) | verified 2026-09-14 | on OLMo 2 7B checkpoints, n = 5 gives a sample standard deviation within ±1σ of the true noise for almost all benchmarks (App. A.3.2, Table 2) |
| FormatSpread analysis | LLaMA-2 7B-70B, Falcon-7B(-Instruct), GPT-3.5 | eval-gate | formats per task | 10 sampled formats (§4.2); 320 formats for GPT-3.5 (§1) | arXiv:2310.11324v2 §1, §4.2 | verified 2026-09-15 | no ablation reported; the authors state that 10 sampled formats give a lower bound on the true spread (§4.2) |

**Starting point for a small general-purpose run.** Every value below comes from a verified row above, with the conditions under which the source used it. For each capability axis, keep the Tülu 3 pairing of development and unseen benchmarks and do not read unseen scores until decisions are final (Tülu 3 at 8B and 70B post-training of Llama 3.1). Decontaminate training prompts with the 8-gram rule (>50% token overlap per instance, >2% instance overlap per set) against both suites. For base-model checkpoints, use OLMES settings: 1,000 sampled instances where a benchmark has more than 1,500, curated 5-shot prompts, both formulations with the better one reported (OLMES tested base models from 1B to 70B). When comparing SFT recipes, train more than one seed; Tülu 3 used 5 seeds at 8B and 3 at 70B. For noise estimates on pretraining runs of 60M-750M parameters, use the final 5 checkpoints (DataDecide models, 60M-750M).

## Generalization lens

**(a) What increases measured breadth.**
- Declaring an unseen suite per axis and reporting Δ_uns next to Δ_dev ([[tulu-3]] §7.4; [[olmo-2]] §2.5). Tülu 3 found its math data transferred (+8.0 dev, +9.0 unseen) and its Persona data did not transfer on instruction following (IFEval +19.2 dev, IFEval-OOD −0.4 unseen) (Table 32).
- Covering more axes and metrics: HELM raised per-model core-scenario coverage from 17.9% to 96.0% ([[helm]] Abstract).
- Using measurements with higher signal-to-noise: BPB raised the 30-task average SNR from 10.0 to 31.5 for DataDecide models ([[signal-and-noise-eval]] Fig. 6); for 7B seed models trained to 210B tokens, MMLU monotonicity over checkpoints is 0.95 with the cloze formulation and 0.09 with the standard format ([[benchmark-variance-quantified]] Table 1).

**(b) What causes narrowing or false breadth claims.**
- Tuning against a fixed constraint set or format: IFEval versus IFEval-OOD and IFBench ([[tulu-3]] Table 33; [[ifbench]] Fig. 1).
- Training steps that improve a target and lower untargeted axes: IF-RLVR lowered AlpacaEval 2 from 33.5 to 21.3 ([[ifbench]] Table 3); math-only SFT lowered IFEval from 69.2 to 42.3 ([[transferability-of-llm-reasoning]] Table 1).
- RLVR that raises pass@1 while leaving 13.3% of AIME24 problems solvable only by the base model at k = 1024 ([[rlvr-beyond-base-model]] Table 2).
- Contamination: +14.8 estimated points on HellaSwag at 8B ([[llama-3]] Table 15).
- Reading averages: a +1.0 average can hide a −9.0 axis (§2 worked example).

**(c) How to measure generality at each stage.**
- Pretraining: macro-averaged per-domain perplexity or BPB ([[paloma]]); CF or BPB task scores until MCF is above chance ([[olmes]]); benchmarks selected for SNR at the model scale in use ([[signal-and-noise-eval]]).
- Mid-training: declared development and held-out base-model benchmarks ([[olmo-2]] Table 9).
- SFT, preference optimization, RL: Tülu 3 development and unseen suites; forgetting table against the starting checkpoint; pass@1 and pass@k up to k = 1024; over-refusal on safe prompts ([[xstest]]).
- Agentic evaluation: a do-nothing agent and trivial-output checks against the grader ([[agentic-benchmark-checklist]]); tasks from repositories outside the benchmark ([[swe-bench-illusion]]).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Treating a development benchmark gain as a capability gain | Δ_dev beyond noise, Δ_uns near zero or negative | compute (Δ_dev, Δ_uns) per axis on an unseen benchmark with a different formulation |
| Comparing unseen and development score levels | "unseen average is 35 points lower, so the model overfits" | compare changes between checkpoints, not levels across suites |
| Reporting one average | average rises while an axis falls | report every axis with its standard error |
| Declaring a winner from a 1-3 point difference on a small benchmark | differences flip between reruns or seeds | paired SE (§6.1) and a seed or final-checkpoint standard deviation (§6.2) |
| Using MCF for early pretraining ablations | scores near chance, low monotonicity across checkpoints | evaluate CF or BPB in parallel; check whether MCF exceeds chance |
| Changing prompt format or answer extraction between runs | a jump that coincides with a harness change | pin and report the format; rescore old checkpoints with the new harness |
| Lowering temperature to reduce variance | variance falls but scores shift | keep the evaluated temperature; average K samples per item instead ([[adding-error-bars-evals]] §3.3) |
| Reporting pass@1 only after RL | pass@1 rises, pass@k at large k falls below the base model | plot pass@k for k from 1 to at least 128 for base and trained models |
| Reporting a combined safety average only | fewer unsafe completions, more refusals of safe prompts | report refusal on safe prompts (XSTest safe set) separately |
| Skipping decontamination of public chat data | high scores on AlpacaEval-style or jailbreak benchmarks | 8-gram overlap between training prompts and every evaluation ([[tulu-3]] §3.2, Table 37) |
| Attributing a fall to forgetting without reading outputs | near-zero scores on a benchmark after a format change | inspect outputs for parse failures before and after the step |

## Check your understanding

1. Tülu 3's persona data raised IFEval by 19.2 points but not IFEval-OOD. Explain which features of the persona data and of IFEval could produce this pattern, and what additional measurement would distinguish constraint overfitting from a harder unseen benchmark.
2. Explain why an unseen benchmark stops being unseen after it influences one decision, using Chollet's distinction between system-centric and developer-aware generalization.
3. A 7B pretraining ablation raises standard MMLU from 25.9 to 26.8. Using the monotonicity and seed results of [[benchmark-variance-quantified]], explain why this comparison is uninformative and what measurement should replace it.
4. Two SFT runs differ by 2.3 points on HumanEval. Explain under which condition on the per-item scores the paired standard error is smaller than the unpaired one, and why neither accounts for seed variance.
5. The first principal component explains nearly 80% of variance across 77 base models. Explain why this does not imply that a single training intervention moves all capability axes together.
6. After RLVR, pass@1 rises and pass@1024 falls below the base model. Explain the mechanism in terms of how probability mass is redistributed across solutions, and what this means for coverage of problems.
7. A safety-tuning step raises the Tülu 3 Safety average. Explain how over-refusal could still increase, and which component scores would show it.

## Connections

- **Previous (array order):** none. ch-00 opens the course.
- **Previous in this learner's resume path:** ch-24 — Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data.
- **Next (array order):** ch-01 — Optimizers for LLM Training: AdamW, Update Size, and Retention of Prior Ability.
- **Next in this learner's resume path:** ch-04 — Sequence Packing, Loss Masking, and Chat Templates.
- **Chapters that depend on ch-00:** ch-01 — Optimizers for LLM Training: AdamW, Update Size, and Retention of Prior Ability; ch-04 — Sequence Packing, Loss Masking, and Chat Templates; ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability; ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix; ch-29e — Instruction Tuning and Generalization to Unseen Tasks; ch-47 — Evaluation Harness and Suite Design for General Capability.
- **Chapters that extend one section:** §3 long context → ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation; §6 → ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions; §7.1 → ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control; §7.2-7.3 → ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity, and ch-43 — Entropy, Output Diversity, and KL Control in RL; §7.4 → ch-52 — Safety Evaluation, Over-Refusal, and Red-Teaming; §7.5 → ch-48 — Contamination Detection and Its Effect on Reported Scores; §4 and §7.6 → ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation.

## Sources

- [[tulu-3]] — core-skill list (Table 3), development and unseen suites (Table 24, §7.2-7.3), decontamination rule (§3.2), dev-versus-unseen ablations (Tables 31-33, Fig. 24), seed table (Table 14), contaminated public datasets (Table 37).
- [[tulu-3-eval-regime]] — chapter extract of the Tülu 3 evaluation sections with loci and the source-internal inconsistencies.
- [[paloma]] — per-domain perplexity formula, macro averaging, corpus-dependent gaps, perplexity-versus-downstream correlations.
- [[olmes]] — MCF and CF definitions, normalization formulas, ARC-Challenge score disagreement across references, OLMES settings.
- [[helm]] — multi-metric coverage (7 metrics, 16 scenarios, 17.9% → 96.0%), calibration and prompting findings, HellaSwag formulation result.
- [[signal-and-noise-eval]] — signal, noise, SNR definitions and their relation to decision accuracy; BPB and checkpoint averaging.
- [[benchmark-variance-quantified]] — seed variance and interval sizes for 13 benchmarks; MMLU standard versus cloze.
- [[prompt-format-sensitivity-formatspread]] — spread definition, FormatSpread procedure, median spread and ranking reversals.
- [[capability-structure-factors]] — three-factor structure across 29 models and correlations with instruction tuning and size.
- [[measure-of-intelligence]] — skill versus generalization, system-centric versus developer-aware generalization, degrees of generalization.
- [[gpt-2-unsupervised-multitask]] — zero-shot transfer as a generality measurement; 8-gram overlap analysis.
- [[observational-scaling-laws]] — low-dimensional capability space across 77 base models.
- [[adding-error-bars-evals]] — standard errors, paired and clustered comparisons, variance reduction.
- [[olmo-2]] — base-model development and held-out split, GSM* partial hold-out, mid-training changes (arXiv 2501.00656v3; card not yet verified, loci read from the primary text).
- [[rlvr-beyond-base-model]] — pass@k estimator and base-versus-RLVR coverage table (arXiv 2504.13837v5; loci read from the primary text).
- [[transferability-of-llm-reasoning]] — math-only SFT versus RL effects on non-math axes (arXiv 2507.00432v2 Table 1; loci read from the primary text).
- [[ifbench]] — IFEval-to-unseen-constraint gap and IF-RLVR forgetting table.
- [[xstest]] — over-refusal measurement and system-prompt effects.
- [[llama-3]] — contamination estimates and exclusion of benchmark training sets from annealing.
- [[swe-bench-illusion]] — memorization evidence for SWE-Bench Verified.
- [[echo-chamber-rl-post-training]] — format convergence and pass@64 under RL with different KL coefficients.
- [[finetuning-compromises-safety]] — safety regression after benign fine-tuning.
- [[helmet]] — long-context category correlations.
- [[bfcl]] — function-calling mode and category differences.
- [[agentic-benchmark-checklist]] — grader validity checks for agentic benchmarks.
- [[multichallenge]] — multi-turn benchmark and judge-rubric agreement.
- [[scaling-laws-unreliable-downstream]] — share of downstream tasks that scale predictably with loss.

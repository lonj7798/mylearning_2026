<!-- chapter: ch-47
     track: eval
     kind: content
     title: Evaluation Harness and Suite Design for General Capability
     deps: [ch-00, ch-36]
     sources: [[tulu-3]], [[tulu-3-eval-regime]], [[olmes]], [[helm]], [[mmlu-pro]], [[ifeval]], [[prompt-format-sensitivity-formatspread]], [[multi-prompt-evaluation]], [[thinkingmachines-defeating-nondeterminism]], [[signal-and-noise-eval]], [[helmet]], [[nolima]], [[context-length-alone-hurts]], [[chroma-context-rot]], [[lost-in-the-middle]], [[holistic-agent-leaderboard]], [[agentic-benchmark-checklist]]
     figures: figures/eval-config-matrix.html
     revised: 2026-09 (generality revision)
-->

# Chapter 47 — Evaluation Harness and Suite Design for General Capability

> **Core insight.** A reported score is produced by three things together: the item set, the way each item is presented and scored, and the inference configuration that generated the answer. Published numbers for one model on one task show the size of the second and third: OLMES Table 1 lists seven ARC-Challenge scores for Llama2-7B, from 43.2 to 54.2, each from a different reference with a different number of shots, formulation, and normalization ([[olmes]] Table 1). Measuring *general* capability adds a fourth requirement, from ch-00: the suite must cover the capability axes the model is claimed to have, and each axis needs a benchmark that was never read during development. Tülu 3's final 8B checkpoint scores 68.8 on its development suite and 32.4 on its unseen suite; those two numbers answer different questions and both are needed ([[tulu-3-eval-regime]] Table 31).
>
> **Guideline.** When a number is reported, report with it the task id, the formulation and normalization, the matcher, the number and source of few-shot examples, the generation settings, the harness version, and the model revision, because each of these has a measured effect of 1 point or more in the sources below. When a suite is built for a general-purpose model, allocate one development and one unseen benchmark per capability axis and record which decision each task supports, because a suite with no unseen half cannot separate a skill gain from a benchmark-specific gain ([[tulu-3]] §7.4). When base checkpoints are compared during pre-training, use the cloze formulation or bits-per-byte, because the multiple-choice formulation stays near chance until the model acquires the format ([[olmes]] §3.4; [[signal-and-noise-eval]] §5.3); once the model is above chance on the multiple-choice formulation, report the higher of the two ([[olmes]] §3.4). When a claim is about robustness rather than peak ability, evaluate several prompt templates per task and report the average and the spread, because a single template gives a median spread of 7.5 accuracy points over ten plausible alternatives ([[prompt-format-sensitivity-formatspread]] §4.2). When scores are compared across runs at the 1-point level, pin the inference stack as well, because at temperature 0 the same prompt on the same weights produced 80 distinct completions out of 1,000 under varying server batch size ([[thinkingmachines-defeating-nondeterminism]]).

## Why this chapter matters for a general-purpose model

Ch-00 defines general capability as measured performance on tasks that were neither targeted by training nor read during development, and gives the measurement errors that apply to any such number. Ch-36 produced an SFT checkpoint together with a held-out split. This chapter builds the instrument that both of those depend on: a suite (which items, in which splits, supporting which decisions) and a harness (the program that turns a checkpoint plus a task definition into a score and a per-item record).

Two failures motivate the work, and both are measurable. The first is comparison failure: two numbers for the same task and the same model are not comparable when their setups differ, and the setups usually are not published. HELM found that before its own standardization, the 30 models it studied had on average been evaluated on 17.9% of its 16 core scenarios, and that even shared datasets were run under different conditions ([[helm]] Abstract, §1). The second is generality failure: a suite that is read during development becomes a target, so its scores stop measuring the skill and start measuring fit to the benchmark. Tülu 3 measured this directly by keeping a second suite unread until decisions were final ([[tulu-3]] §2.2, §7.4).

The harness sits at the end of the pipeline (pre-training → mid-training → SFT → preference optimization → RL → evaluation), but every earlier stage reads from it. A data-mixture ablation, a checkpoint-selection rule, and a reward design are all decided by differences between harness outputs. An instrument that is imprecise or unrecorded at the 1-3 point level makes those decisions at that level unreliable, which is the subject of ch-51.

## §1 From a capability coverage map to a suite

**Definition.** A *capability coverage map* is the list of capability axes a model is claimed to have, with, for each axis, the benchmarks that measure it and the split each benchmark belongs to. An *evaluation suite* is the coverage map made executable: task ids, item counts, splits, settings, and the decision each task is allowed to support.

**The problem, stated measurably.** Without a coverage map, benchmark selection follows availability, and axes with no cheap benchmark go unmeasured. A model can then improve its reported average while losing ability on an unmeasured axis, and no number in the report moves.

**Mechanism.** Building the suite has five steps.

1. List the capability axes. HELM's taxonomy makes each scenario a triple of task, domain, and language, and separates 16 *core* scenarios (where every metric is measured) from 26 *targeted* scenarios that isolate one skill ([[helm]] §1). The axis list used in this course follows ch-00: knowledge, reasoning, mathematics, code, instruction following, long context, multilingual, tool use and agentic work, factuality and calibration, safety and over-refusal.
2. Decide which metrics are measured on every axis, not only accuracy. HELM measures 7 metric categories (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency) on its core scenarios and reports 98 of the 112 possible (scenario, metric) pairs, 87.5% ([[helm]] Abstract, §1).
3. For each axis, choose one development benchmark and one unseen benchmark that require the same skill but differ in source, format, or constraint set. Tülu 3's pairs are MMLU → MMLU-Pro and GPQA, BBH and DROP → AGIEval English, MATH and GSM8K → DeepMind Mathematics, HumanEval → BigCodeBench, IFEval and AlpacaEval 2 → IFEval-OOD and HREF ([[tulu-3-eval-regime]] Table 3).
4. Record, for each task, which decision it supports: data-mixture ablation, checkpoint selection, release gate, or report only. A task used for checkpoint selection is a development task by definition, whatever its name.
5. Freeze the unseen half. Tülu 3 states that unseen scores were not examined during development, which is what makes the dev-to-unseen comparison in §7.4 interpretable ([[tulu-3]] §2.2).

**Evidence.** Reading the two halves against each other separates general gains from benchmark-specific ones. For Tülu 3 8B across SFT → DPO → final: the development average moves 64.9 → 68.3 → 68.8 and the unseen average 29.9 → 31.9 → 32.4, so the pipeline improves both. Per axis the picture is not uniform: instruction following moves 72.8 → 82.4 on IFEval and 17.6 → 24.3 on IFEval-OOD, while coding moves 86.2 → 83.9 on HumanEval and 11.5 → 7.4 on BigCodeBench at 8B, against 12.2 → 21.6 at 70B ([[tulu-3-eval-regime]] Table 31). **Result (single study).**

The mechanism behind one of those gaps is instructive because it is a format effect rather than a skill effect. Scaling the DPO preference data improved MATH more than DeepMind Mathematics; the authors' explanation is that the models learned to emit LaTeX-formatted reasoning and answers, which DeepMind Mathematics does not ask for, and this interfered with the reasoning and with answer extraction ([[tulu-3]] §7.4.1, Fig. 24). **Interpretation (paper's).**

**The coverage map as a table.** The rows below combine the ch-00 axis list with the benchmark assignments each source published. The first six rows are Tülu 3's own pairs ([[tulu-3-eval-regime]] Table 3, Table 24); the last four rows are assembled by this course from the sources named in their cells, because Tülu 3's suite does not cover those axes.

| Axis | Development benchmark | Unseen benchmark | Shape | Metric | Decision it supports |
|---|---|---|---|---|---|
| Knowledge | MMLU, PopQA, TruthfulQA | MMLU-Pro, GPQA | multiple choice / short answer | EM, MC2 | data mixture, checkpoint selection |
| Reasoning | BBH, DROP | AGIEval English | generation with CoT | EM, F1 | data mixture |
| Mathematics | MATH, GSM8K | DeepMind Mathematics | generation, verified | flex EM, EM (Sympy) | data mixture, RL reward design |
| Code | HumanEval, HumanEval+ | BigCodeBench (hard, 148 of 1,140) | generation, executed | pass@10 at T=0.8 | data mixture |
| Instruction following | IFEval, AlpacaEval 2 | IFEval-OOD (52 constraints), HREF | generation, program-checked / judged | prompt-level loose, LC win rate | checkpoint selection |
| Safety and over-refusal | Tülu 3 Safety (macro over 6 sets) | none in the source | generation, classifier-scored | refusal / compliance accuracy | release gate |
| Long context | RAG and recall categories at 8K-128K ([[helmet]] §2.1) | downstream categories at 128K; low-overlap needles ([[nolima]]) | generation at several lengths | SubEM, judge score | position-scaling and long-context data decisions |
| Multilingual | not covered by the sources used here | not covered | — | — | — |
| Factuality and calibration | calibration as a metric on every core scenario ([[helm]] §1) | — | any | ECE, selective accuracy | release gate |
| Tool use and agentic | task-specific benchmark with a fixed scaffold ([[holistic-agent-leaderboard]] §3) | held-out environments (§8) | multi-step, environment-scored | task success with cost | scaffold and agent-data decisions |

The last column is the part most often left implicit. A benchmark used to pick a checkpoint cannot also serve as evidence that the checkpoint generalizes, and recording the role at suite-definition time is what prevents the two uses from being confused later.

**Conditions and limits.** Tülu 3 has no unseen safety benchmark ([[tulu-3-eval-regime]] Table 3), so its safety numbers are development-only. An unseen benchmark also stops being unseen once a decision is made from it; after that it is a development benchmark and a new one is needed.

**Sizing the suite.** Item counts are a budget decision, and larger is not always better. [[signal-and-noise-eval]] reports that a 1,000-question ARC Easy subset gave higher decision accuracy than full MMLU with 90% fewer instances, and that AutoBencher with 33K instances had the highest checkpoint noise of the benchmarks studied (App. B.2). HELM caps evaluation at 1,000 instances per scenario, and OLMES samples 1,000 instances when a dataset has more than 1,500, with the stated reason that extra statistical signal is dominated by other sources of score variation ([[olmes]] §3.1). A planning example: 10 axes × 2 benchmarks × 1,000 items = 20,000 items per full suite run; running five prompt templates (§5) instead of one on the four cheapest axes adds (4 axes × 2 benchmarks × 1,000 items) × 4 extra templates = 32,000 further generations, which is more than the base suite costs.

## §2 Benchmark shape, formulation, and metric

**Definition.** *Formulation* is how a multiple-choice item is presented to the model. In the *multiple-choice formulation* (MCF) the options appear in the prompt with letter labels and the model is scored on the label. In the *cloze formulation* (CF, also called completion) each answer string is scored separately by the model's probability of that string given the question ([[olmes]] §2.1).

**The problem.** The two formulations give different numbers for the same weights, and which one is higher depends on the model's strength, so a comparison across references can invert a ranking.

**Mechanism and formulas.** CF requires a length normalization, because the raw probability of a string falls with its length. OLMES compares four ([[olmes]] §3.3):

- none: `ln P(a_i | q)`
- token: `ln P(a_i | q) / num_tokens(a_i)`
- character: `ln P(a_i | q) / num_characters(a_i)`
- pmi: `ln [ P(a_i | q) / P(a_i | u) ]`, with `u = "Answer:"`

where `a_i` is answer choice *i*, `q` is the formatted question prompt, and `u` is an unconditional prompt that removes the question.

**Worked numeric example** (constructed for this chapter; check by hand). An item has two choices, "iron" (4 characters) and "magma" (5 characters). Suppose `ln P(iron|q) = −6.0` and `ln P(magma|q) = −7.0`, and unconditionally `ln P(iron|u) = −5.0`, `ln P(magma|u) = −7.5`.

| normalization | score of "iron" | score of "magma" | prediction |
|---|---|---|---|
| none | −6.00 | −7.00 | iron |
| character | −6.0/4 = −1.50 | −7.0/5 = −1.40 | magma |
| pmi | −6.0 − (−5.0) = −1.0 | −7.0 − (−7.5) = +0.5 | magma |

Three scoring rules over one unchanged forward pass, two different predictions. OLMES fixes the rule per task from experiments over 15 base models: pmi for ARC-Challenge, CommonsenseQA, OpenBookQA; character for ARC-Easy, HellaSwag, PIQA, Social IQa, MMLU; none for BoolQ and WinoGrande. The per-task choice is within 0.0-1.1 points of the per-model best ([[olmes]] Table 2, Table 3).

**Evidence for the formulation effect.** Llama3-8B on ARC-Challenge: 60.2 with 25-shot CF on the Hugging Face Open LLM Leaderboard and 78.6 with 25-shot MCF in the Llama 3 model card, both correct within their own setup ([[olmes]] Table 1). Llama3-70B scores 93.7 with MCF and 69.0 with CF, which the authors describe as a near-5x difference in error rate (6.3% against 31%) ([[olmes]] §3.4). In the other direction, the 8 weakest of 15 base models score near random with MCF on ARC-Challenge and above random with CF ([[olmes]] §3.4, Fig. 2). HELM reports the same phenomenon from the opposite side: OPT (175B) scores 79.1% on HellaSwag when each choice is scored in a separate 0-shot prompt, 54.8% for the calibrated separate variant, and 30.2% when the choices are presented jointly in a 5-shot multiple-choice prompt ([[helm]] §8.2, Fig. 33; Finding 23). **Replicated** across two independent studies.

**Timing matters for pre-training monitoring.** On MMLU, OLMo-7B-0424 stays near random in MCF until roughly 400B training tokens and gives the stronger signal after that, while CF carries signal from the start ([[olmes]] Fig. 1). OLMES therefore runs both formulations and reports the higher ([[olmes]] §3.4). For loss-like alternatives, [[signal-and-noise-eval]] reports that scoring with bits-per-byte (negative log-likelihood of the correct answer divided by its UTF-8 byte count) raised the 30-task average signal-to-noise ratio from 10.0 to 31.5 and small-scale decision accuracy from 77.0% to 83.7% (§5.3, Fig. 6).

**Choosing benchmarks that can separate checkpoints.** A benchmark is useful for a decision only when the difference it shows exceeds its own noise. [[signal-and-noise-eval]] defines signal as the relative dispersion of scores across a population of similarly trained models, noise as the relative standard deviation over the final *n* checkpoints of one run, and SNR as their ratio (§3, Eq. 2); SNR predicts decision accuracy with R = 0.791 (R² = 0.626) while signal or noise alone does not (§4.1, Fig. 2). MMLU-Pro attacks the same problem from the item side: ten options instead of four, 12,032 questions, and two rounds of expert review; the GPT-4o to GPT-4-Turbo gap is 1% on MMLU and 9% on MMLU-Pro ([[mmlu-pro]] §1).

**Worked numeric example** (constructed; the guessing model is an assumption, not a measurement). Suppose a model knows the answer to 50% of items and guesses uniformly on the rest. With four options its expected score is 0.5 + 0.5 × 0.25 = 62.5%; with ten options it is 0.5 + 0.5 × 0.10 = 55.0%. Two models that know 50% and 55% of items differ by 3.75 points under four options and by 4.5 points under ten, before any noise is considered. More distractors lower the score and widen the gap to be detected.

**Implication for a general-purpose model.** The shape of a benchmark decides which capability is measured. Scoring mathematics as a multiple-choice ranking measures discrimination among given options, not derivation. The axis-to-shape assignment is part of the coverage map, and it is the reason Tülu 3 evaluates coding as pass@10 over sampled generations at temperature 0.8, instruction following with a program-checkable constraint metric, and knowledge with exact match ([[tulu-3-eval-regime]] Table 24).

## §3 The matcher is part of the metric

**Definition.** The *matcher* is the function that maps a model output to a score for one item: exact match, a normalizing extractor, a program that executes the output, a program that checks a constraint, or a model that judges it.

**The problem.** A matcher has false negatives (a correct answer written in an unexpected form) and false positives (an incorrect answer that satisfies the check). Both shift the reported score without any change to the model.

**Mechanism and evidence, with the effect size measured.** IFEval reports the same 541 prompts under two matchers. The *strict* matcher tests the constraint on the response as produced. The *loose* matcher applies eight transformations (remove markdown bold markers, remove the first line, remove the last line, all pairs, all three, identity) and counts the constraint as followed if any transformed response passes. For GPT-4: prompt-level strict 76.89%, prompt-level loose 79.30%; instruction-level strict 83.57%, instruction-level loose 85.37% ([[ifeval]] §2.2, Table 3).

**Worked numeric example.** 541 × 0.7689 = 416.0 prompts pass under the strict matcher, 541 × 0.7930 = 429.0 under the loose one. Thirteen prompts, 2.41 points, are decided by the eight text transformations and not by the model. The IFEval authors state that the loose criterion reduces false negatives and introduces false positives, and treat it as a complement rather than a replacement ([[ifeval]] §2.2) — so a report that does not name which of the four numbers it quotes is ambiguous by up to 8.5 points (76.89 to 85.37 for one model).

A second measurement, from a different team and task: Tülu 3's MATH evaluation uses a "flex" extraction that tries the Minerva format, then the last `<ans>` tag, then the text between the last two `$` delimiters; moving from Minerva-only to flex "can sometimes improve reported scores by up to 10 points" ([[tulu-3-eval-regime]] §7.2). **Replicated** as a phenomenon (two independent reports), though the two effect sizes are from different tasks and are not comparable.

**Model-based matchers.** When no program can check the answer, the matcher is a model, and it needs its own validation. HELMET's summarization judge (GPT-4o-2024-05-13) is validated against human judgments with Cohen's κ of 0.76 and 0.72 for recall and 0.91 and 0.83 for precision on two datasets, and the authors note the judge is more lenient than humans on partially supported points ([[helmet]] App. B.6). The same section shows what a surface metric misses: a Mistral-7B-Instruct-v0.3 summary consisting of one sentence repeated hundreds of times scores ROUGE-L 12.3 and judge score 0.0. Tülu 3's HREF uses Llama 3.1 70B Instruct as judge against a Llama 3.1 405B Instruct baseline, with composite agreement of 69.4% against humans compared with 67% between humans ([[tulu-3-eval-regime]] §7.3.2). Judge bias, calibration, and judge-specific overfitting are the subject of ch-49; the point here is that the judge model, its version, and its prompt are harness coordinates that belong in the record.

**Choosing a matcher.** The families differ in what they can verify and in the error they introduce. Use the strongest one the capability admits.

| Matcher family | Verifies | Introduces | Attested use |
|---|---|---|---|
| Exact or substring match | the answer string appears as written | false negatives on paraphrase and formatting | HELMET RAG categories score SubEM ([[helmet]] Table 3) |
| Normalizing extractor | the answer after a defined set of extraction attempts | false positives from over-permissive extraction | Tülu 3 MATH flex extraction, up to 10 points over Minerva-only ([[tulu-3-eval-regime]] §7.2) |
| Program that checks a constraint | a property of the output that is decidable | false negatives from surface markup; false positives after transformation | IFEval strict and loose, 2.41 points apart on GPT-4 ([[ifeval]] Table 3) |
| Execution against tests | the output's behavior | dependence on test coverage; sandbox and timeout effects | HumanEval and BigCodeBench pass@10 in the Tülu 3 suite ([[tulu-3-eval-regime]] Table 24) |
| Environment-state predicate | the final state of a task environment | grader flaws that trivial agents can satisfy | do-nothing agent passes 38% of τ-bench Airline ([[agentic-benchmark-checklist]] App. E.2) |
| Model as judge | agreement with a reference or a rubric | judge bias, leniency, version drift | HELMET judge with κ 0.72-0.91 against humans ([[helmet]] App. B.6); HREF judge agreement 69.4% against 67% between humans ([[tulu-3-eval-regime]] §7.3.2) |

**Detection rule.** Sample 50 items scored incorrect and read the outputs. If the answer is present but in another form, the matcher has a false-negative problem; if the answer is absent but the item scored correct, it has a false-positive problem. Both are matcher bugs, not model results.

## §4 Inference configuration and numeric determinism

**Definition.** The *inference configuration* is everything that governs generation: temperature, top-p, maximum new tokens, stop sequences, number of samples, chat template, system prompt, thinking mode and budget, position-scaling settings, and the serving stack and precision.

**The problem.** These are not fixed by the task definition, so two harnesses that implement the same benchmark can generate different text and therefore score differently.

**Attested settings, from reports that publish them.** Tülu 3 runs HumanEval and HumanEval+ at temperature 0.8 with pass@10, AlpacaEval 2 with greedy decoding up to 8,192 tokens, IFEval with greedy decoding and prompt-level loose scoring, and MMLU 0-shot with chain of thought ([[tulu-3-eval-regime]] Table 24, §7.2). HELMET uses greedy decoding at five input lengths and adds two in-context demonstrations for every task except ICL and RULER so that base models produce parseable output ([[helmet]] §3, §2.3). MMLU-Pro uses 5-shot chain-of-thought prompting ([[mmlu-pro]] §4). OLMES restricts inputs including the completion to 2,048 tokens and evaluates at the model's default precision ([[olmes]] §3.5).

These settings, collected by shape, are the fields a harness has to expose per task. Every value below is quoted from the source named in the last column; a blank means that source does not fix the field.

| Field | Multiple choice, scored by log-probability | Generation, verified | Generation, judged | Long context |
|---|---|---|---|---|
| temperature | not applicable | 0.8 for HumanEval pass@10 ([[tulu-3-eval-regime]] Table 24) | greedy for AlpacaEval 2 ([[tulu-3-eval-regime]] Table 24) | greedy ([[helmet]] §3) |
| maximum new tokens | not applicable | not reported | 8,192 for AlpacaEval 2 ([[tulu-3-eval-regime]] Table 24) | not reported |
| samples per item | 1 | 10 for pass@10 ([[tulu-3-eval-regime]] Table 24) | 1 | 1 |
| few-shot examples | 5 curated ([[olmes]] §3.2) | 8 for GSM8K, 4 for MATH ([[tulu-3-eval-regime]] Table 24) | 0 ([[tulu-3-eval-regime]] Table 24) | 2, except ICL and RULER ([[helmet]] §2.3) |
| chain of thought | no | yes for GSM8K, MATH, BBH; 5-shot CoT for MMLU-Pro ([[tulu-3-eval-regime]] Table 24; [[mmlu-pro]] §4) | no | task-dependent |
| input cap and truncation | 2,048 tokens including completion ([[olmes]] §3.5) | not reported | not reported | five lengths to 131,072 tokens, truncation from the end ([[helmet]] §2.1, §3) |
| precision | model default ([[olmes]] §3.5) | not reported | not reported | BF16 on H100 ([[helmet]] App. D) |

**Few-shot demonstrations are part of the configuration, with a measured and non-uniform effect.** HELMET at 128K tokens: Llama-3.1-8B base moves from 77.3 to 98.0 on JSON KV and from 0.1 to 7.5 on MS MARCO when two demonstrations are added, while GPT-4o-05 drops from 99.3 to 36.7 on JSON KV with the same change (3 seeds, [[helmet]] Table 8). A single shot count is therefore not neutral across models.

**Test-time compute is a coordinate too.** For models with a thinking mode, the reasoning budget changes the score in both directions: HAL paired low and high reasoning effort for four models across benchmarks and found equal or lower accuracy with more reasoning in 21 of 36 model-agent-benchmark combinations ([[holistic-agent-leaderboard]] §4.1 item 5, Fig. 3). **Result (single study).** A comparison between a thinking model and a non-thinking model with unreported budgets is confounded.

**Numeric determinism.** At temperature 0 the sampler is deterministic given the logits, but the logits depend on the batch the request shared. [[thinkingmachines-defeating-nondeterminism]] attributes this to kernels that are not batch-invariant: matmul, RMSNorm, and attention change their reduction order with batch size, and server load varies. Measurement: Qwen3-235B-A22B-Instruct-2507, one prompt, temperature 0, 1,000 completions of 1,000 tokens each gives 80 unique completions; the most common appears 78 times; all 1,000 agree for the first 102 tokens, and at the divergence point 992 continue "Queens, New York" while 8 continue "New York City". With batch-invariant kernels all 1,000 completions are identical. Throughput cost on Qwen3-8B for 1,000 short sequences: 26 s (vLLM default) → 55 s (unoptimized deterministic) → 42 s (improved attention kernel).

**Worked numeric example** (this course's arithmetic, **Interpretation**; the source measures completions, not benchmark scores). Take the divergence rate at the observed branch: 8 of 1,000 completions took the other continuation. On a 500-item generation benchmark, if each item's output changes with probability 0.008 and half of those changes flip correctness, the expected movement is 500 × 0.008 × 0.5 = 2 items = 0.4 points. That is below the 1-3 point differences usually acted on, but it is not zero, and it is invisible in a report that only records "temperature 0".

The figure [figures/eval-config-matrix.html](figures/eval-config-matrix.html) collects the measured effects from this section and from §2, §3, §5, and §7 into one panel. Selecting a coordinate — formulation, adaptation method, matcher variant, prompt format, benchmark construction, numeric determinism, perturbation, evidence position — shows how far the reported number moved in the cited study with the weights held fixed, and the second panel prints the Tülu 3 8B development and unseen scores per skill across the three pipeline stages.

## §5 Prompt-format robustness as a quantity the harness reports

**Definition.** *Prompt-format sensitivity* is the variation of a score across prompt templates that a competent practitioner would consider equivalent: separators, casing, spacing, descriptor wording, enumeration style.

**The problem.** A single template produces one sample from a distribution whose spread is larger than many of the differences that decisions are made from.

**Mechanism.** FormatSpread defines a grammar over formats, treats each format as an arm of a multi-armed bandit, and searches within a fixed budget for the highest- and lowest-scoring formats, using Thompson sampling and requiring no access to model weights ([[prompt-format-sensitivity-formatspread]] §3.1-3.2). Performance spread is `max_i m(p_i, D) − min_i m(p_i, D)` over formats `p_1…p_n`, dataset `D`, and metric `m`.

**Evidence.** Across 53 Super-NaturalInstructions tasks: spread up to 76 accuracy points for LLaMA-2-13B; median spread 7.5 points with only ten sampled formats per task, which the authors call a lower bound; 20% of tasks have spread of at least 15 points in every LLaMA-2 setting; the spread persists with larger models, more shots, and instruction tuning; and rankings reverse — LLaMA-2-13B and -70B reverse by at least d = 0.02 with probability 0.141 ([[prompt-format-sensitivity-formatspread]] §4.2, Fig. 4). [[multi-prompt-evaluation]] measures the same effect over 6.5M instances, 20 models, and 39 tasks: a Friedman test gives statistically significant differences across templates for 21 of 25 tasks, most Kendall's W values are below 0.85, and a single edit of "." to ":" at the end of one LMentry template moves nous-hermes from 0.04 to 0.65 (§4, Tables 4-5). **Replicated.**

**What the harness should report.** [[multi-prompt-evaluation]] proposes three aggregates over a template set `I`: `MaxP = max_i ε(M,T,i)` for a downstream application with one fixed template; `AvgP = (1/|I|) Σ_i ε(M,T,i)` for robustness; and `CPS = Sat · MaxP` with `Sat = 1 − (MaxP − AvgP)` when both peak and robustness matter (§5). Here `ε(M,T,i)` is the score of model `M` on task `T` under template `i`.

**Worked numeric example.** Model A scores 0.74, 0.30, 0.10, 0.05 over four templates; model B scores 0.60, 0.58, 0.55, 0.52. Model A: MaxP = 0.74, AvgP = 1.19/4 = 0.2975, Sat = 1 − 0.4425 = 0.5575, CPS = 0.74 × 0.5575 = 0.413. Model B: MaxP = 0.60, AvgP = 2.25/4 = 0.5625, Sat = 0.9625, CPS = 0.578. Model A ranks first by MaxP and second by CPS. The paper reports this pattern in real data: on LMentry's rhyming-word task Falcon-Instruct-7b and Vicuna-13b lead on MaxP at 0.74 with AvgP of 0.17 and 0.15 ([[multi-prompt-evaluation]] §6, Fig. 6).

**A benchmark can also be built to reduce the sensitivity.** Under 24 reasonable prompts, score ranges are generally 4-5% on MMLU (peak 10.98%) and about 2% on MMLU-Pro (maximum 3.74%) ([[mmlu-pro]] §6.3, Fig. 5). Option order is a second axis of the same kind and is covered in ch-47a together with perturbation audits.

**Cost control.** Full multi-template evaluation multiplies cost by the number of templates. Two reductions are attested: FormatSpread's bandit search reports a spread for GPT-3.5 across 320 formats and 53 tasks at under $10 per task on average ([[prompt-format-sensitivity-formatspread]] §1), and [[multi-prompt-evaluation]] evaluates each template on a random 100-item subset (§3). A practical compromise is to run the full item set on the canonical template and a small template set on a fixed subsample.

## §6 The reproducibility record

**Definition.** The *reproducibility record* is the set of fields stored with every run, plus the per-item outputs, such that a third party can reproduce the number and the owner can re-slice it without regenerating.

**The problem.** Two failures follow from an incomplete record. A number cannot be reproduced, so a later difference cannot be attributed to the model rather than the harness. And a question asked later — how does the score split by subject, by length, by evidence position, by failure mode — requires a rerun, which costs the full generation budget again.

**Mechanism: what to store.**

```jsonl
{"task_id":"arc_challenge","harness":"olmes","harness_commit":"<sha>","format":"mcf",
 "norm":"pmi","shots":5,"shot_source":"curated-train","split":"test","n_items":1172,
 "model_revision":"<hf-revision-sha>","precision":"bf16","server":"vllm-<version>",
 "gen":{"temperature":0.0,"top_p":1.0,"max_tokens":16,"stop":["\n\n"],"n":1},
 "matcher":{"type":"argmax_choice_logprob"},"score":0.793,"decision_role":"report_only"}
{"task_id":"ifeval","harness":"open-instruct","harness_commit":"<sha>","shots":0,
 "matcher":{"type":"constraint_checker","variant":"prompt_level_loose"},
 "gen":{"temperature":0.0,"max_tokens":2048},"score":0.824,"decision_role":"dev_selection"}
```

Per item, store the rendered prompt, the raw output, the extracted answer, the score, and the item metadata the suite will later slice by (subject, length bucket, constraint type, evidence position, environment id). Ch-50 consumes exactly these records.

**Evidence that the stack itself drifts.** HAL documents, from a $40,000 evaluation campaign: a provider replaced a DeepSeek R1 endpoint with R1-0528 under the same name; one router served FP4 on one call and FP8 on another; silent rate-limit failures were scored as wrong answers; and an API removed the `stop` argument ([[holistic-agent-leaderboard]] App. A3). The same campaign found that the official τ-bench few-shot file contained test-set examples, discovered after about $1,000 of evaluations, and all results from that scaffold were excluded ([[holistic-agent-leaderboard]] App. A5). A model revision id and a serving-stack version in the record are what make such an event attributable after the fact.

**Suite hygiene belongs to the record.** Tülu 3 decontaminates training data against the suite with 8-gram matching on prompts: a test instance overlaps a training instance when more than 50% of its tokens have 8-gram matches with that instance, and a training set counts as contaminated when it overlaps more than 2% of an evaluation's instances; sets contaminated against the unseen suite are removed entirely ([[tulu-3-eval-regime]] §3.2, App. B.2). The reported overlaps are large enough to matter: Evol CodeAlpaca covers 70.7% of HumanEval and LMSys Chat 1M covers 46.5% of AlpacaEval (Table 37). Detection methods and their error rates are the subject of ch-48; the harness's part is to store the decontamination report id next to the scores it applies to.

**Per-item failures are the input to the next stage.** The stored records of failed items are the raw material for failure bucketing (ch-50), for rejection-sampling data construction, and for the perturbation audits of ch-47a. This is the one sense in which an evaluation harness produces negative examples; it does not apply a negative gradient itself, so the mechanisms of the negative-feedback chapter do not apply here.

## §7 Long-context evaluation: harness coordinates

Ch-32c covers claimed versus effective context length and the long-context suites themselves. This section covers what the harness must expose and record for those numbers to mean anything.

1. **Input length and truncation.** Length is an axis, not a setting: HELMET evaluates at 8,192 / 16,384 / 32,768 / 65,536 / 131,072 Llama-2 tokens and truncates long documents from the end ([[helmet]] §2.1, §3). Record the tokenizer used to define the length, the maximum input, and the truncation side.
2. **Position-scaling configuration.** Changing the position embedding at inference is an evaluation hyperparameter with two-sided effects: HELMET reports that Llama-3-Instruct with the RoPE base changed from 500,000 to 16,000,000 and Qwen2-Instruct with YaRN both degrade past 32,768 tokens, and that changing the position embedding can also lower short-length scores ([[helmet]] App. E.3).
3. **Demonstrations and answer format.** Base models need in-context examples to produce parseable output at long lengths; the effect is large and model-dependent ([[helmet]] Table 8, quoted in §4 above).
4. **Metric choice.** Surface-overlap metrics break down on long outputs: ROUGE-L for GPT-4o stays within 2 absolute points across lengths while the judge score rises, and a degenerate repeated-sentence summary scores ROUGE-L 12.3 with judge score 0.0 ([[helmet]] §2.2, App. B.6).
5. **Slices to store.** Evidence position (accuracy is highest at the start or end of the input and lowest in the middle: GPT-3.5-Turbo with 20 documents scores 75.8% when the answer document is first, 53.8% when tenth, 63.2% when twentieth, against a closed-book 56.1%, [[lost-in-the-middle]] Table 1, App. G Table 6), task category, and length bucket.
6. **Task family.** One long-context number is not a summary of long-context ability: across 35 instruction-tuned models at 128K, HELMET's category correlations run as low as 0.34 (Cite against ICL), and no synthetic task averages above 0.8 correlation with the downstream categories ([[helmet]] Fig. 3, Fig. 5). Lexical overlap between question and needle is a confound to control: with overlap removed, 11 of 13 models that claim 128K fall at or below half their short-context score at 32K, and GPT-4o goes from 99.3 to 69.7 ([[nolima]] Abstract, §4.4). Length alone degrades accuracy even when retrieval is verified: accuracy falls 13.9%-85% as input grows, and Llama-3.1-8B-Instruct recites the evidence exactly for 970 of 1,000 MMLU problems at 30k tokens while MMLU accuracy drops 24.2 points ([[context-length-alone-hurts]] Abstract, §1, App. Table 6). [[chroma-context-rot]] adds two harness requirements from 18 models: hold task difficulty fixed while varying length, and score abstentions separately from wrong answers.
7. **Pair every long-context run with a short-context regression run** on the same tasks, because position-scaling changes can lower short-input scores ([[helmet]] App. E.3). Decide the pair with the interval machinery of ch-51.

## §8 Agentic evaluation: harness coordinates

Ch-51a covers agent generality and reliability. The harness coordinates specific to agentic tasks are these.

1. **The scaffold is part of the measured system.** HAL reports that on Online Mind2Web, SeeAct with GPT-5 Medium costs $171 and Browser-Use with Claude Sonnet 4 costs $1,577 for a two-point accuracy difference, that Claude models score higher with Browser-Use and OpenAI models with SeeAct, and that task-specific scaffolds beat a generalist scaffold on 9 of 12 runs (CORE-Bench Hard) and 11 of 12 (SWE-bench Verified Mini) while the generalist costs less in 20 of 24 comparisons ([[holistic-agent-leaderboard]] §4.1 items 6-7). A model comparison with unequal scaffolds measures the pair, not the model.
2. **Record the budgets and the environment image.** HAL's generalist agent is specified as a smolagents CodeAgent with a planning interval of 4 steps, at most 200 steps, and a named tool set ([[holistic-agent-leaderboard]] §3, App. A8). Step, time, and token budgets bound the score; the environment image digest fixes the task.
3. **Report cost next to accuracy.** In HAL, the most costly model is on the cost-accuracy Pareto frontier for 1 of 9 benchmarks, and fewer than one third of models are on the frontier on average ([[holistic-agent-leaderboard]] §4.1 items 1-2).
4. **Validate the grader before trusting the score.** Running a do-nothing agent and an answer-enumerating agent against the grader is a cheap check: an agent that returns nothing passes 38% of τ-bench Airline tasks, and an agent that overwrites SWE-Lancer's test files scores 100%; across ten benchmarks the checklist found 7 with task-validity flaws, 7 with outcome-validity flaws, and 10 with reporting limitations ([[agentic-benchmark-checklist]] §1, §5.2, App. E.2, E.4).
5. **Hold out environments, not only tasks.** An agent suite whose environments all appear in training measures in-distribution reliability. A held-out environment split is the agentic form of the unseen suite from §1.
6. **Store trajectories.** HAL's analysis of 2,184 transcripts found agents retrieving benchmark answers from HuggingFace or arXiv in eight cases and hard-coded "plausible" solutions on two benchmarks, none of which is visible in the accuracy column ([[holistic-agent-leaderboard]] §4.2, App. A7-A8).

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OLMES standard (15 base models, 1B-70B) | — | eval-gate | few-shot examples | 5 curated shots per task, from the training set, label-balanced | arXiv:2406.08446v2 §3.2 | verified 2026-09-15 | §3.2: beyond 5 shots gives no meaningful difference (cites Brown et al. 2020; Barton 2024) |
| OLMES standard | — | eval-gate | instance sampling | test split if labels public, else validation; sample 1,000 if the set exceeds 1,500, `Random(1234).sample` | §3.1, Table 2 | verified 2026-09-15 | §3.1: extra statistical signal is dominated by other sources of score variation |
| OLMES standard | — | eval-gate | CF normalization | pmi for ARC-C, CSQA, OBQA; character for ARC-E, HellaSwag, PIQA, SIQA, MMLU; none for BoolQ, WinoGrande | Table 2, Table 3 | verified 2026-09-15 | Table 3: win percentage over 15 models; 0.0-1.1 points from the per-model oracle |
| OLMES standard | — | eval-gate | formulation | run MCF and CF, report the higher | §3.4 | verified 2026-09-15 | Fig. 1 (MMLU during OLMo-7B-0424 training), Fig. 2 (15 models) |
| OLMES standard | — | eval-gate | input cap; precision; MMLU aggregation | 2,048 tokens including completion; model default precision; macro average over 57 subjects | §3.5 | verified 2026-09-15 | no ablation reported |
| Tülu 3 8B / 70B | 8B, 70B | eval-gate | development settings | MMLU 0-shot CoT EM; PopQA 15-shot EM; TruthfulQA 6-shot MC2; BBH 3-shot CoT EM; DROP 3-shot F1; GSM8K 8-shot CoT EM; MATH 4-shot CoT flex EM; HumanEval and HumanEval+ pass@10 at T=0.8; IFEval prompt-level loose; AlpacaEval 2 length-controlled win rate, greedy ≤ 8,192 tokens | arXiv:2411.15124v5 Table 24, §7.2 | verified 2026-09-15 | §7.2: flex extraction improves reported MATH scores by up to 10 points over Minerva-only |
| Tülu 3 8B / 70B | 8B, 70B | eval-gate | unseen suite | MMLU-Pro, GPQA, AGIEval English, DeepMind Mathematics, BigCodeBench (hard subset, 148 of 1,140), IFEval-OOD (52 constraints), HREF; 0-shot, CoT where marked | Table 3, Table 24, §7.3 | verified 2026-09-15 | §7.4 Table 31: dev-to-unseen comparison of SFT, DPO, and final checkpoints |
| Tülu 3 | — | eval-gate | decontamination threshold | 8-gram prompt match; instance overlaps when > 50% of its tokens match one training instance; training set contaminated when it overlaps > 2% of an evaluation | §3.2, App. B.2 | verified 2026-09-15 | Table 37: overlaps such as Evol CodeAlpaca 70.7% of HumanEval |
| HELMET (59 long-context models) | 1B-405B and closed | eval-gate | long-context protocol | lengths 8,192 / 16,384 / 32,768 / 65,536 / 131,072 Llama-2 tokens; greedy decoding; 2-shot demonstrations except ICL and RULER; judge GPT-4o-2024-05-13; 100-600 samples per dataset | arXiv:2410.02694v3 §2.1-2.3, §3, App. D | verified 2026-09-15 | Table 8: 0-shot vs 2-shot for base and instruction-tuned models, 3 seeds |
| MMLU-Pro (50+ models) | — | eval-gate | prompting | 5-shot chain of thought | arXiv:2406.01574v6 §4 | verified 2026-09-15 | §6.2 Table 3: CoT above direct answering on MMLU-Pro |
| HAL generalist agent | 9 models | eval-gate | agentic scaffold | smolagents CodeAgent; planning interval 4 steps; ≤ 200 steps; search, browse, Python, bash, text inspector, file editor, VLM tools | arXiv:2510.11977 §3, App. A8 | verified 2026-09-15 | §4.1 item 7: task-specific scaffolds win 9 of 12 and 11 of 12 runs; generalist cheaper in 20 of 24 |
| Batch-invariant vLLM (Qwen3-8B) | 8B | eval-gate | numeric determinism | batch-invariant RMSNorm, matmul, and attention kernels | Thinking Machines Lab, Sep 2025, "Performance" and "How nondeterministic are completions?" | verified 2026-09-15 | 1,000 of 1,000 completions identical; 26 s → 55 s → 42 s for 1,000 short sequences |

**Starting point for a small general-purpose run.** Every number here comes from a verified row above, with the conditions under which its source used it. For a 1B-8B checkpoint evaluated during and after SFT: multiple-choice axes under the OLMES standard (5 curated shots, per-task normalization, both formulations with the higher reported, 1,000 sampled instances above 1,500, 2,048-token cap, default precision — developed on 15 base models from 1B to 70B); mathematics and code under the Tülu 3 development settings (GSM8K 8-shot CoT, MATH 4-shot CoT with flex extraction, HumanEval pass@10 at temperature 0.8 — used for 8B and 70B Llama 3.1-based post-training); instruction following with IFEval prompt-level loose, quoting the variant by name; one unseen benchmark per axis from the Tülu 3 unseen list, read once, after the decision; long-context runs at the five HELMET lengths with a paired short-context regression run. Two coordinates that these reports do not fix must be fixed locally and recorded: the serving stack and precision, and whether batch-invariant kernels are enabled.

## Generalization lens

**(a) What increases breadth of measurement.** Multi-metric coverage over a fixed scenario list makes trade-offs visible instead of deferring them to separate reports: HELM measures 98 of 112 (scenario, metric) pairs and reports, for example, that TNLG v2 (530B) falls from 72.6% to 38.9% on NarrativeQA under robustness perturbations ([[helm]] Finding 4). Per-axis unseen benchmarks turn "the model improved" into a claim that can be false: Tülu 3's unseen average rises with the development average at both sizes, which is evidence for the pipeline and not only for the benchmarks ([[tulu-3-eval-regime]] Table 31). Multi-template evaluation converts format luck into a reported quantity ([[prompt-format-sensitivity-formatspread]]; [[multi-prompt-evaluation]]).

**(b) What causes narrowing.** Reading one suite during development narrows the model toward it; the Tülu 3 8B coding column (HumanEval 86.2 → 83.9 with BigCodeBench 11.5 → 7.4) and the MATH-to-DeepMind-Mathematics LaTeX formatting effect are two measured instances ([[tulu-3-eval-regime]] Table 31; [[tulu-3]] §7.4.1). Selecting benchmarks by convenience narrows the coverage map, leaving axes unmeasured. Tuning the harness — picking the formulation, normalization, template, or matcher variant that gives the best number — narrows the report itself; the size of what is available to tune is 11.0 points on one ARC-Challenge row ([[olmes]] Table 1), up to 8.5 points across the four IFEval variants ([[ifeval]] Table 3), and a median of 7.5 points across ten prompt formats ([[prompt-format-sensitivity-formatspread]] §4.2).

**(c) How to measure it for this stage.** Report the development-to-unseen gap per axis and its change over training stages. Report the format spread, not only the best format. Report signal-to-noise for the benchmarks used for small-scale decisions, and prefer high-SNR tasks or bits-per-byte scoring at small scale ([[signal-and-noise-eval]] §4.1, §5.3). For agents, report cost with accuracy and run the do-nothing grader check ([[holistic-agent-leaderboard]]; [[agentic-benchmark-checklist]]). Ch-51 supplies the interval machinery that turns these into go/no-go decisions.

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Comparing a number to a published number from another harness | The gap is a few points and stable across checkpoints | Re-run the baseline model in your own harness; compare only numbers produced by the same task id and settings ([[olmes]] Table 1) |
| Reporting a multiple-choice score without the formulation | Early-checkpoint curves sit at chance and then jump | Run both MCF and CF; plot both against training tokens ([[olmes]] Fig. 1) |
| Quoting one IFEval-style number without the variant | Two reports of the same model differ by 2.4 to 8.5 points | Name the metric: prompt-level or instruction-level, strict or loose ([[ifeval]] Table 3) |
| Matcher false negatives | Failures cluster in items whose answers have unusual formatting | Read 50 incorrect outputs; count answers that are present but unextracted ([[tulu-3-eval-regime]] §7.2) |
| Tuning the prompt template on the reported benchmark | A gain appears on one template and not on paraphrases | Evaluate 5-10 templates on a subsample; report AvgP and the spread ([[multi-prompt-evaluation]] §5) |
| Assuming temperature 0 means reproducible | Re-running the same eval moves the score by a fraction of a point | Re-run one task twice with different server load; compare per-item outputs ([[thinkingmachines-defeating-nondeterminism]]) |
| Treating an unseen benchmark as permanent | The dev-to-unseen gap shrinks over months with no change in method | Log each read of the unseen suite; retire a benchmark after it informs a decision ([[tulu-3]] §2.2) |
| One long-context number | High needle scores with weak downstream long-context behavior | Report per-category and per-length results; add a low-lexical-overlap test ([[helmet]] Fig. 3; [[nolima]] §4.4) |
| Comparing agents across different scaffolds | The ranking changes when the scaffold changes | Fix the scaffold, or evaluate each model under several and report the pairs with cost ([[holistic-agent-leaderboard]] §4.1) |
| Trusting an agent benchmark's grader | A trivial agent scores far above zero | Run a do-nothing agent and an enumerating agent through the grader ([[agentic-benchmark-checklist]] §5.2) |

## Check your understanding

1. OLMES reports the higher of the MCF and CF scores rather than fixing one formulation. Explain why fixing MCF would make a pre-training monitoring curve misleading, and what the reported quantity means once the maximum is taken.
2. The four IFEval numbers for GPT-4 span 76.89 to 85.37. Derive how many of the 541 prompts separate the prompt-level strict and loose numbers, and explain why the loose matcher can both reduce false negatives and create false positives.
3. Tülu 3 8B coding moves 86.2 → 83.9 on HumanEval and 11.5 → 7.4 on BigCodeBench across the pipeline, while at 70B BigCodeBench moves 12.2 → 21.6. What does the difference between the two sizes rule out as an explanation for the 8B result, and what does it not rule out?
4. A team reports a 2-point gain on a development benchmark from a data change, evaluated with one prompt template and temperature 0 on a shared endpoint. List the three measured sources of variation from this chapter that could each account for the gain, with the study and number that bounds each.
5. Why does MMLU-Pro's move from four to ten options increase the measured gap between two models, and under what assumption about the model's behavior does the argument hold?
6. A harness stores only the aggregate score per task. Name two questions from later chapters that cannot be answered without rerunning the evaluation, and state what the per-item record would have to contain to answer them.
7. HAL found that one scaffold pair differed by about 9x in cost for a two-point accuracy difference. Explain why this makes a scaffold a harness coordinate rather than an implementation detail, and what must be reported for an agent score to be comparable.
8. Construct a case in which a model's CPS improves while its MaxP falls, and describe the deployment for which that trade is the right one.

## Connections

- **ch-00 — What General Capability Means and How It Is Measured** (dependency): supplies the capability definition, the development-versus-unseen rationale, and the measurement-error sources that this chapter implements as a suite and a harness.
- **ch-36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split** (dependency): produces the checkpoint and the held-out split that this harness scores.
- **ch-46a — Lab: Small Agentic SFT-then-RL Run with a Generality Gate** (previous chapter): its generality gate is one instance of the suite design formalized here.
- **ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation** (next chapter): the audits that test whether a suite's scores survive perturbation, option reordering, and fresh items.
- **ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation**: owns the long-context suites and the effective-length protocols that §7 configures.
- **ch-51a — Evaluating Agent Generality and Reliability**: owns agent reliability metrics and held-out environment design, configured in §8.
- **ch-48 — Contamination Detection and Its Effect on Reported Scores**: the detection methods behind the decontamination report id stored in §6.
- **ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting**: the validation of the model-based matchers introduced in §3.
- **ch-50 — Slice Analysis, Forgetting Slices, and Failure Bucketing**: consumes the per-item records defined in §6.
- **ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions**: turns the numbers this harness produces into decisions with stated uncertainty.
- **ch-52 — Safety Evaluation, Over-Refusal, and Red-Teaming**: the safety axis of the coverage map, including the over-refusal side that a single refusal score hides.
- **ch-53 — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness**: builds the harness described here.

## Sources

- [[olmes]] — formulation (MCF/CF), the four CF normalizations and their per-task assignment, curated 5-shot prompts, sampling and input caps, and Table 1's spread of published ARC-Challenge numbers.
- [[helm]] — scenario taxonomy, multi-metric coverage (98 of 112 pairs), the 17.9% → 96.0% coverage change, and the adaptation-method finding for HellaSwag.
- [[tulu-3]] — the post-training report whose evaluation framework this chapter follows; §7.4 dev-versus-unseen analysis and the MATH formatting result.
- [[tulu-3-eval-regime]] — development and unseen suite tables, per-task settings, decontamination thresholds, and the Table 31 dev/unseen numbers.
- [[mmlu-pro]] — ten-option construction, discrimination against MMLU, and the 24-prompt sensitivity comparison.
- [[ifeval]] — verifiable-instruction construction and the strict/loose × prompt/instruction metric grid used as the matcher example.
- [[prompt-format-sensitivity-formatspread]] — format-spread definition and search procedure; median 7.5-point spread and ranking reversals.
- [[multi-prompt-evaluation]] — multi-template metrics (MaxP, AvgP, Sat, CPS), Kendall's W over templates, and single-edit effects.
- [[thinkingmachines-defeating-nondeterminism]] — batch invariance as the cause of temperature-0 nondeterminism, the 1,000-completion measurement, and the throughput cost of deterministic kernels.
- [[signal-and-noise-eval]] — signal, noise, SNR, decision accuracy, bits-per-byte scoring, and benchmark-size findings used for suite sizing.
- [[helmet]] — long-context protocol (lengths, demonstrations, judge), category correlations, and position-extension effects at inference.
- [[nolima]] — lexical overlap as a confound in needle tests and the 32K degradation numbers.
- [[context-length-alone-hurts]] — accuracy loss with input length under verified retrieval.
- [[chroma-context-rot]] — controlled length variation with fixed difficulty; abstention scored separately.
- [[lost-in-the-middle]] — evidence-position slicing and the U-shaped position curve.
- [[holistic-agent-leaderboard]] — scaffold, budget, and cost coordinates for agent evaluation; infrastructure drift and the τ-bench few-shot leak.
- [[agentic-benchmark-checklist]] — grader-validity checks (do-nothing and enumerating agents) and the audit of ten agentic benchmarks.

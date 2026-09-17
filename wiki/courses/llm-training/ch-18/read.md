<!-- chapter: ch-18
     track: synthetic
     kind: content
     title: The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix
     deps: [ch-00, ch-10a, ch-12]
     sources: [[self-instruct]], [[nemotron-4-synthetic]], [[nemotron-4-synthetic-recipe]], [[apigen]], [[openmathinstruct-2]], [[openmathinstruct-2-recipe]], [[tulu-1-how-far-can-camels-go]], [[sft-data-composition-dmt]], [[tulu-3]], [[flan]], [[transferability-of-llm-reasoning]], [[prolong]], [[apigen-mt]], [[west-of-n]], [[nathan-lambert-synthetic-data]]
     figures: figures/synth-loop.html
     revised: 2026-09 (generality revision)
-->

# Chapter 18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix

> **Core insight.** Synthetic-data pipelines for post-training can be described with one sequence of steps: generate, filter, deduplicate, verify, select, mix. A prerequisite comes before them: a written target distribution and an evaluation split that the pipeline never tunes on. The published evidence does not support treating any single step as always decisive. APIGen found that adding back samples that failed its execution or semantic checks lowered BFCL accuracy by 4.06 to 12.17 points, while OpenMathInstruct-2 found that up to 20% wrong math solutions cost little accuracy at 256K or more training pairs. Across instruction-tuning mixtures, no single dataset was best on every capability (Tülu 1), which makes the mix step a breadth decision.
>
> **Guideline.** When the target output has an exact executable form (tool calls, code), use execution checks before SFT, because unverified failures lowered BFCL accuracy, more for a 1B than a 7B student ([[apigen]] Fig. 5). When the target is a free-form reasoning trace and the data set is large, spend budget on more unique questions and a stronger teacher before stricter solution filtering, because filtering gave no gain there ([[openmathinstruct-2]] Tables 2-3, Fig. 6). In every case, fix the held-out evaluation suite before generating data, and measure capabilities that the synthetic data does not target, because targeted data raised development metrics that did not transfer to unseen ones ([[tulu-3]] Table 32).

## Corrections to the version you studied

Earlier line references (L…) point to the version at git commit 4a72e54.

1. "GPT-3 (text-davinci-001) prompted with 8 in-context examples (6 seed + 2 generated)" → Self-Instruct generated data with the vanilla GPT-3 "davinci" engine; text-davinci-001 (InstructGPT-001) was the comparison model. The 8 in-context instructions are 6 human-written and 2 model-generated ([[self-instruct]], arXiv:2212.10560 §2.2, §3, footnote 1).
2. "Output: ~252K raw candidates" and "ROUGE-L > 0.7 … Drops ~50% of raw instruction candidates" → the paper reports neither a raw count nor a rejection share. It reports 52,445 instructions and 82,439 instances after filtering (Table 1). "252" in the paper is the number of user-oriented evaluation instructions (§4.4). The excluded keywords are "image, picture, graph" (§2.2).
3. "Minimal [verify] … This is the paper's acknowledged weakness" → the paper measures quality instead: of 200 sampled instructions with one instance each, 92% describe a valid task, 58% have a correct output, and 54% have all fields valid (§3.3, Table 2). Regenerating outputs with InstructGPT-003 raised the share of top-rated responses on the 252 user tasks from 44.4% to 54.4% (§4.5, Fig. 7).
4. "Self-Instruct's diversity stats (Table 2) show root-verb entropy dropping across iterations without the ROUGE filter" → Table 2 is the quality review. §3.2 reports verb-noun structure (26,559 of 52,445 instructions parse into it) and ROUGE-L overlap with the seeds (Fig. 4). No run without the filter is reported.
5. "banning 'graph' wiped anything math-diagram-related in Self-Instruct's 2022 run" → no source supports this; removed.
6. Nemotron-4 "Iterated Nemotron checkpoints as teacher" → the first round of prompts and responses came from Mixtral-8x7B-Instruct-v0.1; later rounds came from 340B intermediate models ([[nemotron-4-synthetic]] §3.2.1, §3.2.4).
7. "the reward-model-as-selector is what lets 20K human examples govern a ~1.4M-example preference corpus" → about 20K human examples were used: 10K for SFT and 10K HelpSteer2 examples for the reward model and preference fine-tuning. Preference data were 160K (DPO) plus 300K (RPO); the ~1.46M total also counts 800K Code SFT and 200K General SFT samples (§3.2, §3.3).
8. Nemotron-4 "Light cross-prompt dedup … task-family partitioning" and "Format validity on code compilation" → the report mentions de-duplication and filtering only for the Genetic Instruct code data, whose fitness check is an LLM, not a compiler (§3.3.1). Synthetic dialogues are filtered by a reward-model score threshold (§3.2.2).
9. "highest-scoring response = chosen, lowest-scoring = rejected. Selection is the entire preference-pair construction step" → where ground truth or a verifier exists, the correct response is chosen and an incorrect one rejected. Otherwise an LLM judge (both response orders, consistent verdicts only) was used early and the reward model later, with Chat-Hard accuracy 0.87 vs 0.54. The reward model also filters for high-quality chosen responses (§3.2.3, §3.3.2).
10. The quotations "reward-model errors compound when the same scorer is reused across iterations" and "a small human anchor set (~20K) can support a much larger synthetic alignment corpus" → neither sentence is in the Nemotron-4 report; both were commentary in an earlier card version (card Verification: "Removed as unsupported").
11. APIGen "Sample (k=1..3) functions … prompt DeepSeek-Coder-V2 or GPT-4 … Diversity sampler upweights rare API categories" → sampling ranges are not reported. The generators were DeepSeek-V2-Chat, DeepSeek-Coder-33B-Inst, Mixtral-8x22B-Inst, and Mixtral-8x7B-Inst at temperature 0.7 ([[apigen]] §3.3, §4.2).
12. "execute the call in a 5-sec Python sandbox against the reference impl … GPT-4 judge" → Python functions run in a subprocess and REST APIs are called; no timeout is reported. The semantic checker is "another LLM" and is not named (§3.2, App. B.2).
13. "Combined rejection ~40% of raw" and "60K verified samples (from ~100K raw at 40% rejection)" → pass rates range from 34.42% (DeepSeek-Coder-33B-Inst) to 84.15% (DeepSeek-V2-Chat) (Table 1). The ~60,000 released samples come from Mixtral-8x22B-Inst and DeepSeek-V2-Chat only (§4.2).
14. APIGen "MinHash over (query, call) pairs" and "Balance by bucket at the end" → neither is reported. The four query styles (simple, multiple, parallel, parallel multiple) are used at generation time (§3.3).
15. "removing the semantic layer costs 6 points of BFCL-V1, removing execution costs 11 points, removing format costs 18 points" → the ablation adds failed data back: +fail-semantic −4.06 and +fail-execution −5.94 for xLAM-7B; −9.59 and −12.17 for xLAM-1B; there is no format arm (§5.2, Fig. 5).
16. "Single-stage SFT on Mistral-7B / Mixtral (xLAM models)" and "xLAM-7B ranks #1 on BFCL among <13B" → xLAM-7B (FC) was fine-tuned from DeepSeek-Coder-7B-instruct-v1.5 and ranked 6th (85.65); xLAM-1B (FC) from DeepSeek-Coder-1.3B ranked 24th (74.41) (§5.1, Table 2).
17. "APIGen's ~$8K teacher-API spend" and "stage 4 alone can exceed stage 1" → not reported.
18. OpenMathInstruct-2 "K=32 CoT solutions per problem at T=1.0, top-p=0.95. 600K problems … from MATH + GSM8K (7.5K each)" → new questions get 32 solutions at temperature 0.7; 1.0 and 0.95 are the ablation settings. The data have 607.3K unique questions, 592K of them synthesized; Table 5 lists 7.4K original questions per source ([[openmathinstruct-2]] §2.2, §3, Table 5).
19. "SymPy symbolic equivalence … Residual false-positive rate ~7% on human-audited sample" → synthesized questions have no gold answer; the majority answer of the 32 solutions is used, with a minimum-vote threshold of 0. Evaluation uses a GPT-4o answer judge. SymPy and a 7% audit rate do not appear (§3, App. C.1, §4).
20. "Regex-extract boxed final answer … Near-duplicate suppression within a problem's accepted solutions" → post-processing is the App. A.2 list (drop multiple \boxed, truncate after the first \boxed sentence, drop solutions over 1024 tokens or under 200 characters, and others); no within-question deduplication is reported.
21. "Single-stage SFT on Llama-3.1-{1.5B, 8B, 70B}. No real-data mixing" → Llama3.1-8B-Base was trained on the full 14M pairs and Llama3.1-70B-Base on a 5M subset; there is no 1.5B model. The data include solutions for the original GSM8K and MATH training questions (§4, Table 5).
22. "avg 23 solutions per augmented problem" → 11.05M pairs / 592.7K new questions = 18.6 (derived from Table 5).
23. "teacher sampling alone consumed ~650K H100-hours" → not reported.
24. "a 14M-example corpus whose downstream gains plateau around 1M" → the paper reports "no signs of saturation" at 14M (§4, Fig. 1).
25. "OpenMathInstruct-2's gains over OMI-1 are roughly equal parts bigger teacher and tighter SymPy verification" → no such decomposition exists. The teacher ablation gives 37.9 vs 30.1 at matched coverage (Table 2), judge and reward-model filters give no gain (Table 3), and up to 20% wrong solutions cost little at ≥ 256K pairs (Fig. 5).
26. "rerun verification with a stronger judge (West-of-N)" → West-of-N builds synthetic preference pairs from the best and worst of N sampled responses under an existing reward model ([[west-of-n]], Abstract).
27. "Nathan Lambert … 'synthetic data can do almost all of the work' given a strong base model plus robust verification" → the June 2024 post says "I think synthetic data can do almost all of the work" about fine-tuning a strong open-weights model; the verification condition is not part of that sentence (interconnects.ai/p/frontiers-in-synthetic-data, opening section; [[nathan-lambert-synthetic-data]]). It is a practitioner opinion without a controlled experiment. The "scarce resource is verifiable prompts" wording was not found in a checked post and is not used here.
28. "Long context (LongAlign, ProLong): stage 1 = document chunking … stage 4 = needle-in-a-haystack check" → ProLong tested synthetic long instruction data at SFT and found short UltraChat data alone scored highest: 55.7 at 0% synthetic long tokens vs 43.3 at 50% ([[prolong]] §5, Table 8).
29. The "verification is the bottleneck" argument of old §6 ("APIGen's per-layer ablation showing 3 of 3 layers", "OMI-2's 14M solutions" as evidence) → rebuilt in §5 below on the reported ablations, which show that the value of verification depends on the output type, student size, data scale, and training objective.
30. Cross-references → iteration and self-improvement are ch-45 (not ch-28); judges are ch-49 and ch-42 (not ch-26); IFD, LESS, and Superfiltering are ch-22 (not ch-25); model collapse is ch-23 (not ch-14); textbook-style synthesis is ch-21 (not ch-22). The old §7 track map is replaced: ch-20 distillation, ch-21 taxonomy and textbook synthesis, ch-22 selection, ch-23 collapse and verification, ch-24 to ch-28 reasoning traces, multi-turn, tool calling, agentic trajectories, and long context; multi-skill mixtures are ch-30b (not ch-27). ch-02 to ch-08 are training foundations, not base-model capability. ch-17 is the filter and mixture ablation lab; curation pipelines are ch-10, ch-10a, and ch-12.

## Why this chapter matters for a general-purpose model

In some open post-training pipelines, most of the data is generated by other models: Nemotron-4-340B-Instruct used over 98% synthetic SFT and preference data ([[nemotron-4-synthetic]] §3.2). The course goal is a model that performs well on tasks that were not targeted. Synthetic data works against that goal in a specific way: a pipeline produces the most data where it can check the data most cheaply (math answers, executable code, tool calls), and the training mixture then over-represents those skills.

The pattern in this chapter applies at every post-training stage that consumes generated data: SFT data (this chapter and ch-24 to ch-28), preference pairs (ch-39), and RL prompt pools (ch-16). It also applies to synthetic pretraining and mid-training data (ch-21, ch-29a). The chapter gives one vocabulary for the steps, the evidence for each step, and a check for each failure mode.

## §1 The loop and its prerequisite: a target distribution and a held-out split

**Definition.** A synthetic-data pipeline is a sequence of operations that turns a small amount of trusted input (seed tasks, APIs, documents, questions) into training examples. This course uses seven named steps: step 0 states what the data is for and how success is measured; steps 1-6 produce the data.

**The problem it addresses.** Without step 0, each later step is tuned against whatever metric the team looks at during development. Tülu 3 measured this directly: its development and unseen suites cover the same skills with different benchmarks, and removing its Persona instruction-following data lowered IFEval (development) from 72.8 to 53.6 while IFEval-OOD (unseen) moved from 17.6 to 18.0 ([[tulu-3]] arXiv:2411.15124v5 §7.4.1, Table 32). The data raised the development metric by 19.2 points and did not raise the unseen one. Result (single study).

| Step | Question it answers | Mechanisms in the sources | Check that the step works |
|---|---|---|---|
| 0 Target and held-out split | Which capabilities, and measured on what the pipeline never sees? | FLAN held-out task clusters; Tülu 3 development vs unseen suites; decontamination | Unseen-suite score reported for every data decision |
| 1 Generate | What does the generator produce? | Few-shot bootstrapping; question augmentation; API-conditioned generation; multi-model responses | Diversity statistics of the raw output |
| 2 Filter | What is dropped for format or surface reasons? | JSON parse, keyword and length rules, truncation rules | Acceptance rate per task family |
| 3 Deduplicate | What is dropped as redundant? | ROUGE-L threshold, MinHash (ch-12) | Nearest-neighbour similarity distribution |
| 4 Verify | What is dropped as wrong? | Execution, ground-truth match, majority vote, LLM or reward-model judge | False-accept and false-reject rate on a labelled sample |
| 5 Select | Which survivors are kept, in what amount? | Reward-model filtering, generator choice, question-count budgets (ch-22) | Held-out gain per selected subset |
| 6 Mix | How does the set combine with other data and stages? | Concatenation, staged SFT, replay fractions | Score on untargeted capabilities before and after |

The interactive figure [figures/synth-loop.html](figures/synth-loop.html) lets the reader click each step to see its failure symptom, its check, and what each of the four case-study pipelines does at that step, and to compute APIGen's per-stage yield for any generator.

**Held-out tasks as the measure of generality.** FLAN defines an unseen task by task cluster: a dataset counts as unseen only if no dataset from its cluster was used in instruction tuning ([[flan]] arXiv:2109.01652 §2.2). With three clusters held out, the average held-out score rose from 49.9 with one training cluster (11 datasets) to 63.5 with seven clusters (39 datasets) (§4.1, Fig. 6). The same study found that instruction tuning lowered held-out performance for models of 8B parameters and smaller (§4.2, Fig. 7). Result (single study, one model family). Self-Instruct evaluated on 119 unseen SuperNI tasks and on 252 new user-oriented instructions written by the authors ([[self-instruct]] §4.3, §4.4). Tülu 3 did not examine scores on its unseen suite while developing its models, so that the suite could measure overfitting to development evaluations ([[tulu-3]] §2.2).

**Decontamination is part of step 0.** A held-out split is held out only if the generated data does not contain its items. Tülu 3 marks a test instance as matched when more than 50% of its tokens lie in 8-grams shared with one training instance, and treats a training set as contaminated when it matches more than 2% of any evaluation's instances ([[tulu-3]] §3.2).

Worked example. A 20-token test prompt shares 8-grams with one training prompt, and those 8-grams cover tokens 1-12. Coverage is 12/20 = 60% > 50%, so the test instance is matched. GSM8K test has 1,319 problems; 2% is 26.38, so a training set that matches 27 or more of them is contaminated under this rule.

N-gram rules miss paraphrases. Tülu 3 notes that embedding methods can in principle find paraphrase contamination (citing Yang et al., 2023) but that it could not separate paraphrase from topical similarity (§3.2). OpenMathInstruct-2 used embeddings to retrieve the top-5 most similar test questions for each synthesized question and then asked Llama3.1-405B-Instruct whether each pair was a paraphrase, in both orders; this removed 569K → 519K new questions ([[openmathinstruct-2]] §3.1, App. C.2). Contamination detection is taught in ch-48.

## §2 Four pipelines placed on the loop

Each cell states what the source reports, with its locus. "Not reported" means the source was checked and does not describe the step.

| Step | Self-Instruct (arXiv 2022-12) | Nemotron-4 340B (arXiv 2024-06) | APIGen (arXiv 2024-06) | OpenMathInstruct-2 (arXiv 2024-10) |
|---|---|---|---|---|
| 0 Held-out | 119 SuperNI unseen tasks; 252 new user instructions (§4.3-4.4) | MT-Bench, IFEval, MMLU, and others per stage (Table 6); no unseen split | BFCL only; no held-out-API split (§5) | 1K MATH train split held out for ablations; LLM paraphrase decontamination (§2.2, §3.1) |
| 1 Generate | GPT-3 davinci; 175 seed tasks; 8 in-context instructions (6 human, 2 generated) (§2.2) | Mixtral-8x7B-Instruct-v0.1 prompts from 3K topics and keyword lists; responses from intermediate models (§3.2.1, §3.2.4) | 4 open generators, 40,000 target samples each, T = 0.7; 3,673 APIs (§4.1-4.2) | Llama3.1-405B-Instruct; 5-shot "similar question" augmentation; 32 solutions per new question at T = 0.7 (§3) |
| 2 Filter | Length heuristics; output repeating input; keywords "image, picture, graph" (§2.2) | Reward-model threshold on dialogues (§3.2.2) | Format check: JSON with query and answer; calls must use given functions (§3.2) | Post-processing rules, e.g. > 1024 tokens dropped (App. A.2) |
| 3 Deduplicate | New instruction kept only if ROUGE-L < 0.7 with all existing ones (§2.2) | De-duplication of Genetic Instruct code data; method not described (§3.3.1) | Not reported | Not reported for solutions; question diversity studied instead (§2.2.4) |
| 4 Verify | None beyond filters; quality audited: 54% all fields valid (§3.3) | Ground truth or Python verifier where available; LLM judge, then reward-model judge (§3.2.3) | Execution check, then LLM semantic check (§3.2) | Majority vote over 32 solutions, minimum-vote threshold 0 (§3, App. C.1) |
| 5 Select | All survivors (§2.2) | Reward model selects high-quality chosen responses (§3.3.2) | Release uses the two strongest generators (§4.2) | Question count prioritized over solution filtering (§2.2.3-2.2.4) |
| 6 Mix | Whole set, 2 epochs (§4.1) | Code SFT ~800K → General SFT 200K with 2% code replay → DPO 160K → RPO 300K ×3 (§3.3) | Mixed with 8,000 relevance-detection examples (App. B.3) | Math-only SFT; 8B on 14M pairs, 70B on 5M (§4) |
| Output | 52,445 instructions, 82,439 instances (Table 1) | Over 98% synthetic alignment data (§3.2) | ~60,000 released samples (§4.2) | 13.97M pairs, 607.3K questions (Table 5) |

Three observations follow from the table. First, only one of the four (OpenMathInstruct-2) reports a held-out split used for data decisions, and it is a split of the target domain, not of other capabilities. Second, "verify" means different operations: execution (APIGen), agreement among samples of the same teacher (OpenMathInstruct-2), and a learned scorer (Nemotron-4). They differ in what errors they can catch. Third, evaluation breadth follows the verifier: the pipelines with the strongest checks (APIGen, OpenMathInstruct-2) evaluate only the domain they check.

## §3 Generate and filter: yield per stage

**Definition.** The yield of a pipeline is the share of generated samples that survive every check. For a sequence of checks it is the product of the conditional pass rates.

**Formula.**

    yield = N_verified / N_generated = Π_s (1 − f_s)

- N_generated: samples requested from the generator.
- N_verified: samples that pass all checks.
- f_s: the fraction of samples reaching check s that fail it (conditional failure rate).

**Worked example ([[apigen]] Table 1).** DeepSeek-Coder-33B-Inst produced 40,000 samples: 4,311 failed format, 15,496 failed execution, 6,424 failed the semantic check, 13,769 passed (sum 40,000).

1. Format: f₁ = 4,311 / 40,000 = 10.8%; 35,689 remain.
2. Execution: f₂ = 15,496 / 35,689 = 43.4%; 20,193 remain.
3. Semantic: f₃ = 6,424 / 20,193 = 31.8%; 13,769 remain.
4. Yield = 0.892 × 0.566 × 0.682 = 0.344, matching the printed 34.42%.

DeepSeek-V2-Chat under the same checks: 817 format, 3,359 execution, and 2,165 semantic failures, 33,659 passed, a yield of 84.15%. To obtain 10,000 verified samples, the first generator needs 10,000 / 0.3442 = 29,053 generations and the second 10,000 / 0.8415 = 11,884. Generator choice changes both cost and which check does the work: for the weaker generator most rejections come from execution.

**Evidence on generator quality.** Self-Instruct's audited 54% fully valid rate came from a base model (GPT-3 davinci) ([[self-instruct]] Table 2). When Tülu 1 trained LLaMA 13B on that dataset, MMLU fell from 42.3 to 30.4 and the six-evaluation average was 21.8, against 45.2 for the Human+GPT mixture ([[tulu-1-how-far-can-camels-go]] Table 3). The authors attribute the result to the weaker generator (Interpretation, §5.1).

**Conditions and limits.** Acceptance rates are properties of a generator-checker pair. A higher yield from a stronger generator is not evidence that the surviving data is more diverse; APIGen reports no diversity statistics per generator.

**Implication.** Log per-stage failure counts per generator and per task family. A task family whose yield is lower than the other families' is either hard for the generator or mis-specified in the prompt, and both reduce its share of the final data unless step 5 corrects for it.

## §4 Deduplicate: near-duplicates and question diversity

**Definition.** Deduplication removes a candidate whose similarity to an already accepted item exceeds a threshold. Exact and MinHash methods are taught in ch-12; this section covers the instruction-level variant used by Self-Instruct.

**Formula (ROUGE-L, F-measure form).** ROUGE-L scores the longest common subsequence (LCS) of tokens. The Self-Instruct paper does not print the formula; the standard F-measure with equal weights is:

    P = LCS(a, b) / |a|,   R = LCS(a, b) / |b|,   ROUGE-L = 2PR / (P + R)

- a: the candidate instruction's tokens; b: an accepted instruction's tokens; |·| is token count.
- LCS: length of the longest subsequence of tokens that appears in both, in order, not necessarily adjacent.

**Worked example.** a = "Write a short poem about the sea" (7 tokens); b = "Write a poem about the ocean" (6 tokens). The LCS is "Write a poem about the" = 5. P = 5/7 = 0.714, R = 5/6 = 0.833, ROUGE-L = 2 × 0.595 / 1.547 = 0.769. Because 0.769 ≥ 0.7, Self-Instruct would not add the candidate ([[self-instruct]] §2.2). A candidate "Compose verses describing waves at night" shares no tokens with b and would be added, though it requests the same kind of output. Surface-overlap thresholds therefore bound lexical repetition, not task repetition.

**Evidence that diversity matters.** OpenMathInstruct-2 held the pair count fixed at 256K and varied unique questions: 1K questions scored more than 10 points below 6.5K questions on MATH validation ([[openmathinstruct-2]] §2.2.4, Fig. 6). Result (single study, Llama3.1-8B-Base, math only). FLAN's cluster ablation (§1) is the task-level version of the same observation.

**Implication.** Measure diversity at the unit that generalization is measured at: unique questions or task types, not token n-grams.

## §5 Verify: what the evidence says about its value

**Definition.** A verifier labels a generated sample as correct or incorrect using information beyond the sample's surface form: an execution result, a reference answer, agreement among independent samples, or a learned scorer.

**The measurable problem.** Wrong samples used as SFT targets are trained on with the same cross-entropy weight as correct ones. The question is how much accuracy they cost, and whether removing them is worth the verifier's false rejections and compute.

**Evidence, by setting.**

1. **Tool calls, small and mid-size students.** Adding samples that failed APIGen's checks back into training lowered BFCL overall accuracy by 4.06 (+fail semantic) and 5.94 (+fail execution) points for xLAM-7B, and by 9.59 and 12.17 for xLAM-1B ([[apigen]] §5.2, Fig. 5). The paper does not state whether the execution arm also contains the semantic failures. Result (single study).
2. **Math reasoning traces, 8B student.** Replacing 10%, 20%, 40%, or 80% of solutions with wrong-answer solutions, at 64K to 1024K pairs, caused little to no degradation up to 20% at ≥ 256K pairs ([[openmathinstruct-2]] §2.2.3, Fig. 5). Removing solutions flagged by two Llama3.1-405B judge prompts or by Nemotron-4-340B-Reward removed 6% to 12% of a 128K set and gave 43.0 to 43.8 accuracy versus 43.6 ± 1.7 unfiltered (Table 3). About 60% of 20 manually checked flagged solutions were in fact incorrect (footnote 4). Result (single study). The authors conclude that imprecise filtering is acceptable for SFT in this setting.
3. **Generator strength in the same math setting.** At matched coverage, data from Llama3.1-405B-Instruct gave 37.9 ± 0.6 and data from Llama3.1-8B-Base gave 30.1 ± 0.6 (Table 2). The authors' preliminary analysis links the gap to more solutions with a correct answer and incorrect reasoning from the weaker model (§2.2.2). A final-answer check does not detect that error type.
4. **Output quality in instruction data.** Replacing Self-Instruct outputs with InstructGPT-003 outputs raised top-rated responses from 44.4% to 54.4% ([[self-instruct]] §4.5, Fig. 7). This regenerates outputs rather than filtering them.
5. **Preference labels.** On RewardBench Chat-Hard, Nemotron-4-340B-Reward as judge scored 0.87 against 0.54 for the LLM judge, and the team switched to the reward model for later preference data ([[nemotron-4-synthetic]] §3.2.3). For preference data the label is the training signal, so label accuracy is not a filter choice but the content of the data.

**Majority vote as a verifier without gold answers.** OpenMathInstruct-2 treats the most common final answer among 32 sampled solutions as the reference ([[openmathinstruct-2]] §3). Worked example: of 32 solutions, 14 answer "12", 10 answer "15", and 8 give other answers. The reference is "12" with 14 votes; the 14 solutions ending in "12" are kept. A minimum-vote threshold of 16 would discard the question. The paper compared thresholds 0, 8, 16, and 24, which kept 381K, 339K, 254K, and 160K pairs with accuracy 50.1, 49.2, 44.4, and 42.0; data size was not matched, so the drop mixes stricter agreement with less data (App. C.1, Table 9). A majority vote shares the generator's errors: if most samples make the same mistake, the mistake becomes the reference.

**Conditions and limits (Interpretation, this course).** The two results in items 1 and 2 are consistent once their settings are separated. APIGen's failures are malformed or non-executing calls, the target output is an exact string, and the students are 1.3B and 7B. OpenMathInstruct-2's injected errors are wrong final answers inside otherwise well-formed reasoning, the data set has at least 256K pairs, and the metric is final-answer accuracy. Neither paper measures the effect of wrong data on untargeted capabilities. Open question: how the tolerance to wrong SFT targets changes with student size, error type, and training stage (SFT versus preference optimization versus RL, where a wrong label changes the sign of the update).

**Implication.** Measure the verifier before trusting it: sample accepted and rejected items, label them, and report the false-accept and false-reject rates. When a verifier is expensive, run a noise-injection ablation like OpenMathInstruct-2's on a subset first; the answer determines whether the verifier is worth its cost.

## §6 Select: which survivors are kept

**Definition.** Selection chooses a subset of verified samples or sources according to an explicit criterion. Methods based on quality scores, diversity, and gradients are taught in ch-22.

**Mechanism in the four pipelines.**

1. Nemotron-4 uses the reward model to keep preference examples with high-quality chosen responses when ground truth is not available, and applies a "less harsh" filter for the 300K RPO set ([[nemotron-4-synthetic]] §3.3.2).
2. APIGen selects at the generator level: the released ~60,000 samples come only from the two generators with the highest pass rates ([[apigen]] §4.2, Table 1).
3. OpenMathInstruct-2 selects at the question level: more unique questions instead of more solutions per question, and a 5M subset for the 70B model because of compute ([[openmathinstruct-2]] §2.2.4, §4).
4. Self-Instruct keeps every survivor; its data-size curve on the 252 user tasks almost plateaus after 16K instructions ([[self-instruct]] §4.5, Fig. 7).

**Implication.** Selection by generator or by score removes whole regions of the distribution when scores correlate with task type. Check the task-family histogram before and after selection.

## §7 Mix: a breadth decision

**Definition.** Mixing decides which data sets are combined, in what amounts, and in which training stage.

**The measurable problem.** Each skill-targeted data set changes several capabilities at once, in both directions.

**Evidence.**

1. **No single data set is best.** With one recipe on LLaMA 13B, the Human+GPT mixture had the best average (45.2) but not the best score on every evaluation. ShareGPT had the highest AlpacaEval win rate (70.5%) and lowered TyDiQA from 43.2 to 30.5; 6 of 12 single data sets lowered GSM and 8 of 12 lowered TyDiQA below the base model ([[tulu-1-how-far-can-camels-go]] §5.1, Table 3). Result (single study).
2. **Ablations change several skills.** Removing Tülu 3's math data lowered GSM8K from 76.2 to 64.1 and MATH from 31.5 to 23.5, and also lowered the unseen DeepMind Mathematics score from 32.3 to 23.3 and BigCodeBench from 11.5 to 8.8 ([[tulu-3]] Tables 10, 32). Removing safety data lowered the safety average from 93.1 to 74.7 with most other skills within about 2 points (Table 10). Removing WildChat lowered AlpacaEval 2 from 12.4 to 7.5 (Table 10).
3. **Amount and stage matter.** For LLaMA models trained on GSM8K RFT, Code Alpaca, and ShareGPT, each ability scored higher with the mixed sources than with its individual source at small data amounts (1/256) and lower at full size ([[sft-data-composition-dmt]] §3.3, Fig. 3). Training specialized data first and general data last lost math accuracy: at 7B, GSM8K was 47.53 for multi-task training and 32.60 for mixed sequential training (Table 1). Adding back 1/256 of the specialized data in the final stage (DMT) gave GSM8K 41.92, HumanEval 17.68, and MT-Bench 6.08 at 7B. At k = 1/256 that is 430 GSM8K RFT and 78 Code Alpaca samples added to 86,060 ShareGPT conversations (App. A, Table 2). Result (single study, single runs).
4. **Staging in a production report.** Nemotron-4 found that a single SFT stage mixing all behaviours produced conflicts, most strongly for coding, and that reweighting did not fix them; it used Code SFT then General SFT with 2% of the code samples included ([[nemotron-4-synthetic]] §3.3.1). General SFT raised MMLU from 72.2 to 78.3 and lowered HumanEval from 70.7 to 66.5 (Table 6). The report gives no single-stage numbers.

**Conditions and limits.** Items 1-3 use SFT of 7B-70B models on 2023-2024 data sets; item 4 is a 340B model without a controlled comparison. Multi-skill mixture design, including agentic and long-context shares, is taught in ch-30b.

**Implication.** Treat the mixture as an experiment with an untargeted-capability report: for every added or removed source, record the change on each skill's development and unseen benchmark.

## §8 Verifier coverage bias

**Definition.** Verifier coverage bias is the skew in a training mixture that arises because pipelines produce more verified data in domains with cheap checks (math answers, code execution, tool calls) than in domains without them (open-ended writing, advice, multi-document synthesis).

**Mechanism.** Steps 1 and 3 are reported results; steps 2 and 4 are this course's interpretation.

1. Verified yield is higher, and cheaper to measure, where a checker exists (§3, §5).
2. Teams scale the pipelines that report clean yields, so these domains grow fastest.
3. Evaluation often follows the checker: APIGen evaluates only BFCL, OpenMathInstruct-2 only math ([[apigen]] §5; [[openmathinstruct-2]] §4).
4. Losses in untargeted domains are therefore neither caused nor detected by the pipeline's own measurements.

**Worked example ([[transferability-of-llm-reasoning]] arXiv:2507.00432v2 Table 1).** Qwen3-14B-Base was trained on one math data set in two ways: SFT on Qwen3-32B chain-of-thought traces selected by rejection sampling, and RL using only the answer labels (§2.2).

| Model | Math average | Other reasoning average | Non-reasoning average |
|---|---|---|---|
| Qwen3-14B-Base | 27.7 | 30.2 | 45.7 |
| SFT on math traces (think) | 49.8 (+22.1) | 45.3 (+15.1) | 21.1 (−24.6) |
| RL on math answers | 53.8 (+26.1) | 60.0 (+29.8) | 53.2 (+7.5) |

The non-reasoning group is CoQA, MC-TACO, IFEval, and HalluEval; IFEval fell from 69.2 to 42.3 after SFT. The authors attribute the difference to larger representation and output-distribution shift under SFT (Interpretation, §3-4). Result (single study, one base model).

**Conditions and limits.** The comparison mixes two differences, the objective and the source of the tokens (teacher traces versus the model's own samples); ch-38a separates them. Tülu 1 reports the SFT side of the same pattern for 2023 data sets (§7 item 1).

**Implication.** When a pipeline scales a verifiable domain, add an evaluation group outside that domain to the go/no-go gate, and budget non-verifiable data (human-written, judge-scored, or real user prompts) explicitly in step 6.

## §9 Porting the loop to long-context and agentic data

**Long context.** The loop applies, but the evidence for synthetic long instruction data is mixed. ProLong mixed synthetic long QA, RAG, and summarization data generated by Llama-3-8B-Instruct into SFT at 0/1/3/10/50% of tokens and measured 55.7/54.1/53.5/53.9/43.3 on its long-context average; a 70B generator did not change the conclusion ([[prolong]] §5, Table 8, App. B.5). The authors' hypotheses are that earlier positive results had less long-context continued training or larger short instruction sets (Interpretation). Long-context synthesis is taught in ch-28 and ch-29a.

**Agentic trajectories.** APIGen-MT moves verification before generation of the conversation: it first validates a task blueprint with execution, policy unit tests, and an LLM committee, then simulates the user-agent conversation, and keeps a trajectory only if the final environment state and outputs match the blueprint ([[apigen-mt]] §4.1-4.2). Agentic feedback during blueprint generation raised configuration success from 28% to 70%, and 67% of simulated trajectories succeeded (Fig. 4). Agentic data synthesis is taught in ch-27, ch-29c, and ch-29d.

## Negative samples and negative feedback

This section uses the four meanings of "negative" from the course standard: negative marginal value, negative as content, negative as conditioning, and negative as gradient.

1. **Where negatives come from.** Format, execution, and semantic check failures (APIGen); minority answers under majority vote (OpenMathInstruct-2); rejected responses chosen by ground truth, a Python verifier, an LLM judge, or a reward model (Nemotron-4); refusal and "impossible request" prompts (Nemotron-4, APIGen). False-negative rates are not reported by any of the four; OpenMathInstruct-2's check of 20 flagged solutions found about 60% truly incorrect, so about 40% of that small sample were false rejections ([[openmathinstruct-2]] footnote 4).
2. **What the pipelines do with them.**
   - Negative marginal value, discarded: APIGen check failures (§3.2) and filtered Self-Instruct generations.
   - Negative as content: 8,000 APIGen relevance-detection targets that answer with an empty call or a refusal when the tools cannot answer ([[apigen]] App. B.3); Nemotron-4 refusal responses for tasks the model cannot perform ([[nemotron-4-synthetic]] §3.2.5). These are ordinary cross-entropy targets.
   - Negative as conditioning: none of the four pipelines uses it (ch-31a).
   - Negative as gradient: Nemotron-4's DPO and RPO rejected responses (§3.3.2).
3. **Mechanism for negatives used as gradient.** For a softmax over logits z, the gradient of the log-probability of token y is

       ∂ log p_y / ∂ z_j = 1[j = y] − p_j

   - p_j: probability of token j; 1[j = y]: 1 when j is the token being scored, else 0.

   Decreasing log p_y moves the logits by −(1[j = y] − p_j), so every other token j gains in proportion to its current probability p_j. Worked example: p = (0.7, 0.2, 0.1) and the rejected token is the third. The logit update direction is (0.7, 0.2, −0.9): per unit step, the most likely token's logit rises by 0.7 and the second token's by 0.2, so most of the probability removed from the third token moves to the first. Pushing down an already unlikely sample therefore raises the most likely alternative, which may itself be wrong. Derivations and sequence-level effects are in ch-43a.
4. **Evidence.** Discarding negatives helped in APIGen (−4.06 to −12.17 BFCL points when they were added back, Fig. 5) and did not help in OpenMathInstruct-2 (Table 3). For negatives as gradient, Nemotron-4 observed that under DPO the likelihoods of both chosen and rejected responses fell, with the gap increasing, and that longer training overfit; DPO lowered MT-Bench from 7.99 to 7.90 ([[nemotron-4-synthetic]] §3.3.2, Table 6). No source in this chapter measures what share of an improvement comes from negatives; this chapter therefore makes no claim that negatives are the main driver of any gain.
5. **Controls.** Nemotron-4 added a weighted SFT loss on chosen responses (an anchoring positive NLL term) and used RPO, whose target is the reward gap, so that a high-quality rejected response is pushed down less (§3.3.2). West-of-N pairs the best and worst of N samples ([[west-of-n]]); a worst-of-N response is often an easy negative far from the chosen one, which gives a weak signal about near-miss errors (Interpretation). Uncertain labels (a narrow majority vote, a judge disagreement across response orders) can be excluded instead of being used as rejected responses; Nemotron-4 kept LLM-judge pairs only when both orders agreed (§3.2.3).
6. **Diagnostics.** Log chosen and rejected log-probabilities separately during preference training; log per-check failure counts per generator; report false-accept and false-reject rates on a labelled sample; report the refusal rate on benign prompts after adding refusal content.
7. **Effect on generality.** Refusal content can over-generalize: Tülu 1 reports 0.1% toxic generations on ToxiGen for Tülu 13B, with the authors hypothesizing overfitting to refusal behaviour (Interpretation) ([[tulu-1-how-far-can-camels-go]] §5.3); Tülu 3 found contrastive prompts helpful against over-refusal ([[tulu-3]] §4.2). In Nemotron-4's DPO training, an improvement in one metric (for example MT-Bench) usually came with a degradation in another (for example 0-shot MMLU) (§3.3.2). Over-refusal evaluation is taught in ch-52.

## Recipe

Rows quote the chapter-relevant settings from the cards and primary sources. "SFT (data)" marks a data-generation setting.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Self-Instruct data (GPT-3 davinci generator) | 175B | SFT (data) | seed tasks; in-context instructions per step | 175; 8 (6 human-written, 2 generated) | arXiv:2212.10560 §2.2 | verified 2026-09-15 | no ablation reported |
| Self-Instruct data | — | SFT (data) | diversity filter | add instruction only if ROUGE-L < 0.7 with every existing instruction | arXiv:2212.10560 §2.2 | verified 2026-09-15 | no ablation reported |
| GPT3-Self-Inst | 175B | SFT | epochs; prompt loss weight | 2; 0 (other OpenAI API defaults) | arXiv:2212.10560 §4.1, App. A.3 | verified 2026-09-15 | no ablation reported |
| APIGen generators (4 open models) | — | SFT (data) | temperature; target samples per generator | 0.7; 40,000 | arXiv:2406.18518 §4.2 | verified 2026-09-14 | no ablation reported |
| xLAM-7B (FC); xLAM-1B (FC) | 7B; 1.3B | SFT | relevance-detection examples; peak LR; epochs | 8,000; 5e-6 cosine, 50 warmup steps; 4 | arXiv:2406.18518 App. B.3 | verified 2026-09-14 | no ablation reported |
| OpenMathInstruct-2 (Llama3.1-405B-Instruct teacher) | 405B | distill-SFT (data) | solutions per new question; temperature; answer rule | 32; 0.7; majority vote, minimum-vote threshold 0 | arXiv:2410.01560v2 §3; App. C.1 | verified 2026-09-14 | Table 9: thresholds 0/8/16/24 → 50.1/49.2/44.4/42.0 (data size not matched) |
| OpenMathInstruct-2 | — | distill-SFT (data) | decontamination | top-5 embedding neighbours + 405B paraphrase check in both orders; 569K → 519K questions | arXiv:2410.01560v2 §3.1, App. C.2 | verified 2026-09-14 | Table 10: examples missed by n-gram matching |
| OpenMath2-Llama3.1-8B | 8B | SFT | data; LR; epochs; batch (unit not printed) | 13.97M pairs; constant 2e-5; 2; 512 | arXiv:2410.01560v2 §4 | verified 2026-09-14 | Fig. 1: gains at 1M/2M/5M/14M, no saturation |
| Nemotron-4-340B-Instruct | 340B | SFT | Code SFT: samples; epochs; LR; global batch | ~800K; 1; constant 3e-7; 128 | arXiv:2406.11704v2 §3.3.1 | verified 2026-09-14 | Table 6: HumanEval 57.3 → 70.7 |
| Nemotron-4-340B-Instruct | 340B | SFT | General SFT: samples; code replay; epochs | 200K; 2% of Code SFT samples; 3 | arXiv:2406.11704v2 §3.3.1 | verified 2026-09-14 | replay: no ablation reported |
| Nemotron-4-340B-Instruct | 340B | preference | DPO pairs; RPO pairs; RPO iterations | 160K; 300K; 3 | arXiv:2406.11704v2 §3.3.2 | verified 2026-09-14 | Table 6 per-stage metrics |
| Tülu 7B, 13B (Human+GPT mix) | 7B, 13B | SFT | mixing; epochs; peak LR | 7 data sets concatenated, no sampling weights (490,694 instances, derived); 2; 2e-5 | arXiv:2306.04751v2 §3.3, Table 1, App. D | verified 2026-09-14 | Table 3: mix average 45.2 vs best single set 42.0 |
| LLaMA 7B/13B/33B (DMT) | 7B-33B | SFT | stage-2 specialized fraction k; epochs; peak LR | 1/256; 3; 2e-5, 3% warmup | arXiv:2310.05492v4 §3.5, App. C | verified 2026-09-15 | Table 1: DMT vs three other strategies, single runs |
| LLaMA 7B/13B/33B (all runs) | 7B-33B | SFT | batch size (unit not printed) | 16 (§3.1); 128 (App. C) | arXiv:2310.05492v4 §3.1, App. C | conflict | the paper gives both values; no released config checked |
| Tülu 3 SFT mix | 8B, 70B | eval-gate | decontamination rule | 8-gram match; test instance matched if > 50% of tokens matched; training set contaminated if > 2% of an eval's instances match | arXiv:2411.15124v5 §3.2 | verified 2026-09-15 | no ablation of the thresholds reported; Fig. 3: decontamination rounds caused "small drops" |
| FLAN 137B | 137B | SFT | examples per dataset; steps; batch; LR | cap 30k, mixing-rate max 3k; 30k steps; 8,192 tokens; 3e-5 Adafactor | arXiv:2109.01652v5 §2.4 | verified 2026-09-15 | no ablation reported |

**Starting point for a small general-purpose run.** For an instruction-data pipeline, the verified rows support these defaults under the stated conditions. A ROUGE-L < 0.7 rule for new instructions comes from Self-Instruct's GPT-3 175B run with 52K instructions. Tülu 3's 8-gram rule (> 50% of test tokens; > 2% of an evaluation) was applied to 8B and 70B SFT mixtures. For math questions without gold answers, 32 samples at temperature 0.7 with a majority-vote answer comes from a 405B teacher and 8B-70B students. When specialized data is trained before general data, adding back a 1/256 fraction of it in the general stage comes from LLaMA 7B-33B. None of these rows includes an ablation on untargeted capabilities, so each needs the held-out check from §1 when reused.

## Generalization lens

**(a) What increases breadth.**
- More task clusters in instruction tuning raised the held-out average from 49.9 to 63.5 at 137B ([[flan]] §4.1, Fig. 6).
- Mixtures of human and model-generated data had the best average across six capability evaluations ([[tulu-1-how-far-can-camels-go]] Table 3).
- More unique questions at a fixed pair budget raised MATH validation by more than 10 points ([[openmathinstruct-2]] Fig. 6); this is breadth within one domain.
- Diverse real-user chat data (WildChat) raised most Tülu 3 skills slightly and AlpacaEval 2 by 4.9 points ([[tulu-3]] Table 10).

**(b) What causes narrowing or forgetting.**
- Math-only SFT on teacher traces lowered the non-reasoning average of Qwen3-14B-Base from 45.7 to 21.1 ([[transferability-of-llm-reasoning]] Table 1).
- Sequential training on specialized then general data lowered 7B GSM8K from 47.53 (multi-task) to 32.60 ([[sft-data-composition-dmt]] Table 1).
- General SFT after Code SFT lowered HumanEval from 70.7 to 66.5 at 340B ([[nemotron-4-synthetic]] Table 6).
- Instruction tuning lowered held-out performance at 8B and smaller in FLAN ([[flan]] Fig. 7).
- Skill-targeted data can raise a development metric without raising the unseen one (IFEval 53.6 → 72.8 vs IFEval-OOD 18.0 → 17.6 with Persona IF data, [[tulu-3]] Table 32).

**(c) How to measure it at this stage.**
- Fix a development suite and an unseen suite per skill before generating data, and decontaminate against both ([[tulu-3]] §2.2, §3.2).
- Report held-out task clusters, not held-out items of trained tasks ([[flan]] §2.2).
- For every data-mixture change, report each skill's development and unseen score, plus at least one group outside the pipeline's verified domain.
- Known measurement errors: GPT-4 preference win rates correlate with the number of unique tokens in responses (Pearson r = 0.96) ([[tulu-1-how-far-can-camels-go]] §5.4); n-gram decontamination misses paraphrases ([[tulu-3]] §3.2); closed models may have trained on the evaluation suite (§7).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| No held-out split fixed before data work | Development scores rise across data iterations; unseen or new-benchmark scores stay flat | Score every data-ablation checkpoint on an unseen suite ([[tulu-3]] Table 32) |
| Decontamination only by exact or n-gram match | Implausibly high scores on a benchmark whose paraphrases may exist in generated questions | Embedding retrieval + LLM paraphrase check on a sample ([[openmathinstruct-2]] §3.1) |
| Reporting total yield only | A task family disappears from the final data without a visible error | Per-check failure counts per generator and task family ([[apigen]] Table 1) |
| Surface deduplication treated as diversity | Many accepted items request the same task in different words | Count unique task types or questions, not n-gram novelty ([[openmathinstruct-2]] Fig. 6) |
| Trusting a verifier without measuring it | Filtering removes data and the metric does not change | Label accepted and rejected samples; noise-injection ablation ([[openmathinstruct-2]] Table 3, Fig. 5) |
| Majority vote treated as ground truth | Consistent wrong answers in a question family survive filtering | Compare majority answers with gold answers on a labelled subset |
| Verifier coverage bias | Math, code, or tool metrics rise while instruction following or chat metrics fall | Evaluation group outside the verified domain in the gate ([[transferability-of-llm-reasoning]] Table 1) |
| Specialized data trained in an earlier stage without replay | The specialized metric drops after the general SFT stage | Evaluate after each stage; add a replay fraction ([[sft-data-composition-dmt]] Table 1; [[nemotron-4-synthetic]] §3.3.1) |
| Win-rate evaluation used as the only general metric | Longer, more diverse responses score higher while benchmarks fall | Pair win rates with benchmark suites ([[tulu-1-how-far-can-camels-go]] §5.4) |
| Easy worst-of-N responses as rejected samples | Rejected log-probabilities fall quickly while chosen log-probabilities also fall | Log chosen and rejected log-probabilities separately ([[nemotron-4-synthetic]] §3.3.2) |

## Check your understanding

1. APIGen's add-back ablation cost the 1B model more than twice the BFCL points it cost the 7B model. Give two mechanisms that could explain the size dependence, and describe an experiment that separates them.
2. OpenMathInstruct-2 tolerated 20% wrong-answer solutions, but its weak-teacher data scored 7.8 points lower. Explain why both can be true, using the error types that a final-answer check can and cannot detect.
3. Tülu 3's Persona IF data raised IFEval by 19.2 points and did not raise IFEval-OOD. What does this imply about how the data was generated relative to the benchmark, and how would you change the generation step?
4. Explain why a majority vote over samples from the same teacher cannot correct systematic errors, and name one verifier signal from this chapter that can.
5. The SFT and RL models in the transferability study used the same math questions. Why did non-reasoning scores move in opposite directions? State which part of your answer is an interpretation.
6. Why does FLAN hold out whole task clusters instead of held-out examples from trained tasks? What error would the second design introduce into a measurement of generality?
7. DMT adds back only 1/256 of the specialized data in the final stage. Explain, using the scaling results in the same paper, why such a small fraction can be sufficient for recall while larger mixtures conflict.
8. A pipeline pairs the reward model's best and worst of 16 responses as chosen and rejected. Using the softmax gradient, explain what an easy rejected response contributes to the update and why near-miss negatives may carry more information.

## Connections

- Depends on: [[ch-00]] What General Capability Means and How It Is Measured (held-out suites); [[ch-10a]] Model-Based Quality Filtering and Benchmark-Targeted Data Selection (filter step); [[ch-12]] Deduplication: Exact, Near-Duplicate, and Semantic (deduplicate step).
- Previous in the course order: [[ch-32f]] Lab: Annealing and Context Extension with a Short-Context Regression Gate.
- Next: [[ch-19]] Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing (step 1 in depth).
- Later steps of the loop: [[ch-22]] Quality, Diversity, and Gradient-Based Data Selection (step 5); [[ch-23]] Model Collapse and Verification of Synthetic Data (iterated generation); [[ch-30b]] Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares (step 6).
- Domain chapters that reuse the loop: [[ch-24]] Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data; [[ch-26]] Tool and Function-Calling Data; [[ch-27]] Agentic Trajectory Data; [[ch-28]] Long-Context Data Synthesis and Synthetic Evaluation Task Families.
- Measurement and negatives: [[ch-29e]] Instruction Tuning and Generalization to Unseen Tasks; [[ch-30a]] Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control; [[ch-31a]] Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; [[ch-43a]] Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages; [[ch-48]] Contamination Detection and Its Effect on Reported Scores; [[ch-49]] Judge Models: Bias, Calibration, and Judge-Specific Overfitting; [[ch-45]] Self-Improvement Loops and Multi-Stage Reasoning Pipelines.

## Sources

- [[self-instruct]] — seed-based generation, ROUGE-L filter, quality audit (Table 2), unseen SuperNI and user-task evaluation, data-size and output-quality curve (Fig. 7); numbers read in arXiv:2212.10560 on 2026-09-15.
- [[nemotron-4-synthetic]] — synthetic share, generator iterations, preference-label judges, staged SFT with replay, DPO/RPO observations, Table 6 stage effects.
- [[nemotron-4-synthetic-recipe]] — SFT, DPO, and RPO settings for the Recipe table.
- [[apigen]] — three-check function-calling pipeline, per-generator yields (Table 1), add-back ablation (Fig. 5), relevance-detection targets.
- [[openmathinstruct-2]] — teacher ablation, noise tolerance, filtering ablation, question diversity, majority vote, LLM decontamination.
- [[openmathinstruct-2-recipe]] — generation and SFT settings for the Recipe table.
- [[tulu-1-how-far-can-camels-go]] — per-dataset capability trade-offs, win-rate bias, refusal over-generalization.
- [[sft-data-composition-dmt]] — data amount vs ratio, training-strategy comparison, DMT (chapter excerpt; arXiv:2310.05492v4).
- [[tulu-3]] — development vs unseen suites, 8-gram decontamination rule, SFT data ablations (Tables 10, 31, 32); numbers read in arXiv:2411.15124v5 on 2026-09-15.
- [[flan]] — held-out task clusters, cluster-count and scale ablations (chapter excerpt; arXiv:2109.01652v5).
- [[transferability-of-llm-reasoning]] — math-only SFT vs RL transfer (Table 1); numbers read in arXiv:2507.00432v2 on 2026-09-15.
- [[prolong]] — synthetic long instruction data at SFT (Table 8).
- [[apigen-mt]] — verification of task blueprints before trajectory simulation.
- [[west-of-n]] — best-vs-worst-of-N synthetic preference pairs (used only to correct an earlier description).
- [[nathan-lambert-synthetic-data]] — practitioner opinion on synthetic fine-tuning (used only to correct an earlier quotation).

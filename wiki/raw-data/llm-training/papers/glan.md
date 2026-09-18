<!-- scope: GLAN (arXiv 2402.13064) - instruction data generated from a human-verified discipline taxonomy (126 disciplines) expanded by GPT-4 into subjects, syllabi and key concepts; 10M pairs; Mistral 7B SFT; train-vs-test loss contamination check
     deps: [[self-instruct]]
     see-also: [[evol-instruct]], [[persona-hub]], [[orca-2]], [[metamath]], [[wizardmath]], [[hf-cosmopedia]], [[prismatic-synthesis]]
-->

# Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models
- **Core Insight:** Mistral 7B fine-tuned on 10 million question-answer pairs generated from a 126-discipline taxonomy, with no benchmark training sets, scores 80.8 GSM8K, 32.7 MATH, 48.8 HumanEval and 81.1 ARC-C, against 43.4, 10.0, 28.0 and 53.9 for base Mistral 7B (§3.1, Table 1); its MMLU gain is limited to STEM (Table 2).
- **Guideline:** When instruction data must cover many domains without seed examples, expand a human-verified discipline list into subjects, syllabi and key concepts with an LLM and sample questions from combinations of key concepts, because this produced gains on math, code, BBH and ARC without task-specific data (Table 1); also report per-category and per-discipline scores, because the same model is below base Mistral on MMLU humanities (54.9 vs 56.5, Table 2).
- **Authors:** Haoran Li, Qingxiu Dong, Zhengyang Tang, Chaojun Wang, Xingxing Zhang, Haoyang Huang, et al. (Microsoft; Singapore University of Technology and Design; Peking University; CUHK-Shenzhen; CUHK; Tsinghua University)
- **Year:** 2024 (arXiv v1 2024-02-20; arXiv comment "Work in progress"; no venue listed)
- **URL:** https://arxiv.org/abs/2402.13064
- **Source type:** paper
- **Relevant topics:** synthetic instruction data, taxonomy-driven generation, instruction tuning, decontamination, generalization measurement

## Abstract
GLAN (Generalized Instruction Tuning) generates instruction-tuning data from a pre-curated taxonomy of human knowledge and capabilities instead of seed examples or existing datasets. The taxonomy splits knowledge into fields, sub-fields and disciplines, built semi-automatically with LLMs and human verification. For each discipline an LLM lists subjects, then designs a syllabus per subject, whose class sessions carry key concepts. Instructions are generated from sampled sessions and key concepts. Experiments on Mistral show gains in mathematical reasoning, coding, academic exams, logical reasoning and instruction following without task-specific training data. New fields or skills can be added as new taxonomy nodes.

## Key Contributions
- A generation pipeline whose only human-curated input is a discipline taxonomy; all later steps are LLM-generated (§2, Algorithm 1).
- A 10-million-pair question-answer dataset: questions from GPT-4, answers from GPT-3.5-turbo (§3.1).
- GLAN-7B (Mistral 7B base) evaluated on 8 benchmarks, IFEval, the Evol-Instruct test set and a new 6,300-instruction GLAN-Test (§3.3, §3.5).
- A train-vs-test loss comparison to check for exposure to benchmark in-domain data (§3.4, Table 3).

## Key Figures/Tables to Study
- Algorithm 1: build_taxonomy → generate_subjects → generate_syllabus → extract_class_details → generate_instructions.
- Table 1: GLAN vs general, math-specialized and code-specialized 7B/13B models. Table 2: MMLU by category.
- Table 3: train and test losses on ARC-C, ARC-E, GSM8K, MATH. Tables 4-6: IFEval, Evol-Instruct test, GLAN-Test.
- Appendix Tables 7-8: per-skill and per-discipline (126) pairwise scores.

## Technical Details
- **Taxonomy (§2.1, §3.1):** GPT-4 is prompted with several instructions (e.g., "list all fields of human knowledge and capabilities"); human annotators vote to keep or remove each element, and removing a field or sub-field removes its descendants. 126 disciplines were kept after majority voting (§3.1). Example top-level fields are Natural Sciences, Humanities and Services (vocational training) (§2.1). The number of fields and sub-fields is not reported.
- **Subjects (§2.2, §3.1):** GPT-4 acts as an education expert for a discipline and lists subjects; a second prompt converts the list to jsonl with keys subject_name, level, subtopics, because the authors observed that adding format instructions to the first prompt lowered list quality (§2.2). GPT-4 is queried 10 times per discipline with temperature 1.0 and top-p 0.95, giving 100 to 200 subjects per discipline on average; subjects repeated across disciplines are not de-duplicated (§3.1). GPT-3.5-turbo is not used here because long-tail subjects may not be modeled well (§3.1).
- **Syllabus (§2.3, §3.1):** one GPT-4 query per subject (temperature 1.0, top-p 0.95); 10 to 30 class sessions per syllabus, around five key concepts per session; sessions and concepts are extracted with another GPT-4 prompt.
- **Questions (§2.4):** sample one or two class sessions and one to five key concepts, and prompt the LLM with them plus the full syllabus.
  1. Single session with m key concepts: Σ_{i=1..5} C(m, i) combinations (basic questions). For m = 5 this is 31 (derived).
  2. Two sessions with m1 and m2 key concepts: Σ_{i=2..5} C(m1+m2, i) − Σ_{i=2..5} C(m1, i) − Σ_{i=2..5} C(m2, i) combinations (harder questions that must use both sessions).
- **Answers (§2.4, §3.1):** questions and answers are generated separately because the authors observed better quality; questions by GPT-4, answers by GPT-3.5-turbo with temperature 0.7 and top-p 0.95, chosen because it is faster "with reasonably good results" (§3.1). §2 also says question generation may use GPT-4 "or GPT-3.5"; §3.1 names GPT-4.
- **Decontamination (§3.1):** pairs containing questions or input prompts from the test sets, and the training sets, of all evaluated benchmarks are removed.
- **Evaluation protocol (§3.3):** GSM8K, MATH, HumanEval 0-shot; MBPP 3-shot; BBH 3-shot with chain-of-thought; ARC and MMLU 0-shot with rule-based extraction from generated text, instead of the option-probability scoring the authors say previous models mostly used. The paper does not state which protocol was used for the baseline rows.
- **Main results (Table 1):**

| Model | HumanEval | MBPP | GSM8K | MATH | BBH | ARC-E | ARC-C | MMLU |
|---|---|---|---|---|---|---|---|---|
| Mistral 7B (base) | 28.0 | 50.2 | 43.4 | 10.0 | 56.1 | 79.5 | 53.9 | 62.3 |
| Mistral Instruct 7B | 46.7 | 31.7 | 24.4 | 8.2 | 46.0 | 76.9 | 52.0 | 53.7 |
| WizardMath v1.1 7B | 51.2 | 54.1 | 83.2 | 33.0 | 58.2 | 79.8 | 53.2 | 60.3 |
| Mistral CodeAlpaca 7B | 35.4 | 50.2 | 34.6 | 8.3 | 56.1 | 79.1 | 54.2 | 60.9 |
| GLAN 7B | 48.8 | 57.6 | 80.8 | 32.7 | 60.7 | 90.7 | 81.1 | 62.9 |
| GPT-3.5-turbo | 72.6 | 70.8 | 74.1 | 37.8 | 70.1 | 88.9 | 83.7 | 70.0 |

- **MMLU by category (Table 2):** Mistral 7B STEM 52.0 / Humanities 56.5 / Social Sciences 73.3 / Other 70.1; GLAN 60.1 / 54.9 / 71.8 / 68.6. The authors suggest chain-of-thought answers from GPT-3.5-turbo help multi-step STEM questions and may add errors on memorization-heavy questions (§3.3; Interpretation).
- **Instruction following:** IFEval strict prompt-level / strict instruction-level: GLAN-7B 34.0 / 44.8, Mistral-7B-Instruct-v0.1 32.0 / 42.8, GPT-3.5-turbo 53.8 / 64.7 (Table 4). Evol-Instruct test (218 instructions, 29 skills), GPT-4 pairwise score gap averaged over both response orders: +1.41 vs Mistral-7B-Instruct, −0.95 vs GPT-3.5-turbo (Table 7).
- **GLAN-Test (§3.5, Tables 6, 8):** 6,300 held-out instructions from GLAN data, 50 per discipline; GPT-4 pairwise gap +1.61 vs Orca2-7B, +0.43 vs Mistral-7B-Instruct, +0.19 vs WizardLM-13B-V1.2, −0.55 vs GPT-4 (Table 6).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GLAN-7B | 7B | SFT | Base model | Mistral 7B | arXiv:2402.13064v1 §3.2 | verified 2026-09-14 | no ablation reported |
| GLAN-7B | 7B | SFT | Training data | 10 million instruction-response pairs, decontaminated against evaluated benchmarks' test and training sets | §3.1 | verified 2026-09-14 | no ablation reported |
| GLAN-7B | 7B | SFT | Loss masking | instruction and response concatenated into one sequence; loss on response tokens only | §3.2 | verified 2026-09-14 | no ablation reported |
| GLAN-7B | 7B | SFT | Epochs | 3 | §3.2 | verified 2026-09-14 | no ablation reported |
| GLAN-7B | 7B | SFT | Peak LR; schedule | 3e-6; cosine, linear warm-up 1,000 steps, final LR 0 | §3.2 | verified 2026-09-14 | no ablation reported |
| GLAN-7B | 7B | SFT | Batch size | 512 instruction-response pairs | §3.2 | verified 2026-09-14 | no ablation reported |
| GLAN-7B | 7B | SFT | Data-generation sampling | subjects and syllabi: GPT-4, T = 1.0, top-p = 0.95; answers: GPT-3.5-turbo, T = 0.7, top-p = 0.95 | §3.1 | verified 2026-09-14 | no ablation reported |
| GLAN-7B | 7B | SFT | Sequence length, packing, optimizer, weight decay, compute, data release | not reported | checked §2, §3.1-3.5, §5, App. A | not reported | — |

## Findings relevant to generality
- **Contamination check (Result, single study; §3.4, Table 3):** Δ = L_test − L_train and Δ(%) = (L_test − L_train)/L_test, where L_test and L_train are the model's losses on a benchmark's test and training splits; a large positive Δ may indicate exposure to in-domain training data. GLAN-7B Δ(%) is −0.74% (ARC-C), −0.23% (ARC-E), 0.92% (GSM8K), −1.79% (MATH); Orca2-7B is 11.4% and WizardLM-13B-V1.2 is 4.39% on GSM8K. GLAN-7B also has higher absolute losses (ARC-C L_test 4.03 vs 2.02-2.39 for the other models), which the authors read as the model not converging to benchmark style (Interpretation).
- **Specialists narrow:** the authors observe that math- or code-optimized models improve on their target benchmarks "while usually not others" (§3.3); e.g., Mistral CodeAlpaca 7B GSM8K 34.6 vs base 43.4 (Table 1).
- **Where GLAN is weaker:** MMLU humanities, social sciences and other categories are below base Mistral (Table 2); Common-Sense skill −1.33 vs Mistral-7B-Instruct (Table 7); American history, Divinity and Radiology have negative gaps vs Orca-2-7b and Mistral-7B-Instruct on GLAN-Test (Table 8, §3.5).
- **Measurement limits:** GLAN-Test is drawn from GLAN's own generated data (§3.5), so it measures coverage of the taxonomy rather than transfer to an external distribution (Interpretation). Evol-Instruct test and GLAN-Test scores are GPT-4 judgments (§3.5).
- **Not covered:** the data are mostly single question-answer pairs; multi-turn conversations and long documents are listed as future work (§5).

## Connections
- [[self-instruct]] — seed-pool method; §1 says few-shot prompting tends to produce instructions similar to the demonstrations.
- [[evol-instruct]] — rewrites existing datasets; §1 says its domain scope is limited by the input datasets; WizardLM-13B-V1.2 is a baseline.
- [[orca-2]], [[metamath]], [[wizardmath]] — Table 1 baselines that use benchmark training sets; compare their Table 3 loss gaps.
- [[persona-hub]] — its §1 argues that comprehensive key-point lists like GLAN's are hard to curate beyond narrow domains.
- [[hf-cosmopedia]] — also controls topic and audience coverage explicitly, for pretraining data rather than instruction data.
- [[prismatic-synthesis]] — uses a GPT-4 skill taxonomy entropy as one diversity-metric baseline.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2402.13064 (arXiv v1, 2024-02-20; the only version listed).
- Corrections to the previous card version:
  - Title "GLAN: Generalized Instruction Tuning via Taxonomy-Driven Synthesis" → "Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models".
  - "Each step uses GPT-4 (or GPT-4-Turbo)"; "Teacher GPT-4 / GPT-4-Turbo" → GPT-4 builds the taxonomy, subjects, syllabi and questions; GPT-3.5-turbo writes the answers; GPT-4-Turbo is not mentioned (§3.1).
  - "Concept → instruction-response pairs ... verification-friendly answer formats for math/code" → one to five key concepts from one or two sessions produce a question; answers are generated in a separate call (§2.4, §3.1).
  - "Top-level field list, a few dozen entries seeded by the authors + GPT-4" → GPT-4-generated taxonomy with human post-editing and majority voting; 126 disciplines kept; field count not reported (§2.1, §3.1).
  - "Filtering: deduplication + light decontamination; teacher self-verification for math/code" → only benchmark decontamination is described; duplicate subjects are deliberately kept; no self-verification (§3.1).
  - "Output: multi-million pairs ... released alongside the paper" → 10 million pairs (§3.1); release is not stated.
  - "Outperforms same-base models fine-tuned on Alpaca / WizardLM / CodeAlpaca on MATH, GSM8K, HumanEval, MBPP, BBH, ARC, MMLU" → baselines are those in Table 1 (no Alpaca); GLAN is not best everywhere (HumanEval 48.8 vs WizardMath 51.2; GSM8K 80.8 vs 83.2).
  - "Four-level hierarchy Field → Subfield → Discipline → Subject → Session → Concept → Instruction" → human-verified fields/sub-fields/disciplines, then LLM-generated subjects, syllabus sessions and key concepts (§2).
  - "Figure 1 taxonomy diagram and its expansion ratios" → Figure 1 compares the inputs of FLAN, Self-Instruct, Evolve-Instruct and GLAN.
  - Affiliation "Microsoft Research + collaborators" → six institutions listed in the p.1 footnote.
- Removed as unsupported by the source: "ablations on taxonomy depth confirm deeper trees give flatter capability distributions" (no ablation exists); "third synthesis paradigm alongside Magpie" (Magpie is not cited); "demonstrated that adding a taxonomy node adds a capability" (stated as a property, not tested); "category-coverage bar chart"; "table comparing GLAN to Alpaca / Evol-Instruct / WizardLM data"; risks list (teacher bias at the top, audit cost, license encumbrance, no novel capabilities); "precursor of Cosmopedia's taxonomy layer"; "feeds the gradient-space coverage direction".
- Not reported by the source: generation cost; number of fields and sub-fields; questions per concept combination; sequence length, optimizer, compute; ablations of any pipeline step.

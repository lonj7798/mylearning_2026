<!-- chapter: ch-21
     track: synthetic
     kind: content
     title: Taxonomy-Driven and Textbook-Style Synthesis
     deps: [ch-20]
     sources: [[glan]], [[nemotron-4-synthetic]], [[nemotron-4-synthetic-recipe]], [[phi-textbooks]], [[phi-1-5]], [[phi-3]], [[phi-4]], [[cosmopedia]], [[hf-cosmopedia]], [[gsm1k]], [[fineweb]], [[mathscale]], [[persona-hub]], [[metamath]]
     figures: figures/taxonomy-expansion.html
     revised: 2026-09 (generality revision)
-->

# Chapter 21 — Taxonomy-Driven and Textbook-Style Synthesis

> **Core insight.** Taxonomy-driven synthesis fixes the list of topics before any example is generated, and textbook-style synthesis fixes the form of the generated text. GLAN generated 10 million question-answer pairs from 126 human-verified disciplines and raised Mistral 7B from 43.4 to 80.8 on GSM8K and from 53.9 to 81.1 on ARC-C without benchmark training data, but the same model scored below its base on MMLU humanities, 54.9 vs 56.5 ([[glan]] Tables 1–2). The Phi reports show a related split at pretraining scale: a 13B model trained on synthetic data without web data gained on reasoning and code benchmarks and lost 14.8 TriviaQA points relative to phi-3-medium ([[phi-4]] Table 3), and on freshly written GSM1k problems the Phi and Mistral families score below their GSM8K accuracy for almost every release and size ([[gsm1k]] §5.1).
>
> **Guideline.** When a model must cover dozens of domains (GLAN: 126 disciplines) and seed prompts are missing for some of them, generate from a human-verified topic list and sample combinations of concepts, because GLAN's gains on math, code, BBH, and ARC came without task-specific data (Table 1). Report per-category and per-discipline results for such a run, because the aggregate MMLU gain of 0.6 points hid a 1.6-point humanities loss (Table 2). When textbook-style synthetic data is half or more of pretraining tokens (Phi-4: at least 55%), keep filtered web and knowledge-heavy sources in the mixture, because in Phi-4's 7B, 1T-token ablations the synthetic-plus-web allocation had its largest gain on TriviaQA (+6.9) while the synthetic-heavy allocation lost 3.0 TriviaQA points, and the authors added web data for knowledge benchmarks (§3.2, Table 4). When a benchmark resembles the synthetic data, evaluate on a distribution-matched fresh set before claiming generality, because phi-1 solved 81.7% of HumanEval problems with close matches in its exercises and 26.9% of those without ([[phi-textbooks]] Table 3).

## Corrections to the version you studied

1. "decompose human knowledge into a Field → Subfield → Discipline → Subject → Session → Concept tree, generate an instruction per leaf concept" and "Instructions are generated at the concept level, guaranteeing coverage across all branches" (quoted as the paper) → GLAN samples one or two class sessions and one to five key concepts from a syllabus and asks GPT-4 for a homework question; the sentence in quotation marks is not in the paper, which says the authors "expect by randomly combining these class sessions and key concepts to ensure the coverage and diversity" ([[glan]] §2, §2.4).
2. "The top two levels are hand-curated by the authors plus GPT-4; every level below is auto-generated … 'list the children'" and "Level 0: Fields (hand-curated, ~36 entries)" → GPT-4 generates fields, sub-fields, and disciplines; human annotators vote to keep or remove elements, and 126 disciplines are kept; the number of fields is not reported; subjects come from an "education expert" prompt queried 10 times per discipline (100–200 subjects), and syllabi from one query per subject (10–30 sessions, about five key concepts each) ([[glan]] §2.1–2.3, §3.1).
3. The GLAN pseudocode loop `for difficulty in ["easy", "medium", "hard"]` with "verifier-friendly formats for math/code" → there is no difficulty loop and no verification; harder questions come from the two-session sampling strategy, and answers are written in a separate call by GPT-3.5-turbo ([[glan]] §2.4, §3.1).
4. "the final corpus is 'multi-million instruction-response pairs spanning all leaf concepts'" → 10 million pairs; the paper does not report how many concept combinations were sampled ([[glan]] §3.1).
5. "Mistral-7B fine-tuned on GLAN data beats Alpaca, WizardLM, and CodeAlpaca training on the same base across MATH, GSM8K, HumanEval, MBPP, BBH, ARC, and MMLU" → the baselines are released models (Orca 2, WizardLM v1.2 13B, Mistral Instruct, MetaMath Mistral, WizardMath v1.1, Mistral CodeAlpaca); WizardMath v1.1 is higher on GSM8K (83.2 vs 80.8), MATH (33.0 vs 32.7), and HumanEval (51.2 vs 48.8); GLAN is below base Mistral on MMLU humanities, social sciences, and other ([[glan]] Tables 1–2).
6. "Adding a new capability is as cheap as adding a subtree" → the paper states that new nodes can be added without regenerating the dataset; it does not test that an added node adds a capability ([[glan]] §1, Verification).
7. Nemotron-4 "curated a shallow tree of task families (coding, general QA, topic-following, document-based reasoning, function calling, refusal)" → synthetic prompts are generated for open Q&A, writing, closed Q&A, and math and coding, seeded with 3K topics and 12K Python and 17K math keywords; topic-following, document QA, and function-calling data come from existing datasets ([[nemotron-4-synthetic]] §3.2.1, §3.2.5).
8. "using the model's own reward head as the judge" and "trains the RM on 20K human HelpSteer2 anchors" → Nemotron-4-340B-Reward is a separate model built from the base with a five-attribute regression head and trained on 10K HelpSteer2 examples; the other 10K human examples are SFT data; chosen/rejected labels come from ground truth or a verifier where available, from LLM-as-judge in early iterations, and from the reward model later ([[nemotron-4-synthetic]] §3.1, §3.2, §3.2.3).
9. The table marking code SFT, DPO, and RPO data as "RM-judged" with "~1.3% human", and the `nemotron_iteration` loop (best and worst of K=4 samples) → Code SFT data come from Genetic Instruct with an LLM fitness check; the report states "over 98%" synthetic and does not describe a best/worst-of-K loop; preference responses come from several intermediate models and several samples of the best model ([[nemotron-4-synthetic]] §3.2.3, §3.3.1).
10. "Staged SFT — code SFT first, general SFT second — shows up in Phi-4, Qwen, and Tülu too. … installs structured-format following" → among this chapter's sources only Nemotron-4 reports two-stage SFT, and its stated reason is conflict between behaviors when trained together, most strongly for coding; Phi-4 reports one SFT round; no source here supports code-first staging for Qwen or Tülu 3 ([[nemotron-4-synthetic]] §3.3.1; [[phi-4]] §4).
11. "Pretraining loss is quality-bounded long before it is quantity-bounded … rivals models ~10× bigger trained on ~100× more tokens" (quoted as the Phi-1 thesis) → not in the paper; phi-1 reports 50.6% HumanEval with a dataset of about 7B tokens against StarCoder (15.5B, 1T tokens) at 33.6% ([[phi-textbooks]] Table 1, Verification).
12. The Phi table: Phi-1 "~7B" pretraining tokens; Phi-1.5 "~27B" tokens and "~20B synthetic (20K-topic taxonomy, GPT-3.5)"; Phi-3-mini "~trillion-scale synthetic" with "GPT-4-class teachers"; Phi-4 "~unspecified, synthetic weighted ~10%" → phi-1-base saw a little over 50B tokens (about 8 epochs over under 7B); phi-1.5 trained 150B tokens over a 30B-token dataset and the report does not name the generator; the Phi-3 report gives no phase split and no generator; Phi-4 pretrained on about 10T tokens with synthetic data at 40% and web rewrites at 15% of training tokens ([[phi-textbooks]] §2.3; [[phi-1-5]] Table 1, Verification; [[phi-3]] §2; [[phi-4]] §3, Table 5).
13. "Phi-1 trained a random-forest on a hand-labeled 'educational vs non-educational' seed" → GPT-4 annotated about 100k samples for educational value, and a random forest on codegen-model embeddings extended the label ([[phi-textbooks]] §2.1).
14. "a carefully curated list of 20,000 topics covering common-sense, grade-school science, logic, everyday reasoning — each expanded by GPT-3.5" (quoted) → the report says the authors "carefully selected 20K topics to seed the generation" and use web samples in the prompts for diversity; the topic description and the generator are not in the report ([[phi-1-5]] §2.2).
15. "Phi-4 … needed only 90 GRPO steps … all the gain is in the SFT data; RL is polish" → Phi-4-reasoning-plus trained 90 GRPO steps over about 6k examples, and that training raised AIME by more than 10%; the authors add that responses were clipped at 31k tokens, which limits what GRPO can change ([[phi-4]], arXiv:2504.21318 §4.2).
16. "later analyses of the Phi line flag non-trivial overlap between synthetic exercises and HumanEval prompts. The 50.6% is under suspicion." → no such analysis is cited; phi-1's own §5 finds 4 HumanEval 13-gram matches (all false positives), scores 45.1–50.6% after pruning similar exercises, and shows 81.7% vs 26.9% on similar vs non-similar problems; published evidence of Phi overfitting is on GSM8K via GSM1k ([[phi-textbooks]] §5, Table 3; [[gsm1k]] §5.1).
17. "Phi-1.5 … is weakest on the harder benchmarks (MMLU-Pro, MATH)" → phi-1.5 was not evaluated on MMLU-Pro or MATH; it scores below Llama2-7B on HellaSwag (0.476 vs 0.571) and MMLU (0.376 vs 0.453) ([[phi-1-5]] Table 3).
18. "their first pass produced 30M prompts with too many duplicate-class outputs; the fix was restructuring the prompt taxonomy" and "Deduplication is the real bottleneck" → the blog describes iterating prompts before generation, adding explicit instructions on how content should differ by audience and style, and reaching over 30M prompts with less than 1% duplicate content; it says most time went to prompt engineering ([[cosmopedia]]).
19. Cosmopedia's web branch "is an audit move … It also doubles the deduplication workload" → web data was used because curated sources did not scale (16,000 OpenStax units, 250,000 Stanford units) and supplied over 80% of prompts (23M); a deduplication workload is not discussed ([[cosmopedia]]).
20. "No tree expansion emits that combination naturally" → GLAN deliberately combines key concepts from two sessions of one syllabus; combinations across different subjects or disciplines are not generated by either sampling strategy ([[glan]] §2.4).
21. Scaling rules of thumb ("30–50 root fields × … ≈ 3M–30M samples", "20K flat topics × 5–10 passes … × ~100K tokens/document", "6 families × ~million-scale generations", "> 40% dedup", "< 1 week of an expert's time") → none of these values is in a source; the reported quantities are listed in §2, §5, and §6.
22. "[[ch-32]] / [[ch-34]] — Phi-3/4 case study in the SFT recipes section; Nemotron-Ultra as another case" → ch-32 is Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL; the Phi reports are covered in ch-34 and teacher-data insertion in ch-35 (outline).
23. Figure claims "Taxonomy-only pipelines plateau" and the coverage bar → no source reports either; the figure is rewritten.

## Why this chapter matters for a general-purpose model

A general-purpose model needs training data for domains that nobody collected prompts for. Seed-based generation (ch-19) produces examples near its seeds; GLAN's authors state that few-shot prompting "tends to generate new instructions similar to its demonstrations" and that rewriting methods are limited by their input datasets ([[glan]] §1). Taxonomy-driven synthesis addresses this by listing the target domains first. Textbook-style synthesis addresses a second problem: web text presents knowledge in forms that are hard to learn from next-token prediction, and the Phi authors argue that generated expositions and exercises are easier to learn (Interpretation, [[phi-4]] §2.1).

Both methods sit in two places in the pipeline: pretraining and mid-training (Phi, Cosmopedia) and SFT (GLAN, Nemotron-4). Both raise the same generality question. The topic list and the teacher decide what the student sees, so the gains may be concentrated in the listed topics and in benchmarks that resemble the generated format. This chapter reports what the sources measured about that concentration.

## §1 Terms and the coverage problem

**Seed-based synthesis** generates new examples by prompting a model with existing examples (ch-19). **Taxonomy-driven synthesis** generates examples from an explicit hierarchy of topics, such as disciplines, subjects, and class sessions. **Textbook-style synthesis** generates expository text and exercises, usually for pretraining, conditioned on a topic and an audience. A **teacher** is the model that writes the data; the **student** is the model trained on it.

**Coverage** in this chapter means a measurable property of a dataset or a model: the share of target categories that have training examples, and the share of target categories on which the trained model does not regress. A **held-out coverage evaluation** tests each category with examples not used for training or model selection.

The problem, stated as a measurement: for a category list with K entries, a seed-based dataset may contain zero examples for some entries. Example with illustrative counts: a target list of 126 disciplines and a seed pool whose examples fall into 90 of them has a data coverage of 90/126 = 71.4%. A taxonomy-driven generator that writes questions for every discipline has 126/126 = 100% data coverage by construction. Model coverage is a separate number: GLAN writes data for every discipline, yet its GLAN-Test gaps are negative for American history, Divinity, and Radiology against all three smaller baselines (Orca-2-7B, Mistral-7B-Instruct-v0.1, WizardLM-13B-V1.2) and against GPT-4 ([[glan]] Table 8). Data coverage does not guarantee model coverage.

## §2 GLAN: from a discipline list to homework questions

**Definition.** GLAN (Generalized Instruction Tuning) generates instruction data from a taxonomy of human knowledge and capabilities, with no seed examples or existing datasets as input ([[glan]] Abstract).

**Mechanism.** Algorithm 1 of the paper:

```
D ← build_taxonomy()                     ▷ list of disciplines (§2.1)
for each discipline d ∈ D do
    S ← generate_subjects(d)             ▷ §2.2
    for each subject s ∈ S do
        A ← generate_syllabus(s, d)      ▷ §2.3
        C, K ← extract_class_details(A)  ▷ class sessions and key concepts
        Q ← generate_instructions(A, C, K, d)   ▷ sample sessions and key concepts (§2.4)
        L ← L ∪ Q
return L
```

1. **Taxonomy.** GPT-4 is prompted with instructions such as "list all fields of human knowledge and capabilities". Human annotators vote to keep or remove each element; removing a field or sub-field removes its descendants. 126 disciplines remain. Top-level fields include Natural Sciences, Humanities, and Services (vocational training) (§2.1, §3.1).
2. **Subjects.** GPT-4, acting as an education expert for a discipline, lists subjects; a second prompt converts the list to jsonl, because format instructions in the first prompt lowered list quality. 10 queries per discipline at temperature 1.0 and top-p 0.95 give 100 to 200 subjects per discipline; duplicates across disciplines are kept (§2.2, §3.1).
3. **Syllabus.** One GPT-4 query per subject produces a syllabus of 10 to 30 class sessions with around five key concepts each (§2.3, §3.1).
4. **Questions.** One or two sessions and one to five key concepts are sampled, and GPT-4 writes a homework question given them and the full syllabus (§2.4).
5. **Answers.** GPT-3.5-turbo answers each question in a separate call at temperature 0.7, chosen because it is faster "with reasonably good results" (§3.1).
6. **Decontamination.** Pairs containing questions or input prompts from the test and training sets of all evaluated benchmarks are removed (§3.1).

**Formula.** The number of distinct concept sets available for one question:

- One session: `N_1(m) = Σ_{i=1..5} C(m, i)`
- Two sessions: `N_2(m1, m2) = Σ_{i=2..5} C(m1+m2, i) − Σ_{i=2..5} C(m1, i) − Σ_{i=2..5} C(m2, i)`

Here m, m1, m2 are the numbers of key concepts in the sampled sessions, i is the number of concepts in the question, and C(n, i) is the binomial coefficient. The subtractions remove sets that draw all concepts from one session, so every two-session set uses both sessions.

**Worked example.** With m = 5: N_1 = 5 + 10 + 10 + 5 + 1 = 31. With m1 = m2 = 5: Σ_{i=2..5} C(10, i) = 45 + 120 + 210 + 252 = 627; Σ_{i=2..5} C(5, i) = 10 + 10 + 5 + 1 = 26; N_2 = 627 − 26 − 26 = 575. With m1 = 3 and m2 = 4: 112 − 4 − 11 = 97. A syllabus of 20 sessions with five key concepts each offers 20 × 31 = 620 single-session concept sets and C(20, 2) × 575 = 190 × 575 = 109,250 two-session concept sets (derived).

**Corpus arithmetic (derived from reported ranges).** At 150 subjects per discipline and 20 sessions per syllabus: 126 × 150 = 18,900 syllabi and 378,000 class sessions. 10 million pairs then average about 529 pairs per syllabus and 26.5 per session. The paper does not report the per-discipline split or how many concept sets were sampled.

[figures/taxonomy-expansion.html](figures/taxonomy-expansion.html) lets the reader change m1 and m2 to see both combination counts, change the subject and session ranges to see the derived corpus sizes, and recompute the GSM1k gap test used in §7.

**Evidence and limits.** Mistral 7B trained 3 epochs on the 10M pairs is the GLAN-7B model (§3.2). The paper reports no ablation of any pipeline step (Verification). Answers are unverified GPT-3.5-turbo outputs. The field and sub-field counts are not reported, so the size of the human-verification step cannot be checked. The data are single question-answer pairs; multi-turn conversations and long documents are listed as future work (§5).

**Implication.** GLAN's input is a discipline list, not a task list. That choice controls which subjects receive data, but not which task formats (multi-turn, strict formatting, tool calls) appear, and not whether answers are correct.

## §3 What GLAN's results measure

**Main table** (Table 1; 0-shot except MBPP and BBH 3-shot):

| Model | HumanEval | MBPP | GSM8K | MATH | BBH | ARC-E | ARC-C | MMLU |
|---|---|---|---|---|---|---|---|---|
| Mistral 7B (base) | 28.0 | 50.2 | 43.4 | 10.0 | 56.1 | 79.5 | 53.9 | 62.3 |
| Mistral Instruct 7B | 46.7 | 31.7 | 24.4 | 8.2 | 46.0 | 76.9 | 52.0 | 53.7 |
| WizardMath v1.1 7B | 51.2 | 54.1 | 83.2 | 33.0 | 58.2 | 79.8 | 53.2 | 60.3 |
| Mistral CodeAlpaca 7B | 35.4 | 50.2 | 34.6 | 8.3 | 56.1 | 79.1 | 54.2 | 60.9 |
| GLAN 7B | 48.8 | 57.6 | 80.8 | 32.7 | 60.7 | 90.7 | 81.1 | 62.9 |

**Result (single study).** Among the models in Table 1 built on Mistral 7B, GLAN is the only one that improves over the base on all eight columns. The math-specialized WizardMath is higher on GSM8K, MATH, and HumanEval, and the code-specialized CodeAlpaca drops GSM8K from 43.4 to 34.6. The authors observe that specialized models improve on their target benchmarks "while usually not others" (§3.3).

**Per-category view.** MMLU by category (Table 2): base Mistral STEM 52.0, Humanities 56.5, Social Sciences 73.3, Other 70.1; GLAN 60.1, 54.9, 71.8, 68.6. The aggregate MMLU change is +0.6, composed of +8.1 on STEM and losses on the other three categories. The authors suggest that chain-of-thought answers from GPT-3.5-turbo help multi-step STEM questions and may add errors on memorization-heavy questions (Interpretation, §3.3).

**Instruction following.** IFEval strict prompt-level accuracy is 34.0 for GLAN-7B, 32.0 for Mistral-7B-Instruct-v0.1, and 53.8 for GPT-3.5-turbo (Table 4). On this benchmark of format constraints, GLAN-7B is 2.0 points above Mistral-7B-Instruct-v0.1 and 19.8 points below GPT-3.5-turbo.

**Train-versus-test loss check.** GLAN checks whether its model was exposed to benchmark-style training data:

`Δ = L_test − L_train`, `Δ(%) = (L_test − L_train) / L_test`

where L_test and L_train are the model's average losses on a benchmark's test split and training split. A model trained on the training split (or text like it) has lower L_train than L_test, so a Δ(%) higher than that of comparable models on the same benchmark suggests exposure to in-domain training data.

Worked example with made-up losses: L_train = 1.00 and L_test = 1.10 give Δ = 0.10 and Δ(%) = 0.10 / 1.10 = 9.1%; L_train = 1.00 and L_test = 1.01 give 1.0%. Reported GSM8K values (Table 3): GLAN-7B 0.92%, WizardLM-13B-V1.2 4.39%, Orca2-7B 11.4%. Orca 2 and WizardMath-style models use benchmark training sets ([[glan]] Connections). **Result (single study)**. Leakage of test items lowers L_test and makes Δ smaller or negative, so a small Δ(%) does not rule out test-set leakage (Interpretation). The check also requires a benchmark with a training split.

**GLAN-Test.** 6,300 instructions held out from GLAN's own generated data, 50 per discipline, scored by GPT-4 pairwise judging: +1.61 vs Orca2-7B, +0.43 vs Mistral-7B-Instruct, +0.19 vs WizardLM-13B-V1.2, −0.55 vs GPT-4 (§3.5, Table 6). Because the test items come from the generator's distribution, GLAN-Test measures how well the student covers the taxonomy's own questions, not transfer to prompts written by users (Interpretation, [[glan]] Findings).

## §4 Nemotron-4 340B: topic-seeded prompts, a separate reward model, and staged SFT

**Definition.** Nemotron-4-340B-Instruct is aligned with SFT and preference data of which "over 98%" is synthetic, using about 20K human-annotated examples: 10K for SFT and 10K HelpSteer2 preference examples ([[nemotron-4-synthetic]] §3.2).

**Roles in the pipeline.**

| Role | Model or signal | Locus |
|---|---|---|
| Prompt generator | Mixtral-8x7B-Instruct-v0.1 (permissive license) | §3.2.1 |
| Topic list | macro-topics → subtopics generated by Mixtral, plus manual topics: 3K topics; 12K Python and 17K math keywords | §3.2.1 |
| Real prompts | LMSYS-Chat-1M, split into disjoint SFT and preference sets | §3.2.1 |
| Response generator | Mixtral in round 1, then each aligned 340B intermediate model | §3.2.4 |
| Chosen/rejected labels | ground truth or Python verifier where available; LLM-as-judge early; reward model later (Chat-Hard 0.87 vs 0.54) | §3.2.3 |
| Reward model | Nemotron-4-340B-Reward: base model with a linear head over five HelpSteer attributes, trained on 10K HelpSteer2 | §3.1 |
| Code data | Genetic Instruct: self-instruction and WizardCoder mutations, LLM fitness check, about 800K samples | §3.3.1 |

Topic seeding is a shallow taxonomy step in the same sense as GLAN: a generated list of topics decides which subjects the prompts cover (Interpretation). Nemotron also measures a difference between synthetic and real prompts: Mixtral responses to synthetic prompts have mean reward-model helpfulness 3.24, against 3.04 for LMSYS prompts, which the report reads as real prompts being harder (Figure 3). **Result (single study)**. Generated topic lists can produce prompts that are easier than user prompts.

**Staged SFT.** The report states that "learning multiple behaviors concurrently can sometimes lead to conflicts between them", most strongly for coding, "where adjusting the sampling weights for the data blend fails to align the model to all coding tasks" (§3.3.1). It trains Code SFT first (about 800K samples, 1 epoch, LR 3e-7) and then General SFT (200K samples including 2% of the code samples "to mitigate the risk of forgetting", 3 epochs). No single-stage numbers are printed.

Stage effects (Table 6):

| After stage | MT-Bench | MMLU | GSM8K | HumanEval | IFEval prompt-strict |
|---|---|---|---|---|---|
| Code SFT | 6.79 | 72.2 | 77.6 | 70.7 | 46.4 |
| + General SFT | 7.99 | 78.3 | 87.9 | 66.5 | 61.4 |
| + DPO | 7.90 | 78.4 | 88.5 | 67.1 | 61.7 |
| + RPO, 3rd iteration | 8.22 | 78.7 | 92.3 | 73.2 | 79.9 |

Code SFT raised HumanEval from the base's 57.3 to 70.7; General SFT then lowered it to 66.5 despite the 2% code replay. **Result (single study)**. The later preference stages recovered HumanEval to 73.2.

**Implication.** A topic list is one control; the reward model, verifier, and stage order are separate controls on which behaviors survive. The staged-SFT evidence is one report at 340B with no single-stage table.

## §5 Phi: textbook-style synthetic data in pretraining

**Definition.** The Phi reports train small models on filtered web or code data plus LLM-generated "textbook-quality" text and exercises.

| Report | Model | Data reported | Tokens | Locus |
|---|---|---|---|---|
| Textbooks Are All You Need (2023-06) | phi-1, 1.3B | filtered code ~6B + GPT-3.5 textbooks <1B; finetuning on CodeExercises <180M tokens (879.5K problems) | little over 50B seen (~8 epochs) | [[phi-textbooks]] §2, §2.3 |
| Textbooks Are All You Need II (2023-09) | phi-1.5, 1.3B | phi-1 data (7B) + ~20B new synthetic tokens seeded by 20K topics; generator not named | 30B dataset, 150B seen | [[phi-1-5]] §2.2, Table 1 |
| Phi-3 Technical Report (2024-04) | phi-3-mini, 3.8B | web filtered by "educational level" + synthetic, two phases; no split, no generator named | 3.3T | [[phi-3]] §2 |
| Phi-4 Technical Report (2024-12) | phi-4, 14B | 50 types of synthetic data, about 400B unweighted tokens; web, web rewrites, code, acquired sources | about 10T | [[phi-4]] §2.2, §3 |

**Phi-1 mechanism.**
1. GPT-4 labels about 100k code samples with the prompt "determine its educational value for a student whose goal is to learn basic coding concepts" (§2.1).
2. A random forest on codegen-model embeddings scores the full pool of over 35B tokens; about 6B tokens are kept (§2.1).
3. GPT-3.5 writes under 1B tokens of textbooks, with diversity from topic and audience constraints (§2.2).
4. GPT-3.5 writes CodeExercises, docstring-completion problems diversified by function names; phi-1-base is finetuned on them to give phi-1 (§2.2).

At 350M parameters, unfiltered data reaches 12.19% HumanEval, the filtered subset 17.68%, and filtered plus synthetic textbooks 20.12% (§2.1). phi-1 reaches 50.6% HumanEval and 55.5% MBPP, and phi-1-base 29% (Table 1, §2). **Result (single study)**.

**Phi-1.5.** The authors "carefully selected 20K topics to seed the generation" and "use samples from web datasets for diversity" (§2.2). phi-1.5 scores 0.734 WinoGrande and 40.2 GSM8K (via coding) against 0.691 and 14.6 for Llama2-7B, and scores below Llama2-7B on HellaSwag (0.476 vs 0.571) and MMLU (0.376 vs 0.453) (Tables 2–4). phi-1.5-web, trained with 40% filtered web, 20% phi-1 code, and 40% synthetic data for 300B tokens, scores 44.6 GSM8K and 41.4 HumanEval against 40.2 and 34.1 for phi-1.5 (§2.4, Table 4). The two runs differ in tokens seen (150B vs 300B), so this is not an equal-token comparison.

**Phi-4 mixture.** Table 5 gives the final pretraining mixture as a share of training tokens and the unique tokens per source:

| Source | Share of training tokens | Unique tokens | Epochs |
|---|---|---|---|
| Web | 15% | 1.3T | 1.2 |
| Web rewrites | 15% | 290B | 5.2 |
| Synthetic | 40% | 290B | 13.8 |
| Code data | 20% | 820B | 2.4 |
| Acquired sources | 10% | 580B | 1.7 |

The epoch column is the allocated tokens divided by unique tokens (§3.2). Worked check with the reported total of about 10T tokens: synthetic 0.40 × 10T = 4T, and 4T / 290B = 13.8 epochs; web 0.15 × 10T = 1.5T, and 1.5T / 1.3T = 1.15, printed as 1.2. Web rewrites are "a sub-category of synthetic data" (footnote 5), so synthetic sources account for at least 55% of training tokens by this table; code data is itself a mixture of synthetic and raw code (§3.2). The report does not reconcile the 400B unweighted tokens of §2.2 with the 290B + 290B unique tokens of Table 5.

**Phi-4 ablations of synthetic versus web data.**
- 13B models without web data, relative to phi-3-medium (Table 3): synthetic only MMLU +0.8, HumanEval +12.1, MATH +4.9, TriviaQA −14.8; synthetic plus web rewrites TriviaQA −7.7.
- 7B models at a 1T-token horizon, 75% of tokens reallocated, relative to the final mixture (Table 4): synthetic-heavy (S) average +0.8 with TriviaQA −3.0; synthetic plus web (S+W) average 0.0 with TriviaQA +6.9; uniform allocation average −2.2.
- The authors chose the final mixture with "targeted and knowledge-heavy filtered web data sources to improve knowledge benchmarks", and report that the gap to synthetic-heavy runs "largely closes" after post-training, without numbers (§3.2).
- §3.1: "Models trained only with synthetic data underperformed on the knowledge-heavy benchmarks and demonstrated increased hallucinations."

**Result (three reports from one lab; not an independent replication).** Synthetic-heavy pretraining is weaker on knowledge recall than on reasoning in phi-1.5 (HellaSwag, MMLU vs Llama2-7B; [[phi-1-5]] Table 3), phi-3-mini (TriviaQA 64.0 vs 82.2 for Mixtral 8x7B, which the authors attribute to limited capacity; [[phi-3]] §3, §6), and Phi-4's ablations ([[phi-4]] Tables 3–4).

## §6 Cosmopedia: an open replication of textbook-style synthesis

**Definition.** Cosmopedia is a Hugging Face dataset of synthetic textbooks, blog posts, stories, and WikiHow articles generated by Mixtral-8x7B-Instruct-v0.1, built to "reproduce the training data used for Phi-1.5": over 30M files, 25B tokens, over 10k GPU hours ([[cosmopedia]]). **Source reliability: official** (Hugging Face built the dataset and trained cosmo-1b; open code, data, and model; no ablation tables).

**Mechanism.**
1. **Curated seeds.** Units from Stanford course outlines, Khan Academy, OpenStax, and WikiHow. These did not scale: 16,000 OpenStax units and 250,000 Stanford units, against a target of at least 20 million prompts.
2. **Audience and style.** Four audiences (young children, high school students, college students, researchers) and three styles (textbooks, blog posts, wikiHow articles) give up to 12 prompts per topic. Changing only the audience or style words "was insufficient to prevent a high rate of duplicate content"; the fix was explicit instructions on how format and content should differ.
3. **Web seeds.** Millions of web samples clustered into 145 clusters; Mixtral names each cluster from 10 random extracts and scores its educational value; low-value clusters are removed (112 topics kept). Prompts ask for a textbook related to a web sample, conditioned on the cluster topic 50% of the time. This gives 23M prompts, over 80% of the total.
4. **Stories.** "In our initial assessments of models trained using the generated textbooks, we observed a lack of common sense and fundamental knowledge typical of grade school education." Stories seeded from UltraChat and OpenHermes2.5 were added.
5. **Decontamination.** Candidates with a 10-gram overlap are compared with `difflib.SequenceMatcher`; a sample is discarded when `len(matched_substrings) / len(benchmark_sample) > 0.5`.

Worked example of the decontamination rule: a benchmark question of 200 characters whose matched substrings in a generated document total 120 characters gives 120 / 200 = 0.6 > 0.5, so the document is removed; 80 matched characters give 0.4 and the document is kept. The rule removed, for example, 386 samples matching 41 unique BoolQ items in the web, Stanford, and OpenStax group.

**Result.** cosmo-1b (1B, Llama2 architecture) is reported above TinyLlama 1.1B on ARC-Easy, ARC-Challenge, OpenBookQA, and MMLU, with "some performance gaps compared to Phi-1.5". Scores appear only in a figure, and training tokens are not given.

**Limits and implication.** Step 4 is a coverage gap found by evaluating a trained model rather than by inspecting the topic list: textbook prompts built from educational sources and web topics lacked grade-school common sense. The blog also reports that Mixtral "may sometimes hallucinate", for example on historical facts and mathematical reasoning.

## §7 Do the gains hold beyond the targeted benchmarks?

**(a) Similarity to the synthetic data.** phi-1's authors split HumanEval by similarity to CodeExercises (AST match at τ = 0.95). phi-1 solves 81.7% of the 71 similar problems and 26.9% of the 93 non-similar problems; StarCoder-Prompted solves 57.7% and 29.0% ([[phi-textbooks]] Table 3). **Result (single study)**. On the non-similar subset, phi-1 is 2.1 points below StarCoder-Prompted, so its advantage is concentrated on problems that resemble its exercises (Interpretation). On 50 new problems graded by GPT-4, phi-1 scores 52% and StarCoder 51% (Table 2).

**(b) Fresh, distribution-matched problems.** GSM1k has 1,205 grade-school problems written by annotators "without assistance from any LLM", matched to GSM8K on solution steps, answer magnitude, and human solve rate ([[gsm1k]] §1, §3). The authors report that "the Phi and Mistral families of models, show systematic tendencies to perform stronger on GSM8k compared to GSM1k for almost every release and scale" (§5.1), while frontier models show minimal gaps (§5.2).

The gap is tested with a two-proportion z-test:

`z = (p_8k − p_1k) / sqrt( p̄ (1 − p̄) (1/n_8k + 1/n_1k) )`, with `p̄ = (p_8k n_8k + p_1k n_1k) / (n_8k + n_1k)`

where p_8k and p_1k are accuracies on GSM8K and GSM1k, n_1k = 1,205, and n_8k = 1,319 GSM8K test problems ([[metamath]] §4.1).

Worked example for phi-2 (App. F): p_8k = 0.566, p_1k = 0.504. p̄ = (0.566 × 1,319 + 0.504 × 1,205) / 2,524 = 0.5364. Standard error = sqrt(0.5364 × 0.4636 × (1/1,319 + 1/1,205)) = sqrt(0.2487 × 0.001588) = 0.0199. z = 0.062 / 0.0199 = 3.12; the appendix prints 3.167 from unrounded accuracies. Other rows: Phi-3-mini-4k-instruct gap 0.040 (z 2.385), Phi-3-medium-128k-instruct 0.044 (z 3.103), phi-1.5 0.051 (z 2.814), Mistral-7B-v0.1 0.027 (z 1.421), gpt-4 −0.012. Under an alternative prompt, Phi-3-mini-4k-instruct's gap falls to 0.007 and Phi-3-mini-128k-instruct's rises to 0.035 (App. F), so a single-prompt verdict for one checkpoint is not stable.

**Conditions and limits.** The authors find a Spearman correlation of 0.36 between the gap and each model's per-character likelihood of the GSM8K test set, conclude that "data contamination is likely not the full story", and name benchmark-like training data and benchmark-based checkpoint selection as other possible causes (§5.4). They also report that phi-2 "is still able to correctly solve over half of GSM1k problems" (§5.3). GSM1k does not identify which part of the Phi training data produced the gap.

**(c) Post-cutoff contests.** Phi-4 was evaluated on the November 2024 AMC-10/12 (78 questions released on or after November 6, 2024, after its training data were collected) and averages 91.8 of 150, against 77.4 for Qwen 2.5 14B-Instruct and 89.8 for Gemini Pro 1.5 ([[phi-4]] §1.1, Fig. 1). The authors read this as evidence that phi-4's MATH score is not due to overfitting. Footnote 8 discloses that all three final candidates scored above 89 and that the final model was chosen after the other two candidates' scores were seen. **Result (single study)**, with the fresh set partly involved in model selection.

**(d) Knowledge recall and hallucination.** Phi-4 scores 3.0 SimpleQA F1 and 63.0 IFEval in Table 1. The authors name SimpleQA, DROP, and IFEval as its weakest benchmark scores, consider the first two reductive, call IFEval "a real weakness", and attribute it to synthetic data focused on Q&A and reasoning (§6, §8). Its synthetic-only 13B ablation lost 14.8 TriviaQA points (Table 3).

**(e) Filtered real data as the comparison.** FineWeb-Edu keeps web pages that a classifier trained on Llama-3-70B-Instruct annotations scores at 3 or above. At 1.71B parameters and 350B tokens it raises MMLU from 33% to 37% and ARC from 46% to 57% over FineWeb ([[fineweb]] §4). The threshold of 3 was chosen as a trade-off against benchmarks such as HellaSwag, and the filter shifts topics toward education and history and away from business, entertainment, and travel (§4, §4.1). **Result (single study)**. Classifier filtering of real text and textbook-style synthesis both move scores toward knowledge and reasoning benchmarks and change the topic distribution. **Open question:** no source in this chapter compares FineWeb-Edu and Cosmopedia at equal tokens on a broad suite; the only equal-token comparison of synthetic and web data here is Phi-4's Table 4.

## §8 Measuring coverage and the bias of a curated taxonomy

**Sources of bias in a taxonomy.**
1. **Selection by the curator.** GLAN's disciplines are GPT-4 proposals filtered by annotator majority vote; which fields were removed is not reported ([[glan]] §3.1).
2. **Classifier drift toward some topics.** Phi-4's web classifier "tends to over-index on STEM-related keywords", so the authors built a separate pipeline to amplify non-STEM content such as arts, history, travel, culture, and entertainment ([[phi-4]] §2.3).
3. **Hard-to-list domains.** The Persona Hub authors argue that comprehensive key-point lists like GLAN's are hard to curate beyond narrow domains ([[persona-hub]] Connections, §1; Interpretation).
4. **Easier generated prompts.** Nemotron-4's synthetic prompts received higher helpfulness scores than LMSYS prompts, 3.24 vs 3.04 ([[nemotron-4-synthetic]] Figure 3).

**Evidence that concept coverage matters.** In MathScale's 25K-example ablation on LLaMA-2 7B, halving the knowledge points lowered the MWPBench macro average by 8.6% relative, halving the topics by 2.3%, and halving the seed questions by 2.9% ([[mathscale]] Table 6). MathScale builds its concept graph from seed questions rather than from a curated list, and its CollegeMath seeds cover only algebra, precalculus, and calculus. MathScale-7B scores 0.6 on differential equations and 5.0 on linear algebra, but GPT-4 also scores 1.2 on differential equations, so the low scores cannot be attributed to missing seed topics alone (App. A.5, Table 10). **Result (single study)**.

**Procedure for a coverage audit (course recommendation, Interpretation).** It combines GLAN's per-discipline reporting with real-prompt comparison:
1. Map a sample of real user prompts (for example LMSYS-Chat-1M, which Nemotron-4 uses) to taxonomy leaves with a classifier or an LLM, and record prompts that map to no leaf.
2. Compute data coverage per leaf (training examples per leaf) and the unmapped share.
3. Build a held-out evaluation set per leaf, including prompts written independently of the generator, and report per-leaf scores against the base model.
4. Add fresh or perturbed versions of any benchmark that resembles the generated format (§7b, §7c).

Worked example with illustrative counts: 1,000 real prompts, of which 870 map to leaves and 130 do not, give an unmapped share of 13.0%. If the model improves on 100 of 126 disciplines, regresses on 20, and is unchanged on 6, the report lists the 20 regressions by name, as GLAN's Table 8 does for American history, Divinity, and Radiology, instead of averaging them into one score.

## Negative samples and negative feedback

Negative signals take four forms: negative marginal value (a sample that hurts when used as a positive target, and is discarded), negative as content (a failure or refusal used as an ordinary target), negative as conditioning (a failure trained under a control token), and negative as gradient (an explicit decrease of a sample's likelihood). Only the last removes probability mass from the sample. None of this chapter's sources uses negatives as conditioning. Full derivations are in ch-43a.

**Where negatives come from in these pipelines.**
- *Discarded (negative marginal value):* Cosmopedia removes benchmark-overlapping documents (ratio > 0.5) ([[cosmopedia]]); Nemotron-4 drops dialogues below a reward-model threshold (threshold not given) ([[nemotron-4-synthetic]] §3.2.2); Phi-4 discards seed questions whose sampled answers all agree or are entirely inconsistent ([[phi-4]] §2.2). GLAN discards nothing by quality; wrong GPT-3.5-turbo answers remain as SFT targets ([[glan]] §3.1).
- *Negative as content:* Nemotron-4 trains refusals for tasks the model cannot do as ordinary SFT targets (§3.2.5); Phi-4 trains (question, refusal) where the base model is usually wrong and (bogus question, refusal) ([[phi-4]] App. A.1).
- *Negative as gradient:* Nemotron-4's DPO and RPO rejected responses (§3.3.2); Phi-4's Pivotal Token Search pairs, where the rejected completion is one token that lowers the estimated success probability, and refusal DPO pairs (refusal > wrong) on the first 5 tokens ([[phi-4]] §4.3, App. A.1).

**Mechanism.** For a softmax over logits z, `∂ log p_y / ∂ z_j = 1[j = y] − p_j`, where p_j is the probability of token j and y is the pushed-down token. A gradient step that lowers log p_y changes the logits by `Δz_j = η (p_j − 1[j = y])`: every other token's logit rises in proportion to its current probability. Worked example with η = 1 and p = (0.6, 0.3, 0.1), pushing down token 3: Δz = (0.6, 0.3, −0.9), and the new probabilities are (0.710, 0.263, 0.026). Token 3 loses 0.074, token 1 gains 0.110, and token 2 loses 0.037. The removed mass goes to the most likely alternative. When that alternative is also wrong, or when chosen and rejected responses share tokens, both can fall; Nemotron-4 reports both chosen and rejected likelihoods falling under DPO (§3.3.2).

**Evidence.**
- Nemotron-4: DPO changed MT-Bench from 7.99 to 7.90; RPO, whose target is the reward-model score gap, and an added SFT loss on chosen responses reached 8.22 MT-Bench and 79.9 IFEval (Table 6). No controlled DPO-versus-RPO ablation is reported.
- Phi-4 (Table 9): after the PTS DPO stage GPQA is 53.6 and MATH 80.5, against 52.4 and 77.6 when only the judge-guided DPO stage is run. The authors state that PTS avoids the noise of full-length pairs in which low-probability tokens receive strong gradients (§4.3).
- Phi-4 refusal data (Fig. 6): SimpleQA correct / not attempted / incorrect moves from 6.8 / 3.2 / 90.0% for the base to 3.0 / 81.1 / 15.8% for the final model. Correct answers fell by 3.8 points while incorrect answers fell by 74.2 points, and the F1 score fell, which the authors discuss in App. A.1.

**Controls and diagnostics.** The controls used in these sources are localization of the negative (PTS single tokens; refusal DPO on the first 5 tokens), a positive NLL anchor (Nemotron's SFT loss on chosen responses), and a question filter of 0.2 ≤ p(success) ≤ 0.8 so that pivotal tokens exist (PTS). When negatives are used as gradient, log chosen and rejected log-probabilities separately and report correct, abstained, and incorrect rates rather than one F1 number, because Nemotron-4 observed both likelihoods falling under DPO (§3.3.2) and Phi-4's F1 fell while incorrect answers fell by 74.2 points (Fig. 6).

**Effect on generality.** Refusal training trades correct answers for fewer hallucinations on questions beyond the model's knowledge (Fig. 6); over-refusal on answerable questions is not reported for Phi-4.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GLAN-7B (Mistral 7B) | 7B | SFT | data | 10M question-answer pairs; decontaminated against test and training sets of evaluated benchmarks | arXiv:2402.13064v1 §3.1 | verified 2026-09-14 | no ablation reported |
| GLAN-7B | 7B | SFT | epochs; batch; LR schedule; loss masking | 3; 512 pairs; 3e-6 cosine, 1,000 warm-up steps, final 0; response tokens only | §3.2 | verified 2026-09-14 | no ablation reported |
| GLAN data generation | — | SFT data | subjects; syllabus; questions; answers | GPT-4, 10 queries per discipline, T 1.0, top-p 0.95; GPT-4 one query per subject; 1–2 sessions and 1–5 key concepts; GPT-3.5-turbo T 0.7, top-p 0.95 | §2.4, §3.1 | verified 2026-09-14 | no ablation reported |
| GLAN-7B | 7B | SFT | sequence length, packing, optimizer, compute | not reported | checked §2, §3, §5, App. A | not reported | — |
| Nemotron-4-340B-Instruct | 340B | SFT | Code SFT examples; epochs; LR; global batch | ~800K (Genetic Instruct); 1; constant 3e-7; 128 | arXiv:2406.11704v2 §3.3.1 | verified 2026-09-14 | Table 6: HumanEval 57.3 → 70.7 |
| Nemotron-4-340B-Instruct | 340B | SFT | General SFT examples; code replay; epochs; batch; LR | 200K; 2% of Code SFT samples; 3; 128; searched in [1e-7, 5e-7] | §3.3.1 | verified 2026-09-14 | Table 6: MMLU 72.2 → 78.3, HumanEval 70.7 → 66.5; staging vs single stage: no numbers |
| Nemotron-4-340B-Instruct | 340B | SFT + preference | human data | ~20K (10K SFT, 10K HelpSteer2) | §3.2 | verified 2026-09-14 | — |
| Nemotron-4-340B-Reward | 340B | reward-model | data; head | 10K HelpSteer2; linear projection to 5 attributes | §3.1 | verified 2026-09-14 | RewardBench 92.0 (Table 4) |
| phi-1-base | 1.3B | pretrain-stable | data; seq length; LR; warmup; weight decay; batch | ~6B filtered code + <1B synthetic textbooks; 2048; 1e-3; 750 steps; 0.1; 1024 (unit not stated) | arXiv:2306.11644v2 §2, §2.3 | verified 2026-09-14 | Fig. 2.1: CodeTextbook vs The Stack+ |
| phi-1-base | 1.3B | pretrain-stable | planned steps; released checkpoint | 36,000; step 24,000, ~8 epochs, a little over 50B tokens | §2.3 | verified 2026-09-14 | selection rule not reported |
| phi-1 | 1.3B | SFT | data; batch; LR; warmup; weight decay; steps | CodeExercises ~180M tokens; 256; 1e-4; 50; 0.01; 6,000 | §2.3 | verified 2026-09-14 | Fig. 2.1: largest HumanEval gain from this stage |
| phi-1.5 | 1.3B | pretrain-stable | mixture; dataset; tokens seen | 80% new synthetic, 20% phi-1 data; 30B; 150B | arXiv:2309.05463v1 §2.3, Table 1 | verified 2026-09-14 | no ablation of the split |
| phi-1.5 | 1.3B | pretrain-stable | LR; optimizer; batch | 2e-4 constant, no warmup; Adam (0.9, 0.98), eps 1e-7, weight decay 0.1; 2048 (unit not stated) | §2.3 | verified 2026-09-14 | no ablation reported |
| phi-1.5-web | 1.3B | pretrain-stable | mixture; dataset; tokens seen | ~40% filtered web, 20% phi-1 code, 40% synthetic; 100B; 300B | §2.4, Table 1 | verified 2026-09-14 | Tables 2–4 vs phi-1.5 (different tokens seen) |
| phi-3-mini | 3.8B | pretrain-stable | tokens; phase split; LR | 3.3T; not reported; not reported | arXiv:2404.14219v4 §2 | verified 2026-09-14 (tokens); not reported (split, LR) | no ablation reported |
| phi-4 | 14B | pretrain-stable | tokens; schedule; peak LR; weight decay; global batch | ~10T; linear warm-up and decay; 3e-4; 0.1; 5760 (unit not stated) | arXiv:2412.08905v1 §3 | verified 2026-09-15 | tuned by interpolation from shorter runs (§3); no table |
| phi-4 | 14B | pretrain-stable | mixture (share of training tokens; unique tokens; epochs) | web 15%, 1.3T, 1.2; web rewrites 15%, 290B, 5.2; synthetic 40%, 290B, 13.8; code 20%, 820B, 2.4; acquired 10%, 580B, 1.7 | §3.2, Table 5 | verified 2026-09-15 | Table 4 (7B, 1T tokens): S +0.8, S+W 0.0, uniform −2.2 average vs final |
| phi-4 | 14B | pretrain-stable | synthetic data types; unweighted tokens | 50 types; about 400B | §2.2 | conflict (not reconciled with Table 5 unique counts in the report) | no ablation reported |
| phi-4 | 14B | long-context | context; tokens; data; RoPE base; LR | 4K → 16K; 250B; 30% new long data, 70% recall tokens; 250K; max LR ÷ 10 | §3.3 | verified 2026-09-15 | "former [natural long data] perform better" than padded samples; no numbers |
| phi-4 | 14B | SFT | LR; tokens | 1e-6; about 8B | §4.1 | verified 2026-09-15 | Table 9 stage scores |
| phi-4 | 14B | preference | rounds; pairs | PTS DPO (Table 7 mixture); judge-guided DPO ~850K pairs, GPT-4o judge | §4.2 | verified 2026-09-15 | Table 9: PTS stage GPQA 53.6 vs stage-2-only 52.4 |
| phi-4 | 14B | preference | PTS question filter | 0.2 ≤ p(success) ≤ 0.8 | §4.3 | verified 2026-09-15 | stated reason: pivotal tokens are rare outside this range; no ablation |
| Phi-4-reasoning-plus | 14B | RL | algorithm; steps; examples; trajectories; checkpoint rule | GRPO; 90; ~6k; 8 per example; best AIME 2024 | arXiv:2504.21318v1 §4.2 | verified 2026-09-15 | Fig. 7a: AIME +>10% at 90 steps, no further gain |
| Cosmopedia (March 2024 release) | — | pretrain data | generator; files; tokens; compute | Mixtral-8x7B-Instruct-v0.1; over 30M; 25B; over 10k GPU hours | huggingface.co/blog/cosmopedia (2024-03-20) | verified 2026-09-15 | cosmo-1b vs TinyLlama (figure only) |
| Cosmopedia (March 2024 release) | — | pretrain data | decontamination rule | 10-gram candidates; discard if matched / benchmark length > 0.5 | same blog | verified 2026-09-15 | removal counts table; no effect measured |
| cosmo-1b | 1B | pretrain-stable | tokens; LR; batch | not reported | checked the blog | not reported | — |
| FineWeb-Edu | — | pretrain data | classifier threshold | ≥ 3 (1.3T tokens kept) | arXiv:2406.17557v2 §4, App. F.2 | verified 2026-09-14 | FW-Edu-2/3/4 at 28B tokens; trade-off vs HellaSwag |

Rows dated 2026-09-15 were read in the primary PDFs or blog on that date because the library cards [[phi-4]] and [[hf-cosmopedia]] had not been verified; the chapter excerpts [[phi-4]] and [[cosmopedia]] hold the checked extracts.

**Starting point for a small general-purpose run.** For SFT of a 7B base on taxonomy-generated question-answer pairs, the verified reference is GLAN-7B: 3 epochs, batch 512 pairs, LR 3e-6 with cosine decay to 0 after 1,000 warm-up steps, loss on response tokens only. These values were used for Mistral 7B on 10M pairs; none was ablated, and sequence length and optimizer are not reported. The same run lowered MMLU humanities, social sciences, and other categories (Table 2), so the per-category evaluation in the Generalization lens is part of the recipe.

## Generalization lens

**(a) What increases breadth.**
- Discipline-wide generation without benchmark data: GLAN-7B improves all eight Table 1 columns over base Mistral, while math and code specialists lose on other columns (CodeAlpaca GSM8K 34.6 vs 43.4) ([[glan]] Table 1, §3.3).
- Concept combinations: halving knowledge points costs more than halving seed questions, −8.6% vs −2.9% relative ([[mathscale]] Table 6).
- Adding knowledge-heavy web data to a synthetic-heavy mixture: TriviaQA +6.9 for S+W vs −3.0 for S at 7B, 1T tokens ([[phi-4]] Table 4).
- Filling a gap found by evaluation: Cosmopedia added common-sense stories after finding that textbook-trained models lacked grade-school knowledge ([[cosmopedia]]); no before/after numbers are reported.
- Real prompts alongside generated ones: Nemotron-4 found LMSYS prompts harder than its synthetic prompts ([[nemotron-4-synthetic]] Figure 3).

**(b) What causes narrowing or forgetting.**
- Topic list without format coverage: GLAN IFEval 34.0 vs Mistral-Instruct 32.0 and GPT-3.5-turbo 53.8 ([[glan]] Table 4); Phi-4 IFEval 63.0, named by the authors as a weakness of Q&A-focused synthetic data ([[phi-4]] §6).
- Knowledge recall: TriviaQA −14.8 for synthetic-only pretraining ([[phi-4]] Table 3); phi-3-mini TriviaQA 64.0 vs 82.2 ([[phi-3]] §3); GLAN MMLU humanities 54.9 vs 56.5 ([[glan]] Table 2).
- Benchmark-shaped data: GSM8K-over-GSM1k gaps across Phi and Mistral releases ([[gsm1k]] §5.1); phi-1 81.7% vs 26.9% on similar vs non-similar HumanEval problems ([[phi-textbooks]] Table 3).
- Stage order: General SFT lowered Nemotron-4 HumanEval from 70.7 to 66.5 despite 2% code replay ([[nemotron-4-synthetic]] Table 6).
- Classifier topic drift: STEM over-indexing in Phi-4's web classifier ([[phi-4]] §2.3); FineWeb-Edu down-samples business, entertainment, and travel ([[fineweb]] §4.1).

**(c) How to measure it for this stage.**
- Per-category and per-discipline scores next to every aggregate ([[glan]] Tables 2, 8).
- A similarity split of each benchmark against the synthetic data ([[phi-textbooks]] Table 3).
- A distribution-matched fresh set with a significance test and more than one prompt format ([[gsm1k]] App. F), and post-cutoff contests held out from model selection ([[phi-4]] §1.1, footnote 8).
- A train-versus-test loss gap where benchmarks have training splits ([[glan]] §3.4).
- Knowledge-recall and abstention metrics reported as correct / not attempted / incorrect ([[phi-4]] Fig. 6).
- Known measurement errors: n-gram decontamination misses rephrased items ([[phi-4]] §5); GLAN-Test is drawn from GLAN's own generated data and PhiBench is written by the Phi team; GPT-4 judges add judge bias ([[glan]] §3.5).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Reporting only the aggregate after taxonomy-driven SFT | MMLU rises while humanities or social-science categories fall | per-category MMLU and per-discipline held-out scores ([[glan]] Table 2, Table 8) |
| Treating data coverage as model coverage | every discipline has data, some disciplines regress | per-leaf evaluation against the base model (§8) |
| Using unverified teacher answers for factual domains | errors on memorization-heavy questions | sample answers per discipline and check against references; compare STEM and humanities changes ([[glan]] §3.3) |
| Synthetic-heavy pretraining without knowledge-heavy web data | reasoning scores rise, TriviaQA and SimpleQA fall, more hallucination | TriviaQA or similar recall benchmark at each mixture ablation ([[phi-4]] Tables 3–4) |
| Generating exercises in a benchmark's format | accuracy on benchmark items similar to the training data exceeds accuracy on non-similar items (phi-1: 81.7% vs 26.9%) | AST or embedding similarity split ([[phi-textbooks]] Table 3) |
| Claiming generality from GSM8K or HumanEval alone | score gap on a fresh matched set | GSM1k-style set, z-test, two prompt formats ([[gsm1k]] App. F) |
| Using a fresh set to pick among final candidates | fresh-set score is no longer independent | record which candidates were scored before selection ([[phi-4]] footnote 8) |
| Varying only audience or style words in prompts | near-duplicate generations | duplicate rate per prompt family; add explicit content differences ([[cosmopedia]]) |
| Mixing all SFT behaviors in one stage when one behavior conflicts | reweighting does not fix one task family | stage-wise evaluation table as in [[nemotron-4-synthetic]] Table 6 |
| Relying on n-gram decontamination only | benchmark paraphrases remain | paraphrase-aware check or fresh sets ([[phi-4]] §5) |

## Check your understanding

1. GLAN's discipline list gives 100% data coverage of its 126 disciplines. Explain why GLAN-7B can still regress on MMLU humanities, and which step of the pipeline could produce that regression.
2. Using the two-session combination formula, explain why sampling concept pairs across sessions increases question variety more than sampling within one session, and which combinations neither strategy can produce.
3. In Phi-4's Table 4, the synthetic-heavy mixture had the highest average, yet the authors chose a mixture with more web data. What measurement drove the choice, and what does the post-training remark imply about when mixture ablations should be evaluated?
4. phi-1 solves 81.7% of HumanEval problems similar to its exercises and 26.9% of the others. Why does this split not show contamination, and what does it show about the source of phi-1's advantage over StarCoder?
5. GSM1k finds a Spearman correlation of only 0.36 between the gap and GSM8K test-set likelihood. Give two mechanisms other than test-set leakage that could make a model trained on textbook-style math exercises score higher on GSM8K than on GSM1k.
6. Nemotron-4's General SFT stage included 2% of the code data and still lowered HumanEval. Explain why replay at a small share may not prevent the loss, and what evidence from the same table indicates the loss was recoverable.
7. Phi-4's refusal training lowered SimpleQA correct answers from 6.8% to 3.0%. Using the softmax gradient, explain why DPO pairs of (refusal > wrong answer) on the first 5 tokens can also remove probability from correct answers.

## Connections

- Previous: ch-20 — Distillation as Data: Explanation Traces and the R1-Distill Lineage (teacher ceilings apply to GLAN's GPT-4 and GPT-3.5-turbo data and to Phi-4-reasoning's o3-mini traces).
- Next: ch-22 — Quality, Diversity, and Gradient-Based Data Selection (how to select from taxonomy-generated and seed-generated pools).
- ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix; ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing (seed-based alternatives and Persona Hub).
- ch-23 — Model Collapse and Verification of Synthetic Data (unverified answers in §2); ch-24 — Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data (MathScale and concept-level math data).
- ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection (FineWeb-Edu and Phi classifiers); ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination (Phi-4's 13.8 epochs over synthetic data).
- ch-29e — Instruction Tuning and Generalization to Unseen Tasks; ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares (Nemotron-4's staged SFT).
- ch-34 — Case Studies B: Generality versus Specialization in Qwen, OLMo, and Phi Reports; ch-35 — Distillation in Practice A: Where Labs Insert Teacher Data.
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation; ch-48 — Contamination Detection and Its Effect on Reported Scores.

## Sources

- [[glan]] — pipeline, Algorithm 1, combination formulas, generation and training settings, Tables 1–4, 6, 8, loss-gap check (card verified 2026-09-14; chapter excerpt of the same name).
- [[nemotron-4-synthetic]] — human versus synthetic data, prompt synthesis with 3K topics, reward-model roles, staged SFT, Table 6, DPO and RPO observations.
- [[nemotron-4-synthetic-recipe]] — SFT, DPO, and RPO settings with loci.
- [[phi-textbooks]] — phi-1 classifier, synthetic textbooks and exercises, filter ablation, similarity split, training settings.
- [[phi-1-5]] — 20K-topic seeding, phi-1.5 vs phi-1.5-web vs Llama2-7B tables, training settings.
- [[phi-3]] — two-phase data description, TriviaQA gap, tokens.
- [[phi-4]] — chapter excerpt read from arXiv:2412.08905v1 and arXiv:2504.21318v1: synthetic data types, Tables 3–5 and 9, midtraining, PTS, refusal data and Fig. 6, AMC evaluation, weaknesses, Phi-4-reasoning-plus RL steps. The library card of the same name was not verified and was not used for numbers.
- [[cosmopedia]] — chapter excerpt read from the Hugging Face blog: prompt curation, audience and style, web clusters, stories, decontamination rule and counts, cosmo-1b result.
- [[hf-cosmopedia]] — library card for the same blog; not verified at the time of writing and not used for numbers.
- [[gsm1k]] — chapter excerpt of arXiv:2405.00332v4: construction, family-level overfitting, likelihood correlation, App. F rows and prompt sensitivity.
- [[fineweb]] — FineWeb-Edu classifier, threshold trade-off, MMLU and ARC gains, topic shift.
- [[mathscale]] — knowledge-point versus seed ablation and per-topic gaps.
- [[persona-hub]] — argument that key-point lists are hard to curate beyond narrow domains.
- [[metamath]] — GSM8K test-set size used in the z-test.

<!-- scope: concept-graph synthesis of 2M math question-answer pairs (MathScaleQA) with GPT-3.5; MWPBench 10-dataset math evaluation
     deps: [[metamath]]
     see-also: [[wizardmath]], [[mammoth]], [[glan]], [[openmathinstruct-2]]
-->

# MathScale: Scaling Instruction Tuning for Mathematical Reasoning
- **Core Insight:** Fine-tuning LLaMA-2 7B on 2M GPT-3.5-written math question–answer pairs, generated from sampled combinations of extracted topics and knowledge points, gives 35.0 micro / 37.5 macro average accuracy on the 10-dataset MWPBench, versus 24.5 / 26.1 for MetaMath-7B (Table 5); accuracy grows about logarithmically with dataset size up to 2M examples (§5.1, Fig. 3).
- **Guideline:** When a math SFT set must grow beyond the variety of its seed questions, generate new questions from sampled topic and knowledge-point combinations, because in the 25K-example ablation halving the knowledge points lowered macro accuracy by 8.6% relative and halving the seed questions by 2.9% (Table 6). The final data has no solution check: GPT-4 marked 26% of a 5K sample of solutions as incorrect, and correcting them did not improve a 5K-example run (§5.3, Table 7).
- **Authors:** Zhengyang Tang, Xingxing Zhang, Benyou Wang, Furu Wei (The Chinese University of Hong Kong, Shenzhen; Microsoft Research Asia; Shenzhen Research Institute of Big Data)
- **Year:** 2024 (arXiv v1 2024-03-05; ICML 2024, PMLR vol. 235)
- **URL:** https://arxiv.org/abs/2403.02884
- **Source type:** paper
- **Relevant topics:** math reasoning, synthetic question generation, concept graph, distillation from GPT-3.5, benchmark construction, data scaling

## Abstract
MathScale is a method for creating math reasoning data with a frontier LLM (GPT-3.5). It extracts topics and knowledge points from seed math questions, builds a concept graph from their co-occurrence, samples concept combinations from the graph, and asks GPT-3.5 to write new questions with answers (Abstract, §3). The resulting MathScaleQA has 2M question–answer pairs. The authors also build MWPBench, a benchmark of ten math word problem datasets at K-12, college, and competition level, evaluated with one protocol. Fine-tuned on MathScaleQA, MathScale-7B has the highest micro and macro averages among LLaMA-2 7B models, 42.9% and 43.7% above the best same-size peer in relative terms (Abstract, Table 5).

## Key Contributions
- A three-step pipeline: concept extraction, concept graph construction, and generation from graph random walks (§3.1–3.3, Fig. 1).
- MathScaleQA: 2M GPT-3.5-generated pairs combined with the MWPBench training set (§4.1).
- MWPBench: 10 datasets with 20,857 training and 18,408 test examples, including the new CollegeMath (1,281 train / 2,818 test from nine textbooks) (Tables 1–2).
- A scaling curve to 2M examples and ablations on seed questions, concept counts, and solution validation (§5.1–5.3).
- Fresh-GaokaoMath-2023: 30 questions from the June 2023 Gaokao, released after LLaMA-2 and GPT-3.5-Turbo (§5.4).

## Key Figures/Tables to Study
- Tables 3–4: prompt templates for concept extraction and for question generation.
- Table 5: MWPBench accuracy per dataset for LLaMA-2 7B, 13B and Mistral 7B bases.
- Fig. 3: accuracy vs MathScaleQA size.
- Table 6: seed-question and concept ablations (25K examples). Table 7: validation ablation (5K examples).
- Tables 9–10 (App. A.5): per-topic accuracy on MATH and CollegeMath.

## Technical Details
Concept extraction
- Seeds: the MWPBench training set, about 20K questions from several source datasets (§3.1, §4.1).
- GPT-3.5-Turbo-0613, prompted to "Act as a Math Teacher", returns 1–2 topics and 1–5 knowledge points (KPs) per question (§3.1, Table 3).
- Topics and KPs that occur only once are removed. The result is 2,018 topics and 8,892 KPs (§3.1, §4.1).

Concept graph
- Nodes are topics and KPs. Edges (topic–topic, topic–KP, KP–KP) connect two concepts extracted from the same seed question (§3.2).
- Edge weight: f_co(u, v) = log(w_uv + ε) (Eq. 1), with ε = 1e-5 (§4.1).
  - w_uv: number of seed questions from which both u and v were extracted; ε: small constant that keeps the logarithm finite.
- Walk probability: p_uv = exp(f_co(u, v)) / Σ_{v′ ∈ N(u)} exp(f_co(u, v′)) (Eq. 2).
  - N(u): neighbours of u in the current sub-graph. Because exp(log(w_uv + ε)) = w_uv + ε, the walk moves to v with probability proportional to w_uv + ε, so frequently co-occurring concepts are sampled more often (derived from Eqs. 1–2).
- Worked example: if topic u has two neighbours with counts 3 and 1, the walk picks them with probabilities 3/4 and 1/4 (ε ignored).

Generation
- One composition: start at a topic (implemented by enumerating all topics for many epochs); walk 1–2 steps in the topic graph; 1 step in the topic–KP graph; 0–4 steps in the KP graph (§3.3).
- Approximately 1K epochs over all topics give 2M unique compositions; GPT-3.5-Turbo-0613 writes one question and its solution per composition in a single prompt (§4.1, Table 4).
- Few-shot examples in that prompt are seed questions selected by Jaccard distance between KP sets (§3.3).
- Decontamination removes all MWPBench test questions from the generated data; the matching method is not described (§3.3, §4.1).
- The GPT-4 validation step of §3.4 is excluded from the final pipeline (§4.1).

Evaluation and results
- MWPBench protocol: zero-shot, greedy decoding, Alpaca template, fuzzy answer matching; multiple-choice items converted to word problems; Chinese items translated to English with GPT-3.5-Turbo (§2.2, App. A.1–A.2).
- MathScale-7B: GSM8K 66.3, MATH 31.1, micro 35.0, macro 37.5. MetaMath-7B: 66.2, 20.6, 24.5, 26.1. MAmmoTH-7B: micro 15.6, macro 17.2 (Table 5).
- MathScale-13B: micro 37.1, macro 39.1. MathScale-Mistral-7B: 38.7, 40.8. GPT-3.5-Turbo: 39.8, 41.5. GPT-4: 52.0, 54.2 (Table 5).
- Ablations (LLaMA-2 7B, 25K examples, MWPBench macro average): full pipeline 14.5; 50% of seed questions 14.0 (−2.9%); seeds from GSM8K and MATH only 13.9 (−3.5%); 50% of topics 14.1 (−2.3%); 50% of KPs 13.2 (−8.6%) (Table 6). Percentages are relative changes.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| MathScale-7B (LLaMA-2 7B), MathScale-13B (LLaMA-2 13B), MathScale-Mistral-7B | 7B, 13B | distill-SFT | batch size (unit not stated) | 128 | arXiv:2403.02884v1 §4.1 | verified 2026-09-14 | no ablation reported |
| same three models | 7B, 13B | distill-SFT | epochs | 3 | arXiv:2403.02884v1 §4.1 | verified 2026-09-14 | no ablation reported |
| same three models | 7B, 13B | distill-SFT | learning rate | 2e-5 | arXiv:2403.02884v1 §4.1 | verified 2026-09-14 | no ablation reported |
| same three models | 7B, 13B | distill-SFT | training data | MathScaleQA: 2M GPT-3.5-Turbo-0613 pairs + MWPBench training set (20,857, Table 1) | arXiv:2403.02884v1 §4.1 | verified 2026-09-14 | Fig. 3 (7B only): near-logarithmic gain up to 2M |
| same three models | 7B, 13B | distill-SFT | prompt format; code base | Alpaca prompt; adapted from open-instruct | arXiv:2403.02884v1 §4.1 | verified 2026-09-14 | no ablation reported |
| same three models | 7B, 13B | distill-SFT | solution validation | none in final data | arXiv:2403.02884v1 §4.1, §5.3 | verified 2026-09-14 | Table 7 (7B, 5K): GPT-3.5 solutions 10.6/11.5; 26% GPT-4-corrected 10.2/11.1; all GPT-4 9.8/10.9 (micro/macro) |
| MathScaleQA generation | n/a | distill-SFT | topics; KPs; ε; epochs over topics | 2,018; 8,892; 1e-5; approximately 1K | arXiv:2403.02884v1 §4.1 | verified 2026-09-14 | Table 6 (7B, 25K): fewer KPs −8.6%, fewer topics −2.3% |
| same three models | 7B, 13B | distill-SFT | LR schedule, warmup, optimizer, max sequence length, packing, loss masking, compute | not reported | checked v1 §4, App. A | not reported | none |
| same three models | 7B, 13B | eval-gate | decoding | zero-shot, greedy, Alpaca template | arXiv:2403.02884v1 §2.2 | verified 2026-09-14 | not applicable |

## Findings relevant to generality, negative feedback, distillation
Generality
- MWPBench includes three sets without a training split (GaokaoBench-Math, AGIEval-Gaokao-Math, AGIEval-SAT-Math); the authors state MathScale-7B outperforms the other open 7B models on them (§4.3, Table 5).
- CollegeMath seeds cover only algebra, precalculus, and calculus. MathScale-7B scores 27.2 on vector calculus, 7.9 on probability, 5.0 on linear algebra, and 0.6 on differential equations; GPT-4 scores 1.2 on differential equations (App. A.5, Table 10).
- Restricting seeds to GSM8K and MATH lowers the 25K-example macro average by 3.5% relative (Table 6).
- Contamination check: on Fresh-GaokaoMath-2023 (30 questions), MathScale-7B 30.0, MetaMath-7B 16.6, WizardMath-7B 13.3, GPT-3.5-Turbo 40.0, GPT-4 43.3 (§5.4, Table 8).

Negative samples and distillation
- On 100 manually annotated generated questions, GPT-3.5-Turbo answers 69% correctly and GPT-4 87% (§5.3).
- Incorrect GPT-3.5 solutions stay in MathScaleQA as ordinary training targets; no negative-gradient term is used. In the 5K-example test, replacing them with GPT-4 solutions did not raise accuracy (Table 7).
- The authors interpret this as distillation: incorrect solutions still help the student imitate GPT-3.5's output distribution, as in sequence-level distillation for machine translation (Kim & Rush, 2016) (§5.3; Interpretation). Whether this holds at 2M examples is not tested.

## Connections
- [[metamath]] — augmentation baseline (answer augmentation, rephrasing, self-verification, FOBAR) that MathScale describes as staying close to the seed questions (§1, Table 5).
- [[wizardmath]] — evol-instruct-based baseline (§1, §6, Table 5).
- [[mammoth]] — baseline trained on 13 datasets with CoT/PoT rationales; evaluated here with CoT only (§4.2, Table 5).
- [[glan]] — taxonomy-driven synthesis of general instruction data (arXiv:2402.13064) whose author list includes all four MathScale authors; neither v1 cites the other.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2403.02884 (v1, 2024-03-05, full PDF including App. A); venue from https://proceedings.mlr.press/v235/tang24k.html.
- Corrections to the previous card version:
  - "MathScale-7B reaches 66.3% on the MwpBench" → 66.3 is GSM8K; MWPBench micro / macro averages are 35.0 / 37.5 (Table 5).
  - "Seed input: GSM8K, MATH, MetaMathQA" → the MWPBench training set, about 20K questions (§3.1, §4.1).
  - "~2K topics × ~5K knowledge points" → 2,018 topics and 8,892 knowledge points (§4.1).
  - "authors prioritize rare edges" → walk probability is proportional to co-occurrence count + ε, which favours frequent pairs (Eqs. 1–2).
  - "GPT-3.5 solves its own generated problem" as a separate step → one prompt writes question and solution together (Table 4).
  - "+5 on GSM8K and +3 on MATH over the same base trained on MetaMathQA" → versus MetaMath-7B: GSM8K 66.3 vs 66.2, MATH 31.1 vs 20.6 (Table 5).
  - "diminishing returns past 1.5M — concept graph saturates" → near-logarithmic growth up to 2M; the authors expect further gains (§5.1).
  - "Correctness verifier: NONE … chief weakness" → the paper tested GPT-4 validation and excluded it because it did not help at 5K (§5.3, Table 7).
  - Affiliations "Microsoft Research / CUHK-SZ" → CUHK-Shenzhen, Microsoft Research Asia, Shenzhen Research Institute of Big Data.
- Removed as unsupported by the source: MinHash near-duplicate filter (Jaccard > 0.7); novelty audit "~5% overlap (MinHash > 0.5), 95% new"; filter dropping unparseable solutions; trace lengths (200–800, ~400 tokens); API cost (~$20K); "MathScaleQA publicly released" (v1 states only a plan to open-source the evaluation framework, §2.2); "balanced coverage across algebra, geometry, probability, number theory"; "geometry with diagrams poorly served"; "difficulty skews easy-to-medium"; "used downstream in WizardMath-style recipes and GLAN"; Self-Instruct as ancestor (not cited by the source).
- Not reported by the source: correctness rate of the full 2M set; decontamination matching method; LR schedule, optimizer, sequence length, compute; number of runs or seeds for ablations.

<!-- scope: safety evaluation — hierarchical safety taxonomy, attack/defense-enhanced question subsets, and two fine-tuned judge models
     deps: [[harmbench-data]]
     see-also: [[wildguard-data]]
-->

# SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models
- **Core Insight:** A safety benchmark of 21,318 base questions organized into 6 domains, 16 tasks, and 66 categories, plus 5,000 attack-enhanced and 200 defense-enhanced questions and ~4,000 multiple-choice questions, shows that model safety rates collapse under attack: Gemini falls from 88.32% to 19.98% and GPT-4 from 93.49% to 80.28% (Table 5, Table 9).
- **Guideline:** When reporting LLM safety, report the base-set and attack-enhanced rates separately and break them down by taxonomy level, because the attack-enhanced subset reorders models: Claude2 leads on both sets (99.77% and 88.02%) while Gemini is close to GPT-3.5 on the base set and 53 points below it under attack (Table 5).
- **Authors:** Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, et al. (Shanghai AI Laboratory, Harbin Institute of Technology, Beijing Institute of Technology, CUHK, HK PolyU)
- **Year:** 2024 (arXiv v1 2024-02; v4 2024-06-07)
- **URL:** https://arxiv.org/abs/2402.05044
- **Source type:** paper
- **Relevant topics:** safety taxonomy, hierarchical benchmark, jailbreak evaluation, safety judge models, attack success rate

## Abstract
SALAD-Bench is a safety benchmark for evaluating LLMs, attack methods, and defense methods together. Its questions are organized by a three-level taxonomy of 6 domains, 16 tasks, and 66 categories. Beyond plain harmful questions it provides attack-enhanced, defense-enhanced, and multiple-choice subsets. Two evaluators accompany the data: MD-Judge, an LLM-based judge fine-tuned for question-answer pairs with emphasis on attack-enhanced queries, and MCQ-Judge, a regex- and in-context-learning-based parser for the multiple-choice subset. Experiments cover safety rates of many open and closed models, the effectiveness of six attack families, and the effectiveness of seven defense prompts. Data and evaluator are released at https://github.com/OpenSafetyLab/SALAD-BENCH.

## Key Contributions
- **Three-level taxonomy** — 6 domains, 16 tasks, 66 categories (§2.1, Figure 2).
- **Base set of 21,318 questions** assembled from a fine-tuned GPT-3.5 self-instruct generator plus eight public safety datasets (Table 9).
- **Three derived subsets** — 5,000 attack-enhanced, 200 defense-enhanced, ~4,000 multiple-choice (§3, §2.2).
- **MD-Judge**, a Mistral-7B judge fine-tuned with LoRA that reaches F1 0.818 on SALAD-Base-Test and 0.873 on SALAD-Enhance-Test, above GPT-4 at 0.785 and 0.827 (§4.1, Table 3).
- **MCQ-Judge**, a template-plus-regex evaluator for the multiple-choice subset (§4.2).
- **Joint attack/defense evaluation** — attack success rates for six attack families and safety rates under seven defense methods (Table 6, Table 7).

## Key Figures/Tables to Study
- Table 1 — comparison against ten prior safety benchmarks on size, taxonomy depth, and evaluator.
- Table 3 — MD-Judge F1 against Keyword, LlamaGuard, GPT-3.5, GPT-4 on five test sets.
- Table 5 — safety rate and Elo rating per model on the base set and the attack-enhanced subset.
- Table 6 — attack success rate per attack method on AdvBench-50, base questions, and enhanced questions.
- Table 7 — attack success rate per defense method across four target models.
- Table 9 — data sources and counts for the base set.

## Technical Details
- **Taxonomy:** six domain-level areas — Representation & Toxicity Harms, Misinformation Harms, Information & Safety Harms, Malicious Use, Human Autonomy & Integrity Harms, and Socioeconomic Harms — refined into 16 tasks and 66 categories (§2.1, §B).
- **Base-set composition (Table 9):** self-instructed from a fine-tuned GPT-3.5 15,433; HH-harmless 4,184; HH-red-team 659; AdvBench 359; Multilingual 230; Do-Not-Answer 189; ToxicChat 129; Do Anything Now 93; GPTFuzzer 42. Total 21,318.
- **Generator:** GPT-3.5-turbo fine-tuned on about 500 collected harmful QA pairs, then prompted with category-level taxonomy labels to self-instruct new questions (§2.2).
- **Deduplication:** locality-sensitive hashing over Sentence-BERT embeddings, to remove surface and semantic duplicates (§2.2).
- **Auto labeling:** each question's category is assigned by unanimous agreement of three models — Llama-2 (7B-class), Mistral-7B-Instruct, and TuluV2-dpo-70B — with human verification on a random sample (§2.2).
- **Attack-enhanced subset (§3.1):** base questions are sorted by rejection rate; those with low rejection rate are kept, giving a filtered set of about 4,000; each is enhanced with human-designed jailbreak prompts, red-teaming LLMs, and gradient-based methods; responses are generated on all candidate models and scored; the 5,000 enhanced questions with the highest unsafe score form the final subset.
- **Defense-enhanced subset (§3.2):** the mirror pipeline, keeping the highest-rejection-rate questions, yielding 200 questions with GPT-4-generated safe answers.
- **MCQ subset (§3.3):** K harmful questions sampled per category; safe candidate responses generated by GPT-4 and unsafe ones by the fine-tuned GPT-3.5; after parsing and human re-checking, three options are selected per question and two prompts are built per question (choose-safe and choose-unsafe), giving about 4,000 questions covering all categories.
- **MD-Judge training data (§4.1, §G):** public QA pairs from the BeaverTails, LMSYS-Chat (subset), and ToxicChat training sets, plus generated attack-enhanced QA pairs whose safety labels are assigned by GPT-4. The training-set size is not reported.
- **MD-Judge evaluation (Table 3):** F1 of 0.818 (SALAD-Base-Test), 0.873 (SALAD-Enhance-Test), 0.644 (ToxicChat), 0.866 (BeaverTails), 0.864 (SafeRLHF). GPT-4 scores 0.785 / 0.827 / 0.470 / 0.842 / 0.835; LlamaGuard scores 0.585 / 0.085 / 0.220 / 0.653 / 0.693. On the out-of-distribution HarmBench and LifeTox sets, accuracy is MD-Judge 83.72% / 79.27%, GPT-4 84.46% / 77.43%, GPT-3.5 61.13% / 73.39%, LlamaGuard 64.56% / 58.62% (Table 4).
- **Model safety rates (Table 5):** Claude2 99.77% base / 88.02% attack-enhanced; GPT-4 93.49 / 80.28; GPT-3.5 88.62 / 73.38; Gemini 88.32 / 19.98; Llama-2-70B 96.21 / 66.24; Llama-3-70B 84.45 / 63.72; Mistral-7B-v0.1 54.13 / 2.44; Mistral-7B-v0.2 80.14 / 6.40; Mixtral-8x7B 76.15 / 9.36; Qwen-72B 94.40 / 6.94; Yi-34B 87.13 / 23.74. Models are also ranked by Elo rating.
- **Attack methods evaluated (§5.1, Table 6):** TAP, AutoDAN, GPTFuzz, GCG, CoU, and human-designed jailbreaks. All attacks use Llama-2-7B-Chat as target. Maximized ASR on enhanced questions: human jailbreaks 89.5% (average 11%), GCG suffixes 25.5% (5.5%), GPTFuzzer 34%, AutoDAN 9–11%, TAP 1.5–5%, CoU 2%.
- **Defense methods (Table 7):** GPT-paraphrasing and Self-Reminder give the largest reductions; without defense, ASR is 34.28% on Llama-2-13B and 93.60% on Mistral-7B, and Self-Reminder brings Mistral-7B to 86.20%.
- **MCQ results (Table 8):** GPT-4 reaches 88.96% both overall accuracy and valid accuracy with a 0% rejection rate; Gemini Pro has a 43.85% rejection rate, giving 44.19% overall and 78.71% valid accuracy.
- **Metrics:** safety rate and Elo rating for models; attack success rate, defined as 1 minus the safety rate under MD-Judge, for attacks and defenses; overall accuracy Acc-O and rejection-excluded valid accuracy Acc-V for the MCQ subset (§5).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| MD-Judge | 7B | SFT (judge) | base model | Mistral-7B | arXiv:2402.05044v4 §5.2 Implementation Details | verified 2026-09-18 | §J compares alternative base models (Llama-2-7B, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.2) and template levels |
| MD-Judge | 7B | SFT (judge) | adapter | LoRA, rank 64 | §5.2 | verified 2026-09-18 | no ablation reported |
| MD-Judge | 7B | SFT (judge) | sequence length | 4096 | §5.2 | verified 2026-09-18 | §J notes Llama-2-7B has a shorter context than Mistral-7B |
| MD-Judge | 7B | SFT (judge) | epochs | 2 | §5.2 | verified 2026-09-18 | no ablation reported |
| MD-Judge | 7B | SFT (judge) | batch size | 16 per GPU on 8×A100 | §5.2 | verified 2026-09-18 | no ablation reported |
| MD-Judge | 7B | SFT (judge) | training data | BeaverTails train + LMSYS-Chat subset + ToxicChat train + generated attack-enhanced QA pairs labeled by GPT-4 | §4.1, §G | verified 2026-09-18 | Table 3: enhanced-pair F1 0.873 vs GPT-4 0.827 |
| MD-Judge | 7B | SFT (judge) | training-set size | not reported | §4.1, §G, Table 3 checked | not reported | — |
| Question generator | — | SFT | base model and data | GPT-3.5-turbo fine-tuned on ~500 harmful QA pairs | §2.2 | verified 2026-09-18 | no ablation reported |
| Question generator | — | generation | output | 15,433 self-instructed questions of the 21,318 base set | Table 9 | verified 2026-09-18 | — |
| Evaluated models | — | eval | decoding for open models | greedy sampling with each model's own chat template | §5.1 | verified 2026-09-18 | §5.1: CoU results depend on whether the chat template is applied |
| Attack evaluation | — | eval | target model | Llama-2-7B-Chat for all attack algorithms | §5.1, §M | verified 2026-09-18 | §M: chosen because it is among the safest models in Table 5, on the hypothesis that successful attacks transfer |
| GCG | — | eval | suffixes | 20 pre-searched suffixes, following Robey et al. (2023) | §5.1 | verified 2026-09-18 | no ablation reported |

## Findings relevant to generality and negative feedback
- **Safety generalization does not survive adversarial prompting.** Every evaluated model loses safety rate between the base set and the attack-enhanced subset; the drop ranges from 11.75 points for Claude2 to 87.46 points for Qwen-72B (Table 5).
- **Judge models transfer across distributions only partly.** MD-Judge is above GPT-4 on all five sets in Table 3, with the largest margin on ToxicChat (0.644 vs 0.470), but its absolute ToxicChat F1 is the lowest of the five. On out-of-distribution HarmBench it is 0.74 points below GPT-4 (83.72% vs 84.46%, Table 4), so judge accuracy is distribution-dependent.
- **Rejection behavior confounds accuracy.** On the MCQ subset, Gemini Pro rejects 43.85% of questions, so its overall accuracy (44.19%) and valid accuracy (78.71%) differ by 34.5 points; the paper reports both to separate refusal from capability (Table 8).
- **Attack success is method-dependent, not uniform.** Human-designed jailbreaks reach 89.5% maximized ASR on enhanced questions while TAP variants stay at or below 5% (Table 6), so a single-attack safety claim does not characterize robustness.
- **The evaluated harmful-question distribution is English.** The base set includes 230 items from a multilingual source (Table 9); the paper reports no per-language evaluation.

## Connections
- [[harmbench-data]] — sibling red-team benchmark; SALAD-Bench uses HarmBench as an out-of-distribution test set for MD-Judge (Table 4).
- [[wildguard-data]] — later moderation and refusal dataset with its own judge model.
- Attack families shared with the jailbreak literature: GCG, AutoDAN, TAP, GPTFuzz, CoU (§5.1).

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2402.05044 (arXiv v4, 2024-06-07)
- Corrections to the previous card version:
  - "~30K questions" / "~30K base + ~10K attack-enhanced" → the base set is 21,318 questions (Table 9), described in the abstract-level text as 21k test samples; the derived subsets are 5,000 attack-enhanced, 200 defense-enhanced, and ~4,000 multiple-choice (§3, §2.2). The "30k" figure appears only as a rounded size column in the Table 1 benchmark comparison.
  - "MD-Judge — Base: Llama-2-7B" → MD-Judge is fine-tuned from Mistral-7B with LoRA rank 64 (§5.2). Llama-2-7B appears only as an alternative base in the §J comparison.
  - "Training data: ~3K human-labeled (query, response, safety-label) triples" → training data are public BeaverTails / LMSYS-Chat / ToxicChat pairs plus generated attack-enhanced pairs labeled by GPT-4; no size is given (§4.1, §G).
  - "Evaluation agreement with human: ~89%" / "MD-Judge matches human annotators at ~89% accuracy" → the reported MD-Judge metric is F1 (0.818 base, 0.873 enhanced, Table 3). The 89% figures in the paper belong to GPT-4 valid accuracy on the MCQ subset under the GPT-evaluator (89.07%), MCQ-Judge (88.96%), and human evaluation (89.17%) (Table 13, §K), not to MD-Judge.
  - "Apply 6 attack methods … GCG adversarial suffix, word-level perturbation, human-written jailbreaks (DAN-style), multilingual translation attacks, persona injection, crescendo / escalation attacks" → the evaluated attack methods are TAP, AutoDAN, GPTFuzz, GCG, CoU, and human-designed jailbreaks (§5.1). Word-level perturbation, translation attacks, persona injection, and crescendo attacks are defense-side prompt perturbations or do not appear.
  - "Llama-2-Chat-70B: safe on 95%+ base questions; drops to ~75% under attack" → Llama-2-70B is 96.21% base and 66.24% attack-enhanced (Table 5).
  - "GPT-4: safe on 97%+ base; ~85% under attack" → GPT-4 is 93.49% base and 80.28% attack-enhanced (Table 5).
  - "Best 2024-era open models (Mistral-Instruct-v0.2) on mid-70s under attack" → Mistral-7B-v0.2 is 80.14% base and 6.40% attack-enhanced (Table 5).
  - "Teacher model: GPT-4 for initial synthesis" → base questions are generated by a fine-tuned GPT-3.5-turbo; GPT-4 generates safe MCQ options and labels attack-enhanced judge training pairs (§2.2, §3.3, §4.1).
  - "Deduplication against training data: manual audit against common safety-benchmark sources" → deduplication is locality-sensitive hashing over Sentence-BERT embeddings within the collected pool (§2.2); no decontamination against model training data is reported.
- Removed as unsupported by the source:
  - "SAfety ALignment And Delineation Benchmark" as the expansion — the paper expands SALAD-Bench as "SAfety evaluation for LLMs, Attack and Defense approaches" (§1).
  - "Full leaderboard of 30+ LLMs" — Table 5 lists model safety rates and Elo; no count of 30+ is stated.
  - "finest-grained public safety taxonomy as of release" — Table 1 shows 6-16-66 against Do-Not-Answer's 5-12-60, but no such superlative is claimed.
  - "Judge-model lineage: MD-Judge → WildGuard (Allen AI) → MD-Judge v2" — not in this paper.
  - "Cost: significant — multi-stage curation with human review" — no cost figure is reported.
- Not reported by the source: MD-Judge training-set size; per-language safety rates; annotation cost. Per-category question counts are reported in Table 17, whose caption says "sixty-five unsafe categories" while the body states 66 categories (§2.1); the card keeps the body figure and records the discrepancy.

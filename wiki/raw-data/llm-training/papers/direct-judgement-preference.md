<!-- scope: Wang, Xu et al. (Salesforce, arXiv 2409.14664): training generative LLM judges (SFR-Judge 8B/12B/70B) with DPO + SFT on 680K judgement preference pairs of three types (CoT critique, standard judgement, response deduction); pointers to Self-Taught Evaluators, Con-J, and J1
     deps: [[dpo]], [[judge-llm-bias]]
     see-also: [[generative-reward-models]], [[ultrafeedback]], [[pairrm]], [[self-rewarding-lm]], [[rlaif-scaling]]
-->

# Direct Judgement Preference Optimization
- **Core Insight:** Training Llama-3.1-70B-Instruct with a DPO + SFT loss on 680K judgement preference pairs (70% CoT critique, 15% standard judgement, 15% response deduction) gives SFR-Judge-70B an average accuracy of 84.25 over 7 pairwise benchmarks, compared with 76.78 for GPT-4o and 82.26 for Self-taught-evaluator-Llama-3.1-70B (§4.1, Table 1).
- **Guideline:** When training a generative judge from datasets that have ground-truth verdicts but no written critiques, sample several teacher critiques per input, pair critiques whose verdict matches the ground truth against critiques whose verdict does not, and add verdict-only pairs plus an SFT term on the chosen output, because in the 8B ablation removing the verdict-only pairs lowered pairwise performance and SFT alone was below the full recipe (§5.3, Fig. 4).
- **Authors:** Peifeng Wang, Austin Xu, Yilun Zhou, Caiming Xiong, Shafiq Joty (Salesforce AI Research)
- **Year:** 2024 (arXiv v1 2024-09; v3 2025-09; no venue printed in v3)
- **URL:** https://arxiv.org/abs/2409.14664
- **Source type:** paper (models and code: github.com/SalesforceAIResearch/sfrjudge)
- **Relevant topics:** generative judge, LLM-as-a-judge, generative reward model, DPO on judgements, negative judgements, response deduction, judge evaluation

## Abstract
Existing judge models are mostly trained with SFT on small datasets (<100K samples, §1) for a few evaluation types, which the authors argue limits generalization. The paper trains judges at a larger data scale (680K pairs) with DPO. Four training tasks (single rating, pairwise comparison, classification, response deduction) are used to form three kinds of DPO preference pairs: CoT critiques (to produce useful critiques), standard judgements (to make correct verdicts), and response deduction (to understand what makes a response good or bad). Judges of 8B, 12B, and 70B parameters are evaluated on 13 benchmarks (7 pairwise, 4 single rating, 2 classification). The models have the best aggregate performance, and the 8B model outperforms GPT-4o on the pairwise benchmarks. Further analysis reports factual, actionable critiques and good starting points for continued finetuning.

## Key Contributions
- Three complementary DPO pair types for judge training: CoT critique, standard judgement, and response deduction (§3).
- A 680K-pair training set from 22 human- or model-annotated sources and the SFR-Judge-8B/12B/70B models (§4.1, App. B Table 5).
- A 13-benchmark evaluation suite across pairwise, single-rating, and classification tasks and domains such as safety and summarization (§4.2, App. C).
- Analyses of critique factuality, use as a reward model and reviser, domain-specific continual finetuning, bias, and hard vs easy negatives (§5.2-5.5, App. E).

## Key Figures/Tables to Study
- Figures 1-2: training tasks and pair construction. Figure 3: why verdict-only pairs are added.
- Tables 1-3: pairwise, single-rating, and classification results. Table 7: RewardBench breakdown.
- Figure 4: training-task ablation. Figure 6: β sweep for specialization. Table 11: hard vs easy negatives.

## Technical Details
- **Judge input and output:** x = (p, i, r), with protocol p (task description and rubric), task input i, and one or two responses r; output y = {c, j}, a critique c and a verdict j (§2).
- **D_CoT:** a teacher M_t samples several {c, j} for each x; samples whose j equals the ground-truth j* are chosen (y_w), the others rejected (y_l) (§3.1). Teacher Llama-3.1-70B-Instruct, 20 samples per prompt, temperature 0.7 (§4.1).
- **D_Std:** the critique is removed from D_CoT pairs and the protocol asks for the verdict only; motivation: in a CoT critique only a few tokens decide the verdict, so critique-length targets dilute the signal on those tokens (§3.2, Fig. 3).
- **D_Ded:** given p, i, and a correct {c, j}, generate the original response(s); chosen = the original response(s), rejected = a deduction written by the weaker Llama-3.1-8B-Instruct (§3.3, §4.1). The authors argue any response reconstructed from a critique is weaker than the original because a critique does not keep all of its information (App. B).
- **Loss (§3.4):** L = −log M_s(y_w|x) / (|y_w| + |x|) − log σ(β log[M_s(y_w|x)/M_ref(y_w|x)] − β log[M_s(y_l|x)/M_ref(y_l|x)]). M_s = judge being trained; M_ref = frozen copy of the same instruction-tuned initialization; |·| = token length; σ = logistic function; β = DPO temperature. The first term is the SFT loss on the chosen output.
- **Data:** 17 human-annotated and 5 synthetic sources, focused on 2023-and-later model responses, e.g. Chatbot Arena, HelpSteer, HelpSteer2, HH-RLHF, BeaverTails, RAGTruth, PRM800K, Prometheus, OffsetBias, UltraFeedback (App. B Table 5). Labels are balanced per task; CoT–verdict consistency of teacher samples was not checked (§4.1; App. B).
- **Evaluation protocol:** accuracy for pairwise and classification, Pearson correlation for single rating; greedy decoding for open models. For the 6 non-RewardBench pairwise benchmarks, each is run twice with response order swapped and the better run is reported (§4.3).
- **Pairwise averages:** SFR-Judge-70B 84.25, 12B 81.49, 8B 80.91; GPT-4o 76.78; Skywork-Critic-Llama-3.1-70B 80.03; Con-J-7B 75.51 (Table 1). RewardBench: 92.7 / 90.3 / 88.7; GPT-4o-2024-08-06 86.7 (Table 7).
- **Single rating (avg Pearson):** 0.76 / 0.70 / 0.68; GPT-4o 0.75 (Table 2). **Classification (avg acc):** 85.60 / 84.12 / 85.41; GPT-4o 85.47 (Table 3).
- **Critique quality (MetaCritique Meta-F1, GPT-4-scored):** 77.60 / 74.04 / 69.52; human critiques from Shepherd 64.02; Self-taught-evaluator-70B 62.99 (§5.2, Table 4).
- **CoT at inference:** removing the critique lowers SFR-Judge-8B single-rating Pearson from 0.68 to 0.58 and pairwise average from 80.97 to 80.05 (task-specific prompt) (App. E.6, Table 9).
- **Downstream use (§5.4, Fig. 5):** SFR-Judge-70B scores Llama-3-8B-Instruct responses to UltraFeedback prompts on a 5-point additive scale; best vs worst response becomes DPO data. A second setting refines low-scoring responses with the judge's critiques and uses {refined, original} as DPO pairs. AlpacaEval-2 win rate vs GPT-4 Turbo is higher with SFR-Judge-70B than with PairRM or ArmoRM, and refinement adds to it (Fig. 5; values not transcribed).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | initialization (and frozen reference) | Llama-3.1-8B-Instruct / NeMo-Instruct-12B / Llama-3.1-70B-Instruct | arXiv:2409.14664v3 §4.1, §3.4 | verified 2026-09-14 | no ablation reported |
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | preference pairs; type mix | 680K; D_CoT 70% : D_Std 15% : D_Ded 15% | §4.1 | verified 2026-09-14 | Fig. 4 (8B): removing D_Std or D_Ded changes results; the ratio itself is not ablated |
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | D_CoT sampling | Llama-3.1-70B-Instruct; 20 samples per prompt; temperature 0.7 | §4.1 | verified 2026-09-14 | App. E.8 Table 11: 70B-generated negatives beat 8B-generated negatives (earlier 8B run) |
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | D_Ded rejected-response generator | Llama-3.1-8B-Instruct | §4.1; App. B | verified 2026-09-14 | argument only (App. B); no ablation reported |
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | loss | DPO + length-normalized SFT on chosen | §3.4 | verified 2026-09-14 | §2 and §5.3 (Fig. 4): SFT alone reported as suboptimal; SFT term follows Pang et al. (2024) |
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | DPO β; LR; epochs; batch; max length; optimizer; compute | not reported | checked arXiv v1 and v3 body and appendices | not reported | — |
| SFR-Judge-8B, contextual continual finetune | 8B | reward-model | pairs; mix; sampling | 12,500 pairwise from RAGTruth; D_CoT 80% : D_Std 20%; 20 samples from Llama-3.1-70B-Instruct at temperature 0.7 | §5.5; App. B.1 | verified 2026-09-14 | no ablation reported |
| SFR-Judge-8B, contextual continual finetune | 8B | reward-model | β; epochs; checkpoint rule | β ∈ {0.01, 0.1, 0.3, 0.5, 0.7}; 3 epochs; best checkpoint reported | App. B.1; §5.5 | verified 2026-09-14 | Fig. 6: β = 0.01 most specialized (55.6% ContextualJudgeBench) |
| Evaluation of all judges | — | eval-gate | decoding; order handling | greedy (open models); OpenAI API temperature 0.7, top-p 1; 2 runs with swapped order, better run reported | §4.3 | verified 2026-09-14 | consistency reported separately (App. E.1 Table 6) |

## Findings relevant to negative feedback and generality
- **Type of negative:** negative as gradient. Rejected judgements enter the DPO term; the SFT term anchors the chosen judgement (§3.4).
- **Where negatives come from:** a teacher verdict that disagrees with the human or model annotation (D_CoT, D_Std), or a weaker model's reconstruction of a response (D_Ded) (§3.1-3.3). False-negative rate: not reported.
- **Hard vs easy negatives (Result, single study, 8B):** negatives from the 70B teacher vs from the 8B teacher: pairwise accuracy 78.83 vs 77.56, pairwise consistency 85.94 vs 80.70, Pearson 0.68 vs 0.67, classification 85.48 vs 84.54; the final models use 70B negatives for these pairs (App. E.8, Table 11).
- **SFT vs DPO:** the authors state that SFT alone is suboptimal because it never shows the model incorrect judgements, and include an SFT-only variant in the Fig. 4 ablation (§1, §2, §5.3).
- **Breadth:** Prometheus, Auto-J, and Llama-3-OffsetBias, which the authors describe as trained for or with single-rating data, do poorly on classification relative to SFR-Judges and GPT-4o (§5.1). Fixed prompts (RewardBench-style or PRePair-style) change SFR-Judge pairwise results by small amounts (App. E.2, Fig. 7).
- **Specialization trade-off:** continual finetuning SFR-Judge-8B at β = 0.01 reaches 55.6% on ContextualJudgeBench (o1: 55.3%) with, per the authors, a minimal drop on the 7 general pairwise benchmarks; the same finetune from Llama-3.1-8B-Instruct reaches 45.2% (§5.5, Fig. 6).
- **Measurement caveats:** for the 6 non-RewardBench pairwise benchmarks the paper reports the better of two response-order runs; order-averaged accuracy is not reported, and consistency is given separately (§4.3; App. E.1 Table 6). v1 states that the pre-August-9 LLM-AggreFact version was used because later versions include RAGTruth, which is in the training data (v1 §4.2).
- **Distillation:** gains over the base model are smaller at 70B, which the authors attribute to Llama-3.1-70B-Instruct being the teacher (App. E.3) (Interpretation).
- **Stated limits:** needs annotated judgements that may need refreshing; evaluates complete responses only (not process rewards); English only; CoT adds inference time (Limitations).

## Connections
- [[dpo]]: the preference loss; [[self-rewarding-lm]]: source of the additive 5-point scoring prompt in §5.4.
- [[judge-llm-bias]]: MT-Bench (a single-rating benchmark here) and position bias, which motivates the order-swap protocol.
- [[ultrafeedback]], [[hh-rlhf]]: training sources (Table 5); UltraFeedback prompts also feed the §5.4 experiment. [[pairrm]]: reward-model baseline in §5.4; [[simpo]]: source of the §5.4 baseline numbers.
- [[generative-reward-models]]: the broader generative-verifier line.
- Self-Taught Evaluators (Wang et al., arXiv:2408.02666, no library card): builds rejected responses by having an LLM write a "modified instruction that is highly relevant but not semantically identical" and answer it (§3.3, Fig. 2); SFT on rejection-sampled judgements (N = 15), 5 iterations, RewardBench 75.4 → 88.3 (88.7 with 32-sample majority vote) from Llama-3-70B-Instruct (§4.1, Table 1).
- Con-J (Ye et al., arXiv:2410.03742, no library card): repeated and hint-driven sampling of judgements; pairs a judgement with the correct preference against one with the wrong or no preference; DPO plus a small SFT weight (§3, Fig. 2).
- J1 (Whitehouse et al., arXiv:2505.10320, no library card): online GRPO training of judges on 22K synthetic pairs (17K WildChat, 5K MATH), with WildChat rejected responses built by the Self-Taught Evaluators noisy-instruction method (§2.1, §2.3).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2409.14664 (arXiv v3, 12 Sep 2025, full PDF; recipe fields also checked in v1, 23 Sep 2024); pointer rows checked against arXiv:2408.02666v2, arXiv:2410.03742v2, arXiv:2505.10320v3.
- Corrections to the previous card version:
  - The card mixed four artifacts (Con-J, Self-Taught Evaluators, J1, Skywork) under the generic title "Direct Judgement / Synthetic-Judge Preference Data (2024–2025 line)". It now describes the artifact named by the slug, Direct Judgement Preference Optimization; the other papers are pointers in Connections.
  - "Noisy-negative trick (Con-J): response to a perturbed instruction as rejected" → this is the Self-Taught Evaluators construction (arXiv:2408.02666 §3.3); Con-J pairs correct vs incorrect sampled judgements (arXiv:2410.03742 §3).
  - "Self-Taught Evaluator: judge_{k+1} trained with DPO on its own decisions; crosses GPT-4 after ~3 iterations" → SFT on rejection-sampled correct judgements, 5 iterations, 88.3 at iteration 5; GPT4-0125 (84.3) is first exceeded at iteration 2 (86.0) (arXiv:2408.02666 §3.4-3.5, Table 1).
  - v1 model names SFR-LLaMA-3.1-8B-Judge / SFR-NeMo-12B-Judge / SFR-LLaMA-3.1-70B-Judge are renamed SFR-Judge-8B/12B/70B in v3.
- Removed as unsupported by the source: "~40K synthetic preference pairs (20K SFT + 20K DPO) beat models trained with 2–40× more data"; "J1 highest-accuracy open judge on RewardBench-hard"; "judge generates prefs, then new policy, then new judge, no human labels after bootstrap" as a description of this line; "Con-J robust to label noise and format bias" wording; cost comparison with UltraFeedback; "judge-as-weapon: RewardBench leakage is a known measurement issue"; the "Risks + gotchas" list; the Con-J OpenReview title and ICLR 2025 venue (OpenReview page not readable in this check).
- Not reported by the source: DPO β, learning rate, epochs, batch size, sequence length, and compute for the main SFR-Judge runs; downstream DPO hyperparameters in §5.4.

<!-- scope: Nemotron-4 340B Technical Report (NVIDIA, arXiv:2406.11704) — synthetic alignment data pipeline, HelpSteer2 regression reward model, staged SFT, DPO, and Reward-aware Preference Optimization (RPO); recipe values in nemotron-4-synthetic-recipe
     deps: [[self-instruct]]
     see-also: [[nemotron-4-synthetic-recipe]], [[nemotron]], [[dpo]], [[wizardcoder]], [[ultrafeedback-construction]]
-->

# Nemotron-4 340B Technical Report
- **Core Insight:** Over 98% of the SFT and preference data used to align Nemotron-4-340B-Instruct was synthetic, with about 20K human-annotated examples (§3.2), and the sequence Code SFT → General SFT → DPO → three iterations of Reward-aware Preference Optimization moved MT-Bench (GPT-4-Turbo judge) from 6.79 to 8.22 and IFEval prompt-strict accuracy from 46.4 to 79.9 (Table 6).
- **Guideline:** When most preference pairs are ranked by a reward model rather than by ground truth, the report supports adding an SFT loss on chosen responses and fitting the reward gap (RPO) instead of plain DPO, because it observed DPO lowering both chosen and rejected likelihoods, overfitting with longer training, and trading MT-Bench gains for 0-shot MMLU losses (§3.3.2); the evidence is the stage-by-stage Table 6, not a controlled DPO-vs-RPO ablation.
- **Authors:** Bo Adler, Niket Agarwal, Ashwath Aithal, Dong H. Anh, Pallab Bhattacharya, Annika Brundyn, et al. (NVIDIA)
- **Year:** 2024 (arXiv v1 2024-06-17; v2 2024-08-06)
- **URL:** https://arxiv.org/abs/2406.11704 (also released as https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T_0.pdf)
- **Source type:** official technical report
- **Relevant topics:** synthetic alignment data, reward model as judge and filter, iterative weak-to-strong alignment, staged SFT, DPO, Reward-aware Preference Optimization

## Abstract
NVIDIA releases Nemotron-4-340B-Base, Nemotron-4-340B-Instruct, and Nemotron-4-340B-Reward under the NVIDIA Open Model License Agreement, which permits distribution, modification, and use of the models and their outputs. The models are competitive with open-access models on a wide range of benchmarks and fit on a single DGX H100 with 8 GPUs when deployed in FP8. Over 98% of the data used in the alignment process is synthetically generated. NVIDIA also open-sources the synthetic data generation pipeline used for alignment.

## Key Contributions
- Release of the three models plus training and inference code; generation prompts, the HelpSteer2 human preference dataset, and the reward model are shared for synthetic data work (§1).
- A synthetic data pipeline covering prompt generation, response and dialogue generation, quality filtering, and preference ranking, used for both SFT and preference fine-tuning (§1, §3.2).
- Nemotron-4-340B-Reward: a multi-attribute regression reward model trained on 10K HelpSteer2 examples, with RewardBench overall accuracy 92.0 (Table 4).
- Iterative weak-to-strong alignment: Mixtral-8x7B-Instruct-v0.1 generates the first round of data, and each aligned 340B intermediate model generates the next round (§3.2.4).
- Two-stage SFT (Code, then General) and RPO, a preference loss whose target is the reward model's score gap (§3.3).

## Key Figures/Tables to Study
- **§3.2.1-§3.2.3 and Figure 2:** prompt, dialogue, and preference synthesis; Supplementary Materials B-D hold the prompts.
- **Table 4:** RewardBench scores; **§3.2.3:** RM-as-judge vs LLM-as-judge on Chat-Hard.
- **§3.3.2:** the RPO loss. **Table 6:** every metric after each alignment stage.
- **Figure 5 and Table 7:** human evaluation against GPT-4-1106-preview; **Figure 6:** AEGIS unsafe-response rates.

## Technical Details
- **Reward model:** Nemotron-4-340B-Base with the final softmax layer replaced by a linear projection to five HelpSteer attributes (Helpfulness, Correctness, Coherence, Complexity, Verbosity), aggregated by a weighted sum at inference (§3.1). The report states that multi-attribute regression separates helpfulness from artifacts such as length better than pairwise ranking models (Interpretation; no ablation in the report). RewardBench: overall 92.0, Chat 95.8, Chat-Hard 87.1, Safety 91.5, Reasoning 93.7, Prior Sets 67.4 (Table 4).
- **Synthetic prompts:** Mixtral-8x7B-Instruct-v0.1 generates open Q&A, writing, closed Q&A (from C4 documents), and math/coding prompts, seeded with 3K topics and with 12K Python and 17K math keywords (§3.2.1). Instruction-following prompts append "verifiable" instruction templates from Zhou et al. (2023); two-turn prompts take the first user turn from ShareGPT (§3.2.1).
- **Real prompts:** LMSYS-Chat-1M prompts are split into disjoint SFT and preference sets; prompts flagged as potentially unsafe are removed from the SFT split and kept in the preference split (§3.2.1). Mixtral responses to synthetic prompts have mean RM helpfulness 3.24 vs 3.04 for LMSYS prompts, which the report reads as LMSYS prompts being harder (Figure 3).
- **Dialogues:** three turns; the model alternates assistant and user roles, with user-personality prompts; polite statements are removed from user turns; greedy decoding; dialogues below a reward-model score threshold are dropped (§3.2.2).
- **Preference pairs:** prompts include synthetic, ShareGPT, LMSYS, and GSM8K/MATH training prompts; responses come from multiple intermediate models, and harder pairs from several samples of the best model by MT-Bench (§3.2.3). Ground truth or a Python verifier selects chosen/rejected where available; otherwise LLM-as-judge (both response orders, consistent verdicts only) was used early and Reward-Model-as-judge later, with Chat-Hard accuracy 0.87 vs 0.54 (§3.2.3).
- **Weak-to-strong iterations:** Mixtral-8x7B-Instruct-v0.1 data trains 340B-Interm-1-Base into 340B-Interm-1-Instruct, which surpasses the Mixtral instruct model; that model then generates data for 340B-Interm-2 (§3.2.4). The report observes that, with fixed data, a stronger base gives a stronger instruct model, and with a fixed base, better data does the same (no numbers given).
- **Additional data:** CantTalkAboutThis topic-following dialogues with distractor turns; refusal responses for tasks the model cannot do (internet access, real-time knowledge), generated few-shot from human-written examples; permissively licensed Open-Platypus subsets; FinQA, contextual QA, WikiTableQuestions; a Glaive AI function-calling subset (§3.2.5).
- **Code SFT:** Genetic Instruct evolves samples from a limited number of seeds with self-instruction and WizardCoder mutations, keeps those passing an LLM fitness check, and yields about 800K samples after de-duplication and filtering (§3.3.1).
- **General SFT:** a 200K-sample blend including 2% of the Code SFT samples to reduce forgetting; loss only on assistant turns (§3.3.1).
- **RPO loss (§3.3.2):** L_rpo(x, y_c, y_l) = D[ β log(π(y_c|x)/π_ref(y_c|x)) − β log(π(y_l|x)/π_ref(y_l|x)) ‖ η (r*(x, y_c) − r*(x, y_l)) ], with D[a‖b] = σ(b) log(σ(b)/σ(a)) + (1 − σ(b)) log((1 − σ(b))/(1 − σ(a))). Here π is the policy, π_ref the reference policy, x the prompt, y_c and y_l the chosen and rejected responses, r* the reward model score, β the KL coefficient, η a reward scale, and σ the sigmoid.
- **Base model:** 9T tokens (8T pretraining + 1T continued training), 96 layers, 4,096 sequence length, 768 DGX H100 nodes (§2, Tables 1-2).

## Recipe ledger
Full ledger (pretraining, reward model, both SFT stages, DPO, RPO, generators): [[nemotron-4-synthetic-recipe]]. Stage order and sizes: Code SFT ~800K samples, 1 epoch, LR 3e-7 → General SFT 200K, 3 epochs, LR searched in [1e-7, 5e-7] → DPO 160K pairs, 1 epoch, batch 256 → RPO 300K pairs, LR 3e-7, η = 1, three iterations (§3.3.1-§3.3.2). Verified 2026-09-14.

## Findings relevant to generality, negative feedback, and distillation
- **Staged SFT:** a single mixed SFT stage produced conflicts between behaviours, most strongly for coding, and reweighting the blend did not fix it; the two-stage order gave "superior results across all downstream tasks" (§3.3.1; no single-stage numbers given).
- **Stage effects (Table 6):** General SFT raised MMLU 72.2 → 78.3 while HumanEval fell 70.7 → 66.5; DPO lowered MT-Bench 7.99 → 7.90; the three RPO iterations ended at MMLU 78.7, GSM8K 92.3, HumanEval 73.2.
- **Human evaluation:** on 136 prompts in 10 categories vs GPT-4-1106-preview, win/tie/loss was 28.19% / 46.57% / 25.24%; losses were highest for rewrite (66.67%) and extraction (45.83%) (Figure 5). Annotators rated length "just right" for 79.41% vs 74.02% (Table 7).
- **Negatives as gradient:** DPO and RPO both push down rejected responses; the report saw both chosen and rejected likelihoods fall under DPO and added an SFT loss on chosen responses; RPO scales the target gap by the reward difference to avoid "unlearning" high-quality rejected responses (§3.3.2).
- **Negatives as content:** refusal responses for impossible tasks are ordinary SFT targets (§3.2.5); unsafe LMSYS prompts appear only in preference data (§3.2.1).
- **Residual failures:** AEGIS unsafe responses remain in Criminal Planning and Regulated Substances; Garak probing found partial malware blocking, direct incorrect answers to impossible logic problems and false denials that some primes are prime, and "a pass rate below 30% for attempted jailbreaks" (§3.4.3).
- **Distillation:** first-round data come from a permissively licensed external model (Mixtral-8x7B-Instruct-v0.1), and the report states that the teacher "does not impose a ceiling" on the student (§3.2.4, qualitative); it presents the released models as generators of synthetic data for training smaller models (§1).

## Connections
- [[nemotron-4-synthetic-recipe]] — all printed hyperparameters with loci and status.
- [[nemotron]] — a second library card for the same report (model-reports/); this card is the more-linked one.
- [[dpo]] — the loss that the report extends with an SFT term and replaces with RPO.
- [[rpo]] — a different method with the same acronym (Iterative Reasoning Preference Optimization, Pang et al., 2024); it is not the RPO of this report.
- [[self-instruct]], [[wizardcoder]] — the two mutation sources in Genetic Instruct.
- [[ultrachat-construction]], [[camel]] — the topic-seeded prompt generation that §3.2.1 follows.
- [[glaive-function-calling]] — source of the function-calling subset.
- [[ultrafeedback-construction]] — another preference dataset ranked by an automatic judge.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2406.11704 (arXiv v2, including Supplementary Materials headings); key numbers (98%, 20K, 800K/200K/160K/300K, LRs, Tables 2, 4, 6, 7) also matched in the cloudfront PDF.
- Corrections to the previous card version: "Authors: NVIDIA" → named authors from arXiv (first six listed); URL → arXiv abs page as canonical; "the paper adds SFT loss and then RPO" with RPO unexpanded → RPO is Reward-aware Preference Optimization (§3.3.2), not [[rpo]]; "DPO alone can overfit to reward gaps" → DPO ignores the reward gap, and the report observed falling chosen/rejected likelihoods, overfitting with longer training, and MT-Bench vs MMLU trade-offs (§3.3.2); "adds a small amount of alignment-style QA in continued pretraining to steer low-accuracy areas" → continued training adds a small number of QA-style alignment examples and separately up-weights sources from low-accuracy areas (§2.3); "iterates through Nemotron checkpoints" → the first generator is Mixtral-8x7B-Instruct-v0.1 (§3.2.1, §3.2.4); "prefers RM-based ranking over raw model self-selection" → RM-as-judge replaced LLM-as-judge after Chat-Hard 0.87 vs 0.54 (§3.2.3); "Seed input: task families … function calling, incapable tasks" → synthetic prompts cover open Q&A, writing, closed Q&A, math/coding, instruction-following, and two-turn prompts; topic-following, document QA, and function calling come from existing datasets (§3.2.1, §3.2.5); "keeps topic-following data intentionally noisy" → CantTalkAboutThis dialogues are "intentionally interspersed with distractor turns" (§3.2.5).
- Removed as unsupported by the source: "Reward-model errors compound when the same scorer is reused across iterations"; "the paper emphasizes that the small human anchor set is enough to sustain the pipeline"; "the pipeline is meant to preserve behavior diversity across task families"; "the practical template later reused by other open alignment stacks".
- Not reported by the source: pretraining LR/optimizer; selected values within the General SFT, DPO, and RPO search ranges; RPO batch size and epochs; reward-model training settings and attribute weights; the dialogue filter threshold.

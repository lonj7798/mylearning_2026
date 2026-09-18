<!-- scope: MetaMath / MetaMathQA (Yu et al., ICLR 2024): math SFT data built by answer augmentation plus rephrased and backward (SV, FOBAR) questions generated with GPT-3.5-Turbo from GSM8K and MATH
     deps: [[training-verifiers-to-solve-math-word-problems]]
     see-also: [[wizardmath]], [[mammoth]], [[mathscale]], [[openmathinstruct-2]], [[star]]
-->

# MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models
- **Core Insight:** Fine-tuning LLaMA-2-7B on MetaMathQA (395K GPT-3.5-Turbo samples that add rephrased and backward questions to answer augmentation on GSM8K and MATH) gives 66.5% on GSM8K and 19.8% on MATH, and in the data-scaling analysis 20K more answer-augmented samples added 0.1 points on GSM8K while 20K Rephrasing, FOBAR, or SV samples added 0.4, 2.3, and 2.6 points (Table 2, §4.5).
- **Guideline:** When a math SFT set is expanded from a fixed pool of seed problems, add rewritten and backward questions instead of only sampling more solutions per question, because for LLaMA-2-7B answer augmentation saturated near 80K samples (59.6% GSM8K) while bootstrapped questions kept raising accuracy (§4.5, Figure 2; App. E.2 repeats the trend for LLaMA-2-13B and Mistral-7B).
- **Authors:** Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, et al.
- **Year:** 2023 (arXiv v1 2023-09; ICLR 2024)
- **URL:** https://arxiv.org/abs/2309.12284
- **Source type:** paper
- **Relevant topics:** math reasoning, synthetic SFT data, question bootstrapping, backward reasoning, rejection sampling, data diversity, distillation from GPT-3.5-Turbo

## Abstract
Open-source LLMs such as LLaMA-2 perform poorly on math problems that need multi-step reasoning. MetaMath is a model fine-tuned for math reasoning. The authors bootstrap math questions by rewriting each question from multiple perspectives, which produces the MetaMathQA dataset, and fine-tune LLaMA-2 on it. On GSM8K and MATH, MetaMath outperforms open-source LLMs. MetaMath-7B reaches 66.5% on GSM8K and 19.8% on MATH, 11.5 and 8.7 points above same-size prior models, and MetaMath-70B reaches 82.3% on GSM8K, slightly above GPT-3.5-Turbo. The dataset, models, and training code are released.

## Key Contributions
- Question bootstrapping with one forward rewrite (Rephrasing) and two backward rewrites (Self-Verification, FOBAR), combined with answer augmentation (AnsAug) that keeps the question unchanged (§3.1–3.3).
- MetaMathQA: 395K samples from GSM8K and MATH training questions (Table 1).
- MetaMath-7B, -13B (full fine-tuning) and -70B (QLoRA) results on GSM8K and MATH (Table 2).
- Analyses of question diversity vs accuracy (§4.5), perplexity (§4.4), backward-question accuracy (§4.6), incorrect-answer training data (§4.7), and merging external RFT data (§4.8).

## Key Figures/Tables to Study
- **Table 1:** MetaMathQA composition by operator. **Table 2:** GSM8K and MATH accuracy by model size.
- **Table 3:** augmentation ablation on LLaMA-2-7B. **Figure 2, Figure 4:** accuracy vs data size and vs diversity gain.
- **Figure 5:** adding RFT data lowers accuracy. **Figure 6:** GSM8K vs GSM8K-Backward accuracy.
- **Table 4:** training on reasoning paths with incorrect answers. **App. E Tables 8, 10:** other base models; DROP out-of-distribution test.

## Technical Details
- **Seeds (§4.1):** GSM8K 7,473 training / 1,319 test questions; MATH 7,500 training / 5,000 test questions.
- **Generator (§4.1):** GPT-3.5-Turbo produces rephrased questions and answers for all four operators, temperature 0.7. The per-question sample counts `K_AnsAug`, `K_rephrase`, `K_SV`, `K_FOBAR` are not reported.
- **Operators:** every operator keeps only generated reasoning paths whose final answer equals the ground truth `a*` (Eq. 1–4).
  1. **AnsAug (§3.1):** few-shot chain-of-thought sampling on the original question, following rejection-sampling fine-tuning (RFT, Yuan et al. 2023).
  2. **Rephrasing (§3.2):** an 8-example rephrasing prompt (App. A.1) rewrites the question; GPT-3.5-Turbo then answers it. Its accuracy is 76.30% on rephrased questions vs 80.74% on the original training questions.
  3. **Self-Verification, SV (§3.3):** the question and its answer are rewritten into a declarative statement (prompt in App. A.2), one number is replaced by `x`, and "What is the value of unknown variable x?" is appended.
  4. **FOBAR (§3.3):** one number is replaced by `x`, and "If we know the answer to the above question is {a*}, what is the value of unknown variable x?" is appended; no declarative rewrite is needed.
  Example 3.4: "James buys x packs of beef that are 4 pounds each. The price of beef is $5.50 per pound. How much did he pay? If we know the answer to the above question is 110, what is the value of unknown variable x?" → 5.
- **Composition (Table 1):** MetaMathQA-GSM8K 80K AnsAug / 80K Rephrasing / 40K SV / 40K FOBAR = 240K; MetaMathQA-MATH 75K / 50K / 15K / 15K = 155K; total 155K / 130K / 55K / 55K = 395K.
- **Loss (§3.4):** `L(θ) = Σ_{(q,r,a)∈D_MetaMathQA} log P(r | q; θ)`, where `q` is the question, `r` the reasoning path, `a` the answer, and `θ` the model parameters.
- **Diversity gain (§4.1):** `d_gain = (1/M) Σ_{x_i∈D_new} min_{x_j∈D_base} ‖f(x_i) − f(x_j)‖²₂`. `D_base` has N samples, `D_new` has M samples, and `f` is the text-embedding-ada-002 embedding.
- **Results (Table 2):** MetaMath-7B 66.5 GSM8K / 19.8 MATH; 13B 72.3 / 22.4; 70B 82.3 / 26.6. GPT-3.5-Turbo 80.8 / 34.1. WizardMath 7B 54.9 / 10.7; WizardMath 70B 81.6 / 22.7.
- **Ablation (Table 3, LLaMA-2-7B trained on MetaMathQA-GSM8K subsets, GSM8K accuracy):** SFT on original data 41.6; AnsAug 59.6; Rephrasing 59.7; AnsAug + Rephrasing 60.6; all four 64.4. SV and FOBAR are added together, not ablated separately. Trained on MetaMathQA-MATH subsets, MATH accuracy goes from 4.7 (SFT) to 17.7 (all four). Table 9 repeats the GSM8K ablation on LLaMA-2-13B: 50.9 (SFT) to 72.3 (all four).
- **Diversity analysis (§4.5):** the gains in the Core Insight come with a Pearson correlation of 0.972 between diversity gain and accuracy gain; the extra 20K AnsAug samples had diversity gain 0.05.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| MetaMath-7B, MetaMath-13B | 7B, 13B | distill-SFT | method; optimizer; epochs; batch size | full fine-tuning; AdamW; 3; 128 | arXiv:2309.12284v4 App. B | verified 2026-09-14 | no ablation reported |
| MetaMath-7B, MetaMath-13B | 7B, 13B | distill-SFT | peak LR; warmup; decay shape; hardware | 2e-5; 3% warmup; not reported; 8 NVIDIA A100 | App. B | verified 2026-09-14 (decay: not reported) | no ablation reported |
| MetaMath-70B | 70B | distill-SFT | method; LoRA rank, alpha, dropout; modules | QLoRA; 96, 16, 0.05; attention and MLP layers | App. B; Table 2 footnote | verified 2026-09-14 | no ablation reported |
| MetaMath-70B | 70B | distill-SFT | optimizer; LR; warmup; epochs and batch | AdamW; 1e-4; none; not reported for 70B | App. B | verified 2026-09-14 (epochs, batch: not reported) | no ablation reported |
| All MetaMath sizes | 7B–70B | distill-SFT | data; examples | MetaMathQA, 395K (mixture in Table 1) | Table 1; §4.1 | verified 2026-09-14 | Table 3 (7B) and Table 9 (13B): all four operators > subsets |
| All MetaMath sizes | 7B–70B | distill-SFT | data generator; temperature | GPT-3.5-Turbo; 0.7 | §4.1 | verified 2026-09-14 | no ablation reported |
| All MetaMath sizes | 7B–70B | distill-SFT | target and template | log P(reasoning path | question); Alpaca-style prompt; "The answer is: {gold}" appended to answers | §3.4; App. B Prompt 1 | verified 2026-09-14 | no ablation reported |
| All MetaMath sizes | 7B–70B | eval-gate | evaluation | zero-shot, "Let's think step by step", temperature 0 | App. B Prompt 2 | verified 2026-09-14 | App. B: zero-shot found better than few-shot for fine-tuned models (no table) |
| All MetaMath sizes | 7B–70B | distill-SFT | max sequence length; packing; samples per operator K; test-set decontamination | not reported | checked §3–4, App. A–E | not reported | n/a |

## Findings relevant to generality, negative feedback, distillation
- **Generality:** training only on MetaMathQA-GSM8K raises MATH accuracy (3.0 SFT → 5.7), and training only on MetaMathQA-MATH raises GSM8K accuracy (13.8 → 34.6) (Table 3). On DROP questions with numeric answers (zero-shot), MetaMath-7B scores 37.1 vs WizardMath-7B 31.5, and MetaMath-70B 72.3 vs WizardMath-70B 63.1 (App. E.4 Table 10); the authors read this as no benchmark overfitting. MetaMathQA also raises Mistral-7B from 52.2 to 77.7 on GSM8K (App. E.1 Table 8). Adding about 47K RFT samples to MetaMathQA subsets of 20K–100K lowered accuracy (§4.8, Figure 5). Accuracy is lower on longer questions (§4.9, Figure 7). Existing models score lower on the 1,270-question GSM8K-Backward set than on GSM8K (§4.6, Figure 6).
- **Negative samples:** wrong-answer paths are discarded (negative marginal value; Eq. 1–4). In §4.7, 7,473 reasoning paths with incorrect final answers, used as ordinary SFT targets, gave 43.6% GSM8K vs 41.6% for SFT on the original data and 52.2% for the same number of correct paths (LLaMA-2-7B, Table 4). The authors hypothesize that correct intermediate steps in those paths provide useful supervision.
- **Distillation:** all MetaMathQA answers come from GPT-3.5-Turbo. MetaMath-70B exceeds GPT-3.5-Turbo on GSM8K (82.3 vs 80.8) but not on MATH (26.6 vs 34.1) (Table 2).

## Connections
- [[training-verifiers-to-solve-math-word-problems]]: the GSM8K dataset, one of the two seed sets (§4.1).
- [[wizardmath]], [[mammoth]]: baselines in Table 2 (MAmmoTH-CoT 7B 50.5 / 10.4; WizardMath 7B 54.9 / 10.7).
- [[lima]]: §4.4 frames the perplexity analysis with LIMA's Superficial Alignment Hypothesis.
- [[star]]: also keeps only self-generated rationales with correct answers; in MetaMath the generator is GPT-3.5-Turbo, not the trained model.
- [[mathscale]], [[openmathinstruct-2]]: later math-data synthesis work used alongside this card in ch-24.
- [[s1]], [[limo]]: small curated reasoning SFT sets; contrast with MetaMathQA's 395K-sample scale.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2309.12284 (arXiv v4, 2024-05-03).
- Corrections to the previous card version:
  - "rewriting the question four ways (self-verification, FOBAR, answer-augmentation, rephrasing)" → three question-bootstrapping operators plus answer augmentation, which does not change the question (§3.1–3.3).
  - SV described as "Given Q and the candidate answer A', determine correctness; if not, fix" with template "Is the following answer correct? …" → SV rewrites question and answer into a declarative statement, masks a number as x, and asks for x (§3.3, Example 3.3, App. A.2).
  - Invented FOBAR example ("Jane has 3 apples…") → replaced with the paper's Example 3.4.
  - "AnsAug alone ~4-point gain; Rephrasing +3; SV +2; FOBAR +3; additive" → Table 3: 41.6 → 59.6 (AnsAug), 60.6 (+Rephrasing), 64.4 (+SV and FOBAR together).
  - "GSM8K (7.5K)" → 7,473 training questions (§4.1). "~25× the seed count" → 395K / 14,973 = 26.4× (derived from §4.1 and Table 1).
  - "Teacher: GPT-3.5-turbo (text-davinci rewrites in ablations)" → GPT-3.5-Turbo at temperature 0.7 for all operators; no text-davinci use (§4.1).
  - "Correctness verifier: numeric exact-match (GSM8K), symbolic match (MATH)" → paths are kept when the answer equals ground truth, extracted after "The answer is:" (Eq. 1–4, App. B); the MATH matching procedure is not described.
  - Guideline operator names "backward-reasoning, variable-abstracted, mask-and-solve" → not terms used in the paper.
  - Internal inconsistency in the source: abstract and §1 give +11.5 / +8.7 points for MetaMath-7B over the previous best; §4.2 gives 11.6 / 9.1, which matches Table 2 (66.5 − 54.9, 19.8 − 10.7).
- Removed as unsupported by the source: "reused downstream in OpenMathInstruct-2, WizardMath, and MAmmoTH"; "[[wizardmath]] adopts Rephrasing"; "[[mammoth]] composes MetaMathQA into its mix"; "[[mammoth]] conceptual ancestor"; "average trace length 300–600 tokens"; "non-reflective"; "cost $5–15K"; "final-answer accuracy ~30% on MATH level-5"; "ambiguous FOBARs leak in"; "rephrasing drift … filter catches most but not all"; "reduces direction overfitting"; "inoculates against memorization".
- Not reported by the source: samples per operator (K), sequence length, LR decay shape, decontamination against test sets, generation cost.

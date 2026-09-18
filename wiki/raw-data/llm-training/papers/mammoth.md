<!-- scope: math instruction tuning on MathInstruct (260K pairs, hybrid chain-of-thought + program-of-thought rationales from 13 datasets); in-domain vs out-of-domain math evaluation
     deps: [[self-instruct]]
     see-also: [[mammoth-2]], [[metamath]], [[wizardmath]], [[mathscale]], [[openmathinstruct]]
-->

# MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning
- **Core Insight:** On Llama-2 7B, training on all of MathInstruct gives a 9-dataset average of 47.9, compared with 32.0 for its chain-of-thought subset alone and 41.0 for its program-of-thought subset alone (Table 6); across sizes, the gain over the best prior open model is +21 to +32 average points on out-of-domain sets (Table 4) and +16 to +28 on in-domain sets (Table 3).
- **Guideline:** When a math SFT set must transfer beyond its training benchmarks, combine several source datasets and both natural-language and program rationales, because in the 7B ablations adding sources raised accuracy on four of five out-of-domain sets (Table 5) and the hybrid set outperformed each format alone (Table 6). These results were measured only on Llama-2 and Code Llama bases with short human- or GPT-4-written rationales.
- **Authors:** Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, et al. (Yu Su, Wenhu Chen); University of Waterloo, The Ohio State University, HKUST, University of Edinburgh, 01.AI
- **Year:** 2023 (arXiv v1 2023-09-11; v3 2023-10-03; the arXiv record lists no venue)
- **URL:** https://arxiv.org/abs/2309.05653
- **Source type:** paper
- **Relevant topics:** math reasoning, instruction tuning, chain-of-thought (CoT), program-of-thought (PoT), GPT-4 distillation, out-of-domain evaluation

## Abstract
MAmmoTH is a series of open LLMs fine-tuned for general math problem solving on MathInstruct. MathInstruct is compiled from 13 math datasets with intermediate rationales; six of them have rationales newly written by prompting GPT-4 (Abstract, Table 1). It mixes CoT rationales (step-by-step natural language) and PoT rationales (a Python program whose execution prints the answer) and covers several math fields. The authors report that MAmmoTH models outperform existing open models on nine math datasets at every tested scale, with average accuracy gains of 16 to 32 points (Abstract). The abstract's headline MATH numbers (33% at 7B, 44% at 34B, above GPT-4 CoT) correspond to MAmmoTH-Coder-7B (33.4) and MAmmoTH-Coder-34B (43.6) versus GPT-4 at 42.5 (Table 3).

## Key Contributions
- MathInstruct: 260K (instruction, response) pairs in an Alpaca-like format from 13 datasets, six with new GPT-4 rationales (§2.2, §2.3, Table 1).
- GPT-4 PoT programs kept only when their executed result matches the human-annotated answer (§2.2).
- Hybrid decoding: try PoT first; if the program is not executable, fall back to CoT (§2.4, Table 7).
- An evaluation split into 4 in-domain and 5 out-of-domain test sets (Table 2); more than 50 models and baselines trained or evaluated (§1).
- Ablations over data subsets (Table 5), rationale format (Table 6), and decoding format (Table 7).

## Key Figures/Tables to Study
- Table 1: MathInstruct composition (rationale type, annotator, size, validation).
- Tables 3 and 4: in-domain and out-of-domain accuracy at 7B, 13B, 30-34B, 65-70B.
- Table 5: adding GSM8K, MATH, Camel, AQuA, NumGLUE subsets one at a time; removing the six new subsets.
- Table 6 / Fig. 2: CoT-only vs PoT-only vs hybrid training at 7B.
- Table 7: CoT vs PoT vs hybrid decoding per model.

## Technical Details
Data
- Table 1 rows (type, annotator, size): GSM8K CoT human 7K; GSM8K-RFT CoT Llama 28K; AQuA-RAT CoT human 90K; MATH CoT human 7K; TheoremQA CoT GPT-4 600 (new); Camel-Math CoT GPT-4 50K (unvalidated); College-Math CoT GPT-4 1.8K (new, unvalidated); GSM8K PoT GPT-4 14K (new); AQuA-RAT PoT GPT-4 9.7K (new); MATH PoT GPT-4 7K (new); TheoremQA PoT GPT-4 700 (new); MathQA PoT human 25K; NumGLUE PoT human 13K (Table 1).
- The listed rows sum to 184.4K CoT and 69.4K PoT (253.8K total), against a printed total of 260K; CoT is about 73% of the listed rows (derived: 184.4 / 253.8, Table 1).
- New CoT: GPT-4 writes CoT for TheoremQA questions; College-Math question–CoT pairs come from Self-Instruct with seed exemplars found online (§2.2).
- New PoT: GPT-4 writes programs for MATH, AQuA, GSM8K, TheoremQA; a program is kept only if its execution result equals the human-annotated ground truth (§2.2). Augmented samples whose answers disagree with the original annotation are removed (Table 1 caption).

Evaluation
- In-domain: GSM8K (1,319), MATH (5,000), AQuA-RAT (254), NumGLUE (1,042). Out-of-domain: SVAMP (1,000), Mathematics (1,000), SimulEq (514), SAT-Math (220), MMLU-Math (974) (Table 2).
- MAmmoTH models are evaluated zero-shot. Baselines use the better of 8-shot and zero-shot on in-domain sets and 5-shot on out-of-domain sets. Maximum decoding length is 2,048 tokens (§3.2).
- CoT is the default output; appending "Let's write a program to solve the problem" to the question triggers PoT (§2.4).

Results
- 7B: MAmmoTH GSM8K 53.6, MATH 31.5; Llama-2 7B 14.6, 2.5; WizardMath-7B 54.9, 10.7 (Table 3).
- 70B: MAmmoTH GSM8K 76.9, MATH 41.8; WizardMath-70B GSM8K 81.6, MATH 22.7. MAmmoTH-70B is 4.7 points below WizardMath-70B on GSM8K (Table 3).
- MAmmoTH-Coder-34B has a higher out-of-domain average (63.2) than MAmmoTH-70B (62.5) (Table 4, §3.3).
- Rationale format at 7B, 9-set average: base 19.9, WizardMath 27.0, CoT-only 32.0, PoT-only 41.0, hybrid 47.9. MATH: CoT-only 9.9, PoT-only 28.9, hybrid 31.5. AQuA: CoT-only 42.2, PoT-only 28.6, hybrid 44.5 (Table 6).
- Decoding, 9-set average: MAmmoTH-7B CoT 33.0, PoT 46.1, hybrid 47.9; MAmmoTH-70B CoT 49.2, PoT 60.1, hybrid 63.4 (Table 7). The text says hybrid decoding improves every test set (§3.4), but Table 7 has exceptions, for example MAmmoTH-7B SimulEq: PoT 48.2, hybrid 41.2.
- Removing the six newly curated subsets lowers the overall score by 9 points (§3.4, Table 5).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| MAmmoTH-7B, MAmmoTH-13B (Llama-2); MAmmoTH-Coder-7B, -13B (Code Llama) | 7B, 13B | SFT | peak learning rate | 2e-5 | arXiv:2309.05653v3 §2.3 | verified 2026-09-14 | no ablation reported |
| MAmmoTH-70B (Llama-2); MAmmoTH-Coder-34B (Code Llama) | 34B, 70B | SFT | peak learning rate | 1e-5 | arXiv:2309.05653v3 §2.3 | verified 2026-09-14 | no ablation reported |
| all MAmmoTH and MAmmoTH-Coder models | 7B–70B | SFT | batch size (unit not stated) | 128 | arXiv:2309.05653v3 §2.3 | verified 2026-09-14 | no ablation reported |
| all MAmmoTH and MAmmoTH-Coder models | 7B–70B | SFT | schedule; warmup | cosine; 3% warm-up | arXiv:2309.05653v3 §2.3 | verified 2026-09-14 | no ablation reported |
| all MAmmoTH and MAmmoTH-Coder models | 7B–70B | SFT | epochs | 3 | arXiv:2309.05653v3 §2.3 | verified 2026-09-14 | no ablation reported |
| all MAmmoTH and MAmmoTH-Coder models | 7B–70B | SFT | training data | MathInstruct, 260K (instruction, response) pairs, Alpaca-like format | arXiv:2309.05653v3 §2.2–2.3, Table 1 | verified 2026-09-14 | Tables 5–6 (Llama-2 7B only): full hybrid set above each subset |
| MAmmoTH-70B; MAmmoTH-Coder-34B | 34B, 70B | SFT | distributed training | DeepSpeed ZeRO stage 3 | arXiv:2309.05653v3 §2.3 | verified 2026-09-14 | not applicable |
| all MAmmoTH and MAmmoTH-Coder models | 7B–70B | SFT | optimizer, weight decay, max training length, packing, loss masking, compute | not reported | checked v3 §2.3, §3, App. A–C | not reported | none |
| all MAmmoTH and MAmmoTH-Coder models | 7B–70B | eval-gate | decoding | zero-shot; PoT first, CoT if program not executable; max 2,048 tokens | arXiv:2309.05653v3 §2.4, §3.2 | verified 2026-09-14 | Table 7: hybrid average ≥ PoT and CoT for every model |

## Findings relevant to generality, negative feedback, distillation
Generality
- The authors state that dataset-specific fine-tuning (RFT, WizardMath) raises GSM8K accuracy by more than 30 points but lowers out-of-domain accuracy on sets such as MMLU-Math or AQuA by up to 10 points (§1). In the tables, WizardMath-70B scores 20.0 on AQuA versus 40.9 for Llama-2-70B (Table 3), and 13.2 on SAT-Math versus 51.3 (Table 4).
- At 7B, training on the GSM8K subset only gives MATH 9.2 and MMLU-Math 25.2; adding MATH, Camel, AQuA and NumGLUE gives 28.9 and 44.5 (Table 5). SVAMP stays near 65 in both settings.
- Stated limits: no proof-type training data; not suitable for mathematical analysis, complex analysis, graph theory, or numerical analysis (App. C).

Negative samples
- GPT-4 programs whose executed answer is wrong, and augmented samples inconsistent with the original annotation, are discarded (§2.2, Table 1 caption). They are not used as training signal. The discard rate is not reported.

Distillation
- Six subsets are GPT-4 rationales written for this work; Camel-Math (50K) and College-Math (1.8K) GPT-4 rationales are not validated (Table 1). Removing the six new subsets costs 9 overall points (§3.4).

## Connections
- [[mammoth-2]] — follow-up sharing three authors (Xiang Yue, Ge Zhang, Wenhu Chen) that harvests 10M instruction–response pairs from the web instead of curating existing datasets (arXiv:2405.03548 Abstract, §1).
- [[metamath]] — contemporaneous GSM8K/MATH augmentation dataset (arXiv:2309.12284), cited in App. A.2.
- [[wizardmath]] — main dataset-specific baseline in Tables 3–4.
- [[self-instruct]] — used to create the College-Math question–CoT pairs (§2.2).
- [[openmathinstruct]] — later 1.8M-pair dataset with code-interpreter solutions; its abstract names MAmmoTH as built from closed-model outputs (arXiv:2402.10176 Abstract).
- [[mathscale]] — evaluates MAmmoTH as a baseline on MWPBench with CoT-only prompting (MathScale Table 5).
- Program-of-Thoughts (Chen et al., arXiv:2211.12588) and PAL (Gao et al., 2023) — the program-rationale prompting methods MAmmoTH trains on (§1); no card in this library.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2309.05653 (v3, 2023-10-03, full PDF including App. A–C); project page https://tiger-ai-lab.github.io/MAmmoTH/ (states no venue).
- Corrections to the previous card version:
  - "problems from 14 datasets" including simuleq, asdiv, mawps, scienceQA, ApEdi → 13 datasets (Table 1); SimulEq is an out-of-domain test set, not a training source (Table 2).
  - "Overall ratio PoT:CoT ≈ 57:43" and guideline "70/30 or 60/40 PoT/CoT" → listed rows give 184.4K CoT and 69.4K PoT, about 73% CoT (derived from Table 1).
  - "MAmmoTH-7B/13B/70B, Mistral, Llama-2 variants" → Llama-2 7B/13B/70B and Code Llama 7B/13B/34B; no Mistral model in v3 (§2.3).
  - "MAmmoTH-70B: GSM8K 76.9 … SOTA open 2023" → WizardMath-70B has 81.6 on GSM8K; MAmmoTH-70B has the top 70B MATH score (Table 3).
  - "CoT-only drops MATH by 8 points; PoT-only drops MATH by 4 but drops AQuA by 15" → CoT-only MATH 9.9 vs 31.5 (−21.6); PoT-only MATH 28.9 (−2.6); PoT-only AQuA 28.6 vs 44.5 (−15.9) (Table 6).
  - "transfer to TheoremQA (not seen in training)" → TheoremQA is a training source (Table 1) and not an evaluation set (Table 2).
  - "CoT: prompt GPT-4 with 2-shot examples; accept if matches gold" → GPT-4 CoT for TheoremQA and Self-Instruct for College-Math; Camel-Math and College-Math are unvalidated (§2.2, Table 1).
  - "MathInstruct 260K … first large-scale hybrid CoT+PoT math SFT set" → the paper claims a unique hybrid, not "first"; wording changed (Abstract).
- Removed as unsupported by the source: per-problem-type PoT/CoT mixing strategy; `def solution():` template; token-length ranges (CoT 200–600, PoT 100–400, "~300 tokens"); GPT-4 API cost estimate ($30–50K); OpenAI terms-of-service risk; PoT "return 42" shortcut risk; "short-CoT insufficient for long-CoT distillation"; "trace style dual template".
- Not reported by the source: optimizer and weight decay; training sequence length; compute; filter discard rates; test-set decontamination procedure.

<!-- scope: OpenMathInstruct-2 (NVIDIA, arXiv:2410.01560) — 14M math CoT question-solution pairs from Llama3.1-405B-Instruct, with SFT ablations on solution format, teacher strength, low-quality solutions, and question diversity
     deps: [[openmathinstruct]]
     see-also: [[openmathinstruct-2-recipe]], [[numina-math]], [[metamath]], [[xwin-math]], [[nemotron-4-synthetic]]
-->

# OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data
- **Core Insight:** Fine-tuning Llama3.1-8B-Base on OpenMathInstruct-2 (14M question-solution pairs synthesized by Llama3.1-405B-Instruct) gives 67.8% on MATH, 15.9 points above Llama3.1-8B-Instruct (51.9%), and the ablations find SFT robust to up to 20% low-quality solutions while cutting unique questions from 6.5K to 1K at 256K pairs lowers MATH-validation accuracy by more than 10 points.
- **Guideline:** When a math SFT set has a fixed pair budget and an 8B-class student, spend the budget on more unique questions and a stronger teacher rather than on stricter solution filtering, because at 256K pairs going from 1K to 6.5K unique questions added 10.5 MATH-validation points (§1, Fig. 6), a 405B teacher beat an 8B-Base teacher by 7.8 points at matched coverage (Table 2), and judge or reward-model filtering gave no gain (Table 3).
- **Authors:** Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, Igor Gitman (NVIDIA)
- **Year:** 2024 (arXiv v1 2024-10-02; v2 2024-10-05; no venue stated on arXiv v2)
- **URL:** https://arxiv.org/abs/2410.01560
- **Source type:** paper
- **Relevant topics:** math reasoning SFT, synthetic question and solution generation, teacher choice, noisy SFT data, LLM-based decontamination

## Abstract
The authors build a math-reasoning SFT dataset with the Llama3.1 model family and run ablations on data synthesis. They find that (a) solution format matters and excessively verbose solutions hurt SFT, (b) data from a strong teacher beats equally sized data from a weak student model, (c) SFT is robust to low-quality solutions, so imprecise filtering is acceptable, and (d) question diversity is needed for data-scaling gains. They release OpenMathInstruct-2: 14M question-solution pairs with about 600K unique questions, described as nearly eight times larger than the previous largest open-source math reasoning dataset. Llama-3.1-8B-Base fine-tuned on it scores 67.8% on MATH versus 51.9% for Llama3.1-8B-Instruct. Code, models, and data are released under a commercially permissive license.

## Key Contributions
- The dataset: 13.97M pairs over 607.3K unique questions, of which 592K are synthesized questions (Table 5; §1).
- The "OpenMath CoT" solution format: 44.5 vs 40.6 MATH-validation accuracy against the Llama CoT format (Table 1).
- "Matching Coverage", a downsampling operation that equalizes the unique questions and the solutions per question of two compared datasets (§2.1).
- LLM-based decontamination that removes paraphrases of GSM8K, MATH, AMC 2023, and AIME 2024 test questions (§3.1).
- Released OpenMath2-Llama3.1-8B and -70B models, NeMo-Skills code, and the data (§1, footnote 1).

## Key Figures/Tables to Study
- Figure 3 and Table 5: pipeline and composition. Tables 1-3 and Figure 5: format, teacher, filtering, added noise.
- Figure 6: unique-question ablation. Table 4: final results. Table 9: majority-vote threshold.

## Technical Details
- Ablation setup: 1K validation split held out of MATH train; the other 6.5K MATH train problems seed SFT data; student Llama3.1-8B-Base; sampling temperature 1.0, top-p 0.95; accuracy averaged over 4 runs (§2.2). Training settings: [[openmathinstruct-2-recipe]].
- Format adherence: with the instruct template and few-shot OpenMath CoT examples, 57% of Llama3.1-405B-Instruct outputs still used the Llama CoT format; a "base" template without Llama special tokens (Fig. 8) reduced this to 0.1% (§2.2.1; App. A.1).
- Format ablation: 64 samples per question; zero-shot gave 350K and few-shot 268K solutions; coverage matching left 260K pairs (§2.2.1). Mean length 331.3 (Llama CoT) vs 237.0 tokens (Table 1). The paper calls this "40% less verbose"; 331.3/237.0 = 1.40, i.e. OpenMath CoT is 28.5% shorter (derived).
- Teacher ablation at matched coverage: Llama3.1-405B-Instruct 37.9 ± 0.6 vs Llama3.1-8B-Base 30.1 ± 0.6 (Table 2). The authors' preliminary analysis links the gap to more correct-answer, incorrect-reasoning solutions from the weaker model (§2.2.2); the filters flag 45% to 67% of 8B-Base solutions (App. B, Table 6).
- Filtering low-quality solutions from a 128K fair-downsampled set: two Llama3.1-405B-Instruct judge prompts and Nemotron-4-340B-Reward (helpfulness or correctness ≥ 3 on a 0-4 scale) remove 6% to 12% of data; accuracy 43.0 to 43.8 vs 43.6 ± 1.7 unfiltered (§2.2.3, Table 3). About 60% of 20 manually checked flagged solutions were incorrect (footnote 4).
- Added noise: wrong-answer solutions or "Incorrect Pairing" (correct solutions attached to unrelated questions) at 10/20/40/80% and 64K to 1024K pairs; little to no degradation up to 20% at ≥ 256K pairs, and Incorrect Pairing remains strong at 40% (§2.2.3, Fig. 5).
- Question diversity: 256K pairs with 1K, 2K, 4K, or 6.5K unique questions; 1K questions lose more than 10 points (§2.2.4, Fig. 6).
- Question-solution augmentation: 5 few-shot examples pairing an original question with a similar new one written by the authors; no instruction to raise difficulty; 32 solutions per new question at temperature 0.7; the majority-vote answer replaces ground truth (§3; App. D.2). Minimum-vote thresholds 0/8/16/24 kept 381K/339K/254K/160K pairs with accuracy 50.1/49.2/44.4/42.0, so 0 was chosen; data size was not matched (App. C.1, Table 9).
- Decontamination: top-5 test questions by embedding similarity (multi-qa-MiniLM-L6-cos-v1), then Llama3.1-405B-Instruct paraphrase checks in both orders (10 calls per question); 569K → 519K new questions (§3.1; App. C.2).
- Post-processing: drop solutions with multiple \boxed entries; remove the "My Solution:" prefix; truncate after the first sentence with \boxed; remove incorrect arithmetic; split complex arithmetic into steps; drop solutions over 1024 Llama3.1 tokens or under 200 characters (App. A.2).
- Composition (unique questions / pairs): GSM8K solution augmentation 7.4K / 0.46M; GSM8K question-solution augmentation 73.6K / 2.11M; MATH solution augmentation 7.4K / 2.46M; MATH question-solution augmentation 519.1K / 8.94M (Table 5).
- Evaluation: GSM8K (1.3K), MATH (5K), AMC 2023 (40), AIME 2024 (30), Omni-MATH (4.4K); zero-shot greedy and majority@256 at temperature 0.7; GPT-4o judges answer equivalence (§4).
- Results, greedy (maj@256) (Table 4): OpenMath2-Llama3.1-8B GSM8K 91.7 (94.1), MATH 67.8 (76.1), AMC 16/40 (23/40), AIME 3/30 (3/30), Omni-MATH 22.0 (24.6); OpenMath2-Llama3.1-70B 94.9 (96.0), 71.9 (79.6), 20/40 (24/40), 4/30 (6/30), 23.1 (27.6). Qwen2.5-Math-7B-Instruct has 83.6 MATH (Table 4).
- Data scaling: 8B models trained on 1M, 2M, 5M, and the full set show consistent gains and "no signs of saturation" at 14M (§4, Fig. 1).
- Checkpoint averaging: averaging the last 4 checkpoints gave more than 2 points over the last checkpoint in one ablation run (App. A.4, Fig. 9).

## Findings relevant to generality, negative feedback, distillation
- Incorrect solutions used as positive SFT targets (a test of negative marginal value, §6.1 type 1 of the course standard): up to 20% wrong-answer or mismatched solutions cost little to no accuracy at ≥ 256K pairs, and removing flagged solutions gives no gain (Fig. 5, Table 3). Result (single study): Llama3.1-8B-Base student, MATH validation.
- Distillation: the stronger teacher wins at matched coverage (Table 2). arXiv v1 called the weak-model data "on-policy data" and claimed insight into "effectiveness of on-policy training"; v2 replaced this with "equally-sized data generated by a weak student model" (v1 vs v2 Abstract, §1, §4).
- Generality: all evaluations are math benchmarks; no non-math or forgetting evaluation is reported. The 70B model improves only on a subset of benchmarks; the authors hypothesize that choices tuned on 8B validation accuracy suit weaker models (§4, Interpretation).
- Contamination: Omni-MATH was released after training and was not decontaminated; about 1.4% of its test questions are in the training data (§4, footnote 7). Retained questions similar but not equivalent to MATH test questions are shown in Table 11.

## Connections
- [[openmathinstruct]] — predecessor: 1.8M code-interpreter solutions, questions only from GSM8K and MATH train sets (§1, §5).
- [[numina-math]] — compared in Table 4; described as restrictively licensed (§1).
- [[metamath]], [[xwin-math]], [[mammoth-2]] — math SFT datasets cited as related work (§5).
- [[nemotron-4-synthetic]] — Nemotron-4 340B report (cited in §1); Nemotron-4-340B-Reward is one filter in §2.2.3, cited there via HelpSteer2 [34].
- [[llama-3]] — teacher Llama3.1-405B-Instruct and students Llama3.1-8B/70B-Base.
- [[training-verifiers-to-solve-math-word-problems]] — GSM8K, a seed and evaluation set.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2410.01560 (v2 PDF; diffed against v1), https://huggingface.co/datasets/nvidia/OpenMathInstruct-2 (license, splits), and NeMo-Skills training docs (in [[openmathinstruct-2-recipe]]).
- Corrections to the previous card version:
  - Teacher ablation "Llama-3.1-405B at 1M samples beats Mixtral at 10M" and "405B > 70B > Mixtral" → the only teacher ablation is 405B-Instruct vs 8B-Base at matched coverage, 37.9 vs 30.1 (Table 2); Mixtral does not appear.
  - "scale the solutions-per-problem before scaling problem count" → at a fixed 256K pairs, 1K unique questions (therefore more solutions per question) score more than 10 points below 6.5K (§2.2.4, Fig. 6).
  - Ablations "(a) teacher, (b) format, (c) problem-set composition, (d) per-problem sample count" → format, teacher, low-quality solutions, question diversity (Abstract).
  - "Apache-2.0" → "commercially permissive license" (Abstract); the dataset card declares CC-BY-4.0.
  - Models "1.5B/8B/70B" → 8B and 70B (§1, Table 4).
  - Seeds "MATH train 7.5K + GSM8K train 7.5K" → Table 5 lists 7.4K unique original questions with solutions for each of GSM8K and MATH; question augmentation "paraphrasing + topic/difficulty-tag conditioning" → few-shot "similar" new questions, no difficulty instruction (§3).
  - "K ≈ 32 ... temperature 1.0, top-p 0.95" → 32 solutions at temperature 0.7 for new questions (§3); 1.0/0.95 is the ablation setting (§2.2).
  - "SymPy canonicalization / numeric-string match" → the paper names no symbolic checker; new questions use majority voting (§3), evaluation uses a GPT-4o judge (§4).
  - "~23 solutions per augmented problem" → Table 5 gives 11.05M pairs / 592.7K new questions = 18.6 (derived).
  - "linear in log(size) up to ~5M; flat beyond" → "no signs of saturation" at 14M (§4).
  - "SOTA among open 8B math models" → "one of the strongest sub-10B open-source models"; Qwen2.5-Math-7B-Instruct is higher (§1, Table 4).
  - "text-CoT with minimal code usage; pure text outperforms TIR at 405B" → OpenMath CoT vs Llama CoT only; no tool-integrated comparison.
- Removed as unsupported by the source: "~650K H100-hours"; "BF16 inference via vLLM"; "avg ~700 tokens", "300-1200 tokens typical"; "near-duplicate suppression"; "+4 MATH points from question augmentation"; "~7% false-positive rate on a human-audited sample"; "within a point of closed-teacher distillations"; "closes most of the open-vs-closed gap"; "students do not acquire backtracking".
- Source inconsistencies: Llama3.1-8B-Instruct MATH is 51.9 in the Abstract but 51.8 in Table 4; the 70B gain is "3.9%" in §1 but 71.9 − 67.9 = 4.0 in Table 4.
- Not reported by the source: teacher-sampling compute, samples per original question in the final dataset, top-p for question augmentation, SFT sequence length, packing, warmup, loss masking.

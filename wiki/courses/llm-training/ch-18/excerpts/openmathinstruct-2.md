---
chapter: ch-18
course: llm-training
phase: read
excerpt_of: "OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data (Toshniwal, Du, Moshkov, Kisacanin, Ayrapetyan, Gitman, NVIDIA, 2024)"
source_url: https://arxiv.org/abs/2410.01560
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: OpenMathInstruct-2 — teacher strength, noise tolerance, question diversity, majority vote

Rewritten for the 2026-09 revision of ch-18 to match the verified library card (checked 2026-09-14 against arXiv v2).
The earlier version of this excerpt contained SymPy verification, a ~7% false-positive audit, K = 32 at temperature 1.0
for all solutions, ~650K H100-hours, a 1.5B model, and a plateau near 1M examples; none is in the paper
(ch-18 read.md Corrections 18-25).

- **Authors:** Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, Igor Gitman (NVIDIA)
- **Year:** 2024 (arXiv v1 2024-10-02; v2 2024-10-05)
- **Source type:** paper

## Ablation setup (§2.2)
- Student Llama3.1-8B-Base; 1K MATH training problems held out as validation; the other 6.5K seed the SFT data.
- Teacher sampling temperature 1.0, top-p 0.95; accuracy averaged over 4 runs.

## Findings ch-18 uses
1. **Teacher strength (Table 2):** at matched coverage, Llama3.1-405B-Instruct data 37.9 ± 0.6 vs Llama3.1-8B-Base data 30.1 ± 0.6. The authors' preliminary analysis links the gap to more correct-answer, incorrect-reasoning solutions from the weaker model (§2.2.2).
2. **Filtering (Table 3):** two Llama3.1-405B-Instruct judge prompts and Nemotron-4-340B-Reward removed 6% to 12% of a 128K set; accuracy 43.0 to 43.8 vs 43.6 ± 1.7 unfiltered. About 60% of 20 manually checked flagged solutions were incorrect (footnote 4).
3. **Added noise (Fig. 5):** wrong-answer solutions or mismatched question-solution pairs at 10/20/40/80%, 64K to 1024K pairs; little to no degradation up to 20% at ≥ 256K pairs.
4. **Question diversity (Fig. 6):** at 256K pairs, 1K unique questions scored more than 10 points below 6.5K.

## Data construction (§3, App. A.2, App. C)
- Question augmentation: 5 few-shot examples of an original question paired with a similar new one; no instruction to raise difficulty.
- 32 solutions per new question at temperature 0.7; the majority-vote answer replaces ground truth; minimum-vote threshold 0. Thresholds 0/8/16/24 kept 381K/339K/254K/160K pairs with accuracy 50.1/49.2/44.4/42.0 (Table 9; data size not matched).
- Decontamination: top-5 test questions by multi-qa-MiniLM-L6-cos-v1 embeddings, then Llama3.1-405B-Instruct paraphrase checks in both orders; 569K → 519K new questions (§3.1, App. C.2).
- Post-processing: drop solutions with multiple \boxed entries; truncate after the first sentence with \boxed; drop solutions over 1024 tokens or under 200 characters; other arithmetic clean-up (App. A.2).
- Composition (Table 5): 13.97M pairs over 607.3K unique questions; 592K synthesized questions; 11.05M pairs on new questions (18.6 per new question, derived).

## Final models (§4, Table 4)
- OpenMath2-Llama3.1-8B: Llama3.1-8B-Base on the full data; MATH 67.8 greedy vs 51.9 for Llama3.1-8B-Instruct (Abstract; Table 4 prints 51.8). OpenMath2-Llama3.1-70B: 70B-Base on a 5M subset; MATH 71.9.
- 8B trained on 1M, 2M, 5M, and the full set shows "no signs of saturation" (Fig. 1).
- Evaluation answers are judged by GPT-4o (§4).

## Limits relevant to ch-18
- All evaluations are math benchmarks; no non-math or forgetting evaluation is reported.
- Omni-MATH was not decontaminated; about 1.4% of its test questions are in the training data (§4, footnote 7).

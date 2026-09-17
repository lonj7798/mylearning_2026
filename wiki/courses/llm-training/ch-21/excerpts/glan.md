---
chapter: ch-21
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/glan.md
source_url: https://arxiv.org/abs/2402.13064
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: GLAN — "Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models"

This excerpt was rewritten on 2026-09-15 to match the verified library card `glan` (checked 2026-09-14 against
arXiv:2402.13064v1). The April 2026 version contained claims that are not in the paper (a hand-curated
"~36-field" root, a difficulty loop per concept, verification-friendly answer formats, same-base Alpaca and WizardLM
baselines, and depth ablations); they are removed.

- **Authors:** Haoran Li, Qingxiu Dong, Zhengyang Tang, Chaojun Wang, Xingxing Zhang, Haoyang Huang, et al.
- **Date:** arXiv v1 2024-02-20 ("Work in progress").

## Pipeline (§2, Algorithm 1)

```
D ← build_taxonomy()                     # list of disciplines (§2.1)
for each discipline d ∈ D:
    S ← generate_subjects(d)             # §2.2
    for each subject s ∈ S:
        A ← generate_syllabus(s, d)      # §2.3
        C, K ← extract_class_details(A)  # class sessions and key concepts
        Q ← generate_instructions(A, C, K, d)   # sample sessions and key concepts (§2.4)
        L ← L ∪ Q
return L
```

1. **Taxonomy (§2.1, §3.1).** GPT-4 is prompted with instructions such as "list all fields of human knowledge and
   capabilities". Human annotators vote to keep or remove elements; removing a field or sub-field removes its
   descendants. 126 disciplines were kept after majority voting. Example top-level fields: Natural Sciences,
   Humanities, Services (vocational training). The number of fields and sub-fields is not reported.
2. **Subjects (§2.2, §3.1).** GPT-4, prompted as an education expert, lists subjects; a second prompt converts the
   list to jsonl (subject_name, level, subtopics). 10 queries per discipline, temperature 1.0, top-p 0.95; 100 to 200
   subjects per discipline on average; subjects repeated across disciplines are kept.
3. **Syllabus (§2.3, §3.1).** One GPT-4 query per subject; 10 to 30 class sessions; around five key concepts per session.
4. **Questions (§2.4).** Sample one or two class sessions and one to five key concepts; GPT-4 writes a homework
   question given the sampled items and the full syllabus.
   - One session with m key concepts: Σ_{i=1..5} C(m, i) combinations.
   - Two sessions with m1 and m2 key concepts: Σ_{i=2..5} C(m1+m2, i) − Σ_{i=2..5} C(m1, i) − Σ_{i=2..5} C(m2, i).
5. **Answers (§3.1).** Generated in a separate call by GPT-3.5-turbo (temperature 0.7, top-p 0.95). No answer
   verification step is described.
6. **Decontamination (§3.1).** Pairs containing questions or input prompts from the test and training sets of every
   evaluated benchmark are removed. Total: 10 million pairs.

## Training (§3.2)
Mistral 7B; loss on response tokens only; 3 epochs; LR 3e-6, cosine, 1,000 linear warm-up steps, final LR 0; batch 512 pairs.

## Results
| Model | HumanEval | MBPP | GSM8K | MATH | BBH | ARC-E | ARC-C | MMLU |
|---|---|---|---|---|---|---|---|---|
| Mistral 7B (base) | 28.0 | 50.2 | 43.4 | 10.0 | 56.1 | 79.5 | 53.9 | 62.3 |
| WizardMath v1.1 7B | 51.2 | 54.1 | 83.2 | 33.0 | 58.2 | 79.8 | 53.2 | 60.3 |
| Mistral CodeAlpaca 7B | 35.4 | 50.2 | 34.6 | 8.3 | 56.1 | 79.1 | 54.2 | 60.9 |
| GLAN 7B | 48.8 | 57.6 | 80.8 | 32.7 | 60.7 | 90.7 | 81.1 | 62.9 |

(Table 1; other rows in the card.)

- MMLU by category (Table 2): Mistral 7B STEM 52.0 / Humanities 56.5 / Social Sciences 73.3 / Other 70.1;
  GLAN 60.1 / 54.9 / 71.8 / 68.6.
- IFEval strict prompt / instruction level: GLAN-7B 34.0 / 44.8; Mistral-7B-Instruct-v0.1 32.0 / 42.8 (Table 4).
- GLAN-Test (§3.5): 6,300 held-out instructions from GLAN data, 50 per discipline, GPT-4 pairwise judging:
  +1.61 vs Orca2-7B, +0.43 vs Mistral-7B-Instruct, +0.19 vs WizardLM-13B-V1.2, −0.55 vs GPT-4 (Table 6).
  American history, Divinity, and Radiology have negative gaps vs Orca-2-7b and Mistral-7B-Instruct (Table 8).
- Loss-gap check (§3.4, Table 3): Δ(%) = (L_test − L_train) / L_test. GLAN-7B: −0.74% ARC-C, −0.23% ARC-E,
  0.92% GSM8K, −1.79% MATH; Orca2-7B 11.4% and WizardLM-13B-V1.2 4.39% on GSM8K.
- Future work (§5): multi-turn conversations and long documents.

## Verification
- Every value above appears in the verified card `glan` with the locus given; the Algorithm 1 block and the two
  combination formulas were re-read in the cached arXiv v1 text on 2026-09-15.

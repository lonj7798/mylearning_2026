---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/openmathinstruct-2.md
source_url: https://arxiv.org/abs/2410.01560
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: OpenMathInstruct-2 — question diversity, noise tolerance, and the teacher ablation

**Source library:** `wiki/raw-data/llm-training/papers/openmathinstruct-2.md` and `openmathinstruct-2-recipe.md` (verified 2026-09-14 against arXiv:2410.01560v2).
Revised for ch-24 §2. The earlier excerpt (git 4a72e54) was titled "teacher strength dominates data scale"; the paper's findings do not support that title.

## The four findings stated in the abstract

(a) solution format matters and verbose solutions hurt SFT; (b) data from a strong teacher beats equally sized data from a weak student model;
(c) SFT is robust to low-quality solutions; (d) question diversity is needed for data-scaling gains.

## Ablations (Llama3.1-8B-Base student, 1K MATH validation split, 4 runs; §2.2)

| Ablation | Result | Locus |
|---|---|---|
| Format | OpenMath CoT 44.5 vs Llama CoT 40.6; mean length 237.0 vs 331.3 tokens | Table 1 |
| Teacher at matched coverage | Llama3.1-405B-Instruct 37.9 ± 0.6 vs Llama3.1-8B-Base 30.1 ± 0.6 | Table 2 |
| Filtering (128K) | 43.0–43.8 with judge or reward-model filters vs 43.6 ± 1.7 unfiltered | Table 3 |
| Added wrong-answer or mispaired solutions | little to no degradation up to 20% at ≥ 256K pairs | Fig. 5 |
| Unique questions at 256K pairs | 1K → 6.5K: +10.5 | §1, Fig. 6 |

## Data construction

- New questions: 5 few-shot pairs of an original and a similar question; no difficulty instruction; 32 solutions per new question at temperature 0.7; the majority answer replaces ground truth; minimum vote threshold 0 (§3, App. C.1, Table 9).
- Decontamination: top-5 embedding neighbors, then Llama3.1-405B-Instruct paraphrase checks; 569K → 519K new questions (§3.1).
- Post-processing drops solutions over 1,024 Llama3.1 tokens or under 200 characters (App. A.2).
- Composition: 13.97M pairs, 607.3K unique questions, 592K synthesized (Table 5).
- OpenMath2-Llama3.1-8B: 2 epochs, batch 512, constant LR 2e-5, AdamW weight decay 1e-2 (§4); GSM8K 91.7, MATH 67.8 greedy (Table 4).
- Only math is evaluated; about 1.4% of Omni-MATH test questions are in the training data (§4, footnote 7).

## Removed from the earlier excerpt (not in the source)

- "Llama-3.1-405B at 1M samples beats Mixtral at 10M samples"; "scale solutions per problem before problem count".
- "text-CoT outperforms TIR at 405B"; "paraphrase + topic-tag novel questions"; "~650K H100-hours"; "BF16 via vLLM".
- "+4 MATH points from question augmentation"; "~7% false-positive rate"; "students do not acquire backtracking".

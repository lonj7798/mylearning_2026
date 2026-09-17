---
chapter: ch-29a
course: llm-training
phase: read
excerpt_of: arXiv:2405.19846v7 (Quest, ICLR 2025), §3, §4, Tables 1-5, App. A.4 (chapter-local verified extract; no library card exists yet)
source_url: https://arxiv.org/abs/2405.19846
created_at: "2026-09-15"
---

# Excerpt: Quest — query-centric grouping of documents for long-context continued pretraining

- **Authors:** Chaochen Gao, Xing Wu, Qi Fu, Songlin Hu (IIE CAS; UCAS; Xiaohongshu)
- **Year:** 2024 (arXiv v1 2024-05; v7 2025-02 used here; ICLR 2025)
- **Source type:** paper
- **Used in:** ch-29a §2.1, Recipe

## Method (§3)
1. Query prediction: a doc2query model predicts queries for each document; long documents are segmented and a query is generated per segment.
2. Keyword extraction with RAKE: "we filter out keywords with a Rake score below 3.0", then remove "frequent but non-informative
   keywords such as 'following sentence' or 'best way'"; "we randomly select one of the remaining keywords to serve as the
   representative keyword for the document."
3. Inverted index: "Documents with an identical representative keyword are indexed together and treated as topically similar ones."
4. Index split: keywords are sorted by document count; the top split-ratio% form a short-index set that is oversampled (Eq. 1–2).
5. "We perform sampling without replacement from the documents within a sampled keyword and concatenate the selected documents
   up to the target context length L for training."
- App. A.4: performance first rises then falls as the split ratio grows; "The best results occur when oversampled indexes comprise 10-30%."

## Setup (§4.1)
Pythia 1.4B / 6.9B / 12B; 30B tokens of keyword-indexed documents extracted from the Pile; identical training sets for all methods,
"The only variation between the methods is how to rearrange documents"; batch 4M tokens; LR 5e-5 / 4e-5 / 2e-5.
Baselines: Standard (random concatenation), KNN (each document with its top-k retrieved neighbours), ICLM (traveling-salesman ordering).

## Results
- LongBench average at 32K (Table 1), Standard / KNN / ICLM / Quest: 1.4B 20.94 / 19.97 / 19.82 / 22.06; 6.9B 22.48 / 21.65 / 20.86 / 23.23;
  12B 24.85 / 22.95 / 24.07 / 25.24.
- Longbook QA at 128K (Table 2): 1.4B 9.94 / 10.36 / 10.70 / 11.30; 6.9B 14.47 / 13.38 / 14.92 / 17.95; 12B 17.81 / 16.42 / 18.44 / 18.92.
- Short text, 7-task average (Table 3): Pythia 0.4830; Standard 0.4802; KNN 0.4769; ICLM 0.4816; Quest 0.4831 (model size not stated in the table).
- Existing long documents vs Quest data, same token amount, LongBench average (Table 5): 21.11 vs 22.06 (Quest value equals the 1.4B row of Table 1).
  The authors attribute the gap to long documents existing "only in a few domains".
- Quest-LLaMA-3-8B-128K Longbook QA 32.39 vs LLaMA-3.1-8B-base 30.11 (Table 4, evaluated by the authors).

## Interpretation by the authors (Fig. 2)
High within-context similarity (KNN) gives redundancy; low similarity (Standard) gives little dependency; Quest sits between.

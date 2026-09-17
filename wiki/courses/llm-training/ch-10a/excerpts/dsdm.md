---
chapter: ch-10a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dsdm.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2401.12926
created_at: "2026-09-15"
---

# Excerpt: DsDm: Model-Aware Dataset Selection with Datamodels

**Authors:** Logan Engstrom, Axel Feldmann, Aleksander Mądry (MIT)
**Version read:** arXiv:2401.12926v1 (23 Jan 2024).
**Status:** no library card existed for this slug on 2026-09-15; values read in the v1 PDF text.

## Method (§2)
- Task-optimal selection (Eq. 1): S* = argmin_{S⊂𝒮, |S|=k} L_Dtarg(S), with L_D(S) = E_{x∼D}[ℓ(x; A(S))].
- Linear datamodel for target example x: τ_θx(1_S) = θ_xᵀ 1_S, with 1_S the indicator vector of the subset; parameters estimated with TRAK (App. B).
- DsDm selection: Ŝ_DM = arg bot-k( (1/n) Σ_i θ_{x_i} ): keep the k candidates with the smallest average predicted effect on target loss.
- Candidate pool: English C4 sliced into 1024-token examples, |𝒮| ≈ 217,000,000 (§3.1 footnote 2).

## Results
- §3 (125M; §3.1 text says "6 billion tokens", while App. A Table 2 lists 2.6 × 10¹⁰ tokens and 25,000 batches for the Section 3 models; targets LAMBADA, CS-Algorithms, SQuAD, Jeopardy): DsDm improves target log-probability on all four; Classifier and DSIR "do not consistently outperform randomly selecting data (e.g., on SQuAD and CS-Algorithms)". DsDm's selections for SQuAD are not Wikipedia-like; its least-likely samples "also often contain QA text" and training on them "is worse than selecting randomly" (§3.2).
- §4 (targets LAMBADA + SQuAD + Jeopardy; 125M proxy datamodels; Chinchilla-optimal 125M-1.3B models, plus 1.8B random baseline at 2× 1.3B compute): "no baseline selection method outperforms selecting data at random"; DsDm 1.3B matches the random-data 1.8B model on mean accuracy (Fig. 3).
- App. D.1: targeted methods (DsDm, Classifier, DSIR) select data for four epochs; Random and SemDeDup use one epoch.
- Fig. 4: targeting LAMBADA alone "reduces world knowledge accuracy compared to randomly selecting data".

## Table 1 (1.3B, accuracy %; delta vs Random 1.3B as printed)
| Category | Benchmark | DsDm | Random | Classifier | DSIR | SemDeDup | Random 1.8B |
|---|---|---|---|---|---|---|---|
| Commonsense | copa | 63.0 (+1) | 62.0 | 66.0 (+4) | 67.0 (+5) | 68.0 (+6) | 64.0 (+2) |
| Commonsense | openbook_qa | 31.2 (–2) | 33.4 | 32.0 (–1) | 32.0 (–1) | 32.2 (–1) | 33.6 (+0) |
| Commonsense | piqa | 69.0 (+0) | 68.9 | 69.4 (+1) | 65.7 (–3) | 69.7 (+1) | 71.5 (+3) |
| Lang. understanding | cbt | 88.2 (+2) | 86.4 | 85.1 (–1) | 92.4 (+6) | 86.2 (+0) | 88.4 (+2) |
| Lang. understanding | hellaswag | 42.3 (–3) | 44.9 | 42.7 (–2) | 40.4 (–5) | 44.9 (+0) | 50.1 (+5) |
| Lang. understanding | winogrande | 51.1 (–1) | 52.2 | 50.5 (–2) | 55.3 (+3) | 50.3 (–2) | 50.9 (–1) |
| Reading comp. | boolq | 58.0 (+3) | 54.9 | 60.9 (+6) | 61.0 (+6) | 49.9 (–5) | 53.4 (–2) |
| Reading comp. | coqa | 25.5 (+7) | 18.8 | 16.7 (–2) | 16.5 (–2) | 22.9 (+4) | 24.9 (+6) |
| Reading comp. | news_qa | 15.6 (+8) | 7.5 | 5.1 (–2) | 5.5 (–2) | 8.6 (+1) | 9.5 (+2) |
| Symbolic | bb_copy_logic | 3.1 (+0) | 3.1 | 0.0 (–3) | 0.0 (–3) | 3.1 (+0) | 3.1 (+0) |
| Symbolic | bb_dyck_lang | 11.9 (–2) | 13.5 | 3.4 (–10) | 1.0 (–13) | 7.3 (–6) | 8.9 (–5) |
| Symbolic | bb_operators | 13.3 (+3) | 10.5 | 6.7 (–4) | 10.5 (+0) | 11.4 (+1) | 9.5 (–1) |
| World knowledge | arc_easy | 47.6 (+3) | 44.8 | 44.7 (+0) | 39.6 (–5) | 43.5 (–1) | 48.5 (+4) |
| World knowledge | bb_qa_wikidata | 48.1 (+8) | 40.6 | 48.3 (+8) | 37.7 (–3) | 45.5 (+5) | 53.6 (+13) |
| World knowledge | trivia_qa | 7.1 (+3) | 3.7 | 2.5 (–1) | 3.5 (+0) | 2.4 (–1) | 4.1 (+0) |

## Training details (App. A.4, Table 2)
GPT-2 style, LLM-Foundry; Adam (0.9, 0.95, 1e-8); sequence length 1024; batch 1024; cosine with 200 warmup batches; grad clip 1. §4 models: LR 6e-4, WD 4e-4; 1.3B: d_model 2048, 16 heads, 24 layers, 2.6e10 tokens.

## How ch-10a uses it
§5.4 (datamodel selection and its formula), §6 (target choice), Negative samples (predicted harmful data), Recipe row.

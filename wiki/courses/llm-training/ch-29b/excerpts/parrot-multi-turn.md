---
chapter: ch-29b
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/parrot-multi-turn.md (library card not present on 2026-09-15; content taken from the primary source)
source_url: https://arxiv.org/abs/2310.07301
primary_version: arXiv:2310.07301v2 (2024-05; v1 2023-10)
created_at: "2026-09-15"
---

# Excerpt: Parrot: Enhancing Multi-Turn Instruction Following for Large Language Models

Authors: Yuchong Sun, Che Liu, Kun Zhou, Jinwen Huang, Ruihua Song, Wayne Xin Zhao, et al. (Renmin University of China; Kuaishou). Checked against the v2 PDF on 2026-09-15.

## Method (§3)
- Motivation: in ShareGPT "approximately 60% of the data does not exceed three turns", and ChatGPT prompted to act as a user produces "information-complete queries that lack common features found in human queries, such as anaphoras and ellipses" (§1).
- Parrot-Ask: LLaMA-13B trained on 70K ShareGPT sessions with the loss on user-query tokens only (Eq. 3, the inverse of response-only SFT Eq. 2). Max length 4,096; 3 epochs; AdamW; initial LR 3e-5, cosine, warmup 0.1 epoch; 8 A100-80G; total batch 32 with 8 gradient-accumulation steps (§4.1.3).
- Data collection: 20K first-turn queries each from ShareGPT and UltraChat; ChatGPT answers, Parrot-Ask asks the next query, repeated to the target turn count; repetitive, short, or sensitive queries removed (§3.2).
- Parrot-40K statistics (Table 1): 40K sessions, 8.71 turns, 3.42 context-dependent queries per session (GPT-4 judged), Self-Rouge 12.5; ShareGPT 6.67 turns and 4.62 context queries; UltraChat 3.85 turns and 1.45; Baize 4.54 and 1.75.
- CaPO negatives (§3.3): 10K context-dependent queries selected by pronoun recognition and GPT-4; negatives from ChatGPT under three strategies: context neglect (answer without history), context hallucination (guess the referent without history), context misunderstanding (pick irrelevant history on purpose). About 30K pairs trained with DPO; effective batch 32, LR 1e-5 (§4.1.3).

## Evaluation and results (§4)
- MT-Bench++: annotators extend MT-Bench to 8 turns (80 sessions, 640 utterances) with ellipsis and anaphora; GPT-4 scores 1-10 per turn.
- Table 3: Parrot-Chat w/o CaPO MT-Bench 6.81, MT-Bench++ 6.56; Parrot-Chat 7.04 and 6.85; LLaMA-2-13B-Chat 6.65 and 6.57; Vicuna v1.5 6.57 and 6.39.
- Table 4 (data source, 20K each): UltraChat-20K 6.09 / 6.17; Parrot-20K(U) 6.33 / 6.36; ShareGPT-20K 6.47 / 6.18; Parrot-20K(S) 6.70 / 6.26. The §4.3 prose describes these as "2.3 scores" and "2.4 scores"; the table differences are 0.23 and 0.24.
- Table 5 (turns kept for SFT): 1 turn 6.59 / 5.90; 3 turns 6.49 / 6.14; 5 turns 6.66 / 6.32; all turns 6.81 / 6.56.
- Table 6 (negative type): none 6.81 / 6.56; neglect 6.84 / 6.72; hallucination 7.06 / 6.73; misunderstanding 6.71 / 6.69; all 7.04 / 6.85.

## Limits
- All scores are GPT-4 judge scores on 80 sessions; no seeds or confidence intervals; no broad capability benchmarks (MMLU, GSM8K) reported.

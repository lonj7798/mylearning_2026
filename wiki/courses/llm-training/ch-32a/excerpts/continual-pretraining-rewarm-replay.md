---
chapter: ch-32a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2403.08763v4 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2403.08763
created_at: "2026-09-15"
---

# Excerpt: Simple and Scalable Strategies to Continually Pre-train Large Language Models

**Paper:** Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, Mats L. Richter, Quentin Anthony, Timothée Lesort, et al.
(Université de Montréal; Concordia University; Mila; EleutherAI). arXiv v1 2024-03; v4 2024-09-04 read; TMLR 06/2024.
Source type: paper. ch-32a uses the schedule, replay, scale, and limitation results.

## Claim (Abstract)
"a simple and scalable combination of learning rate (LR) re-warming, LR re-decaying, and replay of previous data is
sufficient to match the performance of fully re-training from scratch on all available data, as measured by the final
loss and the average score on several language model (LM) evaluation benchmarks" (weak shift Pile → SlimPajama and
strong shift Pile → German at 405M; weak shift at 10B).

## Setup (§4.1–§5.3, Tables 13–14)
- Cosine schedule (Eq. 2): η_t = η_min + (η_max − η_min)/2 · (cos(π · (t − t_ann)/(t_end − t_ann)) + 1).
- Compute-equivalent replay (§4.2): x% of each batch comes from D0 and the D1 token count is reduced by the same amount;
  replay data is read in the original D0 order.
- D0 = 300B Pile tokens. D1 = 300B SlimPajama (weak) or 195.43B-token German Common Crawl split, used as 200B (strong).
- Optimizer states are reset between datasets (§5.2). Batch 1104 sequences, sequence length 2048, AdamW, GPT-NeoX (§5.3).
- 405M: η_max 3·10^-4, η_min 3·10^-5, warmup 1%. 10B (9.6B incl. embeddings): η_max 1.2·10^-4, η_min 1.2·10^-5, warmup 1% (Table 14; batch and sequence length in Table 13).
  Conflict: §6.4 states that both the 405M and 10B continued runs re-warm "to the ηmax of pre-training (3 · 10−4)".

## Schedule results (§6.1)
- Warmup of 0%, 0.5%, 1%, 2%: after 50B tokens, "relatively similar forgetting and adaptation"; short warmups cause a
  transient loss spike (§6.1.1, Fig. 3).
- "the constant ηmin learning rate model achieves the least forgetting on D0"; re-warm and re-decay runs "adapt better to
  the new dataset by a significant margin"; "higher values of ηmax lead to more forgetting and more adaptation" (§6.1.2, Fig. 4).
- Re-warming on the same data (Pile → Pile) raises Pile validation loss by a peak of 0.1 at η_max 3·10^-4 and 0.2 at
  6·10^-4, versus 0.35 and 0.45 when continuing on SlimPajama (§7.1, Fig. 8).

## Replay results (Table 2, 405M; final loss averaged over the last 100 iterations)
| Run | Pile loss | D1 loss | AVG |
|---|---|---|---|
| Pile → SP, 0% / 0.5% / 1% / 5% / 10% / 50% replay | 2.44 / 2.27 / 2.26 / 2.23 / 2.21 / 2.16 | 2.50 / 2.50 / 2.50 / 2.51 / 2.51 / 2.54 | 2.47 / 2.39 / 2.38 / 2.37 / 2.36 / 2.35 |
| 600B Pile ∪ SP | 2.17 | 2.53 | 2.35 |
| Pile → German, 0% / 1% / 5% / 10% / 25% / 50% replay | 3.56 / 2.83 / 2.57 / 2.46 / 2.33 / 2.24 | 1.11 / 1.12 / 1.12 / 1.13 / 1.16 / 1.22 | 2.34 / 1.97 / 1.85 / 1.80 / 1.75 / 1.73 |
| 500B Pile ∪ German | 2.26 | 1.25 | 1.75 |
- Selected settings: 5% replay (weak) and 25% replay (strong), chosen because they see more new tokens than higher replay
  at similar average loss (§6.3). Pile-only 300B baseline: Pile loss 2.17 (Table 3).
- Table 3 LM eval: English average 35.14 (Pile → SP 5% replay) vs 34.30 (union); 32.48 (Pile → German 25%) vs 32.43 (union);
  HellaSwag-DE 31.04 vs 30.45. German evaluations other than HellaSwag are near chance and not averaged (§6.3.2).

## Scale (§6.4, Tables 4–5)
- Pile loss increase without replay: 0.23 (10B) and 0.27 (405M); 5% replay reduces it by 0.19 and 0.21.
- 10B AVG loss: 1.89 (5% replay) vs 1.87 (union). 10B eval average: 47.68 (5% replay), 47.53 (no replay), 48.00 (union);
  MMLU 28.79 (5% replay) vs 37.78 (union), which the authors suspect is due to limited training data (§6.4.2).

## Infinite LR schedules (§7.2–§7.4, Table 14)
- Warmup → cooldown to η_const → constant → final anneal; 405M values η_const 1.65·10^-4, cooldown 60% of iterations.
  Tested only at 405M without distribution shift (3 × 100B SlimPajama splits); all schedules reach similar loss (Fig. 10).

## Practice cited by the authors (§2)
- Glorioso et al. (2024): re-warming, re-decaying, and 60% replay in a 50B-token decay phase. DeepSeek-AI et al. (2024):
  a non-decayed checkpoint and 30% replay for 6T tokens of continued pre-training of DeepSeek-V2. These are the authors'
  descriptions of other reports, not re-verified here.

## Limitations (§8)
Two model sizes; no deduplication between German train and validation; mostly two-task transitions; single seed;
infinite schedules only at 405M without shift. Per-domain training (A.1) gave poor results; the authors advise continual
pre-training on a mixture of domains (§5.2).

## Verification
- Read on 2026-09-15 against arXiv:2403.08763v4 PDF text (Abstract, §1–§9, Tables 1–5, 13–14, Figs. 1, 3–10 captions).
- Not reported: models above 10B; instruction-tuned starting checkpoints; seeds; selection methods for replay samples.

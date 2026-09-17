<!-- excerpt for: ch-32e
     source: OLMo Team, "2 OLMo 2 Furious", arXiv:2501.00656v3 (2025-10-08; v1 2024-12)
     scope: pretraining-to-mid-training handoff, Dolmino Mix 1124 budgets and mixtures, microanneals, checkpoint soups
     library card: [[olmo-2]] (predates the 2026-09 verification; see "Disagreements with the library card")
-->

# OLMo 2 mid-training — verified extract

Checked on 2026-09-15 against the arXiv v3 PDF text. Every value below is followed by its locus.

## Stage boundaries and schedule
- Two base stages: pretraining (">= 90% training FLOPs") and mid-training ("5–10% of training FLOPs"), in which the learning rate is linearly decayed to zero (§2.3).
- Table 3: peak LR 3.0e-4 (7B), 9.0e-4 (13B), 6.0e-4 (32B); warmup 2000 steps; cosine schedule over 5T (7B, 13B) and 6.5T (32B) tokens, truncated after 4T (7B) and after 6T (32B); sequence length 4096 for all sizes; batch 1024 (7B) and 2048 (13B, 32B) instances; gradient clipping 1.0.
- Total tokens: 7B 4.05T (3.90T pretraining); 13B 5.6T (5T pretraining); 32B 6.6T (6.06T pretraining) (§2.3 "Overall").
- The Table 9 caption gives the pretrain checkpoints as 4T (1B, 7B), 5T (13B) and 7T (32B) tokens, which disagrees with 6.06T for 32B in §2.3.

## Budgets and soups
- 7B: three anneals of 50B tokens each with different data orders, averaged. 13B and 32B: three runs of 100B tokens ("same number of update steps as the 7B") and one of 300B tokens; the final model averages all four (§2.3 "Model Merging"; §4.5).
- The report's total-token counts include every souping ingredient: 3.90T + 3 × 50B = 4.05T; 5T + 3 × 100B + 300B = 5.6T (§2.3; arithmetic by this course).
- 1B: one 50B run, no averaging (Table 9 caption).

## Mixtures (Table 13, "Mix %" = share of the Dolmino mix)
| Source | pool tokens | 50B mix % | 100B mix % | 300B mix % | 300B source % (repeats) |
|---|---|---|---|---|---|
| Filtered DCLM | 752B | 47.2 | 50.2 | 51.9 | 20.78 |
| Decontam. FLAN | 17.0B | 16.6 | 16.7 | 11.3 | 200 |
| StackExchange Q&A | 1.26B | 2.45 | 2.47 | 1.68 | 400 |
| peS2o | 58.6B | 5.85 | 9.52 | 19.4 | 100 |
| Wikipedia/Wikibooks | 3.7B | 7.11 | 3.57 | 4.86 | 400 |
| Dolmino Math | 10.7B | 20.8 | 17.5 | 10.8 | 400 |

"Source %" above 100 means repetition, for example 400 = four passes (Table 13 caption).

## Results used in the chapter
- Table 9 (pretraining → pretraining & mid-training): 7B average 53.0 → 62.9, GSM8K 24.1 → 67.5, MMLU 59.8 → 63.7; 13B average 58.9 → 68.3; 32B average 66.3 → 73.3, GSM8K 56.2 → 78.8.
- Table 8: peak LR 3e-4 vs 6e-4, each decayed to 0 over 100B high-quality tokens after 2T tokens: OLMES 73.8 vs 73.9; GSM8K 2.8 points higher for 6e-4 (§4.1 text).
- Microanneals: 50/50 target data and general web data, LR driven linearly to zero; 19 microanneals totalling 130B tokens (§4.4.2). Table 12, experiment 1 (7B, from baseline GSM* 28.5, MMLU 59.8): Math 35/65 → GSM* 63.5 (576M tokens); Math 10/90 → 61.0 (1.72B tokens).
- GSM* is 200 of the 1,319 GSM8K test examples used for development; the remaining 1,119 are reported as held out (§2 footnote 6; §4.4.2). The GSM8K train split is part of the Dolmino Math pool (Table 5).
- Table 14 (7B, 50B mid-training, best single vs 3× soup): mix A GSM* 71.0 → 74.0; mix B 73.0 → 77.0; mix E GSM* 60.5 → 43.0 while OLMES 73.4 → 75.3. The caption states "Souping consistently equals or outperform the single best checkpoint"; the mix E GSM* row does not.

## Disagreements with the library card [[olmo-2]] (for the card maintainer)
- "Context: 4K native, extended to 32K in cooldown" → sequence length 4096 in Table 3; no extension stage is described.
- "Dolmino mix ~50B tokens" → 50B is the 7B budget; 13B and 32B use 3 × 100B + 1 × 300B (§2.3, §4.5).
- "7B ~460K H100 GPU-hours; 13B ~1.9M" → not found; Table 19 reports energy (MWh), not GPU-hours.

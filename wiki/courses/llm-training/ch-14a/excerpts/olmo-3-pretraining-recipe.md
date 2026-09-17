---
chapter: ch-14a
course: llm-training
phase: read
excerpt_of: "Olmo 3 (arXiv:2512.13961v2), front matter, §2.1, §3.4–§3.5.4, Figures 3–4 and 12, Tables 2–5 and 35"
source_url: https://arxiv.org/abs/2512.13961
created_at: "2026-09-15"
note: "The library card model-reports/olmo-3.md has no Verification section as of 2026-09-15 and gives no schedule, batch, or per-source mixture values. Values in ch-14a come from this excerpt."
---

# Excerpt: Olmo 3 — base-model stages, schedule, batch, and stage mixtures

Source type: official technical report (Allen Institute for AI). PDF text of arXiv v2 read on 2026-09-15. Released base checkpoints named on the title page: Olmo-3-1025-7B and Olmo-3-1125-32B.

## Stages and budgets (§2.1, Table 35)
- "three stages of pretraining for up to 5.9T tokens (Section §3.4), midtraining for 100 billion tokens (Section §3.5), and the newly added long-context extension for 50B (Olmo 3 Base 7B) or 100B (Olmo 3 Base 32B) tokens" (§2.1).
- Table 35:

| | 7B pretraining | 7B midtraining | 7B long-context | 32B pretraining | 32B midtraining | 32B long-context |
|---|---|---|---|---|---|---|
| LR schedule | Modified cosine (Fig. 3) | Linear decay | Linear decay | 5.93T cosine truncated at 5.5T | Linear decay | Linear decay |
| Warmup from 0 | 2000 steps | 0 | 200 steps | 2000 steps | 0 | 200 steps |
| Peak LR | 3.0 × 10^−4 | 2.074 × 10^−4 | 2.074 × 10^−4 | 6.0 × 10^−4 | 2.071 × 10^−4 | 2.071 × 10^−4 |
| Final LR | 3.0 × 10^−5 | 0 | 0 | 6.0 × 10^−5 | 0 | 0 |
| Batch (instances) | 512 | 256 | 64 | 1,024 | 512 | 128 |
| Sequence length | 8,192 | 8,192 | 65,536 | 8,192 | 8,192 | 65,536 |
| Batch (tokens) | 4,194,304 | 2,097,152 | 4,194,304 | 8,388,608 | 4,194,304 | 8,388,608 |
| Total tokens | 5.93T | 100B | 50B | 5.5T | 100B (twice) | 100B |
| Peak training temperature (LR²/bsz) | 2.146 × 10^−14 | 2.051 × 10^−14 | 1.026 × 10^−14 | 4.292 × 10^−14 | 1.023 × 10^−14 | 5.113 × 10^−15 |

- Table 35 caption: for the 32B, "double the batch size in all steps, run midtraining twice (with different data order seeds, and average model weights of resulting checkpoints), and increase the long-context extension stage from 50B to 100B tokens."
- Figure 3 caption (7B): "The first half of the learning rate schedule is a cosine schedule over 5T tokens. We stretch the second half of the schedule to reach a target length of one epoch (5.93T tokens). Warm-up is 2000 steps, the peak learning rate is 3 × 10^−4, and the final learning rate is 10% of the peak LR."
- Figure 4 caption (32B): cosine over one epoch (5.93T), truncated at 5.5T; "Due to the truncation, the real final learning rate is 6.210 × 10^−5. Unintuitively, the learning rate for the 32B is higher than for the 7B, but this is somewhat compensated for by the larger batch size of the 32B (8M tokens vs. 4M tokens per batch)." Table 35 prints 6.0 × 10^−5 as the 32B final LR.

## Stage-1 mixture (Table 4)
Dolma 3 Mix, "6T Mix" column (9T pool tokens in parentheses): Common Crawl 4.51T, 76.1% (8.14T); olmOCR science PDFs 805B, 13.6% (972B); Stack-Edu (Rebalanced) 409B, 6.89% (137B); arXiv 50.8B, 0.86% (21.4B); FineMath 3+ 152B, 2.56% (34.1B); Wikipedia & Wikibooks 2.51B, 0.04% (3.69B); total 5.93T (9.31T pool).
- Principles (§3.4): a source is considered for pre-training "if it has potential to yield enough tokens to impact model capabilities at pretraining scale"; structured task data is reserved "only for later stages of midtraining ... and long-context extension" because it "tends to have an outsized impact on evaluation results, potentially confounding data ablations for other sources".

## Stage-2 mixture (Table 5), grouped by the table's Type column
Dolma 3 Dolmino Mix, 100B mix (99.95B tokens): Math (synth) 19.2%; Code 10.0% and Python (synth) 10.0%; QA (synth) 13.9%; Thinking (synth) 8.34%; Instruction (synth) 6.1%; PDFs 5.0%; Web pages 27.5% (Common Crawl HQ subset 22.5% + STEM-Heavy Crawl 5.0%). Group sums are derived from the per-source percentages. The math group includes "Dolmino Math" (10.7B, 10.7%) marked as reuse of previously introduced data.

## Decontamination of mid-training data (§3.5.4, Figure 12)
- "We decontaminate against all splits of all benchmarks" (Figure 12 caption); for DROP this removed "over 60,000 training examples from sources such as Flan".
- A matched 100B anneal on non-decontaminated data was compared with the final decontaminated anneal: "Performance is sometimes, but not always, inflated by contamination"; DROP, Minerva, and SQuAD changed substantially; GSM8K showed "complete leakage" but performance "is in fact better with the decontaminated data".

## Evaluation during and after pre-training
- 7B: "anneal the learning rate to zero at regular intervals throughout training to assess progress"; 32B: "average the weights from four checkpoints, chosen 1,000 steps apart at regular intervals" (§3.4).
- Tables 2 and 3 include an "OlmoBaseEval HeldOut" block (LBPP, BBH, MMLU Pro MC, Deepmind Math); their captions state "Olmo 3 was not evaluated on held-out benchmarks prior to release."
- 32B midtraining soup: merging two midtraining runs gave "nearly a full point" on the MCSTEM cluster and 2.9 and 1.6 points on the Math cluster relative to the first and second runs (§3.5.4).

## Verification
- Read on 2026-09-15 against arXiv:2512.13961v2 PDF text.
- Internal inconsistencies: 32B final LR 6.0 × 10^−5 (Table 35) versus 6.210 × 10^−5 (Fig. 4 caption); 7B midtraining peak LR 2.074 × 10^−4 (Table 35) versus 2.071 × 10^−4 for 32B.
- Not reported: an ablation of the stretched 7B schedule against an unstretched one; per-source ablation numbers for the stage-1 mix.

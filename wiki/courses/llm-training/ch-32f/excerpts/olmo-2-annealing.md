---
chapter: ch-32f
course: llm-training
phase: read
excerpt_of: primary source arXiv:2501.00656v3 (the library card [[olmo-2]] predates the 2026-09 verification and does not contain these tables)
source_url: https://arxiv.org/abs/2501.00656
created_at: "2026-09-15"
---

# Excerpt: OLMo 2 learning-rate annealing, mid-training mixes, microanneals, and soups

**Paper:** Team OLMo: Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, et al. (Allen Institute for AI), "2 OLMo 2 Furious". arXiv v1 2024-12; read at v3 (2025-10-08) on 2026-09-15. Source type: official technical report.

## Schedule (Table 3, §4.1, App. B)

- OLMo 2 7B: sequence length 4,096; peak LR 3.0 × 10⁻⁴; 2,000 warmup steps; cosine schedule over 5T tokens truncated after 4T; gradient clipping 1.0; batch 1,024 (Table 3). RoPE θ = 500,000 (§2.1).
- OLMo 2 1B departs from 7B in layers (16), hidden size (2,048), heads (16/16), batch size (512), and peak LR (4.0 × 10⁻⁴). "We pretrain OLMo 2 1B to 4 trillion tokens on OLMo 2 Mix 1124 and perform a single 50B token anneal on Dolmino Mix 1124", with the schedule set to 5T tokens and truncated at 4T (App. B). Sequence length and RoPE θ are not listed as departures.
- Higher-LR experiment: checkpoints after 300B tokens decayed to zero over 50B (and 100B) tokens; "a higher learning rate does make mid-training more effective, but it does so by exactly the amount that the pretraining is worse" (§4.1, Fig. 11).

## Mid-training mix comparison (Table 11; 7B checkpoint at 4T tokens; 50B-token runs)

| Mid-training mix | OLMES (MCF) | OLMES-Gen | MMLU (MCF) | GSM* |
|---|---|---|---|---|
| n/a (pretrain checkpoint) | 69.6 | 63.2 | 59.8 | 28.5 |
| PT Mix | 74.0 | 64.5 | 61.8 | 27.0 |
| Web FW72 | 75.2 | 63.8 | 63.1 | 28.5 |
| Web FW72 + Math | 75.7 | 69.7 | 62.3 | 52.0 |
| Web FW72 + Math + Ins | 75.7 | 70.2 | 63.1 | 46.5 |

- GSM* is "a random sample of 200 GSM8K questions we use as development set"; "we only allow ourselves to inspect performance on 200 of the 1319 GSM8K examples to inform decisions about data mixtures" (Table 11 caption; §4.4.2).
- The text calls the PT Mix change on GSM* (−1.5) a decrease and the Web FW72 change (+1.5) "within margin of error" (§4.3). The text prints the PT Mix MMLU gain as "+20"; the table gives 59.8 → 61.8.

## Microanneals (§4.4.2, Table 12)

Recipe, quoted: "1. identify a source or small collection of math sources that we want to assess the data quality of; 2. collect roughly the same quantity of data from the general data mix (e.g., DCLM) as from the math sources ...; 3. train this 50/50 mixture as if it were an annealing run, making sure to linearly drive the learning rate down at the proper rate for this smaller collection of data." 19 microanneals used 130B tokens in total, "with results visible after training for less than 10B tokens".

| Experiment | Mix | Web ratio | Tokens | MMLU (avg) | GSM* |
|---|---|---|---|---|---|
| all | Baseline | n/a | n/a | 59.8 | 28.5 |
| 1 | Math 35/65 | 65.0% | 576M | 60.1 | 63.5 |
| 1 | Math 10/90 | 88.3% | 1.72B | 60.9 | 61.0 |
| 2 | 2x Math | 49.3% | 798M | 60.3 | 66.0 |
| 2 | 4x Math | 48.6% | 1.57B | 60.5 | 65.0 |
| 3 | TinyGSM-Inline | 47.9% | 3.17B | 60.4 | 25.0 |
| 3 | TinyGSM-MIND | 52.1% | 6.40B | 61.4 | 65.5 |
| 3 | 2x TinyGSM-MIND | 51.3% | 12.6B | 62.1 | 70.0 |

The math mixture includes GSM8K-Train (§4.4.1). Source-internal difference: the Experiment 2 prose gives "one copy of the math yields a GSM* score of 61", while the table row "1x Math" is 63.5.

## Soups (Table 14, §4.5)

"merging 3 checkpoints annealed on three permutations of the same data mix consistently produces equal or better performance than any individual training run." Example rows (best single → 3× soup): mix B OLMES 75.3 → 77.3, GSM* 73.0 → 77.0; mix E GSM* best single 60.5, soup 43.0 (the E row contradicts "consistently" on GSM*).

## Used in

ch-32f §2 (control arm, 50/50 candidate arms, GSM* half-width example), §5 (run-to-run variation), Recipe rows.

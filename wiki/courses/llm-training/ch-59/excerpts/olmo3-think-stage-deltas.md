<!-- excerpt for [[ch-59]] — per-stage scores, evaluation variance, and RL-mixture findings
     source: Olmo 3 technical report, arXiv:2512.13961v2 (Tables 14, 15, 23; §4.1.1, §4.4.3, §4.5)
     read 2026-09-17 from the cached primary text; see [[olmo-3]], [[allenai-olmo3-open-instruct-scripts-recipe]]
-->

# Olmo 3 Think — stage-by-stage scores and evaluation variance

## Table 14: Olmo 3 Think 32B across training stages

The Table 14 caption states only that these are results for Olmo 3 Think 32B. The evaluation configuration
is given in §4.1.1 (32K max context, temperature 0.6, top-p 0.95, standardized across all models); the
"mean of three runs" wording appears in the caption of Table 15, the 7B counterpart.

| Benchmark | Final SFT | Final DPO | Think 3.0 (RL) | Think 3.1 (extended RL) |
|---|---|---|---|---|
| MATH | 95.6 | 95.9 | 96.1 | 96.2 |
| AIME 2024 | 73.5 | 76.0 | 76.8 | 80.6 |
| AIME 2025 | 66.2 | 70.7 | 72.5 | 78.1 |
| OMEGA | 43.1 | 45.2 | 50.6 | 53.4 |
| BigBenchHard | 88.8 | 89.1 | 89.8 | 88.6 |
| ZebraLogic | 70.5 | 74.5 | 76.0 | 80.1 |
| AGI Eval English | 85.9 | 87.8 | 88.2 | 88.8 |
| HumanEvalPlus | 90.0 | 91.6 | 91.4 | 91.5 |
| MBPP+ | 66.7 | 67.2 | 68.0 | 68.3 |
| LiveCodeBench v3 | 75.8 | 81.9 | 83.5 | 83.3 |
| IFEval | 83.9 | 80.6 | 89.0 | 93.8 |
| IFBench | 37.0 | 34.4 | 47.6 | 68.1 |
| MMLU | 85.3 | 85.2 | 85.4 | 86.4 |
| PopQA | 33.1 | 37.0 | 31.9 | 30.9 |
| GPQA | 55.7 | 57.6 | 58.1 | 56.7 |
| AlpacaEval 2 LC | 69.1 | 78.6 | 74.2 | 69.1 |
| Safety | 64.8 | 65.3 | 68.8 | 83.6 |

Report text on the 3.0 → 3.1 step: "gains of 4+ points on AIME, 4 points on ZebraLogic, 4 points on IFEval,
and 20 points on IFBench … Most other benchmarks remain largely unchanged, with the exception of
AlpacaEval, where we observe a 5-point drop."

## §4.1.1: measured evaluation variance, used to bucket the suite

The report takes "the mean of the standard deviation from 3 runs of 14 models (both baselines and our final
models)" and partitions the suite:

- High variance: GPQA 1.4798, AlpacaEval 3 1.2406, IFEval 0.8835.
- Stable: ZebraLogic 0.5638, Omega 0.5579, AIME 24 (Avg@32) 0.5437, HumanEvalPlus 0.4615, AgiEval 0.4339,
  BigBenchHard 0.3866.
- Very stable: LiveCodeBench (Avg@10) 0.2852, MBPPPlus 0.2749, MATH 0.2522, MMLU 0.2219, PopQA 0.1554.

It also states that during recipe development on 7B versions, "evaluation costs between 10 and 20% of our
compute budget".

## §4.5 (Key Findings): RL data mixture, over-optimization, and the DPO starting point

- "Figure 20 (left) demonstrates that training on specific domains can lead to over-optimization, in which
  performance on evaluations outside that domain drops, while training on a mix yields steady improvements
  across different domains. For example, we observe a trade-off when performing OlmoRL on IFEval alone,
  wherein higher IFEval scores correlate with lower AlpacaEval scores."
- "we observe lower train reward across each domain when training on mixed data as opposed to single-domain
  data, as seen in Figure 21. This suggests that mixing data may in fact reduce the model's tendency to
  over-optimize during training".
- "we find that our RL framework yields greater improvements when applied after contrastive learning with
  DPO rather than directly following SFT (Figure 19)." This sentence is in the §4 contributions list; the
  comparison it summarizes is Table 22 and Figure 19 in §4.5, and it is an Olmo 3 7B comparison.

## Table 23 (§4.4.3): RL infrastructure ablation (2-hour benchmark, 2 × 8 A100 nodes, one train + one inference)

| Configuration | Total tokens (Mtok) | Tokens/second | MFU (%) | MBU (%) |
|---|---|---|---|---|
| OLMo 2 | 6.34 | 881 | 0.30 | 12.90 |
| + continuous batching | 7.02 | 975 | 0.33 | 14.29 |
| + better threading | 9.77 | 1358 | 0.46 | 19.89 |
| + inflight updates (Olmo 3) | 21.23 | 2949 | 1.01 | 43.21 |

Footnote: "While an initial checkpoint took 14 straight days of training across 9 nodes to achieve 1 epoch,
with continuous batching and inflight updates, we could achieve 1 epoch on 5 nodes in 7 days."

---
chapter: ch-18
course: llm-training
phase: read
excerpt_of: "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets (Liu et al., Salesforce AI Research, 2024)"
source_url: https://arxiv.org/abs/2406.18518
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: APIGen — three checks, per-generator yields, and the add-back ablation

Rewritten for the 2026-09 revision of ch-18 to match the verified library card (checked 2026-09-14 against arXiv v1).
The earlier version of this excerpt contained a per-layer removal ablation (−6/−11/−18), a GPT-4 judge, a 5-second
sandbox, MinHash deduplication, a "~40%" rejection rate, and Mistral/Mixtral base models; none is in the paper
(ch-18 read.md Corrections 11-17).

- **Authors:** Zuxin Liu, Thai Hoang, Jianguo Zhang, Ming Zhu, Tian Lan, Shirley Kokane, et al. (Salesforce AI Research)
- **Year:** 2024 (arXiv v1 2024-06)
- **Source type:** paper

## Pipeline (§3)
- API library: 3,539 cleaned ToolBench REST APIs + 134 Python functions = 3,673 APIs in 21 categories (§4.1).
- Samplers for APIs, seed QA examples, and prompt templates; the sampling ranges are not reported (§3.3). Verified samples are added back to the seed set (§3.1).
- Four query styles: simple, multiple, parallel, parallel multiple (§3.3).
- Check 1, format: JSON with "query" and "answer"; calls must use functions and arguments from the given APIs (§3.2).
- Check 2, execution: Python functions run in a subprocess, REST APIs are called; errors, invalid parameters, and timeouts are removed; no timeout value is reported (§3.2).
- Check 3, semantic: another LLM, not named, receives functions, query, calls, and execution results and returns pass yes/no (§3.2, App. B.2).

## Filtering statistics (§4.2, Table 1; 40,000 target samples per generator, temperature 0.7)
| Generator | Verified | Fail format | Fail execution | Fail semantic | Pass rate |
|---|---|---|---|---|---|
| DeepSeek-Coder-33B-Inst | 13,769 | 4,311 | 15,496 | 6,424 | 34.42% |
| Mixtral-8x7B-Inst | 15,385 | 3,311 | 12,341 | 7,963 | 38.46% |
| Mixtral-8x22B-Inst | 26,384 | 1,680 | 5,073 | 6,863 | 65.96% |
| DeepSeek-V2-Chat | 33,659 | 817 | 3,359 | 2,165 | 84.15% |

The Mixtral-8x7B-Inst row as printed sums to 39,000; its pass rate equals 15,385 / 40,000.
The released ~60,000 samples come from Mixtral-8x22B-Inst and DeepSeek-V2-Chat only (§4.2). A human check of 600 released samples found 28 with minor issues (App. A.3).

## Models and results
- xLAM-7B (FC) from DeepSeek-Coder-7B-instruct-v1.5; xLAM-1B (FC) from DeepSeek-Coder-1.3B-instruct (§5.1).
- BFCL leaderboard of 2024-06-15 (Table 2): xLAM-7B (FC) rank 6, 85.65; xLAM-1B (FC) rank 24, 74.41; GPT-3.5-Turbo-0125 (FC) rank 33, 63.88.
- Add-back ablation (§5.2, Fig. 5): adding data that failed the semantic check changed BFCL overall by −4.06 (7B) and −9.59 (1B); adding data that failed execution by −5.94 (7B) and −12.17 (1B). No format arm. The text does not state whether the execution arm also contains semantic failures.
- SFT settings (App. B.3): 8,000 relevance-detection examples; LR 5e-6, cosine, 50 warmup steps; 4 epochs; AdamW; cutoff length 2048; bf16 on 8 × A100 40GB.

## Limits relevant to ch-18
- Evaluation on BFCL only; no held-out-API split and no non-function-calling benchmarks (§5).
- Single-turn generation only; multi-turn is left to future work (§6), later addressed by [[apigen-mt]].

---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/apigen.md
source_url: https://arxiv.org/abs/2406.18518
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against arXiv v1 and the verified card; the earlier version contained a per-layer ablation table that is not in the paper)"
---

# Excerpt: APIGen — three-stage verification of single-turn function-calling data

**Source library:** `wiki/raw-data/llm-training/papers/apigen.md` (verified 2026-09-14)
**Paper:** Liu, Hoang, Zhang, Zhu, Lan, Kokane et al., "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets", arXiv:2406.18518 v1 (2024-06). The arXiv record lists no venue.

## Pipeline (§3.1–§3.3, Fig. 2)

1. Sample APIs, seed QA examples, and a prompt template. Templates include ambiguous or misspelled requests (§3.3). The sampling ranges are not reported.
2. An LLM generator writes JSON with "query" and "answer" fields; one call can return several QA pairs (App. B.1).
3. Stage 1, format checker: the output must parse and use only functions and arguments present in the sampled APIs (§3.2).
4. Stage 2, execution checker: Python functions run in a subprocess; REST APIs are called and status codes read. Type errors, invalid parameters, runtime errors, timeouts, and missing arguments are removed (§3.2). No timeout value is reported.
5. Stage 3, semantic checker: another LLM sees functions, query, calls, and execution results and returns `{"thought", "pass": yes/no}` (App. B.2). The checker model is not named.
6. Verified samples are added back to the seed pool (§3.1).

API library: 3,539 cleaned ToolBench REST APIs plus 134 Python functions = 3,673, merged into 21 categories (§4.1).

## Filtering statistics (Table 1; 40,000 target samples per generator, temperature 0.7)

| Generator | Verified | Fail format | Fail execution | Fail semantic | Pass rate |
|---|---|---|---|---|---|
| DeepSeek-Coder-33B-Inst | 13,769 | 4,311 | 15,496 | 6,424 | 34.42% |
| Mixtral-8x7B-Inst | 15,385 | 3,311 | 12,341 | 7,963 | 38.46% |
| Mixtral-8x22B-Inst | 26,384 | 1,680 | 5,073 | 6,863 | 65.96% |
| DeepSeek-V2-Chat (236B) | 33,659 | 817 | 3,359 | 2,165 | 84.15% |

The Mixtral-8x7B-Inst row sums to 39,000; its printed pass rate equals 15,385 / 40,000 (source inconsistency).

The released xlam-function-calling-60k (about 60,000 samples) comes from Mixtral-8x22B-Inst and DeepSeek-V2-Chat only (§4.2). A human check of 600 released samples found 28 with minor issues (App. A.3). License: CC BY 4.0 (App. A.1).

## Models and results (§5.1, Table 2, Fig. 5)

- xLAM-1B (FC) from DeepSeek-Coder-1.3B-instruct; xLAM-7B (FC) from DeepSeek-Coder-7B-instruct-v1.5.
- BFCL leaderboard of 2024-06-15: xLAM-7B (FC) rank 6, overall 85.65; xLAM-1B (FC) rank 24, 74.41; GPT-3.5-Turbo-0125 (FC) rank 33, 63.88.
- Filtering ablation (Fig. 5): the data rejected at stage 3 or stage 2 is added back to the training set. Printed deltas: xLAM-7B −4.06 (+Fail Semantic), −5.94 (+Fail Execution); xLAM-1B −9.59 and −12.17. There is no format-stage arm and no "remove one stage" table. The text does not say whether the +Fail Execution arm also contains the semantic failures.
- Relevance data: 8,000 examples whose target is an empty call or a refusal when tools cannot answer or arguments are missing (App. B.3).
- SFT settings (App. B.3): LR 5e-6, cosine, 50 warmup steps, 4 epochs, AdamW, cutoff length 2048, per-device batch 6, gradient accumulation 2, bf16, 8 × A100 40GB.

## Not reported

Semantic-checker model, execution timeout, API and example sampling ranges, total SFT examples, global batch, held-out-API evaluation, any non-function-calling benchmark (§5). Generation is single-turn only (§6).

## Connections

- [[toolllm]] — source of the REST APIs.
- [[bfcl]] — the only evaluation benchmark; query styles follow its categories (§3.3).
- [[xlam]] — model family name and training pipeline.
- [[apigen-mt]] — multi-turn follow-up.

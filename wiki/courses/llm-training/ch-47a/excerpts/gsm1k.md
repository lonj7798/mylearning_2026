---
chapter: ch-47a
course: llm-training
phase: read
excerpt_of: "A Careful Examination of Large Language Model Performance on Grade School Arithmetic (arXiv:2405.00332v4, v1 2024-05-01; NeurIPS 2024 Datasets and Benchmarks)"
source_url: https://arxiv.org/abs/2405.00332
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug gsm1k; referenced as [[gsm1k]] by the swe-bench-illusion card). Values read from the v4 PDF on 2026-09-15."
---

# Excerpt: GSM1k — a fresh parallel test set for GSM8K

**Authors:** Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, et al. (Scale AI).

## Construction (§3, §3.1, §3.2)

- 1,205 grade-school math problems written by human annotators, prompted with 3 GSM8K examples each; no language model was used at any point (§3, §3.1).
- Three review layers. Of 2,108 initial problems, 1,419 passed an independent second solve, 1,375 passed a general quality audit, and 1,205 remain after discarding problems to match the GSM8K answer-magnitude distribution (§3.1, §3.2, Fig. 3).
- Difficulty matching: annotators were asked for problems with N resolution steps, where N was estimated from the count of calculator tags in GSM8K solutions, and the requested distribution matched GSM8K's (§3.2).
- Not released at publication time; release committed on the earlier of three open models above 95% on GSM1k or June 2025 (§3).

## Difficulty checks (§3.2.1–3.2.3)

| Check | Result |
|---|---|
| Human distinguishability (19 annotators, 1 GSM1k item hidden among 4 GSM8K items, 1,205 attempts) | GSM1k identified 21.83% of the time; chance is 20% |
| Human solve rate in 15 minutes (14 annotators) | GSM8K 4.07 ± 0.93 problems; GSM1k 4.36 ± 1.11 (standard deviations) |
| Pre-GSM8K models (GPT-2, GPT-NeoX-20B) | "minimal difference" between the two sets (Fig. 12) |

The authors read the solve-rate result as GSM1k being comparable to or slightly easier than GSM8K, so drops on GSM1k are not explained by difficulty (§3.2.2).

## Evaluation settings (§4)

- Fork of EleutherAI's LM Evaluation Harness, default settings except max generated tokens raised from 256 to 1000.
- Both sets use the same prompt: 5 examples drawn at random from the GSM8K **train** split.
- Open models at temperature 0, vLLM where compatible; proprietary models through LiteLLM at provider defaults; API results collected between 2024-04-16 and 2024-07-10.
- Scoring extracts the last numeric answer in the response. An appendix ablation with manual answer extraction did not change the findings (App. J).
- The paper does not state the GSM8K test-set size used by the harness (checked §4 and App. F).

## Results (abstract, §4, App. F table)

Selected rows of the standard-prompt table (Diff = GSM8K − GSM1k accuracy; Z-score from a two-tailed two-proportion Z-test):

| Model | Diff | GSM8k | GSM1k | Z | p |
|---|---|---|---|---|---|
| Yi-6B-Chat | 0.080 | 0.437 | 0.357 | 4.135 | 0.000 |
| math-shepherd-mistral-7b-rl | 0.072 | 0.826 | 0.754 | 4.488 | 0.000 |
| phi-2 | 0.063 | 0.566 | 0.504 | 3.167 | 0.001 |
| Meta-Llama-3-8B-Instruct | 0.062 | 0.752 | 0.690 | 3.532 | 0.000 |
| Phi-3-medium-128k-instruct | 0.044 | 0.869 | 0.825 | 3.103 | 0.001 |
| Mixtral-8x22B-Instruct-v0.1 | 0.026 | 0.872 | 0.846 | 1.913 | 0.028 |
| Meta-Llama-3-70B-Instruct | 0.014 | 0.914 | 0.900 | 1.251 | 0.105 |
| gpt-4-turbo | 0.003 | 0.898 | 0.895 | 0.270 | 0.394 |
| gpt-4o | 0.002 | 0.931 | 0.929 | 0.219 | 0.413 |
| claude-3-opus-20240229 | −0.022 | 0.802 | 0.824 | −1.421 | 0.922 |
| gemini-1.5-flash-preview-0514 | −0.038 | 0.797 | 0.835 | −2.507 | 0.994 |

Worst drop is 8 percentage points (abstract). An alternative-prompt table (App. E) shows the same ordering with different absolute values.

## Analysis (§5)

- Lesson 1: Phi and Mistral families drop on nearly every release and scale; Yi, Xwin, Gemma, CodeLlama to a lesser extent (§5.1).
- Lesson 2: frontier models, including proprietary Mistral Large, show minimal drop (§5.2).
- Lesson 3: overfit models still reason. Phi-2 drops 6 points and still solves over half of GSM1k (§5.3).
- Lesson 4: contamination is not the full explanation. Per-character sequence log-likelihood on the GSM8K test set correlates with the gap: Spearman rank correlation 0.36 (p = 0.03), Pearson r² = 0.26, Kendall τ = 0.29; each additional point of gap is associated with +1.2 × 10⁻² per-character log-likelihood (§5.4, Eq. 1, Fig. 5). Outliers include Math-Shepherd-Mistral-7B-RL (large gap, low log-likelihood; the authors hypothesize the process reward model leaked reasoning chains) and Llemma (high log-likelihood, minimal gap, with a small confirmed number of GSM8K examples in its released corpus).
- Source-internal wording conflict: the abstract writes "Spearman's r² = 0.36" while §5.4 writes "Spearman's rank correlation of 0.36" and gives Pearson r² separately as 0.26.

## Limits stated by the authors

- GSM8K and GSM1k are "highly similar, but not identically distributed" despite the matching effort (§3).
- Proprietary models were evaluated through APIs, so GSM1k items were sent to providers; a held-out reserve set exists in case of leakage (§3).

## Used in

ch-47a §2 (fresh parallel test sets), §7 (audit protocol), Recipe, and the figure `figures/benchmark-audit-gap.html`.

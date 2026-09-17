---
chapter: ch-47a
course: llm-training
phase: read
excerpt_of: "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code (arXiv:2403.07974v2, 2024-06-06; v1 2024-03-12; ICLR 2025)"
source_url: https://arxiv.org/abs/2403.07974
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug livecodebench; referenced as [[livecodebench]] by the swe-bench-illusion card). Values read from the cached v2 PDF on 2026-09-15."
---

# Excerpt: LiveCodeBench — release-date filtering as a contamination control

**Authors:** Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica (UC Berkeley, MIT, Cornell).

## Design (§1, §3, §4)

- Problems are collected continuously from LeetCode, AtCoder, and Codeforces and carry a **release date**, so a model can be scored only on problems published after its training cutoff.
- 511 problems collected from May 2023 to May 2024 in the version evaluated here.
- Four scenarios: code generation, self-repair, code execution, and test output prediction. Metric: pass@1. Open models are run with vLLM, nucleus sampling at temperature 0.2 and top-p 0.95 (§4.3).

## Contamination evidence from date filtering (§5.1, Figs. 1, 10–12)

- DS-Ins-33B (DeepSeek instruct) drops sharply on LeetCode problems released after August 2023, immediately before its release date. The same shape appears in the repair and execution scenarios (Fig. 10).
- DS-Base-33B falls from pass@1 ≈ 60 on May problems to ≈ 0 on September LeetCode problems, which the authors read as competition problems entering DeepSeek pre-training and affecting every instruct model built on it.
- GPT-4-O drops on problems released after November 2023, its stated cutoff.
- Codestral scores pass@1 36.5 on problems released between May 2023 and January 2024 and 28.3 on problems since February 2024.
- The drop appears for LeetCode problems and not for AtCoder problems from the same months (Fig. 11), and GPT-4-Turbo, Gemini-Pro, Mistral-L and the Claude-3 models show no comparable month-to-month structure (Fig. 12).
- Because of this, all headline comparisons in §5.2 use only problems released since September 2023.

## HumanEval overfitting (§5.3, Fig. 5)

- Pass@1 on HumanEval+ correlates with pass@1 on the easy split of LiveCodeBench generation at 0.72.
- Models split into two groups: base and closed-access models lie near the identity line; fine-tuned open-access variants lie above it, scoring well on HumanEval+ and poorly on LiveCodeBench.
- Example: DS-Ins-1.3B reaches 59.8% pass@1 on HumanEval+ and 26.3% on LCB-Easy. DS-Ins-6.7B and CodeQwen beat Claude-3-Sonnet on HumanEval+ and trail it by more than 20 points on LCB-Easy.

## Scenario correlations (§5.2, Fig. 13)

Pass@1 correlations across the four scenarios are above 0.88 for every pair: 0.98 between generation and self-repair, 0.96 between test output prediction and execution, 0.89 between generation and execution. Model order is mostly stable across scenarios, while gaps between specific models change.

## Limits

- The date filter protects only against exposure to the exact problem; it does not control for training on look-alike competition problems.
- A model's stated cutoff date is provider-reported and unverifiable from outside.

## Used in

ch-47a §5 (live and dynamic benchmarks), §6 (benchmark-specific gains), Recipe.

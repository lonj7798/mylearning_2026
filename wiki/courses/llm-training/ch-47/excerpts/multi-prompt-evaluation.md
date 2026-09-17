---
chapter: ch-47
course: llm-training
phase: read
excerpt_of: primary source arXiv:2401.00595v3 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2401.00595
created_at: "2026-09-15"
---

# Excerpt: State of What Art? A Call for Multi-Prompt LLM Evaluation

**Paper:** Mizrahi, Kaplan, Malkin, Dror, Shahaf, Stanovsky (Hebrew University of Jerusalem; University of Haifa). arXiv v1 2024-01, read at v3 (2024-05-06); TACL. Source type: paper. Data and code: github.com/SLAB-NLP/Multi-Prompt-LLM-Evaluation.

## Setup (§3, §4.1, Table 2)
- 6.5M evaluated instances: 20 LLMs, 39 tasks, 3 benchmarks (LMentry, BIG-bench Hard, BIG-bench Lite).
- Instruction paraphrases generated with three GPT-3.5-Turbo prompting methods (rephrase, chain-of-thought, gradual), seeded from each task's original template; on average more than 200 automatic paraphrases per task, then manually verified and filtered; more than 175 manually written paraphrases for BIG-bench Lite tasks (§1, §4.1, §4.3).
- Each instruction template is evaluated on a randomly selected subset of 100 task samples (§3).

## Findings (§4, Tables 4-5, Figs. 2-4)
- Kendall's W over templates as judges and models as ranked objects: most values are below 0.85; the authors report that 10 tasks show only slight to moderate ranking agreement and two show strong agreement, and that a Friedman test gives statistically significant performance differences for 21 of 25 tasks (§4, Table 4).
- Single-token edits move accuracy: replacing "." with ":" at the end of one LMentry template moves nous-hermes from 0.04 to 0.65 (+0.61) and alpaca-13b from 0.61 to 0.19 (−0.42) on another template; replacing "omits" with "lacks" moves ultralm-13b from 0.62 to 0.19 (Table 5).
- For most tasks, the top three models under the original template differ from the top three under the average and maximum metrics (§6).

## Proposed metrics (§5)
- MaxP(M, T, I) = max over templates i in I of the score of model M on task T with template i. Use case: a downstream application with one fixed template.
- AvgP(M, T, I) = mean over templates of the same score. Use case: robustness reporting by model developers.
- Sat(M, T, I) = 1 − (MaxP − AvgP); CPS(M, T, I) = Sat · MaxP. Use case: choosing a model for a platform with many user-visible prompts.
- Example of the difference: on LMentry's rhyming-word task, Falcon-Instruct-7b and Vicuna-13b rank first by MaxP at 0.74 while their AvgP values are 0.17 and 0.15 (§6, Fig. 6).
- Kendall's τ between rankings computed before and after manual filtering of paraphrases is near-perfect to perfect for all tasks except one, so paraphrase-generation noise does not drive the rankings (§6, Table 6).

## Verification
- Read on 2026-09-15 against arXiv:2401.00595v3 PDF text (Abstract, §1, §3-§6, Tables 2-6).

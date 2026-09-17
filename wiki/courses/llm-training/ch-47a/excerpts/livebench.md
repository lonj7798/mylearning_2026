---
chapter: ch-47a
course: llm-training
phase: read
excerpt_of: "LiveBench: A Challenging, Contamination-Limited LLM Benchmark (arXiv:2406.19314v2, v1 2024-06-27; ICLR 2025)"
source_url: https://arxiv.org/abs/2406.19314
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug livebench). Values read from the v2 PDF on 2026-09-15. Code: github.com/livebench/livebench."
---

# Excerpt: LiveBench — frequently updated questions with objective ground truth

**Authors:** Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Benjamin Feuer, Siddhartha Jain, et al. (Abacus.AI, NYU, Nvidia, UMD, USC, Columbia).

Note on the title: the v2 title is "contamination-limited". Earlier public descriptions used "contamination-free"; the paper itself claims limitation, not elimination.

## Three design commitments (§1)

1. Questions come from recent information sources and are updated monthly.
2. Scoring is automatic against objective ground truth, with no LLM judge and no human preference vote.
3. Six categories, so a single score is an average over diverse tasks.

## Composition (§1, §2)

- 18 tasks across 6 categories: math, coding, reasoning, language comprehension, instruction following, data analysis.
- Task sources: recent high-school and olympiad competitions (AMC12 2023, SMC 2023, AIME 2024, USAMO 2024, IMO 2024), recent Leetcode and AtCoder problems via LiveCodeBench, Big-Bench Hard "Web of Lies" made harder, Zebra puzzles, Connections puzzles, typo fixing on recent arXiv abstracts, movie-synopsis unscrambling from recent IMDb entries, instruction-following tasks over recent Guardian articles, and table tasks over recent Kaggle and Socrata datasets.
- Each task holds 40–100 questions, targeting roughly a 30–70% success rate for the top models.
- Prompts use zero-shot chain of thought and require the answer in a parseable form (XML tags or `**double asterisks**`).
- Scoring: each question scores 0 to 1; a task score is the mean over its questions, a category score the mean over its tasks, and the LiveBench score the mean over the six categories (§3).

## Updates and results (§2.7, §3, §3.1, §3.2)

- Update 1 added a 50-question spatial reasoning task, 28 code-generation and 12 code-completion questions, bringing the benchmark to 1,000 questions. Update 2 replaced 132 questions (olympiad fully, AMPS_Hard and math_comp partially) while keeping the total at 1,000.
- 40 models evaluated, from 0.5B to 405B. At the time of the v2 paper no model exceeded 70% overall; o1-preview-2024-09-12 led by 6% over the next model, claude-3-5-sonnet-20240620.
- Category correlations (Fig. 2): math, coding, and reasoning correlate with each other; instruction following correlates weakly with every other category. Among tasks, math_comp correlates highest with the overall score.

## Limits stated by the authors (§6, §7)

- The approach applies only to tasks with a definable objective ground truth.
- Monthly updating requires recurring human work; the paper describes which tasks can be regenerated automatically and which cannot.
- Contamination is limited, not excluded: questions are public once released, so a question is only fresh relative to a given model's cutoff.

## Used in

ch-47a §5 (live and dynamic benchmarks), Recipe, Generalization lens.

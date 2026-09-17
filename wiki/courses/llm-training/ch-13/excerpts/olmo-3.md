---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-3.md (library card not verified on 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2512.13961
primary_version: arXiv:2512.13961v2 (2026-04-14); v1 2025-12
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: Olmo 3 — pretraining, mid-training, and long-context mixtures

Values used by ch-13 `read.md` §1, §4, §5, §6, §7, and Recipe, read in the v2 PDF text on 2026-09-15. Byline: Team Olmo (Allen Institute for AI).

## Dolma 3 Mix (Table 4)
| Source | 9T pool | 6T mix (share) |
|---|---|---|
| Common Crawl | 8.14T | 4.51T (76.1%) |
| olmOCR science PDFs | 972B | 805B (13.6%) |
| Stack-Edu (rebalanced) | 137B | 409B (6.89%) |
| arXiv | 21.4B | 50.8B (0.86%) |
| FineMath 3+ | 34.1B | 152B (2.56%) |
| Wikipedia & Wikibooks | 3.69B | 2.51B (0.04%) |
| Total | 9.31T | 5.93T (100%) |

## Constrained data mixing (§3.4)
- Base procedure "inspired by RegMix (Liu et al., 2024a), Data Mixing Laws (Ye et al., 2025), and CLIMB": "we train 30M-parameter models following the Olmo 3 architecture for 3B tokens (5x Chinchilla), sampling each mix from a Dirichlet distribution centered on the natural (no-mixing) distribution. As a rule of thumb, we launch a swarm of size 5x that of the number of domains."
- "We fit a separate generalized linear model for each task"; the optimizer minimizes average task BPB.
- "Since we ultimately seek a corpus with a 6T token budget, and we avoid repeating any domain more than approximately 4 − 7 times, this naturally imposes maximum ratio constraints on certain domains based on their available token counts."
- Conditional mixing: "treat the existing optimized mix as a single virtual domain with frozen mixing ratios, then re-run the base procedure over this virtual domain plus any new or modified domains." Three rounds: DCLM topics and sources; Stack-Edu languages with web fixed at 75% and Stack-Edu at 25% of that pool; PDF topics.
- Web-topic mixing result: "On 1B-parameter models trains for 5x Chinchilla, this mixture obtains an average improvement of 0.056 and max of 0.209 (in BPB), while only 13 out of 54 tasks show degradations, none of which exceed 0.035."

## Targeted mixes (App. A.2.5, Table 38; 1B models, 100B tokens, OlmoBaseEval Easy BPB, lower is better)
| Mix | QA Easy | Math Easy | Code Easy |
|---|---|---|---|
| Natural distribution | 1.017 | 0.719 | 0.592 |
| QA-heavy | 0.972 | 0.643 | 0.535 |
| Math-heavy | 0.979 | 0.586 | 0.497 |
| Code-heavy | 0.986 | 0.619 | 0.481 |
| Olmix (final objective) | 0.995 | 0.617 | 0.489 |

## Dolma 3 Dolmino Mix (Table 5, largest components of the 99.95B-token mix)
Common Crawl HQ subset 22.4B (22.5%); Dolmino Math 10.7B (10.7%); StackEdu FIM 10.0B (10.0%); CraneCode 10.0B (10.0%); Reddit To Flashcards 5.90B (5.9%); CraneMath 5.62B (5.63%); Nemotron Synth QA 5.0B (5.0%); Dolmino 1 Flan 5.0B (5.0%); olmOCR PDFs HQ 4.99B (5.0%); STEM-Heavy Crawl 4.99B (5.0%). Pool total 2.19T.

## Domain-skewed mid-training mixes (§3.5.4, Table 7; OlmoBaseEval Main)
| Mix | MC STEM | MC Non-STEM | GenQA | Math | Code | FIM |
|---|---|---|---|---|---|---|
| Gen-QA mix | 66.3 | 78.1 | 72.5 | 27.5 | 11.9 | 0.1 |
| Math-code-thinking mix | 62.5 | 69.6 | 65.9 | 60.8 | 35.6 | 37.7 |
| Round 5 (final mix) | 66.4 | 77.4 | 73.1 | 57.3 | 31.2 | 31.7 |

The Gen-QA mix omits math, code, and thinking data; the math-code-thinking mix omits QA and instruction data but keeps web data "to avoid excessive skew away from pretraining distribution".

## Long-context extension (§3.6, Table 11, §3.6.3)
- "We mix 34% long-context data with 66% high-quality short-context data sampled from Dolma 3 Dolmino Mix, and train using this mix for an additional 50B tokens for Olmo 3 7B and 100B tokens for Olmo 3 32B." Context goes from 8,192 to 65,536 tokens.
- Table 11, 50B mix: olmOCR PDFs 8K-16K 4.55%, 16K-32K 3.70%, 32K-64K 9.63%; PDFs + synthetic CWE 3.88%; PDFs + synthetic REX 12.2%; mid-training data mix 66.1%. The 600B+ pool is 639B tokens.
- "Early experiments on a 10B-token extension show that a 66% / 34% mix of long-context to short-context data drops performance on a subset of OlmoBaseEval by 2.5 points; in comparison, a 34% long-context, 66% short-context mix only drops performance by 0.8 points."
- Filtering: gzip compressibility, removing the 20% most compressible and 20% least compressible text (§3.6.1).
- Measurement: RULER is "the primary metric to guide our long-context recipe development"; "We keep HELMET as an unseen evaluation suite and test our final checkpoints on it." A footnote adds: "There is some overlap between RULER and HELMET, so this is not a perfect held-out suite."

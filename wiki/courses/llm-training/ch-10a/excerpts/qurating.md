---
chapter: ch-10a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/qurating.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2402.09739
created_at: "2026-09-15"
---

# Excerpt: QuRating: Selecting High-Quality Data for Training Language Models

**Authors:** Alexander Wettig, Aatmik Gupta, Saumya Malik, Danqi Chen (Princeton University)
**Version read:** arXiv:2402.09739v3 (17 Jul 2024), ICML 2024 (PMLR 235); v1 February 2024.
**Status:** no library card existed for this slug on 2026-09-15; values read in the v3 PDF text.

## Method
- Four criteria (§3.2): writing style, facts & trivia, educational value ("clear explanations, step-by-step reasoning, or questions and answers"), required expertise.
- Judgments (§3.4): GPT-3.5-turbo on 250K text pairs per criterion, both orders, averaged confidence p_{B≻A}; snippets ≤ 512 Llama tokens from 500K SlimPajama documents.
- Bradley-Terry (§3.1): p_{B≻A} = σ(s_B − s_A); QuRater = Sheared-Llama-1.3B with four linear heads, trained with the binary cross-entropy in §3.1; > 93% held-out judgment accuracy (App. B).
- Pairwise vs single ratings (§3.3): Kendall τ with authors' ranking of 10 documents 0.79 ± 0.01 (pairwise) vs 0.61 ± 0.06 (individual).
- Selection (§4): p(d_i) ∝ exp(s_i / τ), sampled without replacement (Gumbel top-k); τ → 0 is top-k, τ → ∞ is uniform. Ratings normalized to variance 1; τ ∈ {0.0, 1.0, 2.0} (§5.2). Domain proportions of RedPajama kept fixed (§5.2).
- App. C: this sampling approximates training on the whole corpus then RLHF toward higher ratings, with τ as the KL weight.

## Setup (§5.1, App. D)
QuRatedPajama: 260B tokens (1024-token sequences). Each run selects 30B tokens; 1.3B model; global batch 2048 sequences; LR 5e-4 cosine to 5e-5; 5% warmup; WD 0.1; Adam (0.9, 0.95); 200 H100 hours per run. Evaluation: SlimPajama validation perplexity (50M tokens) and 10 ICL tasks (5 reading comprehension, 3 commonsense, 2 world knowledge).

## Table 1 (perplexity; RC 5 tasks; Commonsense 3; World knowledge 2; Average 10)
| Method | PPL | RC | CS | WK | Avg |
|---|---|---|---|---|---|
| Uniform | 8.96 | 50.9 | 55.0 | 14.9 | 44.9 |
| DSIR with Wiki | 10.67 | 50.1 | 49.8 | 14.7 | 42.9 |
| Perplexity lowest | 11.92 | 48.3 | 49.6 | 13.7 | 41.7 |
| Writing style top-k | 10.53 | 49.3 | 53.3 | 13.5 | 43.4 |
| Writing style τ=2 | 8.90 | 51.0 | 55.8 | 14.1 | 45.0 |
| Facts & trivia top-k | 10.56 | 54.3 | 51.7 | 15.5 | 45.8 |
| Facts & trivia τ=2 | 8.91 | 52.7 | 55.6 | 15.6 | 46.2 |
| Educational value top-k | 10.59 | 54.7 | 54.9 | 14.4 | 46.7 |
| Educational value τ=2 | 8.91 | 53.3 | 56.3 | 15.7 | 46.7 |
| Required expertise top-k | 11.54 | 52.8 | 48.7 | 14.3 | 43.9 |
| Required expertise τ=2 | 8.93 | 52.7 | 55.5 | 15.0 | 46.0 |
| Uniform +50% data | 8.46 | 52.9 | 57.0 | 15.9 | 46.8 |

## Statements used in ch-10a
- §5.3: top-k "achieves strong performance gains on individual tasks, but there is always a task where it performs worse than uniform selection"; top-k gives "substantially worse perplexity".
- §5.3: educational value with τ = 2.0 improves on uniform in all 10 tasks; instruction-tuned win rate 57.3% vs uniform (Fig. 3).
- §5.3: inverse sampling (lowest ratings): "no criterion meaningfully improves overall ICL performance. However, selecting documents low in facts & trivia or required expertise benefits all 3 commonsense tasks".
- §6.1, Fig. 4: ratings "exhibit a bias towards English" across Wikipedia languages although GPT-3.5 was told to ignore language.
- §6.2: "Educational shortcut": some highly rated documents are about education-related topics but not educational.
- §6.3, Table 2 (AboutMe, 10% selected, τ = 2): educational value amplifies "research, university" (16%) and "students, school" (15%), suppresses "fashion, women" (6%) and "online, store" (7%); required expertise suppresses "mommy", "crafter", "momma" (6%). App. Table 10: retention rates are "far exacerbated" with top-k.
- Limitations: 1.3B scale; transfer to larger models not certain.

## How ch-10a uses it
§5.1 (pairwise ratings, temperature sampling, top-k narrowing), §8 (language and social bias), Negative samples (inverse sampling), Recipe row.

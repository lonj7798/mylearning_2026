---
chapter: ch-47a
course: llm-training
phase: read
excerpt_of: "Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad (arXiv:2503.21934v5, v1 2025-03-27)"
source_url: https://arxiv.org/abs/2503.21934
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug matharena; referenced as [[matharena]] by the swe-bench-illusion card). Values read from the v5 PDF on 2026-09-15. Leaderboard: matharena.ai; code: github.com/eth-sri/matharena."
---

# Excerpt: MathArena / Proof or Bluff — same-day evaluation on an uncontaminated competition

**Authors:** Ivo Petrov, Jasper Dekoninck, Lyuben Baltadzhiev, Maria Drencheva, Kristian Minchev, Mislav Balunović, Nikola Jovanović, Martin Vechev (ETH Zurich; INSAIT, Sofia University).

## Protocol (§2.1–2.3)

- Six proof-based problems from USAMO 2025, evaluated within hours of their release, so the problems cannot be in any evaluated model's training data.
- Each model solves each problem four separate times; solutions are anonymized and converted to PDF for grading. Grok 3 and Gemini-2.5-Pro were graded later and were therefore not fully anonymous.
- Four expert graders (former IMO team members or national team-selection participants). Each problem is graded independently by two of them, following the IMO practice.
- Grading schemes were written by the authors from Art of Problem Solving material, since USAMO publishes none. Each problem scores 0–7 with partial credit; 42 points per run.
- R1 and QwQ were sampled at temperature 0.6, top-p 0.95, as recommended by their authors (App. A.2).

## Results (Table 1; scores averaged over four runs)

| Model | P1 | P2 | P3 | P4 | P5 | P6 | Total (/42) |
|---|---|---|---|---|---|---|---|
| Gemini-2.5-Pro | 6.5 | 0.0 | 0.1 | 3.5 | 0.0 | 0.0 | 10.1 |
| R1 | 0.5 | 0.0 | 0.0 | 1.5 | 0.0 | 0.0 | 2.0 |
| Grok 3 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 |
| Flash-Thinking | 1.5 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 1.8 |
| Claude 3.7 | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.6 | 1.5 |
| QwQ | 1.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.2 |
| o1-pro | 0.5 | 0.0 | 0.0 | 0.0 | 0.2 | 0.4 | 1.2 |
| o3-mini | 0.5 | 0.1 | 0.0 | 0.0 | 0.0 | 0.2 | 0.9 |

- Gemini-2.5-Pro averages 10.1/42 = 24.4%; every other model is below 5% (§3).
- Gemini-2.5-Pro scores 6/7 or above in 6 of its 24 attempts. Among the roughly 175 graded solutions from the other models, one attempt (Grok 3 on Problem 1) received 7/7 (§3).
- The same models score near the top of AIME and HMMT leaderboards, which grade only the final numeric answer (§1, §3).
- Failure modes recorded as the first point of incorrect or inadequately explained reasoning, in four classes: logic, unjustified assumption, lack of creativity, algebra/arithmetic (Fig. 2).

## Automated grading (Table 2, §3.2)

| Model | Human graders | Automated grader A | Automated grader B |
|---|---|---|---|
| Gemini-2.5-Pro | 10.1 | 19.6 | 19.3 |
| R1 | 2.0 | 19.3 | 14.9 |
| Claude 3.7 | 1.5 | 19.0 | 18.4 |
| QwQ | 1.2 | 23.8 | 18.8 |
| o3-mini | 0.9 | 19.5 | 17.1 |

Both automated graders inflate scores, by up to a factor of about 20 (QwQ 1.2 → 23.8). Source-internal inconsistency: the Table 2 caption names "o3-mini and R1" as the graders while the column headers and §3.2 name o3-mini and Claude 3.7; the columns are labelled A and B above for that reason.

## Used in

ch-47a §5 (live and dynamic benchmarks), §7 (what an audit measures), Common mistakes; and as evidence that answer-only scoring and proof scoring do not agree.

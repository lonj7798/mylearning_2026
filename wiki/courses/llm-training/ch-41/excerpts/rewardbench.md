---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: arXiv:2403.13787v2 (RewardBench: Evaluating Reward Models for Language Modeling)
source_url: https://arxiv.org/abs/2403.13787
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because no library card exists yet)"
---

# Excerpt: RewardBench construction and scoring

Used by [[read]] §4.1 and the PPE comparison in §4.3. Authors: Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, et al. (Allen Institute for AI; University of Washington; Berkman Klein Center). arXiv v1 2024-03; checked against v2 (2024-06-08) on 2026-09-15.

## What the benchmark is (§4, Fig. 1)
- Each item is a prompt with a human-verified chosen and rejected completion. The RM scores both; the item is a win when the chosen score is higher. Section accuracy is the percentage of wins; random is 50% (§4.2).
- DPO models are scored by comparing `log π(y|x)/π_ref(y|x)` for the two completions (§3, Eq. 2).

## Sections (Table 1)
| Section | Items | Content |
|---|---|---|
| Chat | 358 | AlpacaEval Easy/Length/Hard; MT-Bench Easy/Medium |
| Chat Hard | 456 | MT-Bench Hard (ratings 7-8 vs 5-6); LLMBar Natural and four adversarial subsets |
| Safety | 740 | refusals (dangerous, offensive), XSTest should-refuse and should-respond, Do-Not-Answer |
| Reasoning | 1,431 | PRM Math (447: human vs buggy LLM answers); HumanEvalPack correct vs buggy code in C++, Go, JavaScript, Java, Python, Rust (164 each) |
| Prior Sets | 17.2k | Anthropic Helpful 6,192; Anthropic HHH 221; SHP 1,741; Summarize 9,000 |

- Section scores except Prior Sets are per-prompt weighted averages over subsets; Prior Sets is weighted 0.5 in the final score because of noise and a lack of well-defined tasks (§4.2, footnote 4).

## Findings used in the chapter
- Prior preference test sets have accuracy ceilings between 60% and 70% because of inter-annotator disagreement (§1).
- Some subsets are solved at 100% by small RMs; others have state-of-the-art near 75% with many models near random (§1).
- DPO models "fail to generalize to popular preference data test sets and present a higher variance in performance" (§1, contribution 2).
- Top model in v2 Table 2: RLHFlow/ArmoRM-Llama3-8B-v0.1, score 89.0 (Chat 96.9, Chat Hard 76.8, Safety 92.2, Reasoning 97.3, Prior Sets 74.3).

## Limits noted by later work
- Chosen and rejected completions often come from models of different strength, so style and source cues can decide a pair ([[rm-bench]] §1).
- In the PPE downstream experiment, RewardBench score correlates negatively with post-DPO Arena score among top models ([[ppe-reward-model-eval]] §2.2, Fig. 4).

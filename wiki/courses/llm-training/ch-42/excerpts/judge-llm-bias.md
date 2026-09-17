---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: arXiv:2306.05685 (Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena), NeurIPS 2023 Datasets and Benchmarks; library card [[judge-llm-bias]]
source_url: https://arxiv.org/abs/2306.05685
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from the primary source because the library card has no Verification section and carries unverified flip rates)"
---

# Excerpt: measured judge biases (Zheng et al.)

Used by [[read]] §4. Checked against the arXiv PDF on 2026-09-15 (tables and section numbers below are from that text).

## Position bias (§3.3, Table 2)
Two similar answers per first-turn MT-bench question (GPT-3.5 sampled twice at temperature 0.7); the judge is asked twice with the order swapped. "Consistency" is the share of pairs where the verdict does not change.

| Judge | Prompt | Consistency | Biased toward first | Biased toward second | Error |
|---|---|---|---|---|---|
| Claude-v1 | default | 23.8% | 75.0% | 0.0% | 1.2% |
| Claude-v1 | rename | 56.2% | 11.2% | 28.7% | 3.8% |
| GPT-3.5 | default | 46.2% | 50.0% | 1.2% | 2.5% |
| GPT-4 | default | 65.0% | 30.0% | 5.0% | 0.0% |
| GPT-4 | rename | 66.2% | 28.7% | 5.0% | 0.0% |

- "Only GPT-4 outputs consistent results in more than 60% of cases." The test is deliberately hard: the two answers are near-identical.
- By category (Table 10): consistency 42.0% (writing), 36.0% (humanities), 44.0% (stem), 86.0% (math), 86.0% (coding).
- By model pair (Table 11): 67.5% for GPT-3.5 vs Claude-v1, 98.8% for GPT-3.5 vs LLaMA-13B — position bias shrinks as the quality gap grows.
- Few-shot judging raises GPT-4 consistency from 65.0% to 77.5% (Table 12) but "high consistency may not imply high accuracy", and prompts are 4× more expensive.
- Mitigation used in the paper: call the judge in both orders and declare a win only if both agree, otherwise a tie (§3.4).

## Verbosity bias (§3.3, Table 3)
"Repetitive list" attack: 23 MT-bench answers containing a numbered list are made longer by prepending a GPT-4 rephrasing of the same items, adding no information. Failure = the judge prefers the longer version.

| Judge | Claude-v1 | GPT-3.5 | GPT-4 |
|---|---|---|---|
| Failure rate | 91.3% | 91.3% | 8.7% |

All three judges correctly return a tie for two identical answers, so the attack is not caught by an identity check.

## Self-enhancement bias (§3.3, Fig. 3b)
Compared with human win rates on the same pairs, "GPT-4 favors itself with a 10% higher win rate; Claude-v1 favors itself with a 25% higher win rate. However, they also favor other models and GPT-3.5 does not favor itself. Due to limited data and small differences, our study cannot determine whether the models exhibit a self-enhancement bias."

## Grading math and reasoning (§3.3, §3.4, Table 4)
Judge failure rate on 10 math questions (LLaMA-13B vs Vicuna-13B, positions swapped; failure = the judge calls an incorrect answer correct):

| Default | CoT | Reference-guided |
|---|---|---|
| 14/20 | 6/20 | 3/20 |

The text describes this as "a significant improvement in failure rate (from 70% to 15%) over the default prompt". With the CoT prompt "in many cases LLM makes exactly the same mistake as the given answers in its problem-solving process".

## Agreement with humans (§4.2, Table 5)
On MT-bench, setup S2 (non-tie votes only): GPT-4 pairwise vs human 85% (first turn) and 85% (second turn); human vs human 81% (first turn), 82% (second turn); random agreement is 50%. Under S1, which counts inconsistent votes as ties, GPT-4 vs human is 66%. Agreement rises with the win-rate gap between the two models compared, "from 70% to nearly 100%" (Fig. 2).

## Corrections to the previous version of this excerpt
- "GPT-4 flips ~22%; GPT-3.5 ~40%" → Table 2 reports consistency 65.0% (GPT-4) and 46.2% (GPT-3.5) under the default prompt, so verdicts change on about 35% and 54% of pairs.
- "Reference-guided grading: +10 pp on MT-Bench objective categories" → Table 4 reports failure rates 14/20 → 3/20 on 10 math questions; no agreement gain in percentage points is printed.
- "GPT-4 vs human agreement ~85% on MT-Bench and ~80% on Arena" → 85% is the S2 (non-tie) figure; the S1 figure is 66%.

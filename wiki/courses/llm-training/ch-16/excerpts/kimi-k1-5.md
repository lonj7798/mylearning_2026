---
chapter: ch-16
course: llm-training
phase: read
artifact: "Kimi k1.5: Scaling Reinforcement Learning with LLMs"
source_url: https://arxiv.org/abs/2501.12599
version: arXiv v4 (2025-06-03)
verified_on: "2026-09-15"
supersedes: the 2026-04 version of this file, which attributed a fixed pass-rate band [0.05, 0.5] and a 500-step re-measurement interval to this report
---

# Excerpt: Kimi k1.5 — RL prompt-set curation, sampling, and partial rollouts

Used by [[read]] §1, §3.1, §5, §6.3 and the Recipe table.

## Three curation properties (§2.1, quoted headings)

- **Diverse Coverage:** prompts spanning STEM, coding, and general reasoning.
- **Balanced Difficulty:** a well-distributed range of easy, moderate, and difficult questions.
- **Accurate Evaluability:** prompts that verifiers can assess objectively, so that performance
  reflects correct reasoning rather than superficial patterns or guessing.

A tagging system categorizes prompts by domain and discipline to keep subject areas balanced. The
report states these properties come from preliminary experiments and gives no ablation numbers.

## Difficulty estimation (§2.1)

"for every prompt, an SFT model generates answers ten times using a relatively high sampling
temperature. The pass rate is then calculated and used as a proxy for the prompt's difficulty — the
lower the pass rate, the higher the difficulty." The stated purposes are prefiltering trivial cases
and allowing different sampling strategies during RL.

## Removing formats that can be hacked (§2.1)

Multiple-choice, true/false, and proof-based questions are excluded because a correct final answer
can be reached through incorrect reasoning. For general question answering, a model is prompted to
guess the answer without any chain-of-thought; if it answers correctly within N attempts the prompt
is removed as easy to hack, and "setting N = 8 can remove the majority easy-to-hack prompts".

## Sampling during RL (§2.3.4)

- **Curriculum sampling:** start on easier tasks and progress to harder ones, using the grade and
  difficulty labels that come with the data.
- **Prioritized sampling:** track the success rate `s_i` of each problem during RL and sample
  problem `i` with probability proportional to `1 − s_i`.

§3.5 compares a curriculum (warm-up on the full mixture, then hard questions only) against uniform
sampling of mixed easy and hard problems and reports higher performance for the curriculum
(Figure 9); the figure's axis values are not printed in the text.

## Partial rollouts (§2.6.2)

A fixed output token budget caps each rollout; a trajectory that exceeds it is saved to a replay
buffer and continued in the next iteration. "only the current iteration (iter n) requires on-policy
computation. Previous segments (iter n-m to n-1) can be efficiently reused from the buffer." The
report adds that certain segments can be excluded from loss computation, and that repeated
sequences are detected, terminated early, and can be penalized. How reused segments enter the
gradient is not stated.

## Negative gradients (§3.5, Figure 10)

Against ReST, which fits the best sampled response without penalizing incorrect ones, the report's
mirror-descent objective shows better sample complexity across 11 evaluation sets, which the
authors attribute to the negative gradients. No share of the gain is quantified.

## Not reported

RL samples per prompt k, batch size, step count, rollout temperature; the τ of the objective; the
partial-rollout token budget; the size of the RL prompt set.

---
chapter: ch-16
course: llm-training
phase: read
artifact: "Reinforcement Learning for Reasoning in Large Language Models with One Training Example"
source_url: https://arxiv.org/abs/2504.20571
version: arXiv v3 (2025-10-24)
verified_on: "2026-09-15"
supersedes: the 2026-04 version of this file, which presented the result without the contamination caveat
---

# Excerpt: one-shot RLVR, read as a limit case

Used by [[read]] §7.

## The result (Abstract, §1, §3)

RLVR with a single training example raises Qwen2.5-Math-1.5B on MATH-500 from 36.0% to 73.6%; the
report states this is an 8.6% improvement beyond format correction. RLVR with two examples
(`{π1, π13}`) reaches 36.6% average across the evaluated benchmarks, above the 35.9% of the 1.2k
DSR-sub subset that contains those examples. The RL algorithm is GRPO with a KL term, which the
authors keep to maintain general language quality.

## Example selection (§3.2)

Examples are ranked by the **historical variance of training accuracy** across epochs of a run on
the full dataset, a quantity the authors relate directly to the reward. `π1` was selected by this
ranking on Qwen2.5-Math-1.5B. The report also states that examples with moderate or low historical
variance can individually produce improvements on MATH-500, so the ranking is not the only source
of usable examples.

## Why the chapter reads it as a limit case

The base model is a Qwen2.5-Math checkpoint and the headline benchmark is MATH-500. On exactly that
pairing, [[reasoning-or-memorization-rl-contamination]] measures 54.60 exact-match reconstruction of
MATH-500 problems from a 60% prefix for Qwen2.5-Math-7B and shows that random and inverted rewards
also raise MATH-500 for Qwen checkpoints while failing on a post-release arithmetic benchmark. The
one-shot result therefore bounds how much can be elicited from such a base with a single prompt; it
is not evidence that the size or composition of an RL prompt set is unimportant.

## What would settle it

The same one-example protocol on a base model from a different family and on a benchmark released
after that base model, with pass@k at large k reported alongside pass@1.

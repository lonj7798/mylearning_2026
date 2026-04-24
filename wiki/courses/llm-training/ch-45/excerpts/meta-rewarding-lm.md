---
chapter: ch-45
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/meta-rewarding-lm.md
source_url: https://arxiv.org/abs/2407.19594
created_at: "2026-04-23"
---

# Excerpt: Meta-Rewarding — push past Self-Rewarding's iter-3 ceiling

**Source library:** `wiki/raw-data/llm-training/papers/meta-rewarding-lm.md`
**Authors:** Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, Sainbayar Sukhbaatar (Meta FAIR + UC Berkeley + NYU)
**Year:** 2024 (arXiv 2407.19594)

---

## Why this source anchors ch-45

Self-Rewarding's iter-3 plateau is the clearest empirical signal that
**policy-as-judge has a calibration drift problem**. Meta-Rewarding is the
direct answer: add a **meta-judge** that evaluates the judge and train
the judge's DPO pair on the meta-judge's output. It is the cleanest example
in the literature of *training the evaluator too*, and it tells ch-45 readers
something structural — self-improvement ceilings are usually filter ceilings,
not actor ceilings.

---

## The three-role loop, attested from source lines 31-37

> Per-iteration data generation:
>   1. Sample K=7 actor responses per prompt from the current policy.
>   2. Sample N=11 judge responses per (prompt, actor-response) pair — each a score + rationale.
>   3. Meta-Judge performs pairwise comparison on those 11 judge responses
>      (per the rubric) to pick the best and worst judgment.
>   4. Actor-DPO uses (best actor response, worst actor response) pairs by aggregated judge score.
>   5. Judge-DPO uses (best judge response, worst judge response) pairs from the meta-judge.

The numerics (K=7 actor, N=11 judge) are worth staring at. Per prompt:

- Generations: 7 (actor rollouts).
- Judge calls: 7 × 11 = 77 (each actor response scored by 11 judge samples).
- Meta-judge calls: 11 choose 2 = 55 pairwise comparisons per actor response, times 7 actor responses = 385 meta-judge calls.

Total per prompt: 7 generations + 77 judgments + 385 meta-judgments = **469 forward passes**.
At Llama-3-8B scale this is affordable; at 70B it is not. The method's scalability
is bounded by the compute budget for the meta-judge leg — which is why the paper
runs on 8B and not 70B.

---

## The headline numbers, source lines 21 and 25

> Lifts Llama-3-8B-Instruct AlpacaEval 2.0 LC win-rate from 22.9 % -> 39.4 %
> over 4 meta-rewarding iterations on zero additional human data.

Breaking that curve across iterations (attested from Table 1):
`22.9 -> 29.8 -> 34.2 -> 37.5 -> 39.4`. Monotone through iter 5, whereas
Self-Rewarding flatlines at iter 3. The judge-human agreement (Table 5)
keeps climbing too — the meta-judge is doing its job.

---

## The length-bias control term — why it is separate from the meta-judge

Source line 37:

> Length-bias control: judge rubric includes "don't reward length for length's sake";
> meta-judge penalizes length-gamed judgments.

Two knobs, one goal. The judge rubric has a rule against rewarding length;
the meta-judge enforces that the judge follows its own rule. This is a
layered defense: plain DPO self-loops inflate response length ~2× over
iterations (Figure 5), and a single rubric rule without the meta-judge
enforcement mechanism is known to be ignored. You need both.

This is the same structural lesson as process-reward models in [[ch-44]]:
a reward signal without a separate mechanism to keep it honest is
reward-hackable by construction.

---

## Why this is structurally a hierarchical-evaluation pattern

Self-Rewarding is a 2-layer stack: Actor, Judge.
Meta-Rewarding is a 3-layer stack: Actor, Judge, Meta-Judge.

You can keep going — imagine an M-Meta-Judge auditing the Meta-Judge.
The paper notes that 3 layers is the practical limit given compute; the
marginal gain from a 4th layer does not justify the `~K · N · M` scaling.

This is the same pattern as hierarchical reward models in multi-step RL —
each layer stabilizes the one below, and the depth is limited by the
compute budget for the deepest layer's rollouts.

---

## Connections

- Direct extension of [[excerpts/self-rewarding-lm]].
- Structural cousin of [[excerpts/self-correct-rl]]'s two-stage Stage I/II design
  (both layer the loss function to stabilize a piece of the trajectory).
- Compared against [[excerpts/spin]] in ch-45 §4: both remove the judge-drift problem
  but by opposite mechanisms (Meta-Rewarding adds a meta-layer; SPIN replaces the
  judge with the data distribution).
- Host chapter: [[ch-45]] §3.

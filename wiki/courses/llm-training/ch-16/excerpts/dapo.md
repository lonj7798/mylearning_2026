---
chapter: ch-16
course: llm-training
phase: read
artifact: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
source_url: https://arxiv.org/abs/2503.14476
version: arXiv v2 (2025-05-20)
verified_on: "2026-09-15"
note: no library card exists for this slug yet; this file is the chapter's verified extract
---

# Excerpt: DAPO — dynamic sampling and the DAPO-Math-17K prompt set

Used by [[read]] §2, §3.2, §3.3 and the Recipe table. Every line below was read at the stated
locus in the cached primary text (`scratchpad/sources/rc03-dapo.txt`, arXiv:2503.14476v2).

## Setting

- Qwen2.5-32B base, math-only RL, verl framework, naive GRPO as the baseline algorithm (§4.1).
- Rollout: prompt batch size 512, 16 responses per prompt; training mini-batch size 512
  (16 gradient updates per rollout step); AdamW, constant learning rate 1×10⁻⁶ with a linear
  warm-up over 20 rollout steps; ε_low 0.2, ε_high 0.28; maximum generation length 20,480 tokens
  (16,384 expected maximum plus a 4,096-token soft-punish cache) (§4.1).
- Evaluation: AIME 2024 repeated 32 times, avg@32, temperature 1.0, top-p 0.7 (§4.1).

## Dynamic sampling (§3.2)

Problem, as stated: "if all outputs {o_i} of a particular prompt are correct and receive the same
reward, the resulting advantage for this group is zero. A zero advantage results in zero policy
gradients". The report adds that "the number of samples with accuracy equal to 1 continues to
increase" during training (Figure 3b), so the effective number of prompts per batch falls.

Remedy: "over-sample and filter out prompts with the accuracy equal to 1 and 0 … Before training,
we keep sampling until the batch is fully filled with samples whose accuracy is neither 0 nor 1."
The constraint in the objective (Eq. 11) is

```
s.t.  0 < |{o_i | is_equivalent(a, o_i)}| < G
```

Algorithm 1, lines 6–8: filter the sampled outputs into a dynamic sampling buffer; if the buffer
size n_b < N, continue generating instead of updating.

Cost: "although more data needs to be sampled due to the filtering out of zero-gradient data, the
overall training time is not significantly affected … the model's convergence time is even reduced,
due to fewer training steps required" (§4.2, Figure 6).

## Progressive ablation (Table 1, AIME24 avg@32)

| Configuration | AIME24 avg@32 |
|---|---|
| DeepSeek-R1-Zero-Qwen-32B (reference) | 47 |
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling (DAPO) | 50 |

Each row includes all rows above it, so the dynamic-sampling contribution (42 → 50) is measured on
top of the other four techniques. One run per row; no seed variance reported.

## DAPO-Math-17K (§3.5, App. A)

Sourced by web scraping and manual annotation from competition homepages. Answers in expressions
or formulas are rewritten so that the expected answer is an integer ("if the original answer is
expressed in the form of (a+√b)/c, we instruct the LLM to modify the question so that the expected
answer becomes a + b + c"), to make rule-based parsing reliable. Result: 17K prompts, each paired
with an integer answer.

## Not reported

Model-side ablation of the filter thresholds; the number of prompts consumed per training step
after filtering; seed variance; any non-math domain.

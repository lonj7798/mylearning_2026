---
chapter: ch-54
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/minibatch-sharing-rl.md
source_url: https://arxiv.org/abs/2402.14740
created_at: "2026-04-23"
---

# Excerpt: B × n minibatch packing — ch-54 §6's attested defaults

**Source library:** `wiki/raw-data/llm-training/papers/minibatch-sharing-rl.md`
**Artifact:** the `1/(n−1)` RLOO variance law and the `n=8` production default across verl / TRL / OpenRLHF.

---

## Why this source anchors ch-54

§6 of ch-54 asks the question a batch-size-knob-tuner always has: with a fixed compute budget `B·n`, how much do I spend on prompt diversity `B` vs rollouts-per-prompt `n`? The synthesis page collects the ablations across RLOO (Ahmadian 2024), DeepSeekMath GRPO, and framework defaults, and they all point to a sweet spot of `n ≈ 8`.

---

## The variance law §6 pins to memory

From the source (line 21):

> **RLOO §4 ablation:** leave-one-out baseline variance scales as `1/(n−1)`; moving from n=2 to n=4 halves variance; n=8 vs n=4 reduces variance further but with diminishing returns.

And (line 42), the analytic form:

```
A_{i,k}^{RLOO} = R_{i,k} − (1/(n−1)) · Σ_{j≠k} R_{i,j}           (unbiased)
A_{i,k}^{GRPO} = (R_{i,k} − μ_i) / (σ_i + ε),   μ_i = mean_k R_{i,k}
```

RLOO is unbiased and has the cleanest `1/(n−1)` scaling; GRPO adds a σ-normalization that introduces mild bias but handles unequal scale across prompts. `n ≥ 4` is where the group baseline stops being dominated by noise.

---

## Framework defaults — the three rows §6 tabulates

From the source (line 23):

> **Framework defaults:** verl `rollout.n=8`; TRL `num_generations=8`; OpenRLHF PPO `n_samples_per_prompt=4` (critic available, so less reliant on group baseline).

ch-54 §6 reproduces these as a table. The OpenRLHF `n=4` is the interesting outlier: OpenRLHF PPO has a value head (critic), so its baseline comes from the value function and `n` only drives variance reduction, not the baseline itself. Critic-free stacks (verl GRPO, TRL GRPO) default to 8.

---

## The prefix-KV economics §6 uses to justify large `n`

From the source (line 40):

> **Prefix KV sharing:** same-prompt rollouts share the prompt prefix forward pass in vLLM — effectively free beyond the first rollout. This makes higher `n` cheaper than linear in compute.

The operational consequence: with a 1024-token prompt and 512-token response, going from `n=1` to `n=8` costs roughly 1.4× wall time, not 8×. The prefix is paid once; only the decode scales with `n`. So `n=8` is not just a variance choice — it is also a pareto-efficient choice on throughput.

---

## The knee — where gains plateau

From the source (line 22):

> **DeepSeekMath GRPO:** uses n=64 per prompt; ablations in appendix show gains plateau past n=16 for math tasks, but 16 is unstable on harder OOD problems.

Interpretation: for easy tasks (GSM8K), `n=8` saturates; for hard tasks (AIME), larger `n` keeps paying. ch-54 §6 treats this as the "tuning knob for task hardness" — if you see GRPO group-reward variance stay high at `n=8` on your task, bump to `n=16`.

---

## The special case §6 warns against

From the source (line 56):

> The `n=1` special case is REINFORCE (no baseline) or requires a learned critic (PPO) — and the critic is what critic-free algorithms are trying to avoid.

So the entire critic-free family (GRPO, RLOO, REINFORCE++) exists precisely to avoid maintaining a separate value head — and `n ≥ 4` is the price. If you want `n = 1`, you are back to PPO with a value head or bare REINFORCE with all its variance.

---

## Connections

- **ch-54 §6** — the B × n table and the `n=8` default story.
- **ch-40 ([[grpo]])** — the group-relative advantage this variance law bounds.
- **[[rloo]]** — the leave-one-out baseline whose `1/(n−1)` scaling §6 quotes.
- **[[async-rollout]]** — async keeps the trainer saturated while the next `B × n` batch generates; this page tells you what `B × n` should be.
- **[[verl-rollout]]** — prefix-KV sharing in vLLM that makes large `n` sub-linear.

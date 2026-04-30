---
chapter: ch-44
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/prm800k.md, wiki/raw-data/llm-training/papers/rlvr-tulu3.md, wiki/raw-data/llm-training/papers/math-shepherd.md, wiki/raw-data/llm-training/papers/rlvr-beyond-base-model.md, wiki/raw-data/llm-training/papers/prorl.md
source_url: multiple
created_at: "2026-04-23"
---

# Excerpt: PRM vs RLVR — the taxonomy, side by side

**Source libraries:**
- `wiki/raw-data/llm-training/papers/prm800k.md`
- `wiki/raw-data/llm-training/papers/math-shepherd.md`
- `wiki/raw-data/llm-training/papers/omegaprm.md`
- `wiki/raw-data/llm-training/papers/rlvr-tulu3.md`
- `wiki/raw-data/llm-training/papers/swe-rl.md`
- `wiki/raw-data/llm-training/papers/rlvr-beyond-base-model.md`
- `wiki/raw-data/llm-training/papers/prorl.md`

---

## Why this synthesis matters

Process supervision and verifiable rewards are often presented as alternatives. They are not — they are orthogonal mitigations for two different Goodhart surfaces in classical RLHF:

| Surface | Mitigation | Mechanism |
|---------|------------|-----------|
| Learned RM drifts OOD under optimisation | RLVR (verifier replaces RM) | No learned proxy; proxy = target. |
| Reward landed at trajectory granularity — credit cannot localise a bad step | PRM (step-level scoring) | Decompose reward along the reasoning chain. |

When both surfaces matter (long chains *and* no ground-truth checker), you need both — a PRM trained on preferences, not on MC rollouts. When neither matters (short outputs, ground-truth checker), RLVR alone is fine. This excerpt crystallises the decision-tree.

---

## Signal density vs cost — the one comparison the chapter's table is built from

| Method | Reward density | Per-example label cost | Goodhart surface |
|--------|----------------|-------------------------|-------------------|
| Preference RM + PPO (ch-41) | trajectory-level, learned | 0 at training time (labels are offline) | RM drift (largest) |
| PRM800K + Best-of-N | step-level, human-calibrated | ~10x outcome label cost | PRM drift (medium) |
| Math-Shepherd PRM + step-PPO | step-level, MC-based | `K` rollouts x `L` steps per trajectory | PRM drift (medium) |
| OmegaPRM + step-PPO | step-level, MC-based | `K * log L` rollouts per trajectory | PRM drift (medium) |
| RLVR (Tülu-3 PPO) | trajectory-level, deterministic | 0 + verifier-engineering amortised | Verifier bugs (smallest) |
| SWE-RL (GRPO + difflib) | trajectory-level, deterministic | 0 + rule amortised | Rule bugs (smallest) |

Four rows share "step-level"; two rows share "deterministic verifier." The step-level rows pay in compute (rollouts) or labels (humans) to get denser signal; the deterministic-verifier rows trade signal density for zero Goodhart gap.

---

## When density matters — long chains

For a 15-step MATH proof that fails at step 14:

- **RLVR** sees one scalar: `r = 0`. The gradient on the first 13 correct steps is zero (they are part of a trajectory that was rewarded 0). The policy cannot distinguish "correct first 13 steps, bad step 14" from "bad first step, the rest was wrong-by-consequence."
- **PRM + step-PPO** (Math-Shepherd / OmegaPRM) sees 13 positive PRM scores, 1 negative, 1 final 0. The policy gets gradient for keeping steps 1-13 as they were, for replacing step 14, and for getting the final answer right.

This is the mechanistic argument for process supervision on long chains. The empirical confirmation is Math-Shepherd's MATH numbers (28.6 -> 43.5) and OmegaPRM's MATH lift (51.0 -> 69.4) — in both cases on tasks where RLVR alone would leave the intermediate steps uncredited.

---

## When Goodhart matters — everything else

From `rlvr-tulu3.md`:

> **Why it sidesteps reward hacking:** the verifier is a fixed, interpretable function. There is no proxy RM to drift; there is no OOD region where the reward spuriously rises. Goodhart's gap (see **[[reward-model-overoptimization]]**) is mechanically zero on verifiable prompts.

A PRM is still a learned proxy. A 10-step MATH solution with a good-looking bad step can have a perfectly high PRM score and a wrong final answer; the PRM does not catch this because it was trained on rollouts whose gold-check weakly correlates with what the PRM learned. RLVR has no such failure mode because there is no learned component between the policy and the reward.

---

## The calibration — pass@k

From `rlvr-beyond-base-model.md`:

> RLVR improves pass@1, base model wins at high `k` ... RLVR mostly redistributes probability mass toward already-existing successful paths, while narrowing exploration and reducing the broader coverage of solvable problems.

And the counter, from `prorl.md`:

> ... sufficiently long RL training, paired with KL divergence control, reference-policy resetting, and diverse tasks, can reveal reasoning strategies inaccessible to the base model even under extensive sampling.

The right reading: RLVR at short horizons with weak KL control sharpens priors; at long horizons with good KL control (ch-43) and task diversity it can expand the boundary. Both claims are consistent with the PRM picture — PRMs also sharpen rather than extend in the short horizon, but can add new local gradients that surface rare-but-correct intermediate strategies.

---

## The chapter's decision tree (distilled)

```
Does the prompt have a programmatic ground-truth check?
  - Yes, short output (math/code/IFEval)              -> RLVR (ch-44 §6, Tulu-3 config)
  - Yes, but reasoning chain is long (deep MATH)      -> RLVR + PRM shaping (Math-Shepherd / OmegaPRM)
  - No check, but a reference output exists (patches) -> rule-based similarity reward (SWE-RL)
  - No check, no reference                            -> preference RM (ch-41) or synthetic judge
```

In all four branches, the KL-control machinery of ch-43 still applies. None of these methods replace KL control; they replace the reward side of the `(reward, KL)` tradeoff.

---

## Carry into ch-44 and ch-46

- §8 of read.md collapses this excerpt's decision-tree into the taxonomy table.
- Ch-46 lab offers "DPO option" vs "RLVR option"; this excerpt tells the learner why the lab does not also offer a "PRM option" — PRM training is its own pipeline, not a drop-in replacement for RLVR at lab scale.
- Ch-45 opens with self-improvement loops; the natural bootstrap is "use the current policy as the Math-Shepherd / OmegaPRM rollout policy to generate the next PRM's labels." That only works if the rollout policy improves faster than the labels calcify — the exact dynamics ch-45 will unpack.

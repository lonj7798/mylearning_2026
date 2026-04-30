---
chapter: ch-44
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/math-shepherd.md
source_url: https://arxiv.org/abs/2312.08935
created_at: "2026-04-23"
---

# Excerpt: Math-Shepherd — MC rollouts as automatic step labels

**Source library:** `wiki/raw-data/llm-training/papers/math-shepherd.md`
**Anchor paper:** Wang et al. 2023 — "Math-Shepherd: Verify and Reinforce LLMs Step-by-Step"

---

## Why this source anchors ch-44

PRM800K needs humans. Math-Shepherd does not. This paper is the moment the process-supervision line went from "use it if you can afford it" to "use it by default." The contribution is a single-line label formula plus the PPO reward that uses the resulting PRM as a dense per-step signal.

---

## The label formula — verbatim

From `math-shepherd.md` §Key Contributions:

> **Auto-labeling algorithm:** for every step `s_t` in a rollout, sample K completions and compute:
>   - `y_hard(s_t) = 1` iff at least one of K completions reaches the correct final answer, else 0.
>   - `y_soft(s_t) = (# correct completions) / K`.

Formally:

```
MC(s_t) = (1/K) * sum_{i=1..K} I[rollout(policy | s_1..s_t) reaches gold]
```

This is the exact formula from ch-44 §4. Three design choices to note:

1. **`policy` is typically a *stronger* generator than the one being trained.** The paper uses DeepSeekMath-7B to label for Mistral-7B. Using a weaker policy as the labeler produces noisy labels because many prefixes recoverable by a good policy look unrecoverable to a bad one.
2. **`K` is 8 or 16** in the paper. OmegaPRM (next excerpt) argues K >= 16 is needed for deep trajectories.
3. **Soft beats hard on MATH.** The paper's Table 5 (cited in ch-44) shows soft labels have lower variance on the `p_correct` estimate at deep steps, which matters more the longer the chain gets. On GSM8K the two are comparable because chains are shorter.

---

## PPO reward composition

From `math-shepherd.md` §Technical Details:

> **PPO reward composition:**
> `R_total = r_final + λ · sum_{t ∈ steps} PRM(step_t)`
> with λ ≈ 0.1–1.0; the final answer correctness reward is still included.

Two things this does that pure RLVR does not:

1. **Dense intermediate credit.** On a 15-step solution that fails at step 14, RLVR gives a single 0 reward; Math-Shepherd gives 13 positive `PRM(step_t)` increments plus one negative `PRM(step_14)`. The gradient signal per rollout is `O(L)` rather than `O(1)`.
2. **Partial-credit rollouts.** A trajectory that fails at step 14 still contributes positive signal on steps 1-13. In RLVR that same rollout is wasted.

`lambda` trades off the final reward against the shaped reward. Too small (~0.01) and you recover pure RLVR; too large (~5.0) and the PRM's calibration errors dominate and the policy learns to produce PRM-pleasing steps that do not compose (a process-level Goodhart, i.e., reward hacking against the PRM rather than against an outcome RM).

---

## Results — verbatim

> Mistral-7B GSM8K: 77.9 -> 84.1 (+PPO), 89.1 (+PRM verify).
> Mistral-7B MATH: 28.6 -> 33.0 (+PPO), 43.5 (+PRM verify).

The asymmetry `verify > PPO` is the interesting empirical fact. A PRM used only at inference beats a PRM used as dense reward during training. Three possible explanations the paper floats:

- The PRM has calibration errors that compound under optimization pressure but not under selection.
- Best-of-N at inference averages over many samples; PPO commits to a single trajectory per rollout.
- PRM-as-reward introduces a new Goodhart surface (the PRM itself), which Best-of-N does not because the policy is frozen.

For ch-46 lab framing, the operational rule is: use the PRM as a verifier first, then consider stepwise reward only if the verify-only number has plateaued.

---

## Aggregation for Best-of-N

From `math-shepherd.md` §Technical Details:

> **Verification at inference:** sample N = 256 or 1024 chains, aggregate step scores with `min`, pick max.

This is where `min` enters the chapter. Math-Shepherd's Table 4 shows `min` > `mean` > `prod` on both benchmarks. The intuition is that for composed reasoning, the solution is as strong as its weakest step; `prod` over-penalises long chains (many near-1.0 probabilities multiply to something small); `mean` lets a great step rescue a broken one.

---

## Risks — verbatim

> Still susceptible to proxy misalignment (**[[reward-model-overoptimization]]**) but less so than preference RMs because labels are grounded in end-to-end correctness.

The PRM's labels trace back to gold answers via MC rollouts, not to preferences. The Goodhart surface is smaller but nonzero: a step can have high `MC(s_t)` while being spurious (a lucky prefix that the rollout policy recovers from), and an honest step can have low `MC(s_t)` because the rollout policy is weaker than the gold chain it was trying to complete.

---

## Carry into ch-44

- Formula for §4 of read.md — the literal MC definition.
- Hyperparameter defaults: K = 8 or 16, `lambda` ~ 0.1-1.0.
- The `verify > PPO` asymmetry is referenced in §4 and flagged for ch-46.
- `min` aggregation is the default the chapter recommends in §2 (PRM800K) even though Lightman's paper uses `prod` — Math-Shepherd's ablation is why.

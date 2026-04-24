---
chapter: ch-19
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/wizardmath.md
source_url: https://arxiv.org/abs/2308.09583
created_at: "2026-04-23"
---

# Excerpt: WizardMath — Bidirectional Math Evol-Instruct + RLEIF

**Source library:** `wiki/raw-data/llm-training/papers/wizardmath.md`
**Heritage:** Luo et al. 2023 (Microsoft + Tsinghua). The math specialization of [[excerpts/evol-instruct]]. Introduces *bidirectional* evolution (downward + upward) and the first joint IRM × PRM RL objective — the latter becoming the template for later math-RL recipes, even after RLEIF itself was superseded.

---

## Why this source anchors ch-19

Ch-19 §4 claims domain specialization matters, and WizardMath is the counterintuitive half of that claim. WizardCoder is intuitive — code has obvious axes (edge cases, libraries) that generic Evol-Instruct misses. WizardMath is less obvious: why would a model learn math better from *easier* problems (downward evolution) mixed with hard ones? The paper's answer is structural — a broader difficulty spectrum smooths the reasoning manifold, teaching the model to recognize difficulty before committing to a solution path.

---

## The bidirectional operator design — verbatim

From the source:

> - **Downward evolution operators** (per Math Evol-Instruct): *reduce constraints*, *replace concepts with simpler ones*, *shorten the chain*, *make arithmetic easier*.
> - **Upward evolution operators**: *add constraints*, *compose with another concept*, *increase reasoning depth*, *require multiple solution steps*.

The upward operators mirror generic Evol-Instruct's In-Depth operators, specialized for math. The downward operators are the novel contribution — and the paper justifies them with a concrete argument:

**Why downward helps.** A model trained only on MATH-level problems learns to *always* attempt deep-reasoning chains, even for problems solvable in one step. This produces over-elaborate solutions on easy problems and paradoxically more errors on the easy split (because the model introduces unnecessary intermediate steps). Training on downward-evolved easy problems teaches the model to *stop early* when the problem admits a one-line solution.

The ablation in the paper confirms this: SFT on upward-only beats vanilla by +5 points on MATH but *loses* 3 points on GSM8K-easy. SFT on downward-only is the inverse. Bidirectional wins on both.

---

## The answer-verifier — the step no open pipeline can skip

The source:

> **Filtering/rescoring:** answer-verifier (exact match for GSM8K-style; symbolic equivalence for MATH) rejects incorrect solutions; duplicate instruction filter.

The answer-verifier is the single most-copied component of WizardMath. Every post-WizardMath math-SFT pipeline uses some version of it:

- **GSM8K-style exact match.** The final-answer number is compared literally. Works because GSM8K answers are integers or simple decimals.
- **MATH-style symbolic equivalence.** Uses SymPy or a symbolic solver to check `sympy.simplify(answer - generated) == 0`. Handles algebraic equivalence ("x²+2x+1" ≡ "(x+1)²").

Without the verifier, synthetic math data is catastrophic — the teacher produces plausible-looking but arithmetically wrong solutions at ~15–20% rate, and SFT on wrong solutions actively degrades reasoning. The verifier is what makes synthetic math tractable.

This is the ancestor of the RLVR (reinforcement learning with verifiable rewards) template that ch-26 will cover: once you have a verifier, the gap between SFT data and RL reward is narrow — both use the verifier as the quality signal.

---

## RLEIF — the IRM × PRM objective

The source:

> **RLEIF step:**
> - **IRM** — trained on pairs of evolved instructions to score instruction quality/evolution success.
> - **PRM** — trained on step-level labels (similar to [[prm800k]] lineage) to score partial-solution correctness.
> - PPO objective: maximize `IRM(instruction, response) × PRM(response_steps)` with KL penalty to SFT reference.

Two reward models, multiplied. Why not summed? The paper's argument: the IRM and PRM measure different things, and failure on either should dominate the overall reward. A solution with a perfect PRM score but poor instruction-following (IRM low) should not be rewarded; same for the inverse. Multiplicative composition enforces AND-gating — both must be high.

The +3-point GSM8K / +2-point MATH gain from RLEIF over SFT-only is modest compared to the SFT gains from bidirectional evolution. The RLEIF template matters more than the numerical gain — the decomposition into (instruction-quality reward × process reward) reappears in many later math-reasoning recipes, often with the RMs swapped for other signals.

---

## The 70B result — context and caveats

The source:

> WizardMath-70B: GSM8K ~81.6 / MATH ~22.7 at release — above GPT-3.5-Turbo, Claude 2.

At August 2023, this was a striking result: an open 70B beat GPT-3.5 on GSM8K by ~5 points and matched Claude 2 on MATH. The context: the 70B base is LLaMA-2-70B, which on its own scores ~56 on GSM8K. WizardMath lifts this +25 points.

The MATH score of 22.7% looks low by 2025 standards (current frontier models clear 70%). It was SOTA for open 70B at release. The gap closed partly because subsequent work (DeepSeek-Math, Qwen-Math) built on the WizardMath template with stronger verifiers and larger synthetic math corpora.

---

## Why RLEIF was superseded — and what survived

The paper's RLEIF is rarely run today. Two successors replaced it:

- **RLVR (verifiable rewards).** Drops the IRM entirely; uses only the answer-verifier as reward. Simpler, and the IRM turned out to be noisy enough that dropping it improved results.
- **GRPO-style process rewards.** Replaces PRM with group-relative advantages across multiple sampled solutions. Avoids PRM's noisy step-level labeling.

What survived from WizardMath:
- Bidirectional evolution is still the standard for synthetic math difficulty control.
- The answer-verifier is the single non-negotiable component of any math-SFT pipeline.
- The multiplicative reward decomposition reappears in modern pipelines as (correctness × format) or (correctness × length-bonus).

---

## The risks the source flags

The source:

> - **Reward hacking**: PRM signal is noisy; overtraining on PRM can produce surface-correct chains with wrong answers.
> - **Benchmark saturation** since release — newer 7B reasoning models now exceed WizardMath-70B.
> - **License:** WizardLM-family data has had access restrictions; check before redistribution.
> - **Process-reward labeling scale**: PRM data is expensive; WizardMath uses model-labeled step correctness, inheriting the teacher's error modes.

The reward-hacking risk is the operational one. A model trained to maximize PRM can learn to produce chains that *look* step-wise correct (each step references the previous, uses math notation, concludes with "therefore") while the actual arithmetic is wrong. The answer-verifier catches this at training time only if it's run on the final answer — which it is in WizardMath, but not in every derivative.

---

## Connections

- [[excerpts/evol-instruct]] — the generic framework WizardMath specializes bidirectionally.
- [[excerpts/wizardcoder]] — sibling code specialization.
- [[excerpts/self-instruct]] — the Self-Instruct → Evol-Instruct → WizardMath chain.
- [[ch-19]] — this excerpt is the foundation of §4's math half and supports the WizardMath row in §9's comparison table.

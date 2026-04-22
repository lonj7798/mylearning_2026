<!-- scope: reasoning-trace synthesis — self-evolving MCTS + PPM loop to build 747K verified traces
     deps: [[rstar]], [[math-shepherd]]
     see-also: [[omegaprm]], [[deepseek-r1]], [[openmathinstruct-2]]
-->

# rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking
- **Core Insight:** A 7B base can reach o1-preview-level math by iterating an MCTS-based self-evolution loop that (a) produces step-level verified reasoning traces via code-augmented MCTS, (b) trains a process preference model (PPM) on the resulting pairwise step preferences, and (c) retrains the generator on only PPM-top solutions — all without a stronger external teacher.
- **Guideline:** For frontier small-model reasoning, couple MCTS rollouts with a code executor for step-level correctness, train a PPM on step-preference pairs extracted from the tree, and iterate generator/PPM for multiple rounds; each round compounds the previous.
- **Authors:** Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, Mao Yang (Microsoft Research Asia)
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2501.04519
- **Relevant topics:** reasoning, MCTS, PPM, self-evolution, code-augmented CoT, small-model frontier

## Abstract
rStar-Math extends rStar into a four-round self-evolution loop that produces 747K training trajectories with step-level correctness labels. Key innovations: (1) code-augmented CoT — every reasoning step produces both natural-language and executable Python, rejecting steps whose code raises errors; (2) per-step Q-values derived from MCTS rollouts train a Process Preference Model (PPM) that scores intermediate steps; (3) four rounds of policy/PPM co-training. Applied to Qwen2.5-Math-7B, rStar-Math hits **90.0% MATH** and **53.3% AIME24**, matching or exceeding o1-preview on several benchmarks.

## Key Contributions
- **Code-augmented MCTS** — step-level code execution as a correctness signal (no gold-answer leakage at step level).
- **Process Preference Model (PPM)** — trained on pairwise step preferences from MCTS Q-values rather than regressed scalar scores.
- **Four-round self-evolution**: each round's top policy samples new trajectories, PPM retrains, generator retrains.
- **747K trajectory dataset** covering 747K math problems after augmentation.
- **Qwen2.5-Math-7B-rStarMath**: 90.0 MATH, 53.3 AIME24, 71.0 USAMO problems attempted.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** ~747K math problems drawn from NuminaMath, MATH, GSM8K, Olympiad, AIME archives.
- **Round-0 generator:** Qwen2.5-Math-7B-Instruct (or DeepSeek-Math-7B) as bootstrap policy.
- **MCTS step construction:** each MCTS node = one reasoning step, containing `(natural-language thought, Python code block)`. Code is executed; execution failure ⇒ node pruned. Step reward uses terminal-only signal (final answer correctness against gold) propagated back via MCTS Q-values.
- **Trajectory filter:**
  - Executable: every step's Python runs without exception.
  - Correct: final answer exact-matches gold.
  - Q-value confidence: MCTS visit-count ≥ threshold.
- **PPM training data:** within each problem, pairs of sibling MCTS steps with high vs low Q-value form step-preference pairs. PPM trained with pairwise ranking loss.
- **Self-evolution loop:** 4 rounds; in each round, the top-K trajectories (by PPM score) train the next generator; PPM retrains on the new MCTS trees.
- **Output shape:** 747K fully-verified trajectories; ~4–10 reasoning steps each; code block per step; ~400–1200 tokens per trace.
- **Teacher model(s):** none external; self-evolution only.
- **Cost / compute:** not fully disclosed; on the order of 100K GPU-hours across four rounds on MSRA clusters.

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** 400–1200 tokens; structured into 4–10 labeled steps, each with code.
- **Trace style:** code-augmented CoT — like OpenMathInstruct TIR but with MCTS-curated step-level correctness.
- **Correctness verifier:** step-level = Python execution success; trajectory-level = gold answer exact match; preference = MCTS Q-value ordering.
- **PPM loss:** pairwise Bradley-Terry on step preferences (step_high, step_low) drawn from MCTS siblings with Q-gap > δ.
- **Error-mode filter:** a step is discarded if its code raises an exception OR if its Q-value falls below round-specific threshold.
- **PPM is not a scalar PRM:** authors argue pairwise training avoids the Goodhart-style issues of scalar reward regression observed in [[math-shepherd]] / [[prm800k]].

## Quality / diversity evaluation
- Qwen2.5-Math-7B-rStarMath: **90.0 MATH, 53.3 AIME24, 58.5 Olympiad**.
- Beats o1-mini on MATH; matches o1-preview on several.
- Round-by-round: MATH improves 58 → 78 → 85 → 88 → 90 across four rounds.
- PPM ablation: replacing PPM with a scalar PRM loses 6 MATH points.

## Risks + gotchas
- **Gold-answer dependence:** final-answer correctness is still ground-truth — the trick is step-level verification, not zero-supervision.
- **Problem pool ceiling:** the 747K problems must contain gold answers; expansion requires human-labeled problems.
- **Compounding distribution narrowing:** four self-evolution rounds risk collapsing to a small region of the solution space; authors mitigate with temperature scheduling.

## Connections
- Direct successor of [[rstar]].
- PPM contrasts scalar PRMs: [[math-shepherd]], [[prm800k]], [[lets-verify]].
- Tree-search synthesis sibling: [[omegaprm]] (Monte-Carlo, no MCTS UCB).
- Code-augmented CoT lineage: [[openmathinstruct]], [[mammoth]].

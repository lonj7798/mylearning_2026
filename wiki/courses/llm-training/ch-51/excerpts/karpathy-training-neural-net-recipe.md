---
chapter: ch-51
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/karpathy-training-neural-net-recipe.md
source_url: https://karpathy.github.io/2019/04/25/recipe/
created_at: "2026-04-23"
---

# Excerpt: Karpathy's recipe — "predict-before-run" as the backbone of the go/no-go memo

**Source library:** `wiki/raw-data/llm-training/blogs/karpathy-training-neural-net-recipe.md`
**Artifact:** training fails silently; predict-outcome-before-run; review worst 10; six-step discipline.

---

## Why this source is ch-51's "discipline anchor"

Ch-51 is a statistical chapter. But the bootstrap math doesn't save you from the failure mode Karpathy names: training (and eval) fail silently. The numbers come out; they look fine; you ship; two rounds later a real regression surfaces that was already visible in the raw data and was masked by a rolling average you set by default.

Source §Core Insight:

> Neural network training is a leaky abstraction that fails silently; the only defense is an obsessive, data-first, incremental workflow where every step is verified against an explicit prediction.

Ch-51's memo template §6 operationalizes this. The "claim" field is a prediction; the "evidence" field is the verification; the "regressions considered and dismissed" field is the pre-commitment to checking where you *expect* not to see a problem (which is where problems hide).

---

## The single rule ch-51 treats as non-negotiable

Source §Technical Details / §5 Tune:

> Predict-outcome-before-run.

Applied to evaluation: before any eval run, write down the expected effect size, the expected CI halfwidth, and the ordering of outcomes you anticipate across slices. File this as `noise_budget.txt` (ch-51 §2). Then run the eval. Surprises — cells that contradict the prediction — are the *only* thing that teaches anything. Ch-51 §6 memo's §4 "One surprise" slot is reserved for exactly this.

---

## "Review the 10 worst validation examples" = per-item audit

Source §Technical Details / Step 6 Squeeze:

> Review the 10 worst validation examples — they reveal systematic errors.

Bootstrap aggregates individual items; it does not tell you *which* items dropped. Ch-51's guideline inherits this: after any regression that the paired bootstrap flags, open the per-item deltas, sort by |s_A - s_B|, and inspect the top-20 negative items. These are where the regression *actually* lives. If they're all one slice (e.g., multi-turn dialogs), the paired-bootstrap p-value was accurate and the regression is real; if they're scattered and idiosyncratic, judge noise or decode noise is plausible — rerun with more seeds.

---

## "Be paranoid about .train() vs .eval()"

Source §Famous Heuristics:

> Be paranoid about `model.train()` vs `model.eval()`.

This is a variance source ch-51 §1 lists explicitly (dropout / stochastic layers during eval, σ 0.2–1.0 pp if forgotten). In 2025 LLM inference stacks (vLLM, SGLang), dropout is typically off by construction, but custom eval harnesses that load HuggingFace models for scoring can re-expose this bug. The assertion "no dropout active at eval" is a 3-line check that prevents a silent 1-pp variance contribution.

---

## "Don't be a hero" — apply the simplest known-good bootstrap

Source §Key Contributions:

> "Don't be a hero" — use the simplest known-good architecture.

Applied to CIs: percentile bootstrap first, BCa only when θ̂ is near the 0/1 boundary, parametric bootstrap never. Novel estimators go in follow-up work, not in the go/no-go memo. Ch-51 §3's "use percentile when θ̂ ∈ [0.2, 0.8] and N ≥ 200; BCa when θ̂ is near the boundary or N < 200" is the Karpathy rule: start with the simplest thing that works, escalate only when it demonstrably fails.

---

## Connections

- **[[gradient-clipping]]** — monitor-and-clip ideology extended to eval metrics: monitor CI halfwidth and abort eval runs where it exceeds the claimed effect size by 2×.
- **[[adam]]** — "3e-4 safe default" analogous to "B=10000, percentile bootstrap, N=500" as ch-51's default.
- **Post-training relevance (2025):** the same silent-failure list applies to DPO / PPO / RLVR evals — wrong tokenizer, wrong prompt template, judge prompt drift.
- **ch-51 §2, §6, §7** — the noise budget, memo template, and failure-mode list are direct translations of this essay's discipline.

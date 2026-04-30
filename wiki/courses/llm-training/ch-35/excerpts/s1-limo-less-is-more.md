---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/s1.md + wiki/raw-data/llm-training/papers/limo.md
source_urls: https://arxiv.org/abs/2501.19393 ; https://arxiv.org/abs/2502.03387
created_at: "2026-04-23"
---

# Excerpt: s1 + LIMO — the "less is more" datapoint

**Source libraries:** `wiki/raw-data/llm-training/papers/s1.md`, `wiki/raw-data/llm-training/papers/limo.md`
**Papers:** Muennighoff et al., *s1: Simple test-time scaling* (2025); Ye et al., *LIMO: Less is More for Reasoning* (2025).

---

## Why this source anchors ch-35

s1 and LIMO are the pair of 2025 papers that establish the lower-bound of distillation-SFT trace count. Both are on Qwen2.5-32B-Instruct bases; both hand-curate; both beat mass-distillation recipes on AIME. From the s1 source:

> **Core Insight:** A very small but carefully curated reasoning SFT set can teach a strong base model how to spend more test-time compute; the big gain comes from inference-time control, not from a large training corpus.

And LIMO:

> **Core Insight:** A tiny set of carefully curated long-CoT examples can unlock strong math reasoning in a capable pretrained base; the bottleneck is not volume, but whether the traces provide the right cognitive template.

Ch-35 §5 uses the s1/LIMO row of the comparison table as the chapter's "extreme low-data" datapoint. This excerpt extracts what the two papers say about *why* 1000 traces beat 17,000 on a reasoning-rich base.

---

## s1 — 1,000 curated + budget-forcing

From the s1 source:

> - **s1K dataset:** 1,000 curated question-trace pairs selected from a much larger candidate pool.
> - **Budget forcing:** an inference-time controller that suppresses early stopping by appending `"Wait"` when the model tries to end thinking.
> - **Candidate pool:** the repo exposes a larger 59K question pool and the scripts used to recreate s1K.
> - **Curation criteria:** difficulty, diversity, and quality, each validated through ablations in the paper.

So the final 1K is the survivor of a 59K -> 1K hand + automated cull. Three axes:

1. **Difficulty** — keep problems that strong baselines get wrong, so the student has headroom to improve.
2. **Diversity** — cover source + topic breadth, not just one benchmark.
3. **Quality** — filter for trace correctness and reflective structure.

The s1 training is minimalist: **26 minutes on 16 H100** with PyTorch FSDP. Total training cost is ~$50 at rental rates.

### Budget forcing — the other half of the result

From the s1 source:

> - **Inference control:** budget forcing is implemented by ignoring the end-of-thinking stop and repeatedly appending `"Wait"`.
> - **Scaling behavior:** the repo shows that forcing more thinking can extend performance beyond the native checkpoint behavior, up to the context limit.

This is the test-time scaling that gives s1 its name. On the same checkpoint, forcing more thinking lifts AIME24 from ~50% to ~57%. Same weights, different inference protocol.

Crucially from the "Risks + Gotchas":

> **Inference hack, not new capability:** budget forcing mostly reallocates compute already latent in the model.

Budget-forcing is a test-time policy, not a training improvement. The 1K-trace SFT unlocks the *ability* to spend more compute; budget-forcing is the *scheduling*. Both are needed.

### Scores

> s1-32B gets 56.7 AIME24, 93.0 MATH500, and 59.6 GPQA-Diamond in the model card table.

AIME24 56.7% with 1K traces beats Sky-T1's 43.3% with 17K traces. The gap is attributed to curation quality, not scale.

---

## LIMO — 817 traces and the activation hypothesis

From the LIMO source:

> The current reported version reaches **63.3% AIME24** and **95.6% MATH500**, while also showing strong out-of-distribution gains.

817 hand-curated traces. AIME24 63.3% — beats Bespoke-Stratos's 17K (~63%) on Qwen2.5-32B-Instruct.

### The "Less-Is-More Reasoning Hypothesis"

From the LIMO source:

> Formalizes the **Less-Is-More Reasoning Hypothesis**: strong latent knowledge from pretraining plus high-quality demonstrations are the two prerequisites.

Two ingredients, not one:

1. **Latent pretrain knowledge.** The base must already contain the domain-relevant facts and reasoning primitives. A weak base (e.g., Qwen2.5-0.5B or a non-math-pretrained model) will not activate from 817 traces.
2. **High-quality demos.** The traces must show *reflective structure* — self-verification, branching, backtracking markers — not just correct final answers.

The first ingredient is the ceiling. LIMO does *not* claim 817 traces work on any base; it claims they work on bases that already have the latent capacity. This is why LIMO and s1 both use Qwen2.5-32B-Instruct — a base already strong at math.

### Quality scoring attested

> - Correctness of the final answer against the gold solution.
> - Presence of self-verification and re-checking segments.
> - Branching or backtracking markers that expose non-linear reasoning.
> - Fine-grained step granularity instead of outline-only answers.
> - Manual curation: hand-filter down to the final small set, removing lucky guesses and traces with subtly broken intermediate logic.

Four automated signals plus one human pass. The human pass removes the "right-answer-wrong-reasoning" traces that automated SymPy filters miss — the same failure mode that OpenR1 flags as the Math-Verify blindspot (ch-20 §5.5).

---

## What this pair means for ch-35

Three non-obvious conclusions:

**1. Trace count and benchmark are not monotone — past a critical quality threshold.** Below the threshold, more traces help. Above it, more traces *dilute*. s1 and LIMO both sit above; Sky-T1 and OpenR1 sit below. Bespoke-Stratos sits at the edge.

**2. Base-model capacity is the ceiling.** Both s1 and LIMO work on Qwen2.5-32B-Instruct. On a 7B base (even Qwen2.5-7B-Instruct) the 1K-trace recipe drops hard; OpenR1's 440K-trace recipe is more competitive at 7B. Small-data recipes scale *up* in base size, not down.

**3. Budget forcing reveals the inference-time capacity ceiling.** s1's +7 AIME from inference-time forcing says the *trained* model has more latent capability than it emits by default. This is consistent with the Qwen 3 thinking-budget mode and Nemotron-Ultra's reasoning-budget-control (ch-34 §3, ch-35 §2) — 2025 converges on *trained-ability + inference-time-scheduler* as two orthogonal knobs.

For the ch-35 decision tree (§6), s1/LIMO occupy the **reasoning-rich-base + strong-curators** quadrant. Reaching the same quality via Bespoke-Stratos-style pipelines costs ~10× more and needs R1 API access. If you have the curators, you save the money.

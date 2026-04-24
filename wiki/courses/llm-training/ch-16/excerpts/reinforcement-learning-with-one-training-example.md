---
chapter: ch-16
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/reinforcement-learning-with-one-training-example.md
source_url: https://arxiv.org/abs/2504.20571
created_at: "2026-04-23"
---

# Excerpt: One-shot RLVR — historical-variance ranking as the limit case of the pass-rate filter

**Source library:** `wiki/raw-data/llm-training/papers/reinforcement-learning-with-one-training-example.md`
**Paper:** Wang, Yang, Zeng, Ren, Liu, Peng, Cheng, He, Wang, Gao, Chen, Wang, Du, Shen (2025), "Reinforcement Learning for Reasoning in Large Language Models with One Training Example."

---

## Why this source anchors ch-16

This paper is the extreme edge of ch-16's thesis. If §2's claim is "filter by pass-rate, variance-weight the survivors," this paper shows that at the limit, *one* well-chosen prompt is enough to lift a base model on MATH500 from 36.0% to 73.6%. That is not a typo; it is the actual measured effect on Qwen2.5-Math-1.5B. The paper matters less for its specific recipe than for what it reveals about where the gradient signal lives in RLVR — and confirms that the signal lives in high-variance examples, not in many-examples-averaged.

Ch-16 cites this paper at §2 as the motivating extreme for the historical-variance ranking heuristic.

---

## The headline result

From the source (Abstract):

> On Qwen2.5-Math-1.5B, a single selected example lifts MATH500 from 36.0% to 73.6% and raises average performance across six math benchmarks from 17.6% to 35.7%, matching the reported 1.2k-example DeepScaleR subset.

The magnitude of this effect is almost implausible, so it's worth checking what it means carefully:

- The training *example* is one prompt-answer pair.
- The training *data* is that pair duplicated to batch size, with RL rollouts sampled fresh every step.
- The optimizer sees full batches; the semantic dataset is a single prompt.
- Gains on MATH500 come from **generalization**, not memorization of the training prompt.

So the paper is not saying "one gradient step lifts MATH500 by 37.6 points." It is saying "the RLVR update direction, computed against one well-chosen prompt's rollouts, aligned with a useful reasoning-improvement direction for a broad benchmark distribution." That is a much more surprising claim, and it's what ch-16 §2 builds on.

---

## Historical-variance ranking as a prompt-selection heuristic

From the source (Key Contributions and Technical Details):

> Introduces a simple selection heuristic based on **historical variance score** to pick the single example that works best.
>
> The best one-shot example is selected by a **historical variance score** computed from training accuracy across epochs on the full dataset. High-variance examples are more informative for RLVR because they expose the model to reward-sensitive decision boundaries.

This is the key mechanism. Ch-16 §2 describes pass-rate as `p̂(x)` — the policy's mean probability of solving `x` at the current temperature. Historical-variance ranking generalizes this to **time-variation of `p̂(x)`** across training epochs: a prompt whose `p̂` fluctuates across checkpoints is one whose gradient signal is currently large.

Why it matters:

- A prompt with stable `p̂ = 0.5` produces consistent gradient signal on each visit but no information about which training epochs are the important ones.
- A prompt with `p̂` that swings from 0.3 to 0.7 across epochs produces gradient signal only in the swing epochs — so the paper's heuristic effectively identifies "prompts that are about to be learned" as the highest-signal examples.

The chapter's §2 includes this as the "derivative form" of the pass-rate filter: variance-over-time, not variance-at-current-step.

---

## Entropy matters — the separable contribution

From the source (Key Contributions, Reported phenomena):

> Finds that **entropy loss** materially helps exploration and that entropy alone can improve MATH500 even without outcome reward.
>
> **Entropy matters:** exploration collapse hurts; a properly tuned entropy bonus is an important stabilizer.

This result is off-axis for ch-16 (entropy regularization is the RL-algorithm chapter's domain) but it does matter for the chapter's §4 curriculum discussion. Part of why one prompt works is that the entropy term is doing exploration work that an expanded prompt pool would otherwise have to do. In the many-prompts regime, diversity *across* prompts replaces diversity *across* rollouts-of-one-prompt; in the one-prompt regime, the rollout-diversity channel has to carry the full exploration burden, and the entropy term is what enables that.

Implication for ch-16: the `[p_lo, p_hi]` band isn't the only lever on exploration; the entropy coefficient is a parallel lever. In curriculum design, tightening the band without adjusting entropy can produce premature exploration collapse.

---

## Post-saturation generalization — the "what else is going on" signal

From the source (Reported phenomena):

> **Post-saturation generalization:** the model keeps getting better on held-out math problems after it has already memorized the training example.

This is the single strangest observation in the paper and worth taking seriously as a check on ch-16's §2 framing. If training accuracy on the one example has saturated at 100%, then in the chapter's filter logic, the prompt has `p̂ = 1` and its reward variance is 0 — it should contribute no gradient. Yet generalization continues to improve.

Two reconciling interpretations:

1. **Sampling-noise residual.** Even at "saturated" training accuracy, stochastic rollouts occasionally fail; the `p̂ = 1` estimate has measurement noise. Residual variance near the saturation point is small but non-zero.
2. **Entropy-driven exploration.** The entropy bonus keeps the policy exploring even after the outcome reward stops supplying signal; that exploration, projected onto the training loss, appears as continued generalization.

Ch-16 does not claim to resolve this; it notes the phenomenon as a reason why the pass-rate filter is a *necessary* condition for good RL, not a *sufficient* one.

---

## What this excerpt unlocks

- **ch-16 §2** — historical-variance ranking as the derivative form of pass-rate; justifies variance-weighting the replay buffer.
- **ch-16 §4(b)** — "annealing requires re-measurement" is softer than one might think; post-saturation generalization suggests keeping solved prompts in a low-weight tail rather than ejecting them entirely.
- **Track 3 (synthetic)** — the one-shot result says a synthetic-prompt generator that produces even a handful of well-chosen prompts can be enough; quantity is over-emphasized compared to quality.

## Connections

- [[excerpts/rlvr-tulu3]] — the standard RLVR framework the paper specializes to one-prompt.
- [[excerpts/replay-buffer-rlhf]] — the variance-weighted sampling is the replay-buffer analog of one-shot's historical-variance ranking.
- [[ch-16]] — §2, §4(b).

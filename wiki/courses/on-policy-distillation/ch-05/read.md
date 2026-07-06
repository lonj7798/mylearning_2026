<!-- chapter: ch-05
     track: tradeoffs
     kind: content
     title: Economics and Failure Modes — When On-Policy Distillation Wins
     deps: [[ch-04]]
     sources: [[tm-on-policy-distillation]], [[nrehiew-sft-rl-opd]], [[qwen3-strong-to-weak-distillation]]
-->

# Chapter 05 — Economics and Failure Modes: When On-Policy Distillation Wins

> **Core insight.** On-policy distillation's headline claim is not "better scores" — it is "**the same or better scores for ~10× less compute than RL**," because a dense per-token teacher signal ([[ch-04]]) delivers O(N) bits per episode against RL's O(1). But it is a priced bet, not a free lunch: it requires teacher-log-prob access and student-sampling infrastructure, it can collapse (entropy crash, style-token domination), and it is *not* the right tool for every domain. This chapter is the trade-off altitude — what the corner buys, what it costs, and the rule for when to enter it.

> **Guideline.** Reach for on-policy distillation when (a) a stronger teacher already exists for the target skill, (b) the outputs are long (so exposure bias dominates), or (c) you must add capability without forgetting (continual learning / personalization). Avoid or hedge it when the reward is cleaner than any teacher (math/code often favor RL) or when you cannot serve teacher log-probs. Watch entropy; clip per-token.

---

## 1. The compute case (the number everyone quotes)

The Qwen3 technical report is the industrial evidence ([[qwen3-strong-to-weak-distillation]], reproduced in [[tm-on-policy-distillation]]). Distilling math reasoning into Qwen3-8B-Base with Qwen3-32B as teacher, on AIME'24:

| Stage | AIME'24 | GPU-hours |
|---|---|---|
| SFT-400K (off-policy baseline) | ~60% | — |
| + Reinforcement Learning | 67.6% | **17,920** |
| + On-policy distillation | **74.4%** | **1,800** |

On-policy distillation reaches a **higher** score at roughly **one-tenth** the GPU-hours of RL. Two more efficiency results from Thinking Machines:

- **vs SFT data-scaling:** reaching a comparable score by scaling SFT data costs **9–30×** more FLOPs than on-policy distillation (depending on whether the teacher's FLOPs are amortized).
- **Self-distillation** (distilling an RL-trained model back into its own base): reaches teacher performance "approximately 7-10x faster than RL," and "Cumulatively, the reduction in compute required is on the order of **50-100x**," thanks to shorter contexts and smaller batches.

> **Interactive companion:** [`figures/compute-and-collapse.html`](figures/compute-and-collapse.html) — tab 1 plots the three rows above (score vs GPU-hours) with the ~10× multiplier live; tab 2 animates the characteristic failure mode — a sudden reward jump coupled to an entropy collapse — and why per-token clipping guards against it.

---

## 2. Why it is so cheap

The efficiency is not magic; it is the three axes of [[ch-01]] paying off at once:

- **Density.** Every token carries a full teacher distribution — O(N) bits per rollout vs RL's single end-of-episode scalar. More signal per sample means fewer samples.
- **On-policy shortcut.** Thinking Machines: "on-policy distillation does not need to model the intermediate strategies during the curriculum of RL." RL has to *discover* good behavior through sparse reward; distillation is handed the teacher's behavior densely and only has to match it in its own states.
- **Cheaper rollouts.** Self-distillation runs at "shorter context and smaller batch sizes" than RL, compounding the per-step savings.

---

## 3. When it wins

Three regimes where on-policy distillation is the right call:

1. **Long sequences.** Exposure bias scales with the horizon ([[ch-03]]), so the longer the output, the more the on-policy property is worth. This is the regime that matters for multi-turn agents.
2. **Continual learning / personalization.** Mid-training a model on new (e.g. internal) documents degrades instruction-following, and *no data mix fully preserves it* — "there is no weighting which maintains the original performance on IF-eval" ([[tm-on-policy-distillation]]). On-policy distillation **recovers** the lost behavior without erasing the new knowledge: "on-policy distillation recovers nearly full performance on IF-eval without losing any knowledge." It is the cleanest known fix for catastrophic forgetting during specialization.
3. **Cheap strong-teacher transfer.** When a capable teacher already exists (a larger sibling, a frontier model), distilling it on-policy is far cheaper than re-running RL — the Qwen3 strong-to-weak story ([[qwen3-strong-to-weak-distillation]]).

---

## 4. Failure modes and the knobs that tame them

**Entropy collapse.** nrehiew observes that OPD's learning curve differs from RL's: "the reward increase is a lot more sudden and corresponds with a drastic collapse in entropy" ([[nrehiew-sft-rl-opd]]). The student can snap onto a narrow mode. Monitor entropy; a cliff is the warning sign.

**Style tokens dominate the loss.** "style tokens have significantly higher per-token KL divergence while Math tokens have the lowest KL divergence" ([[nrehiew-sft-rl-opd]], and see the [[ch-04]] grading figure). Left unchecked, the gradient is dominated by discourse tokens ("wait", "alright") the student and teacher merely phrase differently — not the task-critical tokens. The fix is **per-token clipping**, so no single high-KL token can dominate the update.

**The teacher matters less than you would think.** A striking result: OPD students *beat both* their SFT and RL teachers on the minimal-code-editing task (Pass@1 0.800 and 0.787 vs SFT-teacher 0.775, RL-teacher 0.792), and "both OPD students converged to similar performance despite different teacher quality." nrehiew's takeaway: "the source of the data (ie. via on-policy sampling) matters a lot while the teacher matters perhaps less than expected" ([[nrehiew-sft-rl-opd]]). Corollary: do not over-invest in a perfect teacher; invest in genuinely on-policy sampling.

---

## 5. When NOT to use it (domain dependence)

On-policy distillation is not universal. nrehiew's domain split ([[nrehiew-sft-rl-opd]]): "Math and Code tasks tend to favor RL," while "creative and knowledge domains benefit more from self-distillation or distillation-style methods." The reason is signal quality: where a *clean, verifiable* reward exists (a unit test, a numeric answer), RL's sparse-but-exact signal can push capability past any teacher; where reward is noisy or subjective (style, tone, open-ended knowledge), a dense teacher signal is more reliable than a hackable reward. And there is a hard prerequisite: you must be able to **serve teacher log-probs** over the student's samples every step — no teacher access, no on-policy distillation.

The modern pipelines reflect this: **Pretrain → SFT → RL/Expert → OPD-merge** ([[nrehiew-sft-rl-opd]]), where RL builds per-domain experts and on-policy distillation *merges* them back cheaply. OPD is a stage, not a religion.

---

## 6. Myth killed: "on-policy distillation is always better"

The compute numbers make it tempting to treat OPD as a strictly dominant upgrade. It is not. It is dominated by RL where a clean reward beats any teacher (math/code with verifiers), it is impossible without teacher-log-prob access, and it can collapse without entropy monitoring and per-token clipping. The First-Law-style discipline holds: name the cost. OPD buys sample efficiency and exposure-bias reduction; it costs teacher serving, sampling infrastructure, and collapse risk. Enter the corner when those costs are worth the exposure-bias cure — not by default.

---

## 7. Price the bet: is the boson pipeline in the winning regime?

Run [[ch-03]]'s and this chapter's tests against `boson-agent-synthetic-data-dev`:

- **Long sequences?** Yes — 20–50-turn calls, deep in the exposure-bias regime. ✅ favors OPD.
- **Teacher available?** Yes — Claude / large Qwen can grade seller turns (cross-family serving is [[ch-06]]). ✅ possible.
- **Clean verifiable reward instead?** Only partly — "did the call close / stay compliant" is a *sparse, noisy, end-of-episode* signal, exactly RL's weakness; there is no unit-test oracle for a good sales turn. So a dense teacher signal is more attractive than a reward model here. ✅ favors OPD over RL.
- **Costs?** Teacher-log-prob serving over Korean, tool-calling seller turns each step; entropy-collapse and style-token risk in a chatty domain (where style tokens are abundant — clip aggressively). ⚠️ real, budget for them.

The verdict the capstone will defend: the boson seller sits in OPD's winning regime — long-horizon, teacher-available, no clean reward — but the *engineering* (which tokens, cross-family teacher, compaction/barge-in handling) is what decides whether the bet actually pays. That engineering is [[ch-06]] and [[ch-07]].

---

## Where This Goes

Chapter 6 turns the mechanism and economics into a runnable recipe: TRL's `GKDTrainer` (the `lmbda`/`beta`/`temperature` knobs from [[ch-04]] as real config), the HuggingFace "any model family" GOLD extension that lets a cross-tokenizer teacher grade the student, and the practical bottleneck of serving teacher log-probs. Then the capstone applies all of it to the boson pipeline.

## Additional Reading

- Qwen Team, "Qwen3 Technical Report" (2025) — https://arxiv.org/abs/2505.09388 ([[qwen3-strong-to-weak-distillation]])
- Thinking Machines, "On-Policy Distillation" — https://thinkingmachines.ai/blog/on-policy-distillation/ ([[tm-on-policy-distillation]])
- nrehiew, "SFT, RL, and On-Policy Distillation Through a Distributional Lens" — https://nrehiew.github.io/blog/sft_rl_opd/ ([[nrehiew-sft-rl-opd]])
- Cui et al., "The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models" (2025) — https://arxiv.org/abs/2505.22617 (entropy-collapse background)

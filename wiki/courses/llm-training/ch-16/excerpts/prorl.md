---
chapter: ch-16
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/prorl.md
source_url: https://arxiv.org/abs/2505.24864
created_at: "2026-04-23"
---

# Excerpt: ProRL — reference-policy resets as curriculum in prompt space

**Source library:** `wiki/raw-data/llm-training/papers/prorl.md`
**Paper:** Liu, Diao, Lu, Hu, Dong, Choi, Kautz, Dong (2025), "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models."

---

## Why this source anchors ch-16

ProRL is ch-16's primary source for the claim that curriculum in prompt space — and specifically **reference-policy resets** — can be as important as any algorithm-side choice. Ch-16 §4 treats reference-policy resets as "curriculum in disguise": the reset step does not just recalibrate the KL penalty, it effectively re-curates the prompt pool by widening the pass-rate distribution.

The paper is also an important foil: it pushes back on the (otherwise reasonable) claim from [[rlvr-beyond-base-model]] and [[spurious-rewards-rlvr]] that RL only sharpens pretrained priors. ProRL's answer is that this is an artifact of *short-horizon* RL, not a fundamental limit.

---

## The recipe

From the source (Key Contributions):

> Introduces the **ProRL** recipe: prolonged RL + KL control + reference policy resetting + task diversity.

Four ingredients; the paper's argument is that all four are necessary. Ch-16 §4 tabulates ProRL as a distinct curriculum row because the "reset + diversify" pattern is qualitatively different from Tülu 3's fixed-band or K1.5's prioritized-`1−p` schedules. In the ProRL schedule:

- Early training: broad task suite at medium difficulty. Pass-rate band is wide.
- Mid-training: gradually narrow to harder-only prompts. Band shifts left.
- Reset: reference policy is re-anchored to the current policy. KL penalty is reset. Effectively, the model's "prior" is updated.
- Post-reset: the pass-rate distribution widens again — prompts that had saturated re-enter the active band because the KL geometry has changed — and broadening resumes.

---

## Why reference-policy resets function as prompt-pool re-curation

From the source (Technical Details):

> **Reference policy resetting** to avoid locking the run to a stale anchor policy.

The mechanism is subtle. In KL-regularized RLHF, the reward stream is `r(x, y) − β · KL(π_θ(·|x) || π_ref(·|x))`. As training proceeds, `π_θ` drifts away from `π_ref`, and the KL penalty grows — making the effective reward for any further drift smaller. Eventually the policy converges on a distribution that trades off "do well on the task" against "stay close to `π_ref`."

Resetting `π_ref` to the current `π_θ` does two things:

1. Zeros out the KL penalty for the current policy state.
2. Changes the geometry of the reward landscape for all *future* rollouts.

The second effect is what makes this a prompt-curation event. A prompt that the current policy had solved with `p̂ = 0.95` was only solved that way because `π_θ` had converged near `π_ref`'s local optimum. After the reset, the policy can move further in directions the old KL penalty forbade — which means previously-solved prompts become mid-difficulty again (the policy no longer exploits the memorized mode), and previously-unsolvable prompts enter the reachable set.

In ch-16's §2 language: **resetting `π_ref` is a non-local re-measurement of the pass-rate distribution.** It's not just that `p̂(x)` changes because `π_θ` moved; it's that `p̂(x)` changes because the *reward surface* moved.

---

## The boundary-expansion claim and its operational reading

From the source (Abstract, Key Contributions):

> ProRL ... argues that sufficiently long RL training, paired with KL divergence control, reference-policy resetting, and diverse tasks, can reveal reasoning strategies inaccessible to the base model even under extensive sampling.
>
> Shows cases where RL models solve problems that the base model cannot solve even with aggressive sampling.

Ch-16 does not take a position on whether this is genuinely novel reasoning or sophisticated prior-sharpening. The chapter's operational reading is narrower: **whatever ProRL is doing, it cannot be done without aggressive prompt-pool re-curation.** A schedule that keeps `π_ref` frozen for the entire run will hit a ceiling set by the initial KL geometry. A schedule that resets `π_ref` periodically — and therefore redraws the pass-rate distribution — can push past that ceiling.

This is the useful takeaway for the chapter even if one is skeptical of the boundary-expansion claim.

---

## The "short-horizon RL may be misleading" argument

From the source (Technical Details):

> The paper does not say all RL does this by default.
>
> It says that **training duration and control strategy matter**, which is a narrower and more useful claim.
>
> Practical implication: many negative conclusions about RL may be conclusions about **short-horizon RL**.

This is methodologically important. If a 2024 paper runs RL for 1k steps and concludes "RL only sharpens priors," that conclusion is specific to the short-horizon regime. ProRL's counter is that the regime where interesting things happen may only be reachable after 10k+ steps with periodic resets and task diversity.

For ch-16's curriculum discussion, the lesson is that **horizon interacts with curriculum design**. A schedule that anneals toward harder-only prompts too fast in a short-horizon run will starve the policy; the same schedule with ProRL-style resets can be productive over a longer run.

---

## What this excerpt unlocks

- **ch-16 §4(c)** — "reference-policy resets are curriculum in disguise" is directly from this source.
- **ch-16 §2** — band-tuning is horizon-dependent; a narrow band works only if the reset mechanism keeps the band reachable.
- **RL track (Track 4)** — ProRL's algorithm-side pieces (KL control, entropy bonus) belong to the RL-algorithms chapter, but the data/prompt-curation piece is fully covered here.

## Connections

- [[excerpts/kimi-k1-5]] — prioritized sampling is a within-stage curriculum tool; ProRL's resets are between-stage tools. They compose.
- [[excerpts/on-off-policy-rlhf]] — the distribution-shift framing explains why resetting `π_ref` produces a fresh coverage region.
- [[ch-16]] — §2 (band tuning is horizon-dependent), §4(c) (resets as re-curation).

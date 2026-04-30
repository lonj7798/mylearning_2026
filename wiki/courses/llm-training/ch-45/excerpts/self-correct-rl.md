---
chapter: ch-45
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/self-correct-rl.md
source_url: https://arxiv.org/abs/2409.12917
created_at: "2026-04-23"
---

# Excerpt: SCoRe — self-correction as a two-stage RL target

**Source library:** `wiki/raw-data/llm-training/papers/self-correct-rl.md`
**Authors:** Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, JD Co-Reyes, Avi Singh, et al. (Google DeepMind)
**Year:** 2024 (arXiv 2409.12917)

---

## Why this source anchors ch-45

Every other method in ch-45 bootstraps a **single-turn** policy. SCoRe bootstraps
the **two-turn** structure `(answer, revise)`. The change of action space is not
cosmetic — it exposes a failure mode (mode collapse onto turn-1 answer) that
single-turn self-training cannot exhibit, and it motivates a reward-shaping
recipe that is structurally different from anything in Self-Rewarding, SPIN,
or ReST-EM. SCoRe is the ch-45 reference for *what a custom-shaped reward
looks like when the goal is not a single-shot output but a behavior*.

---

## The two SFT failure modes — source lines 17-18

> Diagnosed two failure modes of SFT-on-self-correction:
> (1) distribution shift — SFT data drawn from a stronger teacher;
> (2) mode collapse — the model learns to produce the correct answer in turn 1
>     and no-op in turn 2.

Both kill the self-correction behavior you wanted to train. (1) is an
IID-assumption violation; (2) is reward-hacking (the reward rewards `r(y_2)=1`,
and `r(y_2)=1` is easiest if `y_1` was already correct). Plain SFT on a trace
dataset cannot avoid either — this is why SCoRe uses RL with a bespoke shape.

---

## Stage I — freeze turn-1 with KL, optimize only turn-2

Source lines 20-22:

> Stage I: RL on turn-2 only, with a heavy KL regularization to the base model on turn-1
> (keeps turn-1 behavior fixed while learning to edit).

The loss (attested from source line 37):

```
grad L_I = grad [ log pi(y_2 | x, y_1) * r(y_2) ]   with KL(pi || pi_ref) applied only on turn 1
```

Why this works. If you freeze turn-1 to pi_ref, the model *cannot* cheat by
producing the right answer up front. The only gradient direction for
`r(y_2) = 1` is to actually learn an editing operation conditioned on the
turn-1 response. This is the same structural trick as [[dpo]]'s KL-to-reference,
but applied **on a specific segment of the trajectory** rather than the whole
rollout.

Figure 5 of the paper ablates skipping Stage I — **mode collapse is immediate
without it**. This is not optional.

---

## Stage II — reward the improvement delta

Source lines 22-23:

> Stage II: joint RL over both turns with a reward-shaping bonus on the improvement
> delta r(y_2) - r(y_1).

The loss (attested from source line 38):

```
R_shaped = r(y_1) + alpha * (r(y_2) - r(y_1)),   alpha = 2.0
grad L_II = R_shaped * grad [ log pi(y_1 | x) + log pi(y_2 | x, y_1) ]
```

The `alpha = 2.0` multiplier amplifies the *improvement between turns* relative
to the raw turn rewards. Because `r(y_2) − r(y_1) ∈ {−1, 0, +1}` (binary outcome),
a successful correction `(0 → 1)` gets reward `0 + 2*1 = 2`, double the reward
of just getting it right in turn 1 `(1 → 1)` which would be `1 + 2*0 = 1`.

This is a **curriculum** encoded into the reward, not the data. The policy
gradient sees a bigger gradient signal for "revise a wrong answer to correct"
than for "already correct, no-op." Ch-45 readers should recognize this as the
same structural pattern Meta-Rewarding uses (layered losses, one for each role)
but applied in time (turn 1 vs turn 2) rather than in role (actor vs judge).

---

## The numerical claim

Source line 22:

> Achieves 15.6 pts of self-correction accuracy gain on MATH with Gemini 1.0 Pro
> and 9.1 pts on MBPP — the first method to cross zero on the self-correction task
> (models historically got worse on self-correction).

Historically, adding a "please revise" turn made models **worse**, because
without appropriate training they would second-guess correct answers and flip
them to wrong ones. SCoRe is the first published crossing of zero. Not
overwhelming, but this is a hard regime change: the sign of the effect flips.

---

## Why offline methods fail here — source line 23

> Establishes the on-policy requirement: off-policy / offline methods systematically
> fail the self-correction task.

If you sample correction trajectories from a teacher and SFT on them, the `y_1`
comes from the teacher's distribution — and your student's `y_1` will look nothing
like it. The conditioning `x + y_1` is out-of-distribution at inference. On-policy
rollouts are the only way to make `y_1 ~ pi(·|x)` match between training and
deployment. This is the same on-policy vs off-policy distinction that separates
PPO from DPO in [[ch-39]], but expressed as "which distribution does turn-1 come from."

---

## Connections

- Shares the Stage I / Stage II layered-loss pattern with [[excerpts/meta-rewarding-lm]]'s
  Actor-DPO / Judge-DPO split.
- Uses GRPO-style rollouts and rule-based reward like [[excerpts/r1-zero-analysis]], but
  with a multi-turn action space.
- Complements [[let-verify]] and [[ch-44]] process-reward models: those produce
  step-level signals; SCoRe trains the policy to *use* such signals to edit.
- Host chapter: [[ch-45]] §6.
- Forward to [[ch-50]] agentic RL where multi-turn trajectories become the norm.

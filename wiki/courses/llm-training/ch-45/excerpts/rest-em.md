---
chapter: ch-45
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rest-em.md
source_url: https://arxiv.org/abs/2312.06585
created_at: "2026-04-23"
---

# Excerpt: ReST-EM — expectation-maximization for reasoning self-training

**Source library:** `wiki/raw-data/llm-training/papers/rest-em.md`
**Authors:** Avi Singh, John D. Co-Reyes, Rishabh Agarwal, et al. (Google DeepMind / Brain)
**Year:** 2023 (arXiv 2312.06585)

---

## Why this source anchors ch-45

ReST-EM is the **verifier-filtered** row of the ch-45 table. It replaces
Self-Rewarding's judge with a rule-based verifier (exact-match answer, unit-test
pass) and DPO with plain SFT. The simplification is the point: when a verifier
exists, you do not need preferences, you do not need a reward model, you do not
even need RL — an EM loop on the verifier-filtered distribution is enough.
ReST-EM is the M-step=SFT twin of R1-Zero's M-step=GRPO.

---

## The EM formalism

Source lines 17-18:

> Formalizes self-training as EM on a latent rationale variable: E-step samples
> rationales, filters by verifier; M-step fine-tunes on survivors.

Written out explicitly:

```
E-step:   z ~ pi_{t-1}( . | x )      sample K rationales
          keep  { z  :  verify(answer(z), x) = correct }
M-step:   pi_t = argmax_pi   sum_{(x, z_kept)}  log pi(z | x)
          = SFT on kept rationales, 1 epoch
```

The latent variable is the rationale. The observable is the answer. E-step
rejection-samples the latent; M-step maximizes the likelihood of the retained
latents under the new policy. This is textbook EM with the verifier standing
in for the exact posterior.

---

## The hyperparameters

Source lines 30-37:

> Base model: PaLM-2-L (~340B active params), also ablated on -S and -XS.
> E-step: sample K=32 solutions per problem at T=1.0, top-p=0.95.
> Verifier: exact-match on ground-truth final answer (MATH) or unit-test pass (APPS).
> M-step: SFT on (problem, correct-solution) pairs; 1 epoch; lr=1e-5; batch 128.
> Diversity cap: keep at most 4 distinct correct solutions per problem
> (prevents memorization of one solution path).
> Iterations: 2 for MATH; gains saturate.

Three knobs are load-bearing and ch-45 readers should internalize them:

| Knob | Value | Fails if |
|---|---|---|
| K samples/problem | 32 | K=4 leaves too few survivors on hard problems |
| Temperature | 1.0 | T=0.7 collapses solution diversity; E-step loses its rejection-sampling character |
| Diversity cap | top-4 distinct per problem | without it, iter-3 regresses via memorization |

The **diversity cap** is the ingredient that separates ReST-EM from naive
self-distillation. Without it, the E-step produces many copies of the same
solution path (the model's favorite one), the M-step memorizes that path,
the next E-step is even more peaked, and the iteration collapses.

---

## The numerical results

Source lines 19-22:

> Shows ReST-EM on PaLM-2-L raises MATH test accuracy from 34.1 % (human-data SFT)
> to 50.6 % (two iterations of ReST-EM).
> Shows the same trick lifts APPS code-generation from 16.4 % -> 31.2 %.
> Demonstrates ReST-EM transfers — training on MATH improves Big-Bench-Hard unrelated tasks.

The MATH curve attested from Figure 2: 34.1 → ~42 → 50.6 → flat at iter 3
(+8, +6, then nothing). This is the canonical **2-iteration saturation** for
verifier-filtered self-training. It is also the strongest evidence in the
EM family that **self-training on model-generated correct solutions can exceed
training on human-written solutions** — 50.6 % vs 34.1 % is a 16-point gap,
and the model's output is the *only* difference.

The BBH transfer (Figure 4) is the surprise: training on MATH improves unrelated
reasoning benchmarks. Not huge (a few points), but present. The implication:
the skill being transferred is "reasoning," not just "MATH-answer-format."

---

## Saturation mechanism

ReST-EM saturates for a different reason than Self-Rewarding or SPIN:
**the E-step distribution stops changing meaningfully**. After iter 2, the
verifier-correct set is approximately stationary (the model gets almost all
the problems it can get right), so the M-step has no new signal. This is
cleaner than Self-Rewarding's drift or SPIN's equilibrium; it is just *exhaustion
of the signal under the fixed verifier*.

The fix R1-Zero takes (see [[excerpts/r1-zero-analysis]]) is replacing the SFT
M-step with GRPO. A gradient update can extract more signal from the same
verifier-correct samples than SFT can — the policy gradient sees the full
distribution, not just the argmax.

---

## Connections

- Descendant of [[star]] (K=1 with rationalize-backward trick); ReST-EM drops the
  rationalization and scales K.
- Sibling of [[v-star]], which uses the failed traces as verifier-training data.
- Direct precursor to [[excerpts/r1-zero-analysis]] — R1-Zero is ReST-EM with
  GRPO as the M-step instead of SFT.
- Motivates the [[ch-44]] outcome-vs-process discussion: ReST-EM uses outcome-only
  supervision exclusively.
- Host chapter: [[ch-45]] §7.

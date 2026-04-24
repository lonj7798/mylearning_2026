---
chapter: ch-45
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/spin.md
source_url: https://arxiv.org/abs/2401.01335
created_at: "2026-04-23"
---

# Excerpt: SPIN — preference signal from SFT data alone

**Source library:** `wiki/raw-data/llm-training/papers/spin.md`
**Authors:** Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu (UCLA)
**Year:** 2024 (arXiv 2401.01335)

---

## Why this source anchors ch-45

SPIN's contribution is structural, not numerical. It shows that **preference-level
gradient signal does not require preference labels** if you are willing to treat
the human SFT response as "chosen" and your previous iterate's output as "rejected."
For the ch-45 filter table, SPIN is the row where the filter is **the data itself** —
not a learned judge, not a verifier, not a game. That puts it in an equivalence class
by itself and makes it the cleanest ablation against Self-Rewarding.

---

## The SPIN loss, attested from source line 19

> Provides a closed-form DPO-equivalent update:
>   L_SPIN = −log sigmoid( beta * ( log pi(y_human)/pi_{t-1}(y_human)
>                                    − log pi(y_gen)/pi_{t-1}(y_gen) ) ).

This is identical to the DPO loss with two substitutions:

- `y_chosen := y_human` (from the SFT dataset)
- `y_rejected := y_gen` (sampled from π_{t−1}, the previous iterate)

The DPO algebra is otherwise verbatim. The innovation is **the preference source**,
not the objective. The implication: any codebase that can run DPO can run SPIN by
swapping the data loader.

---

## The Nash-equilibrium interpretation

Source lines 18 and 26:

> Frames SFT-only post-training as a two-player game: the policy plays against
> its previous iteration; Nash equilibrium is reached when pi_t = data distribution.
>
> Theorem 4.1: Nash equilibrium characterization — when policy generates the
> human distribution, the SPIN loss becomes 0.

This is the **ceiling SPIN imposes on itself**. If the equilibrium is "π_t matches
the human data distribution," SPIN cannot exceed the SFT corpus. It is a
distribution-matching algorithm disguised as a preference algorithm.

Practical consequence for ch-45: SPIN is a warmup, not a terminal stage. [[iterative-sft-rl]]
(Tülu 3) runs SFT → DPO → RLVR; SPIN substitutes for the DPO stage when preferences
are unavailable. It does not substitute for the RLVR stage, because RLVR can exceed
the SFT ceiling and SPIN cannot.

---

## The hyperparameters that matter

Source lines 30-36:

> Per iteration:
>   1. Sample 50K (prompt, response) pairs from pi_{t-1} at T=1.0.
>   2. Build DPO pairs (chosen=y_human, rejected=y_gen) 1:1 with the SFT data.
>   3. DPO-train pi_t from pi_{t-1}: beta=0.1, lr=5e-7, 3 epochs, batch 64.
>   4. Reset reference to pi_{t-1} for the next iteration.

- **T=1.0** on the rollouts. Higher temperature ensures the `rejected` samples
  cover a wide distribution; T=0.7 like Self-Rewarding would produce `rejected`
  that are too close to `chosen` and collapse the DPO margin.
- **beta=0.1** identical to DPO-from-preference-labels. The paper ablates and
  reports 0.05 and 0.3 both underperform.
- **1:1 ratio** of human:generated pairs. Off-ratio (2:1 or 1:2) hurts by ~0.3
  MT-Bench points per iter. This is a hidden constraint — you cannot just pair
  every human response with 4 generated responses.
- **Reference reset to pi_{t-1}**. This is the invariant across ch-45 self-improvement
  methods. Without it, SPIN keeps pulling toward the stale SFT reference and the
  iteration contributes nothing.

---

## The MT-Bench curve

Source line 22:

> Shows monotone improvement across 3 SPIN iterations (MT-Bench 6.39 -> 7.12).

Compared to Self-Rewarding's AlpacaEval climb 9.94 → 20.8, SPIN's MT-Bench climb
6.39 → 7.12 looks modest. But note the input: SPIN used **only** Ultrachat SFT data.
No preference labels, no GPT-4 judge. The Zephyr-7B baseline that matches SPIN
used 60K GPT-4-labeled preferences. SPIN replaces preference labeling with
50K-prompt rollouts per iteration — that trade is what the paper sells.

---

## Why it saturates at 3 iters

Not because of judge drift (there is no judge). Because of **distribution matching**:
once π_t ≈ data distribution, the DPO margin `log(π/π_{t−1})` collapses to zero on
both the chosen and rejected branches, and the gradient vanishes. The paper's
"minor gain after 3" is the algorithm reaching its fixed point.

This is the mechanical difference from Self-Rewarding saturation. Self-Rewarding
saturates because the signal degrades (reward hacking); SPIN saturates because
the signal vanishes (equilibrium reached). If your SFT data is *better* than your
current model, SPIN still has gradient to consume. If it is worse, SPIN will pull
you backward toward it.

---

## Connections

- DPO algebra from [[dpo]]; only the preference source differs.
- Contrasted with [[excerpts/self-rewarding-lm]]: different filter, different saturation mechanism.
- Theoretical cousin of [[self-play-preference]] (Nash-LM): both cast alignment as
  a two-player game; SPIN's equilibrium is distribution-matching, Nash-LM's is
  preference-matching.
- Precursor to the *online DPO* family summarized in [[iterative-sft-rl]].
- Host chapter: [[ch-45]] §4.

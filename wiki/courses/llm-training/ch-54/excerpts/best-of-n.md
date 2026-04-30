---
chapter: ch-54
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/best-of-n.md
source_url: https://arxiv.org/abs/2009.01325
created_at: "2026-04-23"
---

# Excerpt: Stiennon 2020 — the BoN ceiling ch-54 §2 uses as baseline

**Source library:** `wiki/raw-data/llm-training/papers/best-of-n.md`
**Artifact:** the BoN-KL closed form and the Figure-6 KL-matched comparison.

---

## Why this source anchors ch-54

§2 of ch-54 opens with a provocation: before running PPO, measure what Best-of-N with your reward model already delivers. Stiennon 2020 is the source for *why* this is a reasonable ask. The paper's §2 line 20 gives the number that makes the provocation operational:

> **BoN KL formula:** `KL(BoN || base) = log N − (N−1)/N` — derived in appendix; tight for well-calibrated RM.

That closed form turns "BoN vs PPO" into a **KL-matched** comparison rather than an apples-vs-oranges one. Pick N; compute KL_BoN; set PPO's β so attained KL is the same; now you can ask "which produced better human-eval wins at the same KL?"

---

## The number §2 forces into memory

At N = 64: KL_BoN ≈ ln 64 − 63/64 ≈ 4.159 − 0.984 ≈ **3.17 nats**.

Stiennon's Figure 6 at that budget: BoN-64 sits within ~2 human-preference points of a well-tuned PPO, with zero training cost and no KL-runaway risk. That is the baseline your engineering investment has to beat to justify itself.

---

## The ceiling — Figure 4

From the source (line 24):

> **Figure 4 (RM score vs human preference, KL on x-axis):** the overoptimization curve — both BoN and RL rise, then RL keeps climbing in RM score while human preference plateaus or drops.

BoN is monotone in N only if the RM is faithful; past the critical KL, argmax-over-N starts preferring exploit cases (sycophancy, length inflation, formatting hacks). This is the first documented instance of what [[reward-model-overoptimization]] later formalized as "Goodhart on RM", and it applies just as much to BoN deployment as to PPO training.

---

## Procedure (attested)

From the source (line 30):

> **BoN procedure:**
>   - Sample N summaries at T=0.7.
>   - Score each with RM.
>   - Return argmax.

Cheap. Trivially parallelizable (no gradient). The only failure mode is the RM itself — which §3 (ensembling) and ch-41 (architecture) address.

---

## Why §2 of ch-54 treats BoN as the default first move

- **Cost.** Inference only — no trainer GPUs, no loss curves to babysit, no KL controller to tune.
- **Determinism.** Given the same prompt, same temperature seed, same RM, you get the same answer. PPO runs diverge across seeds.
- **Ceiling aligned with RM quality.** BoN's ceiling is a proxy for PPO's — so if BoN-64 is flat, PPO will not save you; the RM is the problem.
- **Production pattern.** [[best-of-n]] §Connections notes BoN "remains the production deployment pattern for many inference-time alignment recipes" — test-time compute research, Cohere chat.

---

## Connections

- **ch-54 §2** — BoN KL formula + Figure 6 KL-matched comparison.
- **ch-54 §4** — BoN has no replay question because it has no gradient.
- **ch-41 §2** — [[reward-model-overoptimization]] (Gao 2022) formalized the inverted-U Stiennon first drew.
- **[[rejection-sampling-finetuning]]** — BoN applied to *training data* instead of inference; Llama 2 post-training.
- **[[west-of-n]]** — BoN generalized into *preference-pair* generation for later DPO-style training.

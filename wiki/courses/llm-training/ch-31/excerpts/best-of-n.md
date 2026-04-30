---
chapter: ch-31
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/best-of-n.md
source_url: https://arxiv.org/abs/2009.01325
created_at: "2026-04-23"
---

# Excerpt: Best-of-N — the inference-time twin and the KL-cost formula RSFT inherits

**Source library:** `wiki/raw-data/llm-training/papers/best-of-n.md`
**Artifact:** BoN as an RL baseline, the BoN-KL closed form, and the overoptimization knee

---

## Why this source anchors ch-31

RSFT is "BoN at training time." The theoretical understanding of what BoN does per round — how much KL it moves the implied policy, where the reward-model overoptimization knee lives, how it compares against PPO at matched KL — all comes from [[best-of-n]] (Stiennon 2020, the TL;DR summarization paper that predated InstructGPT by two years). Without this source on the page, ch-31's claim that "RSFT has a bounded per-round drift" is hand-waving. With it, the claim has a closed-form formula.

---

## The three attested facts ch-31 uses

From the source (lines 7–8, 19–20, 34–35):

> On summarization, at small reward-model KL budgets, **Best-of-N** sampling (generate N candidates, pick the one with the highest reward-model score) is competitive with or superior to full RLHF — it has no training instability, costs only inference, and is monotonic in N until the reward model overoptimizes.

> **BoN KL formula:** `KL(BoN || base) = log N − (N−1)/N` — derived in appendix; tight for well-calibrated RM.

> Compared RLHF against **Best-of-N (BoN)** and showed BoN is a strong, often-overlooked baseline: BoN-64 with a well-trained RM is within 2 points of PPO on human eval at much lower KL.

Read these as a system. (1) BoN is a strong baseline people under-rate. (2) It has a closed-form KL cost. (3) At matched KL, BoN is competitive with PPO and sometimes better.

---

## The BoN-KL formula, worked

`KL(BoN_N || base) = log N - (N-1)/N`

At N=2:  `log 2 - 1/2 \approx 0.69 - 0.50 = 0.19` nats.
At N=10: `log 10 - 9/10 \approx 2.30 - 0.90 = 1.40` nats.
At N=32: `log 32 - 31/32 \approx 3.47 - 0.97 = 2.50` nats.
At N=64: `log 64 - 63/64 \approx 4.16 - 0.98 = 3.18` nats.

Two consequences for ch-31:

1. **Doubling N adds less than log 2 of KL** past N=4 (the (N-1)/N term saturates fast). Past N=16, doubling N is almost exactly `\Delta KL = log 2`.
2. **Each RSFT round cannot exceed the BoN-KL of the selection step as a ceiling on its per-round drift.** If you sample K=10 and keep top-1 per prompt, that round moves the policy at most ~1.4 nats toward the RM's argmax. This bounds the per-round reward-hacking headroom.

This is the quantitative backbone behind ch-31's "RSFT cannot drift arbitrarily far within a round" claim. The RSFT training step may not exactly achieve the BoN-KL (SFT is not a perfect projection onto the filtered distribution), but BoN-KL is the theoretical ceiling.

---

## The overoptimization knee

From the source (line 25):

> **Figure 4 (RM score vs human preference, KL on x-axis):** the overoptimization curve — both BoN and RL rise, then RL keeps climbing in RM score while human preference plateaus or drops.

The practical rule: **pick N on the rising part of this curve, not past it.** For a well-trained RM, Stiennon reports BoN-64 still sits on the rising part on TL;DR; for weaker RMs the knee comes earlier. Ch-31's HTML slider caps at K=64 specifically because past that you are visibly past the knee on most real RMs.

Two mechanisms push the knee left:

- **RM accuracy.** A less-accurate RM has a narrower range of reliable selection. Ch-31's decision tree node 5 ("is RM accuracy above 70% on held-out?") is a proxy for "is your knee past K=10?"
- **Distribution shift.** After a few RSFT rounds the policy has drifted from the RM's training distribution; the RM silently loses accuracy; the knee moves left. This is why Llama-2 refreshes preference data weekly and Llama-3 retrains the RM every round.

---

## BoN as the apples-to-apples RL baseline

From the source (line 26):

> **Figure 6 (BoN vs RL at matched KL):** BoN and RL are nearly coincident; BoN wins at very low KL, RL wins at higher KL.

The BoN-KL formula lets you *directly* match a PPO run's final KL to a BoN-N. If PPO converges at KL=1.4 nats against `\pi_{ref}`, compare it to BoN-10 (which sits at ~1.4 nats). The formula is what makes this comparison clean.

Ch-31's claim ("if you have strong RM and weak RL infra, iterate SFT") is justified here: at matched KL, BoN is within 2 points of PPO and has zero training instability. RSFT distills BoN back into the policy so inference does not have to pay the K-fold generation cost at every request.

---

## Why RSFT is strictly stronger than inference-time BoN at matched N

BoN pays K-fold inference compute at *every* request, forever. RSFT pays it once to build the training set, then the policy internalizes the filtered distribution. At serving time the policy emits one sample that approximates the BoN-N argmax.

The caveat: RSFT's approximation is lossy. The policy trained on top-1-of-K samples does not perfectly recover the BoN-K distribution; it recovers a smoother version. This is why inference-time BoN on top of an RSFT-trained policy is still additive — you compound the two.

Ch-31 does not work through this compounding in prose because production deployments typically pick one (RSFT during training, then greedy at inference, *or* an SFT policy plus inference-time BoN). The point for the chapter is that both paths are legitimate and the BoN-KL formula sets the ceiling for either.

---

## Connections

- [[rejection-sampling-finetuning]] — RSFT is BoN-at-training-time.
- [[reward-model-overoptimization]] (Gao 2022) — formalized the overoptimization curve Stiennon first documented.
- [[west-of-n]] — BoN generalized into preference-pair generation for DPO.
- **ch-31 §2** — the numerical BoN-KL calculations above are what §2 summarizes.
- **ch-31 decision tree node 5** — uses the overoptimization knee as the "is your RM good enough to iterate on?" gate.

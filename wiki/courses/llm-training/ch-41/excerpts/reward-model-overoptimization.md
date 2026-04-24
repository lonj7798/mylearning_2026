---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/reward-model-overoptimization.md
source_url: https://arxiv.org/abs/2210.10760
created_at: "2026-04-23"
---

# Excerpt: Gao 2022 — the inverted-U §2 of ch-41 internalizes

**Source library:** `wiki/raw-data/llm-training/papers/reward-model-overoptimization.md`
**Artifact:** scaling laws `R_gold(d) = d · (α − β·d)` (BoN) and `R_gold(d) = d · (α − β·log d)` (PPO)

---

## Why this source anchors ch-41

§2 is the load-bearing section of the chapter. Without Gao 2022, a learner might read §1's BT derivation and conclude "great, train the RM, run PPO, done." Gao is the empirical demonstration that RMs are *proxies* with bounded generalization — and that proxy and gold reward diverge as a predictable function of `d = sqrt(KL)`.

Every later section of ch-41 is a response to this fact: §3 ensembles to push the peak right, §4 changes RM architecture to reduce proxy error, §5 decomposes into attributes to make the proxy steerable, §6 enumerates when to skip training the RM altogether.

---

## The three equations ch-41 §2 requires you to memorize

From the source (line 18):

> For best-of-n, `R_gold(d) ≈ d · (α_bon − β_bon · d)`; for RL, `R_gold(d) ≈ d · (α_RL − β_RL · log d)`. Proxy reward grows monotonically; gold reward follows an inverted-U.

And (line 34):

> **Best-of-n KL:** `KL_bon(n) = log n − (n−1)/n` — derived analytically, matches observed curves.

Ch-41 §2 reproduces these verbatim and derives the closed-form peaks:
- BoN: `d* = α_bon / (2 β_bon)`.
- PPO: `d* = exp(α_RL / β_RL − 1) / e` (set derivative `α − β(log d + 1) = 0`).

The companion figure [figures/rm-overopt.html](../figures/rm-overopt.html) plots both and marks `d*` live as you slide α and β.

---

## Why sqrt(KL) is the natural axis

From the source (line 20):

> **KL = sqrt(KL) as natural x-axis:** plotting against `d = sqrt(KL)` linearizes many of the relationships, consistent with KL being a squared-distance metric locally.

Ch-41 §2 turns this into operational advice: plot `R_gold(d)` during RL, not `R_proxy(step)`. The x-axis matters. Step-based plots hide the overoptimization curve; `d`-based plots surface it.

---

## The scaling rule §2 calls out

From the source (line 19):

> both α and β shrink smoothly with RM parameters — a 10× larger RM roughly halves the overoptimization slope but does not eliminate the hump.

Ch-41 §2 reports this as "larger RMs push the peak right, not to infinity." §3's ensembling is the cheap complement: if you cannot afford a 10× RM, ensemble K = 3–5 and get a similar peak shift.

---

## The counterintuitive result §6 rests on

From the source (line 22):

> **Policy size barely matters:** bigger policies optimize the proxy faster but hit the same gold peak — this is a property of the RM, not the policy.

Ch-41 uses this to kill a common temptation: "scale up the policy to dodge overoptimization." No — a 70B policy reaches the same `d*` as a 7B policy, just faster. The RM is the bottleneck; the policy is the thing being bottlenecked.

---

## β is not a free lunch

From the source (line 23):

> **KL penalty β is not a free lunch:** varying β in PPO traces out essentially the same front as early-stopping, up to small differences.

Ch-41 §2 encodes this as "KL penalty β and early stopping are the same knob — don't tune both independently." If you have not internalized this, you will tune β, fail to improve, and blame the RM. The correct move is to pick one, hold it, and monitor `d`.

---

## The empirical number ch-41 forces into memory

From the source (line 37):

> at RM size 3M, gold reward peaks near `d ≈ 3` nats^0.5 and loses most of the gain by `d ≈ 8`; larger RMs shift the peak right.

Ch-41 §2 quotes this as the "empirical number to memorize." Modern 7B+ RMs push the peak past `d ≈ 5`, which §3 Coste 2023 confirms shifts to `d ≈ 5–8` with ensembling. The interactive figure lets you sweep α, β, K and watch the peak slide — the goal is to build reflex on "where is d* right now?"

---

## Connections to the rest of ch-41

- **§1** — the "proxy" is the BT RM trained with the loss on line 21 of [[bradley-terry-rm]].
- **§3** — [[reward-ensembling]] is the direct defense; quoted peak shift from `d ≈ 3` → `d ≈ 5–8`.
- **§4** — [[generative-reward-models]] and [[pairrm]] reduce proxy error via better RM architecture.
- **§6** — the decision framework's "ensemble" row exists because of this law.
- **ch-40** — KL control is the mechanism that makes β the budget knob.

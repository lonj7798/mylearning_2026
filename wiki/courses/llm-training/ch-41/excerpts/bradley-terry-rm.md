---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/bradley-terry-rm.md
source_url: https://www.jstor.org/stable/2334029
created_at: "2026-04-23"
---

# Excerpt: Bradley-Terry — the loss §1 of ch-41 derives

**Source library:** `wiki/raw-data/llm-training/papers/bradley-terry-rm.md`
**Artifact:** logistic preference model + InstructGPT RM loss

---

## Why this source anchors ch-41

Every subsequent section of ch-41 assumes you own the BT derivation. §2 (Gao's overoptimization law) is stated over a BT-trained proxy vs a BT-trained gold. §3's ensembling averages BT heads. §4's PairRM and GenRM are *structural alternatives* that only make sense as departures from BT. §5's multi-attribute heads are K parallel BT losses glued together. §6's decision framework is a menu of *when to leave BT behind*.

So §1's job is to lock in the derivation: BT is logistic regression on score differences, and the RM loss is binary cross-entropy with label 1 and logit `r(y_w) − r(y_l)`. Memorize both lines.

---

## The generative model

From the source (lines 15, 18–19):

> if items have latent scores `r_1, …, r_K`, the probability that item `i` is preferred to `j` in a pair is `P(i ≻ j) = σ(r_i − r_j) = exp(r_i) / (exp(r_i) + exp(r_j))`.

Nothing in this is derived from physics. The logit link is an assumption about how latent scalar quality maps to discrete choice — one that Bradley & Terry made in 1952 because logistic regression was tractable, and one that every modern RM inherits without replacement.

The whole "your RM is calibrated" story rests on this single assumption. When BT fails (intransitive preferences, context-dependent preferences, multi-attribute preferences), it fails because this assumption was wrong — not because the training procedure drifted.

---

## The loss ch-41 §1 writes out

From the source (line 21):

> `L_RM(θ) = − E_{(x, y_w, y_l) ~ D}[log σ(r_θ(x, y_w) − r_θ(x, y_l))]`

Ch-41 adds the one observation the source hints at but does not belabor: this is *exactly* binary cross-entropy with label 1 and logit `r_θ(x, y_w) − r_θ(x, y_l)`. That identity is why a BT RM can be implemented as a 10-line patch on any classification pipeline.

---

## Identifiability — the footgun ch-41 §1 calls out

From the source (line 26):

> BT scores are identified only up to an additive constant per prompt; in practice the per-prompt mean is subtracted (or absorbed by the language model).

Ch-41 §1 turns this into a concrete rule: *do not compare `r` values across prompts*. The scalar head absorbs a prompt-dependent offset that you cannot see in a single forward pass. Two `r = 5.0` responses to two different prompts are not comparable without prompt-mean correction.

This is also why Elo ratings are BT scores up to a constant — the Chatbot Arena story is not an analogy, it is the same math [[bradley-terry-rm]].

---

## What ch-41 §1 keeps, changes, drops

| Source emphasis | Ch-41 §1 treatment | Reason |
|-----------------|--------------------|--------|
| Full K-way Plackett-Luce derivation | One-line mention | Pairwise dominates modern pipelines; Plackett-Luce is a footnote |
| Length-bias patches (regression, length-matched pairs) | Listed as attested failure mode | §5's multi-attribute head is the *structural* answer |
| BT ↔ DPO algebra | Pointed to as a connection, not derived | DPO chapters (ch-44+) do the derivation; ch-41 stays on RM side |
| Margin/smoothing variants `−log σ(r_w − r_l − m)` | Dropped | Rarely used in open recipes; adds a knob without insight |
| Connection to Elo | One line | Chatbot Arena is its own story |

---

## The failure modes §1 uses §6 to answer

- *Length bias* → §5 (multi-attribute RM with separate Verbosity head) or §4 (PairRM with length-balanced training pairs).
- *Transitivity violation* → §6 decision table points to judge-LLMs or GenRMs when preferences are obviously intransitive.
- *Context-dependence* → §3 ensembling does not help; §4 GenRM rubric steerability does.

This is the throughline: §1 names the failure surface, §3–§6 pick mitigations.

---

## Connections to the rest of ch-41

- **§2** — Gao's proxy/gold story is told over BT-trained RMs; the loss above is the proxy.
- **§3** — ensemble aggregators (mean, LCB, min) are defined over K independent BT heads.
- **§4** — PairRM replaces `r(x, y_A) − r(x, y_B)` with a joint-encoder logit; GenRM replaces the linear head with a log-prob over verdict tokens.
- **§5** — HelpSteer2's 5-dim RM is 5 parallel BT losses with separate heads.
- **§6** — the decision framework's default row ("scalar BT RM") is exactly the loss on line 21.

---

## Further reading

- [[bradley-terry-rm]] — full source with K-way extension, margin variants, Elo connection.
- [[reward-model-overoptimization]] — what happens to a BT RM once you start optimizing against it.
- [[pairrm]] — the alternative §4 contrasts with.

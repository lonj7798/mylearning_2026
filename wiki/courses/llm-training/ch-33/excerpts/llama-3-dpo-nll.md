---
chapter: ch-33
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
---

# Excerpt: Llama 3's DPO stabilizer — why NLL-on-chosen matters

**Source library:** `wiki/raw-data/llm-training/model-reports/llama-3.md` §DPO
**Artifact:** DPO + auxiliary NLL-on-chosen loss with coefficient 0.2

---

## Why this source anchors ch-33

Vanilla DPO (Rafailov 2023) optimizes a *relative* preference margin between chosen and rejected; nothing in the loss fixes the *absolute* log-probability of the chosen response. In practice this causes a specific failure mode: chosen-logprob can decay while the preference margin still increases, and the model loses confidence in responses it nominally prefers. Llama 3's fix — add an auxiliary NLL term on the chosen sequence, weighted at 0.2 — is a tiny but important novelty that ch-33 §3.1 quotes and that the HTML figure highlights in the Llama 3 column.

---

## The attested DPO configuration

From the source (Table 7 / lines 49–53):

> Learning rate: 1e-5
> Beta (KL coefficient): 0.1
> Auxiliary NLL loss on chosen sequences: coefficient 0.2 — added to stabilize training by preventing chosen-logprob decay.
> Single epoch per round; masks prompts from loss.
> Most-recent-batch preference data only (older batches cause format drift).

Four of these five lines are standard DPO hparams. The third is the one that isolates a specific bug: *chosen-logprob decay*.

---

## What chosen-logprob decay actually looks like

In vanilla DPO the objective is (schematically):

`L_DPO = -log σ( β · ( log π_θ(y_chosen | x) - log π_ref(y_chosen | x)  -  log π_θ(y_rejected | x) + log π_ref(y_rejected | x) ) )`

The loss depends only on the *difference* of the chosen and rejected implicit rewards. You can satisfy DPO by pushing the rejected logprob down faster than you push the chosen logprob down — the margin still widens, the preference-accuracy metric still improves, but the absolute `log π_θ(y_chosen | x)` drops. Llama 3's fix adds:

`L_total = L_DPO + 0.2 · NLL(y_chosen | x)`

which anchors the chosen sequence to non-trivial probability. The 0.2 coefficient is small — the dominant signal is still DPO — but enough to block the pure-down-weighting-of-rejected degenerate path.

---

## Why "most-recent-batch only"

From the same section, the line *"Most-recent-batch preference data only (older batches cause format drift)"* is the other load-bearing rule. Across six rounds, the policy's output distribution changes — the formats it produces, the token sequences it emphasizes, the refusal style it defaults to. A preference batch collected against round-2's policy contains pairs whose *rejected* half reflects round-2 format errors. By round 5, those errors no longer occur; training on old pairs pulls the policy back toward old formats. The fix is simple and absolute: each round's DPO sees only that round's freshly collected preferences.

This is the operational consequence of the "fresh RM every round" rule from [[llama-3-six-rounds.md]]: freshness applies to *both* the RM and the preference batch that DPO trains on.

---

## Reward-model-free or reward-model-driven?

One point of confusion: Llama 3 uses a reward model for *rejection sampling* and for *labelling preference pairs*, but the RL *optimization step* is DPO, not PPO. The RM is a scoring function inside the loop; it is not the reward in a PPO rollout. This is the exact opposite of Tülu 3's RLVR stage, which uses no RM and runs PPO against a deterministic verifier. The two recipes evolved PPO differently: Llama 3 dropped PPO entirely and kept a learned RM; Tülu 3 kept PPO and dropped the learned RM.

Ch-33 §4's final comparison table makes this dichotomy explicit.

---

## What ch-33 keeps from this source

- DPO LR 1e-5, β=0.1, NLL coefficient 0.2 (§3.1).
- The chosen-logprob-decay failure mode the 0.2 coefficient targets (§3.1, implicit in §3.3 reason 5).
- The "most-recent-batch only" rule (§3.1 and §3.3 reason 5).
- The RM-is-for-scoring-not-for-RL-reward distinction (§4 table).

---

## Connections

- **ch-33 §3.1 / §3.3 / §4** — where this excerpt is cited.
- **[[dpo]]** — the Rafailov 2023 base algorithm; Llama 3's NLL add-on is the novel stabilizer.
- **[[llama-3]]** — the tech report this excerpt is drawn from.
- **[[tulu-3]]** — DPO β=0.1 confirmed as a robust default; Tülu 3's DPO uses length-normalized DPO at β=5.0, a different parameterization with the same intent.
- **[[reward-model-overoptimization]]** — the related but distinct failure mode RM retraining guards against.
- **ch-37..ch-46 (RL track)** — DPO and its stabilizers get their own chapter; this excerpt is the case-study anchor.

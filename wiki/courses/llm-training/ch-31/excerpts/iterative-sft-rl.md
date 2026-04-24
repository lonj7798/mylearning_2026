---
chapter: ch-31
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/iterative-sft-rl.md
source_url: https://arxiv.org/abs/2307.09288
created_at: "2026-04-23"
---

# Excerpt: Iterative SFT+RL — the synthesis ch-31 was written to teach

**Source library:** `wiki/raw-data/llm-training/papers/iterative-sft-rl.md`
**Artifact:** the Llama-2 5-round RSFT+PPO schedule and the Tulu-3 SFT->DPO->RLVR stage ablation

---

## Why this source anchors ch-31

This raw-data file is the synthesis note the chapter is structured around. Where [[llama-2]] and [[llama-3]] report their own pipelines and Tulu-3's report describes SFT->DPO->RLVR as its own thing, the [[iterative-sft-rl]] synthesis names the common structure: multi-round alternation, reference-reset between stages, per-round data refresh. The 3-5 point single-pass-vs-multi-round ablation number ch-31 §5 cites is attested here.

---

## The attested synthesis — quoted

From the source (line 7):

> Running *multiple* alternating rounds of SFT-on-filtered-samples and RL outperforms a single SFT + single RL run — because each round's RL policy produces better on-policy data for the next round's SFT, and each round's SFT stabilizes the reference distribution for the next round's RL.

And from the source (lines 17, 30):

> Both works report that the iterative structure is not incidental: ablating it costs ~3–5 points on AlpacaEval / MT-Bench.

> Both papers show large gains from the *iteration*, not just from added data.

The "iteration, not added data" distinction is the one the chapter's §5 table hinges on. The comparison is at matched data — same preference batches, same SFT pool — single-pass vs iterative. 3-5 points is the gap the iteration alone buys. This is *not* a "more data helps" finding; it is an "iteration structure itself adds gain" finding.

---

## The three attested configurations, side by side

From the source (lines 20–29, 38–47):

**Llama-2 (V1..V5):**

> - K=10 samples/prompt from current model; keep top-1 by RM score.
> - SFT on kept samples for 1 epoch.
> - Then PPO with KL-controller for several thousand steps.
> - Repeat 5 rounds.

> PPO: KL coef 0.01 (adaptive), clip 0.2, batch 1K rollouts, lr 1e-6.

**Tulu-3 (SFT -> DPO -> RLVR):**

> - Stage 1 — SFT on Tulu-3-SFT-mix (~1M instances, multi-task, synthetic-heavy).
> - Stage 2 — DPO on Tulu-3-DPO-mix (~300K pairs, synthetic via GPT-4o / UltraFeedback).
> - Stage 3 — RLVR: PPO with binary verifiable reward on math + code + strict instruction-following prompts.
> - Reference reset between stages: DPO uses the SFT checkpoint as reference; RLVR uses the DPO checkpoint.

> SFT lr 5e-6, 2 epochs, linear decay.
> DPO beta=5, lr=5e-7, 1 epoch.
> RLVR: PPO with value head removed -> REINFORCE++-style baseline, binary verifier reward in {0, 1}, KL-to-SFT-ref beta=0.05.

**Llama-3 (6 rounds SFT + rej-sample + DPO):** attested in [[llama-3]]; summarized in [[iterative-sft-rl]] as "generalizes Llama-2 with DPO replacing PPO and 6 rounds."

These three are the canonical reference configurations for ch-31's decision tree. Each row of the tree's §6 flowchart maps to one of them.

---

## The reference-reset trick — the most important one

From the source (line 47):

> Key hyperparameter: reference resets — critical; without them, DPO/RLVR keeps dragging the policy backwards toward the stale SFT reference.

This is the one operational mistake that costs the most hours in open replications. The failure mode: you start round N with `\pi_{ref}` = the SFT checkpoint from round 0. As rounds proceed, `\pi_{current}` drifts toward the good-reward region; `\pi_{ref}` stays frozen at the round-0 SFT distribution. The DPO/PPO KL regularizer keeps pulling `\pi_{current}` back toward `\pi_{ref}` instead of keeping it near the *previous round's* checkpoint.

Fix: before each new round's DPO/RLVR step, reset `\pi_{ref}` to the latest SFT checkpoint. Tulu-3 says "DPO uses the SFT checkpoint as reference; RLVR uses the DPO checkpoint." Llama-3's 6-round loop implies reference reset each round (the DPO ref is always the latest round's SFT checkpoint).

Ch-31 §6 hard-codes this as the one "every path" rule at the bottom of the decision tree. It is the single thing you cannot skip.

---

## Why DPO replaced PPO — the Llama-2 to Llama-3 lesson

From the source (line 8):

> Llama-2 used 5 rounds of RSFT + PPO; Tulu 3 uses SFT -> DPO -> RLVR as three distinct stages.

And implicit in the configurations: PPO at 70B needs a value function, GAE, KL controller, careful lr tuning. DPO is a single loss term over chosen/rejected pairs. Llama-3 switched PPO -> DPO and added an NLL auxiliary on the chosen side; the empirical result is at least parity, often better.

For ch-31's decision-tree node 6 ("3 rounds -> switch to DPO with reference-reset"), the DPO-over-PPO default is justified here: the engineering cost is lower, the training is more stable, and the attested deltas do not favor PPO at matched data. PPO still earns its place in the RLVR stage (binary verifier reward, which is harder to turn into a DPO pair), but for RM-scored preference data, DPO is the default.

---

## The one claim ch-31 inherits wholesale

From the source (line 50):

> Supports the SPIN / ReST-EM / Self-Rewarding thesis that iteration > single-pass.

This single sentence is ch-31's whole argument. The decision tree operationalizes it; the §5 eval-delta table grounds it; the §7 failure-modes list protects it. When in doubt about what the chapter exists to teach, point here.

---

## Connections

- [[llama-2]] — the 5-round RSFT+PPO schedule; ch-31 §1.
- [[llama-3]] — the 6-round SFT+rej-sample+DPO schedule; ch-31 §2.
- [[rest-em]] — verifier-only iteration; ch-31 §4.
- [[rejection-sampling-finetuning]] — the primitive; ch-31 §1.
- **ch-31 §5** — the 3-5 point eval delta is attested here.
- **ch-31 §6** — the decision tree is the operationalization of this synthesis.

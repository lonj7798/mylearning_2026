---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
---

# Excerpt: Llama 3 — the industrial DPO recipe

**Source library:** `wiki/raw-data/llm-training/model-reports/llama-3.md`
**Authors:** Aaron Grattafiori et al. (Meta Llama Team)
**Venue:** arXiv 2407.21783
**Year:** 2024

---

## Why this source anchors ch-39 §10

The Llama 3 technical report is the most-cited deployment of DPO in 2024. Its post-training pipeline, described in §4 of the paper and summarized in the Herd paper's Figure 7, is the industrial answer to: *does DPO actually work at 405B scale, and if so, with which hyperparameters?*

Answer: yes, with two adjustments to the academic recipe — (a) ten times the learning rate, and (b) an auxiliary NLL term on the chosen response. Both of these are explicitly attested in Table 7 of the report.

## The six-round structure

Source lines 34–36 and Figure 7:

> Six rounds of: (a) Reward Model (RM) training, (b) Rejection Sampling to build the round's SFT pool, (c) SFT on curated pool, (d) DPO on preference data collected with the latest RM-ranked generations. Each round uses a fresh batch of ~human preference annotations plus synthetic data resampled from the round-(N-1) best model.

This is the canonical "offline within a round, online across rounds" structure. Each round is a standard offline DPO run on a fixed preference batch; but between rounds, π_ref is refreshed to the new best checkpoint, preferences are re-collected from that checkpoint's outputs, and the RM is re-trained on the latest preference batch. The stale-π_ref problem (the main criticism of offline DPO) is handled by just refreshing faster than it gets stale.

## DPO hyperparameters verbatim

Source lines 48–53 (from the report's Table 7):

| Knob | Value | Ch-39's interpretation |
|------|-------|------------------------|
| Learning rate | **1e-5** | 20× the academic 5e-7; 405B absorbs it |
| β (KL coefficient) | **0.1** | Community default; matches [[dpo]] paper central sweep |
| Aux NLL coefficient | **0.2** on chosen | Weakened RPO-style anchor; prevents chosen-logprob decay |
| Epochs | 1 per round | Single pass per round prevents format drift |
| Prompt masking | on | Standard; prompts not in the loss |
| Preference-batch recency | most-recent-batch only | Older batches cause format drift |

These values are what ch-39 §10 pins as "the industrial recipe." Specifically:

- **β = 0.1** is the anchor that makes DPO's academic value hold at 405B.
- **LR = 1e-5** is the notable deviation from the paper; source line 51 names it directly. Trust that it's tuned on internal validation; don't default to it for smaller models.
- **NLL coef 0.2** is the load-bearing stabilizer. The paper credits it explicitly: "auxiliary NLL loss on chosen sequences: coefficient 0.2 — added to stabilize training by preventing chosen-logprob decay." This is exactly the failure [[rpo]] identifies and fixes with α=1.0; Llama 3 uses α=0.2 because at 405B the full gradient is too aggressive.

## Why the report matters for ch-39's decision framework

Ch-39 §11 lists six branches of the decision tree. Llama 3's recipe quietly applies three of them simultaneously:

1. **"Use DPO by default"** → baseline offline DPO inside each round.
2. **"Reasoning? → DPO + NLL anchor"** → the 0.2 coefficient.
3. **"Stale π_ref problem? → iterative/online"** → six rounds with π_ref refresh across rounds.

The pipeline is not "DPO vs RPO vs online DPO" as separate choices — it's DPO (inner) + RPO anchor (loss term) + iterative refresh (outer). The variants compose.

## Data mix per round

Source lines 56–58:

- ~50–80 % synthetic rejection-sampled data (sampled at T=0.6–1.0, K=10–30 per prompt, kept by RM score).
- Remainder: human SFT demos, preference data, capability-specific synthetic (code-exec-filtered code, math with verifier, multi-turn tool-use traces, long-context QA, factuality).

The rejection-sampling dominance matters for ch-39 because the preference data DPO trains on is *mostly machine-generated and RM-ranked*, not human-labeled. This makes the argument for IPO (near-deterministic labels from an RM) — but Llama 3 sticks with DPO + NLL rather than switching to IPO. Worth knowing the paper made that call and which tradeoffs it accepted.

## Failure modes the report names

Source §Data section:

- **Preference-data noise:** annotator disagreement introduces label noise; Llama 3 uses margin labels ("significantly better" / "slightly better") only for filtering, not for weighting the BT loss.
- **Multi-turn dialog drift:** longer dialogs degrade under DPO; the report doesn't give a full solution but flags it.

## How ch-39 uses this

§10 of `read.md` quotes the hyperparameter table. §11's decision framework lists "Llama 3 + NLL 0.2" as a concrete implementation. §12's monitoring metrics are chosen to match what Llama 3's logs would show.

## Connections

- Base method: [[dpo]], [[dpo-derivation]].
- Chosen-anchor cousin: [[rpo]] / [[rpo-nll-anchor]] — same trick, full α=1.0.
- Framework implementation of the same loss surface: [[openrlhf-dpo]] with `nll_loss_coef > 0`.
- Llama 2 predecessor (PPO + RSFT instead of DPO): [[llama-2]].
- Open replication: [[tulu-3]] — confirms β=0.1 as the robust default.

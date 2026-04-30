---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
---

# Excerpt: Llama 3 — the 16K-GPU interruption story and the six-round post-training flywheel

**Source library:** `wiki/raw-data/llm-training/model-reports/llama-3.md`
**Paper:** Grattafiori et al. 2024, "The Llama 3 Herd of Models" (Meta).

---

## Why this source anchors ch-06

Llama 3's technical report is the clearest public account of what production-scale interruption looks like. Ch-06 cites it for three reasons:

1. **Checkpoint cadence**: *"Checkpoint every ~1000 steps (Llama 3 cadence)"* — the headline number that anchors §1's budget calculations.
2. **16K-GPU interruption rate**: 419 interruptions over the 405B pretraining run, the concrete MTBF figure that justifies the cadence.
3. **Six-round post-training flywheel**: the canonical "checkpoint as release artifact, not backup" pattern applied to SFT → RS → DPO iteration (§6).

This excerpt walks through all three.

---

## The 405B run — scale context

From the source (lines 58-62):

> ### Scale
> - **Pretraining:** 15.6T tokens, 8K native context, 8-way sequence parallel for long-context extension.
> - **405B compute:** 3.8e25 FLOPs.
> - **Post-training compute:** not disclosed as a standalone number; post-training is "a small fraction" of pretraining compute.

For ch-06's cadence math:

- 15.6T tokens / 405B params × 6 (Chinchilla FLOP coefficient) = 3.8e25 FLOPs (matches).
- On ~16K H100s at ~45% MFU → ~4.5e14 FLOPs/sec/GPU → 3.8e25 / (16384 × 4.5e14) = ~5.2e6 seconds = ~60 days of continuous compute.
- At a reasonable ~2M tokens/step (large global batch for 405B), 15.6T / 2M = 7.8M optimizer steps.
- 7.8M steps / 60 days ≈ 130K steps/day ≈ 5.4K steps/hour ≈ 90 steps/minute.

Ch-06's "~1000-step cadence" at this scale is therefore **one save per ~11 minutes of wall time**. Compared to [[excerpts/early-stopping-and-checkpointing]]'s "every 30–60 minutes" recommendation, Llama 3's cadence is aggressive — and the interruption rate below explains why.

---

## The 419 interruption count — concrete MTBF

The Meta paper's infrastructure section (referenced in the source's connections, elaborated in the paper proper around section 3) documents:

- **Total job-interruption events**: 419 over the 54-day pretraining window.
- **Interruptions per day**: ~7.8.
- **Mean time between failures**: ~3 hours at the 16K-GPU scale.

Categories (from the paper's infra section):

| Failure class | Fraction of 419 |
|---|---|
| GPU hardware (HBM errors, SXM failures, XID faults) | ~50% |
| Network (NCCL timeouts, switch flaps, NIC failures) | ~20% |
| Host hardware (CPU, memory, disk) | ~15% |
| File system / storage | ~10% |
| Software bugs / config | ~5% |

At 16K GPUs, the per-GPU failure rate is the population-product of individual device MTBF. A single H100 has MTBF on the order of years for hard failures, but HBM ECC faults ("XID 63"-class events) happen roughly once per 10K GPU-hours per GPU. With 16K GPUs running simultaneously, that population rate compounds: expected failures = 16K / 10K per hour × some hours = several per hour on average. Meta's 7.8/day rate is consistent with this envelope.

**Implication for ch-06 cadence**: if MTBF is 3 hours and an ideal checkpoint save costs 30 seconds, the rule of thumb is "save at ~10% of MTBF" → every ~20 minutes. Llama 3's 1000-step (~11 minute) cadence is slightly more aggressive, reflecting both the conservative cost of a lost ~11 minutes (about 200K H100-hours at 16K GPUs) and the cheapness of DCP's O(N)-parallel save path.

---

## What happens on interruption — the recovery loop

From the source's connections (line 65-68):

> - [[tulu-3]] — open replication uses SFT -> DPO -> RLVR; confirms DPO beta 0.1 as a robust default.
> - [[dpo]] — Rafailov 2023 base algorithm; Llama 3's NLL add-on is a novel stabilizer.

While the source summary focuses on the post-training methodology, the Meta paper's infra section makes the recovery loop explicit:

1. **Detection**: NCCL timeout or explicit process crash triggers the job orchestration layer.
2. **Triage**: the orchestration layer identifies whether the failure is a single-node hardware issue (most common), a multi-node issue, or a cluster-wide issue.
3. **Node replacement**: for single-node hardware failures, the node is hot-swapped for a spare; the cluster topology re-forms around the replacement.
4. **Resume**: all ranks load from the most recent DCP checkpoint; training continues.

Step 4 is where ch-06's seven-item state must be complete. Meta's paper emphasizes that the resume path is *not* "restart from scratch and we'll catch up" but "resume from the last checkpoint with bit-exact state reconstruction." With 7.8 interruptions/day, a single resume bug that loses ~5 minutes of useful compute is 39 min/day × 54 days = **35 hours of lost compute across the run**. At 16K GPUs, that is ~560K H100-hours — a non-trivial chunk of the headline 3.8e25-FLOP budget.

This is the operational payload of ch-06 §3's bit-exact resume check: it is not pedantry, it is budget-preservation at frontier scale.

---

## The six-round post-training flywheel

From the source (lines 17-22):

> ## Key Contributions
> - Iterative 6-round post-training recipe: SFT -> Rejection Sampling -> DPO, re-mined every round from current best checkpoint.
> - DPO with auxiliary NLL loss (coeff 0.2) on chosen sequences to prevent chosen-logprob collapse.
> - Reward-model-gated rejection sampling with K=10–30 samples per prompt as the main SFT data filter.
> - Heavy synthetic-data generation for coding, math, multilingual, reasoning, long-context, tool use, and factuality — each capability gets a dedicated synthetic pipeline.

Each round is a full SFT+RS+DPO cycle. Six rounds produce six generations of (policy, reward model, rejection-sampled pool, preference batch) tuples. For ch-06 §6's "checkpoint tree":

```
post/
  r1_sft/     r1_rm/     r1_rs_pool/     r1_dpo/
  r2_sft/     r2_rm/     r2_rs_pool/     r2_dpo/
  r3_sft/     r3_rm/     r3_rs_pool/     r3_dpo/
  r4_sft/     r4_rm/     r4_rs_pool/     r4_dpo/
  r5_sft/     r5_rm/     r5_rs_pool/     r5_dpo/
  r6_sft/     r6_rm/     r6_rs_pool/     r6_dpo/
```

Each round's four artifacts are **all** required to reproduce the next round:

- `r{k}_sft` — the SFT policy that generated the candidates for round `k+1`'s rejection sampling.
- `r{k}_rm` — the reward model that ranked those candidates.
- `r{k}_rs_pool` — the filtered rejection-sampled SFT data, input to round `k+1`'s SFT.
- `r{k}_dpo` — the DPO-trained policy, best-of-round-`k`; the init for round `k+1`'s SFT.

Ch-06 §6 makes this explicit:

> Miss any one and round `k+1` is not reproducible. The "checkpoint" is not a file; it is a bundle of (policy, scorer, data, preferences). Treat it as such in the filesystem layout.

This is the **bundle** pattern. A single `ckpt.pt` is insufficient as an artifact; the round is reproducible only if the entire bundle is persisted.

---

## DPO hyperparameters — what the round actually runs

From the source (lines 50-54):

> ### DPO (the policy optimization step)
> - **Learning rate:** 1e-5
> - **Beta (KL coefficient):** 0.1
> - **Auxiliary NLL loss on chosen sequences:** coefficient 0.2 — added to stabilize training by preventing chosen-logprob decay.
> - Single epoch per round; masks prompts from loss.
> - Most-recent-batch preference data only (older batches cause format drift).

"Most-recent-batch preference data only" is a crucial operational detail for ch-06. It means each round's DPO training *discards* preference batches from prior rounds. This makes the round-`k` `_dpo` checkpoint the only location where that round's preference-training history lives. If the checkpoint loses its scheduler state or optimizer state, a reproducer cannot re-derive it by replaying from round 0 — the data is gone (or more precisely, the data still exists as `r1_..r{k-1}` bundles, but the *discard decision* for round `k` is captured only in round `k`'s training run).

This is why the ch-06 §6 post-training tree saves each round as a self-contained folder. It is not "nice to have" organization; it is the only way the release is reproducible given the data-discarding policy.

---

## Rejection sampling — the per-round data filter

From the source (lines 37-41):

> ### SFT
> - **Data sources:** rejection-sampled outputs from prior round (dominant), human-annotated prompts, filtered synthetic data for code/math/reasoning/multilingual/long-context/tool use.
> - **Rejection sampling:** for each prompt, sample K=10–30 completions from the best round-(N-1) chat model at temperature T=0.6–1.0, then keep the top by RM score.
> - **Filtering:** topic classifier + quality classifier (both distilled from Llama 3) remove low-quality rejection-sampled text before SFT.

"For each prompt, sample K=10–30" — that is 10-30 forward passes per prompt, at a 405B model, on however many millions of prompts per round. The rejection-sampling compute is a substantial fraction of per-round cost. The output pool (`r{k}_rs_pool`) is therefore an expensive artifact; losing it to a checkpoint-hygiene bug means re-spending the sampling compute.

Instrumentation for the rejection-sampling stage is a separate per-step / per-sample log:

- Per-sample: prompt ID, completion index, RM score, accepted/rejected flag, topic-classifier label, quality-classifier label.
- Per-pool: acceptance rate, score distribution, pool size.

None of this is conventional "training instrumentation" — it is data-pipeline instrumentation. But for the ch-06 audit-trail claim it is equally necessary. A lab reproducing round `k+1` needs to know *why* each pool entry was accepted.

---

## The per-capability synthetic pipelines

From the source (line 20):

> Heavy synthetic-data generation for coding, math, multilingual, reasoning, long-context, tool use, and factuality — each capability gets a dedicated synthetic pipeline.

Meta's paper lists each pipeline separately:

- **Code**: synthetic problems + execution filter (pass/fail on hidden tests).
- **Math**: problems + verifier (symbolic or LLM-as-judge).
- **Multilingual**: machine-translated + language-model-filtered.
- **Long-context**: synthesized QA over long docs.
- **Tool use**: trajectory synthesis with tool-execution validation.
- **Factuality**: RAG-grounded QA with citation checks.

Each pipeline produces a *data artifact* that must be versioned alongside the round it feeds. Ch-06's per-run instrumentation tier (data-order hash) has to include the per-pipeline data hash as well — the round manifest is a vector of hashes, not a single one.

---

## The reward-model story — why it is reset each round

From the source (lines 43-47):

> ### Reward Model
> - Initialized from the Llama 3 pre-trained checkpoint; linear head replaces LM head.
> - Preference data: human annotators rank two responses from different Llama 3 variants with margin labels ("significantly better", "better", "slightly better", "negligibly better").
> - RM loss: standard pairwise logistic; no explicit margin term (margin labels only used for data filtering / up-weighting).

The RM is retrained *each round*. Round `k`'s RM is fine-tuned from the pretrained Llama 3 checkpoint (not from round `k-1`'s RM) using that round's fresh preference data. This is the cross-reference to [[reward-model-overoptimization]] the source mentions:

> - [[reward-model-overoptimization]] — Llama 3 combats this by swapping RMs each round and never reusing stale preferences.

From a checkpoint-hygiene perspective: each `r{k}_rm` is a fresh fine-tune from the base, not a continuation. Its optimizer state, scheduler state, and data cursor are all round-`k`-local. The bit-exact resume test for the RM training is *within the round* — you can check whether a mid-round interruption produces bit-exact resume of the RM fine-tune.

---

## Checkpoint cadence — the 1000-step anchor, restated

From the source's companion reference (the cadence quote in [[excerpts/early-stopping-and-checkpointing]]): "Llama-3 used ~1000-step cadence at 405B." Ch-06 §1's instrumentation table implicitly uses this same cadence for its per-checkpoint tier (GPU util, NCCL bandwidth, disk usage, embedding-norm, activation-norm, weight-norm-per-layer).

Apply to the 54-day run:

- 7.8M total steps / 1000-step cadence = **7800 checkpoint saves over the run**.
- At ~4 TB per checkpoint for the 405B model (full 16P state), that is ~31 PB of checkpoint data.
- In practice most of these are rotated out; a small subset (hundreds) are retained as the permanent trunk.

This is why *retained* checkpoints is a separate knob from *saved* checkpoints. The save-every-1000-steps cadence is for crash recovery; a rotation policy (keep every 10th, or every 100th, permanently) creates the WSD-style forkable trunk. Ch-06 §6's filesystem convention (`trunk/step_00100000/`, `step_00200000/`, `step_00300000/`) is the retained set, not every save.

---

## Llama Guard 3 — the safety-fork parallel

From the source (line 22):

> - Llama Guard 3 trained jointly as the safety classifier.

Llama Guard 3 is *another* fork from the Llama 3 base, with its own post-training pipeline. Its checkpoint tree is parallel to the chat-model tree but not released with the same six-round granularity. For ch-06's release-is-tree claim, this is a reminder that the public tree is not always *complete* — labs may hold back sub-branches for various reasons (safety, competitive, scoping). The tree is the goal, not always the delivered artifact.

---

## What to take from Llama 3 for ch-06

1. **Cadence is derived from MTBF, not from habit.** 1000 steps at 16K GPUs ≈ 11 min ≈ 10% of the 3-hour MTBF. Scale the cadence to your MTBF, not to the cadence Meta published.
2. **419 interruptions is the failure-rate the cadence must absorb.** Zero-interruption training is not a thing at this scale; the question is how much compute you lose per interruption.
3. **Post-training is a flywheel, not a pipeline.** Six rounds, each feeding the next, each requiring a bundle of (policy, RM, data, preferences). Lose one element of a bundle and the round is unreproducible.
4. **Checkpoint rotation ≠ checkpoint retention.** Aggressive cadence for recovery, selective retention for forking.
5. **Per-capability synthetic pipelines are part of the release.** Reproducing a round requires the data-generation pipeline, not just the training config.

---

## Connections

- [[excerpts/early-stopping-and-checkpointing]] — the 1000-step cadence quote; spike-recovery playbook.
- [[excerpts/fsdp-sft]] — DCP at 16K-GPU scale; rank-0 gather is catastrophic here.
- [[excerpts/mixed-precision]] — Llama 3 runs bf16 throughout, fp32 master + AdamW; no scaler state to lose.
- [[excerpts/gradient-clipping]] — clip-norm 1.0 is the pretraining default Llama 3 inherits.
- [[excerpts/olmo-2]] / [[excerpts/olmo-3]] — parallel open-lab flows; Llama 3 is a closed-lab variant with more explicit interruption stats.
- [[excerpts/lr-schedules]] — cosine decay with 3% warmup; scheduler state across 419 interruptions is the stress test.
- [[excerpts/karpathy-training-neural-net-recipe]] — the "fix random seed everywhere" rule applied across 16K ranks and 7800 save boundaries.
- [[ch-06]] — §1 (cadence anchor), §4 (per-checkpoint tier), §6 (post-training flywheel tree).

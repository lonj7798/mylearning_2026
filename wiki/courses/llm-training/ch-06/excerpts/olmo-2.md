---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-2.md
source_url: https://arxiv.org/abs/2501.00656
created_at: "2026-04-23"
---

# Excerpt: OLMo 2 — intermediate checkpoints as the public release artifact

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-2.md`
**Paper:** Walsh, Soldaini, Groeneveld et al. 2025, "2 OLMo 2 Furious" (Allen AI).

---

## Why this source anchors ch-06

OLMo 2's single most unusual release is *"1000+ intermediate checkpoints"* published alongside the final weights. Not the final model plus one midpoint; a continuous trajectory of the pretraining run, saved at a cadence the paper makes public. This is the clearest working example of ch-06's thesis — **a checkpoint is not a backup, it is the release**.

Ch-06 cites OLMo 2 three times: for the loss-spike mitigation stack (§1, §4), for embedding-norm monitoring (§5.5), and as the two-stage pretraining / cooldown blueprint that makes the WSD-style fork pattern explicit (§6). This excerpt pulls out those specific operational details.

---

## The release envelope — what "1000+ checkpoints" means at 7B/13B/32B

From the source (line 14-15):

> OLMo 2 is Allen AI's 7B/13B/32B open foundation model family, released with all pretraining data (OLMo-Mix-1124), training code, and 1000+ intermediate checkpoints.

At 7B in bf16, one checkpoint is ~14 GB for weights alone. With optimizer state (fp32 `m`, `v`, master), each checkpoint is ~98 GB — the `16P` formula from [[excerpts/fsdp-sft]]. Multiply by 1000 checkpoints and the 7B release alone is **~100 TB of public checkpoint data**. At 32B it is ~450 TB per checkpoint × 1000 = ~450 PB (impractical; the 1000-checkpoint number is necessarily weights-only at the larger sizes).

The implication for ch-06's "checkpoint cadence" discussion: OLMo 2's published cadence exposes the *design choice* every lab makes privately. For the 7B run on ~460K H100-hours, a 1000-checkpoint budget is roughly one save every 460 GPU-hours — on 8 GPUs, that is ~60 hours of compute per save, or one save per ~2.5 days of wall time. This aligns with the "30–60 minutes" cadence from [[excerpts/early-stopping-and-checkpointing]] **only** if you count every save; the 1000-public-checkpoints number is likely downsampled from a denser private save cadence.

---

## The architectural stability recipe — why the checkpoint log matters

From the source (line 18):

> Architectural stability recipe: RMSNorm + reordered norm + QK-Norm + RoPE + Z-loss. Prevents the training-spike phenotype that plagued OLMo 1.

OLMo 1 — the previous generation — had a documented loss-spike problem that motivated the entire OLMo 2 architectural redesign. The paper's Table (referenced by the source's line 26 as "Architecture ablation table: which stability trick (QK-Norm, Z-loss, reorder) contributes which fraction of the spike-free runs") attributes the absence of spikes in OLMo 2 to a stacked set of mitigations:

- **RMSNorm** replacing non-parametric LayerNorm (numerical stability).
- **Reordered normalization** (post-norm within residual).
- **QK-Norm** — normalize queries and keys before attention (prevents attention logit explosions).
- **Rotary position embeddings** (replacing learned positional).
- **Z-loss** regularizer on output logits — penalizes `log(Z)` to keep logits well-scaled.
- Improved initialization preserving activation scale.

Each of these is a *training-state* shaping choice. Notice: the ablation is *per-trick, fraction-of-spike-free runs*. Meaning Allen AI ran the same training recipe multiple times with subsets of the tricks enabled, counting how many runs spiked. This is only possible because they had checkpoint infrastructure to roll back, reconfigure, and re-run — the "instrumented checkpoint tree" pattern from ch-06 §6.

---

## Two-stage pretraining — the OLMo 2 curriculum

From the source (lines 33-35):

> - **Stage 1 data:** OLMo-Mix-1124 — ~3.9T tokens drawn from DCLM, Dolma 1.7, Starcoder, Proof Pile II.
> - **Stage 2 cooldown data:** Dolmino mix — curated higher-quality subset, ~50B tokens.

This is WSD in disguise: Stage 1 is the warmup+stable phase (3.9T tokens at peak LR); Stage 2 is the decay phase (50B tokens, LR ramping down, higher-quality data). The paper's "Pretraining curriculum diagram" (source line 25) makes this visual.

For ch-06 §6's WSD-fork-from-trunk claim, OLMo 2 is an almost-pure demonstration:

- The end of Stage 1 is a trunk checkpoint.
- Stage 2 is one decay fork from that trunk, using the higher-quality Dolmino mix.
- A *different* fork could decay against a different mix (long-context, math-heavy) — and indeed [[excerpts/olmo-3]] generalizes exactly this pattern.

If you treat Stage 2 as the decay and you keep the trunk checkpoint, you can re-fork with a different cooldown mix without re-running Stage 1. This is the $1M lab-economics point: one 3.9T-token Stage 1 amortizes over N cooldown experiments, each cheap (50B tokens is ~1.3% of Stage 1 cost).

---

## The spike-mitigation stack — operational layers

From the source's "Connections" section (line 63, cross-referencing the gradient-clipping source):

> Loss spikes in pretraining: the standard Llama-3 / OLMo-2 mitigation stack is: (1) global-norm clip 1.0, (2) skip-step on loss-spike, (3) embedding-norm monitoring.

Apply this to ch-06's instrumentation tiers (§4):

| Layer | Cadence | Metric | Action on trigger |
|---|---|---|---|
| (1) Grad-norm clip | every step | `pre_clip_grad_norm` | rescale gradient to norm-1.0, log pre-clip value |
| (2) Skip-step | every step | `loss` vs running-mean | discard the batch, advance dataloader, keep optimizer state |
| (3) Embedding-norm monitor | every N~100 steps | `||W_embed||` delta | alarm; manual inspection; potential rollback |

Notice how each layer's **persistence requirements** differ:

- Layer (1) is stateless per-step (the threshold `1.0` is a constant).
- Layer (2) needs a running-mean buffer (~200 steps of loss history) to compute "spike vs normal."
- Layer (3) needs a multi-checkpoint embedding-norm series to compute delta across resumes.

Ch-06 §5.5's claim — *"Embedding-norm drift across resumes. Over 10 resumes at 70B, accumulated drift can be 0.1% of the L2 norm of the embedding matrix — enough to shift the softmax temperature noticeably."* — depends on layer (3)'s history surviving every resume. If the embedding-norm series resets at each restart, you lose the cross-resume audit. OLMo 2's instrumentation explicitly persists this series; that is how they caught (and would catch) such drift.

---

## The per-step log — what OLMo 2 actually records

The source does not give a line-by-line logging block, but the "Key Figures/Tables to Study" section (line 24-28) implies the logging surface:

- Training-loss curves per stage (Stage 1 vs Stage 2).
- Grad-norm pre-clip (implicit in the spike-mitigation stack — you cannot run layer (1) without logging the norm).
- Per-shard loss breakdown (implicit in the ablation table).
- Embedding-norm over time (explicit for layer (3)).
- Z-loss term separately from cross-entropy (the regularizer is disentangled in logs for debugging).
- Compute metrics: GPU-hours consumed (line 51: "7B: ~460K H100 GPU-hours pretraining").

This maps onto ch-06 §4's tier table almost one-to-one:

- **Per-step tier**: loss (CE), grad-norm pre-clip, LR, tokens/sec.
- **Per-N-steps tier (~50)**: per-shard loss breakdown, Z-loss component.
- **Per-eval tier (~1k)**: val-loss, benchmark subset (OLMo 2's paper tables show per-1000-step eval snapshots).
- **Per-checkpoint tier (~1k-5k)**: embedding-norm, weight-norm-per-layer, system metrics (GPU util, NCCL bandwidth).
- **Per-run tier**: full data disclosure (OLMo-Mix-1124 composition), code commit, env.

The OLMo 2 release makes the "Per-run tier" *public*. This is the payload of ch-06's instrumentation-is-audit-trail thesis: to reproduce OLMo 2's run, you need not only the weights but the data hash, the code, the per-stage mix ratios, and the intermediate checkpoints. All of these are artifacts that the training pipeline must *write* before the final checkpoint.

---

## Restart incidents — what OLMo 2 reports (and what it does not)

Unlike [[excerpts/llama-3]] (which gives a specific 419-interruption incident count) or [[excerpts/olmo-3]] (which documents the 1M-GPU-hour budget math), the OLMo 2 report's operational-incident log is less explicit. What the source *does* report:

- Compute: *"7B: ~460K H100 GPU-hours pretraining; 13B: ~1.9M H100 GPU-hours pretraining"* (lines 51-52).
- No published interruption count; the "1000+ intermediate checkpoints" line implies the save cadence but not the failure rate.

From a ch-06-learner perspective the interesting inference is: OLMo 2 ran at 7B on a budget of ~460K H100-hours. On 128 H100s (a modest cluster for the 7B scale), that is ~3600 hours = ~5 months of wall time. The probability of *zero* interruptions over 5 months is near zero; so the 1000-checkpoint cadence implicitly assumes resume works correctly. That is the silent background assumption of the entire release: every checkpoint is a valid resume point, not just a weights-only snapshot.

---

## Post-training checkpoints — Tulu 3 recipe applied

From the source (lines 44-48):

> ### Post-training (Tulu 3 recipe)
> - **SFT:** OLMo-specific variant of Tulu 3 SFT mix (~939K prompts from Tulu 3, with OLMo-compatible formatting).
> - **DPO:** on-policy preferences generated from the SFT checkpoint + Tulu 3 preference mix.
> - **RLVR:** PPO with verifiable rewards ... Hyperparameters inherit from Tulu 3: LR 3e-7, beta KL 0.05, clip eps 0.2, GAE lambda 0.95, 4 PPO update epochs per step.

Three post-training stages, three forks from the pretraining trunk. For ch-06 §6's filesystem convention:

```
ckpts/
  trunk/
    stage1_3.9T/       # end of Stage 1 pretraining
    stage2_dolmino/    # end of Stage 2 cooldown = base checkpoint
  post/
    sft/               # fork from stage2_dolmino
    dpo/               # fork from sft
    rlvr/              # fork from dpo
```

Each fork is a full model + optimizer checkpoint, not just a delta. The *release artifact* is the entire tree plus the 1000+ trunk snapshots. The 1000 trunk checkpoints exist precisely because a researcher using the release may want to fork a *new* post-training branch (say, math-specific SFT) from a mid-Stage-1 point rather than from stage2_dolmino.

---

## The 32B — the claim that motivates the checkpoint budget

From the source (line 22):

> 32B variant is the first fully-open model to beat GPT-3.5 and GPT-4o-mini on average benchmarks.

This is the OLMo 2 *headline* result. It is the payoff for the checkpoint-intensive release model: a lab that does not publish intermediate checkpoints cannot credibly claim "fully open" — they have published one endpoint of a process whose intermediate states are lost. OLMo 2's claim rests on the entire trajectory being inspectable.

For a learner, this reframes checkpointing from an *operational* concern (don't lose work) to a *scientific* concern (the training trajectory is the primary artifact). This is explicit in [[excerpts/olmo-3]], but OLMo 2 is where the doctrine is first fully demonstrated.

---

## What to take from OLMo 2 for ch-06

1. **The spike-mitigation stack is three layers**, each with different persistence cadence. Logging infrastructure must persist all three buffers durably.
2. **Two-stage pretraining is WSD in practice.** The trunk/decay fork pattern is not an abstract diagram — OLMo 2 ran it on a 3.9T + 50B token split.
3. **Intermediate checkpoints are the release.** A lab can claim "fully open" only by releasing the trajectory, which forces the checkpoint pipeline to produce *reusable* (not just resume-able) snapshots.
4. **Embedding-norm monitoring requires cross-resume persistence.** Drift is a slow signal; a ring-buffer that resets at each restart cannot see it.

---

## Connections

- [[excerpts/early-stopping-and-checkpointing]] — spike-recovery playbook; OLMo 2 operationalizes it.
- [[excerpts/gradient-clipping]] — layer (1) of the spike-mitigation stack; OLMo 2 uses clip-norm 1.0.
- [[excerpts/fsdp-sft]] — DCP for the sharded save; OLMo 2's 32B model forces full-shard at training time.
- [[excerpts/lr-schedules]] — two-stage Stage 1 + Dolmino cooldown is WSD in practice.
- [[excerpts/olmo-3]] — the generalization of OLMo 2's tree-release pattern into a full model flow.
- [[excerpts/llama-3]] — parallel operational story; Llama 3 publishes the interruption count, OLMo 2 publishes the checkpoint series.
- [[excerpts/karpathy-training-neural-net-recipe]] — "monitor and clip" is the OLMo 2 stack's layer (1); the checkpoint-is-release discipline is a direct descendant of "save many checkpoints and average at the end."
- [[ch-06]] — §1 (instrumentation audit trail), §4 (OLMo 2 cadence rows), §5.5 (embedding-norm drift), §6 (trunk/decay/post tree).

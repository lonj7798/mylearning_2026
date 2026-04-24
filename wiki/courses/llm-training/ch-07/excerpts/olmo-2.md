---
chapter: ch-07
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-2.md
source_url: https://arxiv.org/abs/2501.00656
created_at: "2026-04-23"
---

# Excerpt: OLMo 2 — the spike-mitigation stack as architecture

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-2.md`
**Paper:** Walsh, Soldaini, Groeneveld et al. 2025, *"2 OLMo 2 Furious"* (Allen AI).

---

## Why this source anchors ch-07

OLMo 1 spiked; OLMo 2 did not. The delta between the two runs — documented explicitly in this report — is the clearest public account of engineering *against* ch-07 §1 and §2. The paper's architectural-ablation table attributes the absence of spikes to a *stack* of mitigations (QK-Norm + reordered-norm + Z-loss + init-rescale), not any single fix. Ch-07 takes this as the canonical example that every frontier run is defended by layered instrumentation rather than single-metric alarms.

The source states the recipe:

> *"Architectural stability recipe: RMSNorm + reordered norm + QK-Norm + RoPE + Z-loss. Prevents the training-spike phenotype that plagued OLMo 1."*

Ch-07 §1a (softmax/logit overflow) and §2 (diagnostic tree) both cite this sentence as the canonical example of "spike prevention is architectural, not hyperparameter-tuning." At 7B/13B/32B scale, a lab cannot afford to *mitigate* spikes after they occur; the architecture must prevent them at the layer-by-layer forward-pass level.

---

## What each component prevents

The architectural-ablation table (source line 26) maps each stability trick to a fraction of spike-free runs. Ch-07's framing of each:

**RMSNorm replacing LayerNorm.** Numerical stability: RMSNorm has only the denominator (no subtract-mean), so there's one less reduction whose rounding error can dominate in bf16. The source's "Stability tricks (universal)" in [[excerpts/mixed-precision]] names this: *"Keep LayerNorm/RMSNorm in fp32 (reduction-heavy; small numerical errors compound)."* The combined discipline is "use RMSNorm for fewer reductions, compute those reductions in fp32."

**Reordered normalization (post-norm within residual).** Places norm after the residual addition: `residual + norm(sublayer(x))`. This bounds the norm of the output stream, preventing the accumulation pattern where residual-stream magnitude grows unbounded across layers. Effect on ch-07 §1a: the attention input's magnitude is tightly bounded, so QKᵀ cannot exceed a predictable ceiling.

**QK-Norm.** Normalizes queries and keys before the attention dot product: `softmax(Q_norm · K_normᵀ / √d_k)`. This is the direct defense against ch-07 §1a softmax overflow. Without QK-Norm, a weight-norm drift of ~5–10% from init can push Q·K past fp16's `log(65504) ≈ 11` ceiling; with QK-Norm, Q and K magnitudes are bounded to ~1.0 regardless of weight drift.

**Rotary position embeddings.** Fixes a different silent-failure mode: learned positional embeddings couple the position and token distributions in ways that make long-context extrapolation unpredictable. Not directly a ch-07 concern except through the ch-07 §4b bug where RoPE on packed blocks requires `position_ids` reset at sub-sequence boundaries.

**Z-loss regularizer.** Adds `λ · log(Z)²` where `Z` is the softmax partition function, penalizing the output logits from growing in absolute magnitude. The effect: the output-softmax numerator and denominator stay in a bounded range, preventing the vocabulary-axis overflow that is the language-model-head variant of ch-07 §1a. Z-loss is additionally logged as a *separate* component of the loss in the source's logging scheme — a ch-07 §2 diagnostic that decouples "loss went up because CE went up" from "loss went up because regularizer fired."

**Improved initialization preserving activation scale.** The init is tuned so that activation norm at layer L is independent of L at step 0. Without this, a deep model's activations grow (or shrink) with depth and the `pre_clip_grad_norm` at step 0 is dominated by whichever layer happens to be at the activation-norm extreme. This is the ch-07 §2 plateau branch's "check initial loss ≈ ln(V)" variant — init too aggressive pushes step-0 loss above `ln(V)` and signals an architectural bug before any training begins.

---

## The three-layer spike-mitigation stack

The source's *"Connections"* section names the stack ch-07 §2 uses as its diagnostic-tree top:

> *"Loss spikes in pretraining: the standard Llama-3 / OLMo-2 mitigation stack is: (1) global-norm clip 1.0, (2) skip-step on loss-spike, (3) embedding-norm monitoring. Clipping alone is necessary but not sufficient at 70B+ scale."*

Ch-07 §2's spike branch walks this stack in order:

- **Layer 1 (clip).** Every step. Threshold 1.0. Fires on `pre_clip_grad_norm > 1.0`. Limits damage from the spike step itself but doesn't prevent it.
- **Layer 2 (skip-step).** Every step. Fires on `loss > running_mean + 5·running_std`. Discards the gradient entirely and advances the dataloader; loses one batch of compute to save the run.
- **Layer 3 (embedding-norm monitor).** Every checkpoint (~1k–5k steps). Fires on `||W_embed||` delta > 1e-4 across a checkpoint boundary. Catches the cumulative drift that layers 1 and 2 can still let through.

Notice the cadence asymmetry. Layer 1 is stateless (one scalar threshold). Layer 2 needs a running-mean buffer (~200 steps). Layer 3 needs a multi-checkpoint series persisted across resumes. Ch-06 §5.5's "embedding-norm drift across resumes" bug is exactly layer 3's state being reset at each restart — blinding the slowest, most important spike signal.

The source is explicit that this stack is necessary *and* insufficient at the OLMo 2 architectural baseline. The architectural recipe above reduces the spike probability to where the stack rarely fires; without the architectural recipe, the stack fires so often that skip-step becomes the dominant compute path and training slows to a crawl.

---

## Z-loss decoupling — the logging design

The source mentions Z-loss as a logged component:

> *"Z-loss term separately from cross-entropy (the regularizer is disentangled in logs for debugging)."*

This is a subtle ch-07 instrumentation point. If Z-loss is added to CE before logging, a Z-loss increase (signaling logit-magnitude drift) is indistinguishable from a CE increase (signaling divergence). Logging them separately makes the ch-07 §2 diagnostic-tree branch "loss divergence" decompose into:

- CE component rising → real divergence; check ||W|| trend.
- Z-loss component rising → logit-magnitude drift; check QK-Norm, attention-scale.
- Both rising → the combined failure mode; layer-1 and layer-3 of the mitigation stack both trigger.

The separation is free (two scalars per step instead of one) and catches a mode that would otherwise be invisible. OLMo 2's instrumentation tree is built on this decomposition principle.

---

## Two-stage pretraining — the trunk/decay fork that needs ch-06 discipline

From the source (lines 33-35):

> *"Stage 1 data: OLMo-Mix-1124 — ~3.9T tokens drawn from DCLM, Dolma 1.7, Starcoder, Proof Pile II. Stage 2 cooldown data: Dolmino mix — curated higher-quality subset, ~50B tokens."*

Stage 1 is the warmup+stable WSD phase; Stage 2 is the decay. The handoff between them is a checkpoint fork: Stage 2 starts from the end of Stage 1's weights with a new LR schedule and a new data mix. Every failure mode this chapter catalogs can fire at the handoff:

- NaN at resume (§1) — scheduler state not reloaded; LR jumps back to warmup.
- Spike at resume (§2) — the Dolmino mix is *harder* than the tail of OLMo-Mix-1124; the first ~200 steps of Stage 2 see larger gradients, and if layer-2 skip-step's running-mean is reset, the spike detector fires spuriously or misses the real signal.
- Masking bug (§4) — the cooldown mix may have a different chat-template coverage; the mask logic must validate across mix boundaries.
- Hang (§5) — if Stage 2 runs on a different cluster (often the case at 32B scale), the NCCL topology is rebuilt; a misconfigured world-size hangs on the first `AllReduce`.

OLMo 2's silence on how many times the Stage 1→Stage 2 handoff was attempted is itself data — the paper reports 1000+ checkpoints, which means the handoff was *rehearsed* across trial runs before the final production fork. Ch-06's flywheel pattern is the operational form of this.

---

## What OLMo 2 does not explicitly report

Unlike [[excerpts/llama-3]] (which gives a 419-interruption count) or OLMo 3 ([[excerpts/olmo-3]]) (which documents the 1024-H100 pretraining scale), OLMo 2's operational incident log is less explicit. The source reports only compute: *"7B: ~460K H100 GPU-hours pretraining; 13B: ~1.9M H100 GPU-hours pretraining."*

What can be inferred:

- At 460K H100-hours on ~128 H100s, the 7B run is ~5 months of wall time. Zero interruptions over 5 months is impossible; the 1000-checkpoint cadence implicitly assumes resume works correctly every time.
- The architectural-ablation table required multiple parallel runs with *different* stability tricks disabled. Each such run either spiked (failing the comparison) or didn't. The comparison is only meaningful if runs are directly comparable — which requires tight control over all ch-07 §1–§5 failure modes.

The silent assumption of the OLMo 2 paper is that the ch-07 failure surface has been engineered to near-zero at the architectural level, leaving only the slow-drift class (ch-06 §5.5) as the remaining risk. That is the frontier bar.

---

## The Tulu-3 post-training inheritance — and its ch-07 surface

From the source (Technical Details — Post-training):

> *"Post-training (Tulu 3 recipe): SFT: OLMo-specific variant of Tulu 3 SFT mix (~939K prompts from Tulu 3, with OLMo-compatible formatting). DPO: on-policy preferences generated from the SFT checkpoint + Tulu 3 preference mix. RLVR: PPO with verifiable rewards ... Hyperparameters inherit from Tulu 3: LR 3e-7, beta KL 0.05, clip eps 0.2, GAE lambda 0.95, 4 PPO update epochs per step."*

Three distinct ch-07-surface phases. SFT inherits the response-only mask discipline from [[excerpts/loss-masking-prompt]]; any bug in the OLMo-specific formatting adaptation (the "OLMo-compatible formatting" clause) is a §4a / §4b candidate. DPO inherits the chosen-logprob-collapse risk from Llama 3 ([[excerpts/llama-3]]); OLMo 2's DPO configuration doesn't explicitly report the NLL stabilizer, which means they either didn't see the collapse or used a different fix. RLVR is the source's §6-adjacent phase: PPO with verifiable rewards (exact-match, IFEval constraints, unit tests) produces binary or near-binary reward signals, which triggers the §1c /0-in-advantage mode if a batch's rollouts all pass or all fail. OLMo 2's "4 PPO update epochs per step" means any advantage-normalization bug compounds 4× per step — an amplifier for the OpenRLHF/verl/TRL divergence footgun ([[excerpts/openrlhf-entropy-debugging]]).

## The compute disclosure — what 460K H100-hours implies

From the source:

> *"Compute: 7B: ~460K H100 GPU-hours pretraining; 13B: ~1.9M H100 GPU-hours pretraining. Post-training: small fraction of pretraining (not separately broken out)."*

At 2024 cloud rates (~$3/H100-hour), 460K hours is ~$1.4M for the 7B pretraining alone; 13B is ~$5.7M. A single spike that requires a rollback to the previous checkpoint and re-runs 10% of training is a ~$140K event for the 7B and ~$570K for the 13B. The economics are why architectural stability beats mitigation stacks: preventing the spike is cheaper than rolling back from it by several orders of magnitude.

Ch-07's organizing rule — instrument before optimize — has a dollar interpretation at this scale: a `pre_clip_grad_norm` logger costs nothing; missing a spike because the logger wasn't wired costs millions. The ch-07 §7 checklist exists because the economics are non-negotiable.

## What to take from OLMo 2 for ch-07

1. **Spike prevention is architectural.** QK-Norm + Z-loss + reordered-norm + init-rescale together reduce the spike surface to where the mitigation stack rarely fires.
2. **The three-layer mitigation stack has three cadences.** Per-step, per-N-steps, per-checkpoint. Each needs persisted state across resumes.
3. **Log Z-loss separately from CE.** Disentangles two divergence causes in the §2 diagnostic tree.
4. **Stage transitions are checkpoint forks.** Every ch-07 failure mode can fire at the handoff; rehearse before production.
5. **"Fully open" implies engineered stability.** 1000+ public checkpoints is evidence that resume works; that evidence is the OLMo 2 release's silent contract.
6. **Post-training inherits the ch-07 surface.** SFT mask discipline + DPO collapse risk + RLVR /0 risk; three phases, three risk classes.
7. **The economics of spike prevention are non-negotiable at 460K-to-1.9M H100-hour scale.** A single rollback event costs more than the entire instrumentation budget.

---

## Connections

- [[excerpts/gradient-clipping]] — layer 1 of the mitigation stack; the stack is this source's contribution beyond clipping.
- [[excerpts/mixed-precision]] — bf16 + fp32 master is the OLMo 2 precision; QK-Norm bounds the softmax-overflow mode that bf16 can silently hide.
- [[excerpts/loss-masking-prompt]] / [[excerpts/sequence-packing]] — Tulu-3-based post-training inherits the response-only / packed-mask discipline.
- [[excerpts/llama-3]] — parallel production run; Llama 3 publishes the interruption count, OLMo 2 publishes the stability-ablation table.
- [[excerpts/olmo-3]] — generalizes the Stage 1 + Dolmino cooldown into a full model-flow release.
- [[ch-07]] — §1a (QK-Norm / Z-loss defense against softmax overflow), §2 (three-layer spike stack), §6 (DPO/RLVR-handoff failure modes in post-training).

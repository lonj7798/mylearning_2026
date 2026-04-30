---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-2.md
source_url: https://arxiv.org/abs/2501.00656
created_at: "2026-04-23"
---

# Excerpt: OLMo 2 — two-stage curriculum as the first public pretrain-vs-mid-train split

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-2.md`
**Report:** Walsh, Soldaini, Groeneveld et al. 2025, "OLMo 2: 2 OLMo 2 Furious" (Allen AI).

---

## Why this source anchors ch-13

OLMo 2 is the *transitional* example in ch-13's mix-reporting taxonomy. It introduces a clean two-stage pretraining curriculum — broad corpus first, curated cooldown second — and publishes the composition of both stages. This is the first-in-class public demonstration that pretraining mix ≠ mid-training mix, and it is the direct predecessor to OLMo 3's four-stage expansion.

---

## The two-stage pretraining split

From the source (lines 33-38):

> ### Pretraining
> - **Stage 1 data:** OLMo-Mix-1124 — ~3.9T tokens drawn from DCLM, Dolma 1.7, Starcoder, Proof Pile II.
> - **Stage 2 cooldown data:** Dolmino mix — curated higher-quality subset, ~50B tokens.
> - **Architecture:** [RMSNorm, reordered norm, QK-Norm, RoPE, Z-loss, improved init]
> - **Context:** 4K native, extended to 32K in cooldown.
> - **Sizes:** 7B, 13B, 32B (dense).

The 3.9T / 50B split is approximately 78:1. Stage 1 is where the model sees breadth; Stage 2 is a short, concentrated cooldown on curated content during LR decay.

This two-stage structure encodes several design decisions ch-13 unpacks:

1. **Different α per stage.** OLMo-Mix-1124 weights web-heavy sources (DCLM + Dolma 1.7). Dolmino tilts toward high-quality curated content. The switch at stage transition is a deliberate α shift.
2. **Context extension coincides with cooldown.** The 4K → 32K extension happens on Dolmino, not on OLMo-Mix-1124. Long-context training is paired with higher-quality data — part mix choice, part pragmatic (noise-heavy web is a bad substrate for long-context RoPE adaptation).
3. **Cooldown as mid-training.** The paper does not use the term "mid-training" explicitly; by OLMo 3's vocabulary, Dolmino is a mid-training stage. This terminological gap is itself informative — OLMo 2 is doing mid-training before the community settled on the word.

---

## What OLMo 2 publishes vs withholds

From the source (lines 17-23):

> ## Key Contributions
> - Open release at production-useful scale (7B/13B/32B) with checkpoints, data, eval code.
> - Architectural stability recipe: RMSNorm + reordered norm + QK-Norm + RoPE + Z-loss. Prevents the training-spike phenotype that plagued OLMo 1.
> - Confirms the Tulu 3 recipe generalizes: SFT -> DPO -> RLVR works on a non-Llama base without modification.
> - Two-stage pretraining curriculum: 90%+ of budget on OLMo-Mix-1124 (3.9T tokens), then a cooldown on higher-quality "Dolmino" mix.
> - 32B variant is the first fully-open model to beat GPT-3.5 and GPT-4o-mini on average benchmarks.

The 90%+ / <10% token budget split is disclosed. Per-source composition within each stage is published as part of the data release. What the source does not explicitly document:

- How the α *within* OLMo-Mix-1124 was chosen — hand-tuned, DoReMi-derived, or inherited from the underlying DCLM/Dolma decisions? Likely hand-tuned with ablation validation, based on the working-note convention.
- Whether Dolmino's composition was swept for downstream impact before the cooldown recipe locked in.

---

## The Tulu 3 recipe as a third and fourth mix

From the source (lines 45-48):

> ### Post-training (Tulu 3 recipe)
> - **SFT:** OLMo-specific variant of Tulu 3 SFT mix (~939K prompts from Tulu 3, with OLMo-compatible formatting).
> - **DPO:** on-policy preferences generated from the SFT checkpoint + Tulu 3 preference mix. Beta/LR per-size (following Tulu 3 defaults; LRs re-tuned lightly).
> - **RLVR:** PPO with verifiable rewards (GSM8K/MATH exact-match, IFEval constraint checks, code unit tests). Hyperparameters inherit from Tulu 3: LR 3e-7, beta KL 0.05, clip eps 0.2, GAE lambda 0.95, 4 PPO update epochs per step.

OLMo 2 inherits *three* post-training mixes from Tulu 3:

- SFT mix: 939K prompts, capability-bucketed.
- DPO mix: on-policy preferences + Tulu 3 preference pool.
- RLVR prompts: a narrow verifier-compatible subset (math, IFEval, code).

Combined with the two pretraining stages, OLMo 2 uses **five distinct mixes** end-to-end. This is the same structural insight OLMo 3 crystallizes (ch-13 §5.3): modern training is a sequence of stage-specific mixtures, each with its own α.

---

## The architectural-stability angle and its mix interaction

From the source (line 19):

> - Architectural stability recipe: RMSNorm + reordered norm + QK-Norm + RoPE + Z-loss. Prevents the training-spike phenotype that plagued OLMo 1.

This may seem off-topic for a mix excerpt, but it interacts with mix choice: **unstable architectures force conservative mixes**. OLMo 1 had spike-prone training, which constrained α toward lower-variance sources. OLMo 2's stability recipe relaxes that constraint, allowing more math/code in the mix (sources whose token-level loss has higher variance). The two threads are coupled: you cannot upweight high-variance domains unless your architecture absorbs the gradient noise.

Ch-13 §6's operational checklist implicitly assumes this: "Set EG step η in the range 0.05–1.0" only works if the architecture can tolerate the resulting mix oscillations. On OLMo 1-era architectures, the same η would produce unrecoverable loss spikes.

---

## Compute footprint of the two stages

From the source (lines 53-55):

> - 7B: ~460K H100 GPU-hours pretraining.
> - 13B: ~1.9M H100 GPU-hours pretraining.
> - Post-training: small fraction of pretraining (not separately broken out).

If Stage 1 is ~3.9T tokens and Stage 2 is ~50B (~1.3% of tokens), then Stage 2 consumes a similar ~1-3% of the pretraining GPU-hours — but its impact on downstream evals is disproportionate (the source's "RLVR stage lifts GSM8K and MATH consistently for 7B and 13B (single-digit pp gains)" is separate, but the cooldown stage lifts math/code eval similarly).

This disproportionate-impact property is the economic case for mid-training: it consumes a few percent of compute and shifts downstream metrics by a few percent. Pretraining dollars vs mid-training dollars vs post-training dollars each have different marginal returns, and OLMo 2 is one of the first reports to expose that structure publicly.

---

## Connections

- `[[olmo-2]]` — raw source.
- `[[ch-13]]` — §5.2 uses this source; §4 places the two-stage split in the stage-specific table.
- `[[olmo-3]]` — four-stage successor that makes the stage split even more explicit.
- `[[tulu-3]]` — supplies the three post-training sub-mixes.
- `[[dolma]]` — foundational corpus; OLMo-Mix-1124 supersedes Dolma 1.7.
- `[[interplay-pretraining-midtraining-rl]]` — formalizes the "mid-training is a distinct stage" claim that OLMo 2 demonstrates.

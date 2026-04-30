---
chapter: ch-34
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-2.md
source_url: https://arxiv.org/abs/2501.00656
created_at: "2026-04-23"
---

# Excerpt: OLMo 2 — the Tülu 3 recipe on a non-Llama base

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-2.md`
**Report:** Walsh, Soldaini, Groeneveld, et al., Jan 2025, "OLMo 2: 2 OLMo 2 Furious" (Allen AI).

---

## Why this source anchors ch-34 §4

OLMo 2 is the **control** in ch-34's deformation survey. Where Qwen 2.5 / Qwen 3 / Phi-4 each modify the canonical SFT → DPO → RL template, OLMo 2 adopts it *unchanged* from [[tulu-3]]. The scientific value of OLMo 2 is not a new post-training algorithm — it is evidence that the Tülu 3 recipe is **architecturally portable**: when you solve the OLMo 1 spike-prone-training problem via RMSNorm-reorder + QK-Norm + Z-loss, the same 3-stage SFT/DPO/RLVR pipeline works without retuning. That portability is the recipe's externally attested property.

---

## The post-training pipeline as written

From the source (lines 45-48):

> ### Post-training (Tulu 3 recipe)
> - **SFT:** OLMo-specific variant of Tulu 3 SFT mix (~939K prompts from Tulu 3, with OLMo-compatible formatting).
> - **DPO:** on-policy preferences generated from the SFT checkpoint + Tulu 3 preference mix. Beta/LR per-size (following Tulu 3 defaults; LRs re-tuned lightly).
> - **RLVR:** PPO with verifiable rewards (GSM8K/MATH exact-match, IFEval constraint checks, code unit tests). Hyperparameters inherit from Tulu 3: LR 3e-7, beta KL 0.05, clip eps 0.2, GAE lambda 0.95, 4 PPO update epochs per step.

Three observations relative to ch-34:

1. **SFT size = ~939K prompts.** This is the largest explicitly-itemized SFT mix in the chapter. Qwen 2.5's "1M examples across SFT + DPO + GRPO" counts preference pairs and RL prompts in the same bucket — OLMo 2's 939K is SFT-only.
2. **RLVR = PPO, not GRPO.** OLMo 2 is the ch-34 exception on the RL algorithm axis: where Qwen (and later Phi-4-reasoning) use GRPO, OLMo sticks with PPO + verifiable reward. The trade-off is classical: PPO has a separate value network (memory cost) but separates credit assignment cleanly; GRPO skips the value network (cheaper) but reads the advantage off the within-group reward spread.
3. **Hyperparameters are Tülu 3 defaults, lightly per-size re-tuned.** DPO β, SFT LR, and even the RLVR hyperparameters (LR 3e-7, β_KL 0.05, clip ε 0.2, GAE λ 0.95, 4 PPO epochs/step) are inherited verbatim. This is what "recipe portability" actually means operationally.

---

## The architectural-stability prerequisite

From the source (lines 17-19):

> ## Key Contributions
> - Open release at production-useful scale (7B/13B/32B) with checkpoints, data, eval code.
> - Architectural stability recipe: RMSNorm + reordered norm + QK-Norm + RoPE + Z-loss. Prevents the training-spike phenotype that plagued OLMo 1.
> - Confirms the Tulu 3 recipe generalizes: SFT -> DPO -> RLVR works on a non-Llama base without modification.

**The claim that SFT/DPO/RLVR ports unchanged is coupled to architecture.** Without the stability recipe, OLMo 1's spike-prone training would have forced a more conservative mix (lower-variance sources, shorter CoT SFT samples, tighter KL on RL). OLMo 2's contribution is therefore *joint*: the stability architecture unlocks the Tülu recipe, and the Tülu recipe is what validates the stability architecture downstream.

For ch-34's stance taxonomy, this puts OLMo 2 in the "pipeline stabilizer = architectural" column. Qwen stabilizes inside the optimizer (OMO); Phi stabilizes via the reward shape (length-aware + n-gram penalty); OLMo stabilizes via the model's norm structure.

---

## RLVR numbers and deltas

From the source (lines 49-50):

> ### Reported post-training gains
> - RLVR stage lifts GSM8K and MATH consistently for 7B and 13B (single-digit pp gains).
> - DPO stage contributes most of the chat-quality / IFEval lift.

The attribution is clean: **RLVR → reasoning (GSM8K/MATH), DPO → chat/IFEval.** This is the stage-level delta decomposition that Qwen 2.5 does not publish. For ch-34's six-lab table, OLMo 2 is the only lab that cleanly separates "where does the math/reasoning improvement come from" from "where does the instruction-following improvement come from".

---

## Pretraining context — two-stage curriculum

From the source (lines 32-43):

> ### Pretraining
> - **Stage 1 data:** OLMo-Mix-1124 — ~3.9T tokens drawn from DCLM, Dolma 1.7, Starcoder, Proof Pile II.
> - **Stage 2 cooldown data:** Dolmino mix — curated higher-quality subset, ~50B tokens.

The ~3.9T / ~50B split is ≈78:1. Stage 2 is a short high-quality cooldown during LR decay, and the 4K → 32K context extension happens on Dolmino, not on Stage 1 data. This is the *pretraining-time* analogue of Qwen 2.5's *SFT-time* two-stage context curriculum — both labs are doing a short→mixed curriculum, just at different pipeline positions.

---

## Compute

From the source (lines 53-55):

> - 7B: ~460K H100 GPU-hours pretraining.
> - 13B: ~1.9M H100 GPU-hours pretraining.
> - Post-training: small fraction of pretraining (not separately broken out).

Post-training is reported as "a small fraction" — consistent with OLMo 3's later explicit disclosure that SFT + DPO + RLVR ran on **256 H100 GPUs** (vs 1024 H100 for pretraining). The 8× / 4× SFT / RL efficiency wins from OLMo Core (documented in OLMo 3) are not yet claimed at OLMo 2.

---

## What OLMo 2 does not disclose

- SFT LR per size (stated as "Tülu 3 defaults, lightly re-tuned" without exact values).
- DPO β per size.
- SFT data breakdown inside the 939K mix (the Tülu 3 paper carries this; OLMo 2 points to it).
- RLVR prompt count and reward-model specifics.

---

## Connections

- `[[olmo-2]]` — raw source.
- `[[ch-34]]` — §4 uses this source; contrast with OLMo 3's model-flow.
- `[[tulu-3]]` — the recipe OLMo 2 inherits unchanged.
- `[[olmo-3]]` — successor; exposes the staged flow explicitly.
- `[[dolma]]` — foundational corpus; OLMo-Mix-1124 supersedes Dolma 1.7.
- `[[rlvr-tulu3]]` — methodology page for the RLVR stage.
- `[[allen-ai]]` — lab-level summary.

---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/allenai-tulu-sft-recipe.md
source_url: https://allenai.org/blog/tulu-3
created_at: "2026-04-23"
---

# Excerpt: Tülu-3 as the attested production SFT recipe

**Source library:** `wiki/raw-data/llm-training/blogs/allenai-tulu-sft-recipe.md`
**Companion source:** `wiki/raw-data/llm-training/papers/tulu-3-sft-mix.md`
**Paper:** Lambert et al. 2024 — "Tülu 3: Pushing Frontiers in Open Language Model Post-Training" (arXiv:2411.15124)

---

## Why this source anchors ch-30

Ch-30 presents a five-axis design space. Tülu-3 is one of the two worked examples in ch-30 §6 (the other is Zephyr's smaller recipe). Tülu-3 is the 2024 reference for "what do the five axes look like when you push to the largest fully-open SFT run." Every number here is attested from the Ai2 blog post or the paper; the interest is in how each axis is set and what the ablations say about counterfactual choices.

---

## The mix scale — and why it changes an axis

From `allenai-tulu-sft-recipe.md`, §Overview:

> Tülu 3's SFT mix of 939K prompts with careful contamination filtering matches or beats closed-source instruct models at 8B / 70B.

939K is an order of magnitude larger than the Zephyr / Alpaca reference runs that defined the NEFTune default. From §SFT Hyperparameters:

> | NEFTune | off (found neutral on 939K) | off |

This is the single most important lesson in the recipe for ch-30: the NEFTune axis flips its default based on data scale. The source's ablation finding — "NEFTune gain saturates — no improvement at 939K; small gain ≤ 100K" — is the empirical basis for ch-30's dataset-size rule.

## Notice: the data axis is where scaling happens, not the loss axis

From `allenai-tulu-sft-recipe.md`, §Core Insight:

> Scaling SFT quality is overwhelmingly about *data composition* (math vs code vs chat vs safety mix) and *dedup against eval sets*, not about loss-function tricks.

Ch-30 places this observation between the lines. The five design axes are loss-function / template / packing axes. They are necessary and they have correct defaults. They are *not* the main variable that moves capability at scale. The data mix is. Ch-30 spends its real estate on the axes because they are the cheapest to get wrong, not because they have the highest leverage.

---

## The mix composition — attested ratios

From `allenai-tulu-sft-recipe.md`, §Data Mix (939K prompts):

> | Bucket | Share | Notable sources |
> |--------|-------|-----------------|
> | Chat / general | 27% | OpenAssistant-2, WildChat-1M curated |
> | Math | 21% | Tülu-3 Persona-Math (synthetic), OpenMathInstruct-2 |
> | Code | 14% | OpenCodeInterpreter, Evol-CodeAlpaca |
> | Precise IF | 11% | IFEval-persona + No-Robots |
> | Safety | 10% | WildJailbreak, Tülu-3 Safety |
> | Multilingual | 7% | Aya, Tülu-3 Persona-Multiling |
> | Reasoning / knowledge | 10% | FLAN-v2 subset, SciRIFF |

These ratios are not derived from a theoretical argument; they are the result of Ai2's skill-specific sub-mix construction, then merging, then downsampling. From the companion paper ([[tulu-3-sft-mix]]):

> Ai2 starts with public datasets that have clear provenance and licenses, then manually reviews each candidate source for diversity, hard-skill coverage, and decontamination. The team builds skill-specific data mixtures and models first, keeps the mixes that perform best on individual skills, and then combines them into a preview mix.

---

## The removal ablations — what each bucket costs

From `allenai-tulu-sft-recipe.md`, §Ablation findings:

> - Removing Persona-Math drops GSM8K by 15 pts; removing code drops HumanEval by 12.
> - Removing safety data barely moves capability evals but tanks WildJailbreak from 98% → 52%.
> - 2 epochs > 1 epoch > 3 epochs at this mix size; later epochs hurt IFEval.
> - NEFTune gain saturates — no improvement at 939K; small gain ≤ 100K.
> - Packing: 2.5× throughput, no quality delta.

Three patterns worth extracting:

1. **Capability buckets are mostly additive** — removing math hits math, removing code hits code. There is no substantial cross-bucket spillover in this mix. This is a useful priors-check for ch-31..ch-35: if you add a new capability via SFT, you can expect localized gains, not across-the-board gains.
2. **Safety is orthogonal to capability** — removing safety data does not change MMLU / GSM8K / HumanEval but collapses jailbreak resistance. Safety is a behaviour the model learns via its specific data, not an emergent property of capability training.
3. **Epoch count is non-monotone** — 2 > 1 > 3. More SFT is not universally better; 3 epochs specifically hurts IFEval (instruction following), plausibly because the model overfits to the specific phrasing of the training instructions. The sweet spot is dataset-dependent.

---

## The packing attestation

> Packing: 2.5× throughput, no quality delta.

This is the ablation that backs [[packed-vs-unpacked-ablation]]'s claim. Tülu-3's 8B run used packing; they also ran an unpacked control; the quality was statistically indistinguishable. The 2.5× throughput is the realised speedup (vs the ~6.8× raw speedup the formula predicts for `L_max = 4096, avg(L_i) = 600`); the gap is FlashAttention overhead and memory-bandwidth ceiling, as discussed in ch-30 §4.

---

## The 70B-vs-8B learning-rate scaling

From `allenai-tulu-sft-recipe.md`, §SFT Hyperparameters:

> | Learning rate | 5e-6 | 2e-6 |
> | Distributed | FSDP FULL_SHARD | FSDP + HYBRID_SHARD |

The learning rate halves (roughly) at 70B. This follows a rough `1/√N_params` scaling that is folklore for SFT and well-attested across Llama / Qwen / DeepSeek reports. The distributed strategy also shifts — FULL_SHARD at 8B, HYBRID_SHARD at 70B — because HYBRID keeps one replica per node and all-gathers within a node, reducing inter-node bandwidth pressure that becomes dominant at larger model sizes.

---

## Decontamination as a data-recipe step, not an afterthought

From `allenai-tulu-sft-recipe.md`, §Decontamination:

> - 8-gram overlap ≥ 50% against every eval set → drop.
> - Embedding similarity > 0.9 to eval-set items → drop.
> - Documented "surviving overlap" rates per eval.

And from [[tulu-3-sft-mix]]:

> Decontamination is part of the data recipe, not an afterthought; Ai2 explicitly removes overlap with more than 2% of the eval suite.

This is a point ch-30 mentions but does not belabour: decontamination is not itself one of the five SFT *design* axes, but it is a non-negotiable preprocessing step. The "5-axis design space" assumes the input data has been decontaminated; if it has not, every axis's ablation becomes uninterpretable because the baseline is inflated.

---

## What ch-30 lifts from this recipe

| Recipe element | Ch-30 use |
|----------------|-----------|
| 939K mix scale | Justifies NEFTune-off default for `|D| ≥ 500K` |
| 2.5× packing speedup | Cited in §4 as the realised number |
| LR 5e-6 / 2e-6 for 8B / 70B | Quoted in §6's production-recipe table |
| 2 epochs sweet spot | Quoted in §6 |
| Response-only loss | Confirmed as default |
| FSDP FULL_SHARD / HYBRID_SHARD | Referenced for infra track ch-54..ch-59 |

---

## Connections

- [[excerpts/neftune-regularizer]] — Tülu-3 is the saturation datapoint; this excerpt is the recipe that flipped the default.
- [[excerpts/sequence-packing-contract]] — Tülu-3's 2.5× is the attested realised speedup.
- [[excerpts/sft-pre-determines-rl]] — the Tülu-3 stack is SFT → DPO → RLVR; what SFT shaped bounds what later stages do.
- [[ch-30]] — §6 recipe table and §5 NEFTune rule both depend on this source.
- [[ch-36]] (SFT lab) — Tülu-3 is the full-budget reference; the lab's resource-constrained path is a scaled-down version of this recipe.

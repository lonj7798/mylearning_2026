<!-- scope: HuggingFace SmolTalk — 1M-sample SFT dataset for SmolLM2; Magpie + public mix
     deps: [[magpie]]
     see-also: [[hf-cosmopedia]], [[tulu-3-sft-mix]], [[openhermes]]
-->

# SmolTalk: HuggingFace SmolLM2 SFT Dataset
- **Core Insight:** A 1M-sample SFT mix built mostly from **Magpie-Ultra (400K)** generated with Llama-3.1-405B-Instruct plus targeted public datasets — this recipe gives best-in-class small-model post-training results (SmolLM2-1.7B-Instruct) and validates Magpie-scale synthetic SFT in the open.
- **Guideline:** For a small-model SFT dataset, anchor on ~40% Magpie-style in-house synthetic generated with a strong open teacher (405B-class), then layer targeted public sources for math (MetaMathQA), code (Self-OSS-Instruct), multi-turn (SystemChats2.0), and reasoning (OpenHermes 2.5).
- **Author(s):** Loubna Ben Allal, Anton Lozhkov, Gabriel Martín-Blázquez, Elie Bakouch, Leandro von Werra, Thomas Wolf (HuggingFace Smol team)
- **Year:** 2024/2025
- **URL:** https://huggingface.co/datasets/HuggingFaceTB/smoltalk ; SmolLM2 paper: https://arxiv.org/abs/2502.02737
- **Relevant topics:** SFT mix composition, Magpie, SmolLM2, open data recipe

## Abstract
SmolTalk is the SFT training mix released with SmolLM2. It combines ~400K new synthetic data generated with the [[magpie]] pipeline using Llama-3.1-405B-Instruct (Smol-Magpie-Ultra), plus smaller task-specific synthetic sets (Smol-constraints, Smol-rewrite, Smol-summarize) and curated public datasets (OpenHermes 2.5, MetaMathQA, NuminaMath-CoT, Self-Oss-Starcoder2-Instruct, SystemChats 2.0). The result is the backbone of the SmolLM2-*-Instruct family, and an excellent reference for what a modern open 1M-sample SFT corpus looks like.

## Composition table (the load-bearing numbers)

| Component | Size | Source / Purpose |
|---|---|---|
| **Smol-Magpie-Ultra** | **400K** | Magpie pipeline w/ Llama-3.1-405B-Instruct — core instruction coverage |
| Smol-constraints | 36K | Constraint-following tasks (precise formatting, JSON, word limits) |
| Smol-rewrite | 50K | Rewrite / paraphrase / tone-shift tasks |
| Smol-summarize | 100K | Summarization tasks |
| OpenHermes 2.5 (subset) | 100K | Teknium's open catalogue — general instruction + reasoning |
| MetaMathQA (subset) | 50K | Math word problems |
| NuminaMath-CoT | (subset) | Competition-grade math with CoT |
| Self-Oss-Starcoder2-Instruct | (subset) | Open-source code instructions |
| SystemChats 2.0 | 30K | System-prompt-conditioned dialogues |
| LongAlign | (subset) | Long-context alignment |
| Everyday-conversations | (subset) | Casual dialogue |
| Explore-Instruct-Rewriting | (subset) | Instruction rewriting |

**Total:** ~1M samples.

## Why each piece
- **Magpie-Ultra (40%):** captures the bulk of general instruction diversity via a strong 405B teacher; cheap (no API) because Llama-3.1-405B is open.
- **Smol-constraints / Smol-rewrite / Smol-summarize:** targeted synthesis for capabilities that Magpie underweights.
- **Public math/code datasets:** verifiable domains where synthetic alone isn't enough; hand-curated mixes.
- **SystemChats 2.0:** system-prompt steerability.
- **LongAlign:** long-context.

## Training outcome
- SmolLM2-1.7B-Instruct (trained on SmolTalk) is best-in-class small model at release on IFEval, MMLU-Pro, and BBH for its size.
- For smaller variants (135M, 360M), a size-matched subset of SmolTalk is used.

## Practitioner takeaways
- Open Magpie-style synthesis at ~400K scale is the new "default open SFT anchor."
- Explicit targeted synthesis for constraints/rewrite/summarize is more effective than assuming Magpie covers them.
- Composition table is tunable — for a coding-focused variant, inflate the code share.

## Risks + gotchas
- **Teacher-model inheritance:** Llama-3.1-405B-Instruct biases propagate.
- **Public-subset overlap**: OpenHermes 2.5 already contains Magpie-adjacent data — dedup matters.
- **Long-context subset is small**; downstream long-context performance partly inherited from pretraining.

## Connections
- Validates [[magpie]] at production scale.
- Sibling to [[tulu-3-sft-mix]] (different ratios, same spirit).
- Companion to [[hf-cosmopedia]] on the pretraining side.
- Uses [[openhermes]] (subset) and MetaMathQA / NuminaMath as public anchors.

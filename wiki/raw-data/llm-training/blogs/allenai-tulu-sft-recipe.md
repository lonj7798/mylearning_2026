<!-- scope: Tülu 3 SFT recipe — Allen AI's fully-open post-training stack; scale + ablation depth
     deps: [[sequence-packing]], [[loss-masking-prompt]]
     see-also: [[hf-alignment-handbook]], [[rlvr-tulu3]], [[tulu-3]]
-->

# Allen AI Tülu 3 SFT Recipe
- **Core Insight:** Scaling SFT quality is overwhelmingly about *data composition* (math vs code vs chat vs safety mix) and *dedup against eval sets*, not about loss-function tricks; Tülu 3's SFT mix of 939K prompts with careful contamination filtering matches or beats closed-source instruct models at 8B / 70B.
- **Guideline:** Follow Tülu 3's data recipe: explicitly mix persona / math / code / chat / safety in known ratios, dedup against all eval sets via n-gram + embedding match, and run SFT for 2 epochs at LR ~5e-6 with packing + response-only loss.
- **Authors:** Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, Hannaneh Hajishirzi
- **Year:** 2024
- **URL:** https://allenai.org/blog/tulu-3 ; paper: https://arxiv.org/abs/2411.15124
- **Relevant topics:** fully-open SFT, data curation, contamination dedup, multi-domain mix, scale ablations

## Overview
Tülu 3 is Allen AI's fully-open post-training suite: data, code, checkpoints, evals. The blog post distills the SFT half: mix construction, dedup, loss, hparams, ablations. The paper expands with DPO-then-RLVR.

## Key Contributions
- Open release of the 939K SFT mix with full provenance.
- Explicit decontamination against MMLU, GSM8K, MATH, IFEval, BBH, AlpacaEval, Arena-Hard, HumanEval.
- Systematic skill-level ablation: math / code / chat / safety / precise-IF contributed additively.
- Public 8B, 70B, and (later) 405B SFT + DPO checkpoints.

## Data Mix (939K prompts)

| Bucket | Share | Notable sources |
|--------|-------|-----------------|
| Chat / general | 27% | OpenAssistant-2, WildChat-1M curated |
| Math | 21% | Tülu-3 Persona-Math (synthetic), OpenMathInstruct-2 |
| Code | 14% | OpenCodeInterpreter, Evol-CodeAlpaca |
| Precise IF | 11% | IFEval-persona + No-Robots |
| Safety | 10% | WildJailbreak, Tülu-3 Safety |
| Multilingual | 7% | Aya, Tülu-3 Persona-Multiling |
| Reasoning / knowledge | 10% | FLAN-v2 subset, SciRIFF |

All generated or curated responses are from GPT-4o / Claude / Llama-3.1-70B-Instruct; no human rewrites for response content.

## Decontamination
- 8-gram overlap ≥ 50% against every eval set → drop.
- Embedding similarity > 0.9 to eval-set items → drop.
- Documented "surviving overlap" rates per eval.

## SFT Hyperparameters (8B / 70B)

| Knob | 8B | 70B |
|------|-----|-----|
| Max seq length | 4096 | 4096 |
| Packing | yes | yes |
| Response-only loss | yes | yes |
| Optimizer | AdamW (0.9, 0.95) | same |
| Learning rate | 5e-6 | 2e-6 |
| LR schedule | linear, 3% warmup | same |
| Epochs | 2 | 2 |
| Global batch (prompts) | 128 | 128 |
| Precision | BF16 | BF16 |
| Distributed | FSDP FULL_SHARD | FSDP + HYBRID_SHARD |
| Gradient checkpointing | true | true |
| NEFTune | off (found neutral on 939K) | off |

## Ablation findings (from paper)
- Removing Persona-Math drops GSM8K by 15 pts; removing code drops HumanEval by 12.
- Removing safety data barely moves capability evals but tanks WildJailbreak from 98% → 52%.
- 2 epochs > 1 epoch > 3 epochs at this mix size; later epochs hurt IFEval.
- NEFTune gain saturates — no improvement at 939K; small gain ≤ 100K.
- Packing: 2.5× throughput, no quality delta.

## Post-SFT chain
1. SFT on 939K (this blog).
2. DPO on Tülu-3-Preference (~270K pairs from UltraFeedback + on-policy).
3. RLVR (verifiable-reward RL) for math / IF specialization → [[rlvr-tulu3]].

## Connections
- Upstream SFT mechanics: [[sequence-packing]], [[loss-masking-prompt]].
- HF counterpart (smaller mix, same mechanics): [[hf-alignment-handbook]].
- DPO stage foundation: [[dpo]].
- RLVR follow-up: [[rlvr-tulu3]].
- Full model report: [[tulu-3]] (model-reports).

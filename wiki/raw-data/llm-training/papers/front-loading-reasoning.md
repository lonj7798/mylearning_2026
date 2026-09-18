<!-- scope: controlled 8B study (NVIDIA/CMU, arXiv Sep 2025) of adding reasoning-style QA data in pretraining vs SFT at different scale, diversity, and quality, with GRPO on two backbones; data timing across stages, not a mid-training or annealing method
     deps: [[open-thoughts]]
     see-also: [[interplay-pretraining-midtraining-rl]], [[transferability-of-llm-reasoning]], [[echo-chamber-rl-post-training]], [[rlp-reinforcement-as-pretraining-objective]], [[grpo]]
-->

# Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data
- **Core Insight:** In an 8B hybrid Mamba-Transformer pretrained from scratch on 1T tokens, models with reasoning data in pretraining stayed ahead after SFT (35.92 vs 26.62 average, Table 2) and after GRPO (56.66 vs 37.92, Table 3); the abstract summarizes this as a 19% average gain that later SFT does not recover.
- **Guideline:** When the same reasoning corpora can be placed in pretraining or SFT, the paper's results support the large, diverse corpus in pretraining (ℳLDQ 64.09 vs ℳSHQ 54.98, Table 1) and the small, long-CoT, high-quality corpus in SFT (44.99 vs 31.54, Table 5); this was tested at one model size (8B), one architecture, and one pretraining budget (1T tokens).
- **Authors:** Syeda Nahida Akter, Shrimai Prabhumoye, Eric Nyberg, Mostofa Patwary, Mohammad Shoeybi, Yejin Choi, et al. (NVIDIA, CMU, Boston University, Stanford)
- **Year:** 2025 (arXiv v1 2025-09; no venue listed)
- **URL:** https://arxiv.org/abs/2510.03264
- **Source type:** paper
- **Relevant topics:** reasoning data placement, pretraining mixture, SFT data quality vs diversity, catch-up hypothesis, GRPO, instruction-following trade-off

## Abstract
The paper asks whether reasoning data added during pretraining is better than the same kind of data added during post-training when token counts are controlled, and whether early inclusion causes overfitting. The authors pretrain four 8B models that differ only in the reasoning data mixed into pretraining, fine-tune each on reasoning corpora of different scale, diversity, and quality, and apply GRPO to two backbones. They report a 19% average gain from front-loading reasoning data into pretraining that later SFT does not fully replicate even with more data. They report an asymmetric pattern: pretraining benefits most from diverse reasoning data (11% average gain) and SFT is more sensitive to quality (15% average gain). High-quality data in pretraining shows a latent effect that appears after SFT, and scaling SFT with mixed-quality data can remove the benefit of early reasoning data (abstract).

## Key Contributions
- A crossed design: four pretraining variants, SFT on several reasoning corpora, and GRPO on the two extreme backbones (§2.3, §4).
- A test of the "catch-up" hypothesis: ℳbase with twice the SFT data (34.01) stays below ℳSHQ + SFT (37.33) (§5, Table 4).
- An asymmetric allocation result: scale and diversity help in pretraining (Table 1); quality helps in SFT (Table 5).
- A latent effect: ℳLMQ and ℳLDQ are equal after pretraining (64.07 vs 64.09, Table 1), but ℳLMQ is 4.25 points higher after the same SFT (50.95 vs 46.70, §5, Table 4).
- An SFT scaling ablation: doubling mixed-quality SFT data lowers math by 4.92 points; adding 0.4% more high-quality samples raises the average (§5, Table 6).

## Key Figures/Tables to Study
- Table 1 and Table 7: base-model accuracy per pretraining variant, per benchmark.
- Table 4: catch-up test (SFT 2×) and the latent effect after a shared D_SHQ SFT.
- Tables 5 and 6: SFT corpus quality vs diversity; two SFT scaling strategies.
- Table 3: ℳbase vs ℳLMQ after SFT and GRPO.
- Table 9 and Figure 2 (App. C): answer-length filtering; the same data seen in pretraining and SFT.

## Technical Details
**Data (§2.2, App. C).**
- D_base: the Nemotron Nano 2 pretraining corpus, 6.2T tokens from Common Crawl, math, and code (§2.2).
- D_LDQ (large, diverse, mixed quality): Nemotron-Pretraining-SFT-v1, 336B tokens, about 56% math, 17% code, 27% science and general reasoning (§2.2); average answer length about 550 tokens (App. C).
- D_SHQ (small, high quality): the OpenThoughts data of Guha et al. (2025), 1.2M examples, 71% math, 21% code, 8% science (§2.2); long CoT answers from strong teacher models, average length above 10k tokens (App. C).
- D_LMQ = D_LDQ ∪ D_SHQ (§2.2).
- D_ALF: examples whose answer exceeds 4096 tokens; about 2% of the source corpus and 75% math (§2.2, App. C). The source corpus is written D_LLQ, a symbol the paper does not define. D_ALF′ adds D_SHQ to D_ALF (§5).

**Models and stages.** An 8B hybrid of Mamba-2, self-attention, and FFN layers, pretrained from scratch for 1T tokens on 80% D_base and 20% D_res (§2.1, §2.3). Variants: ℳbase (no D_res), ℳSHQ, ℳLDQ, ℳLMQ; ℳres denotes the average of the three reasoning variants (§2.3). GRPO is applied to ℳbase + SFT and ℳLMQ + SFT, both fine-tuned on D_SHQ (§4).

**Evaluation (§3.2).** Base models use LM Eval Harness: general reasoning (ARC-C, HellaSwag, WinoGrande, RACE; 0-shot), math (GSM8K 8-shot, MATH-500 4-shot), science (MMLU, MMLU-Pro; 5-shot), code (HumanEval and MBPP EvalPlus; 0-shot, avg@32). SFT and RL models use NeMo-Skills and add AIME24/25 (pass@1 averaged over 16 runs), GPQA-Diamond, LiveCodeBench, and IFEval (averaged over 4 runs).

**Results (averages as printed).**
- Pretraining: ℳbase 52.70, ℳSHQ 54.98, ℳLDQ 64.09, ℳLMQ 64.07 (Table 1). ℳLDQ gains +28.4 math and +9 code over ℳbase (§4) and +9.09 average over ℳSHQ (§5).
- After SFT: ℳres 35.92 vs ℳbase 26.62 (+9.3); the largest gap is science, 34.77 vs 20.92 (§4, Table 2).
- Catch-up: SFT(2×) raises ℳbase by 7.39 to 34.01, still 3.32 below ℳSHQ + SFT (§5, Table 4).
- SFT corpus on ℳres: D_SHQ 44.99, D_LDQ 31.54, D_LMQ 31.21; ℳbase + SFT[D_SHQ] 29.92 (Table 5).
- SFT scaling on ℳLDQ: SFT[D_LDQ] 32.84, SFT[2×D_LDQ] 32.99 (math 28.38 → 23.46), SFT[D_ALF] 42.66, SFT[D_ALF′] 43.04 (Table 6).
- After GRPO: ℳbase 37.92 vs ℳLMQ 56.66 (Table 3); the text reports an 18.57% average lead and a 39.32% improvement on AIME (§4).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| ℳbase/ℳSHQ/ℳLDQ/ℳLMQ (not released) | 8B hybrid | pretrain-stable | tokens seen | 1T | arXiv:2510.03264v1 §2.3, §3.1 | verified 2026-09-14 | no ablation reported |
| same | 8B | pretrain-stable | mixture (token share vs sampling weight not stated) | 80% D_base, 20% D_res | §2.3 | verified 2026-09-14 | no ablation of the 20% share reported |
| same | 8B | pretrain-stable | peak LR / min LR | 3e-4 / 3e-6 | §3.1 | verified 2026-09-14 | no ablation reported |
| same | 8B | pretrain-stable | warmup; decay shape | not reported | §3.1, App. A–C checked | not reported | — |
| same | 8B | pretrain-stable | AdamW β1, β2; weight decay | 0.9, 0.95; 0.1 | §3.1 | verified 2026-09-14 | no ablation reported |
| same | 8B | pretrain-stable | global batch; context length | 6M tokens; 8192 | §3.1 | verified 2026-09-14 | no ablation reported |
| same | 8B | pretrain-stable | compute | 512 H100 80GB; 8-way tensor and pipeline parallelism; GPU-hours not reported | §3.1 | verified 2026-09-14 | — |
| each variant + SFT | 8B | SFT | examples | 4.8M reasoning samples from D_res | §3.1 | verified 2026-09-14 | Table 4: 2× SFT tested only for ℳbase |
| same | 8B | SFT | LR; warmup ratio | 5e-6; 0.05 | §3.1 | verified 2026-09-14 | no ablation reported |
| same | 8B | SFT | AdamW β1, β2; weight decay | 0.9, 0.95; 0.01 | §3.1 | verified 2026-09-14 | no ablation reported |
| same | 8B | SFT | batch (unit not stated); context length | 512; 32k | §3.1 | verified 2026-09-14 | no ablation reported |
| same | 8B | SFT | epochs; loss masking; packing | not reported | §3.1, App. checked | not reported | — |
| ℳbase + SFT, ℳLMQ + SFT (D_SHQ) | 8B | RL | algorithm; framework; data; epochs | GRPO; veRL; nemotron-crossthink; 1 epoch | §2.3, §3.1 | verified 2026-09-14 | no ablation reported |
| same | 8B | RL | LR | 1e-6, constant | §3.1 | verified 2026-09-14 | no ablation reported |
| same | 8B | RL | batch; PPO mini-batch; context length | 128; 128; 8192 | §3.1 | verified 2026-09-14 | no ablation reported |
| same | 8B | RL | prompts per step × rollouts; temperature; top-p | 128 × 8; 1.0; 1.0 | §3.1 | verified 2026-09-14 | no ablation reported |
| same | 8B | RL | KL coefficient (placement not stated) | 0.001 | §3.1 | verified 2026-09-14 | no ablation reported |
| same | 8B | RL | clip ε; max response length | not reported | §3.1 checked | not reported | — |

## Findings relevant to generality and distillation
- Breadth at the base stage: the general-reasoning average (ARC-C, HellaSwag, WinoGrande, RACE) ranges only 75.63–76.83 across pretraining variants (Table 1); the authors attribute this to tasks that need commonsense and general knowledge (§5).
- Narrowing in SFT: reasoning-heavy SFT lowers instruction following. ℳbase insSFT falls 43.98 → 32.70 with 2× SFT (Table 4); ℳres insSFT is 43.56 with D_SHQ vs 54.61 with D_LDQ (Table 5); ℳLDQ IFEval is 39.87 with D_ALF vs 57.61 with D_LDQ (Table 9). The authors call this a trade-off between breadth and reasoning depth (App. C).
- Forgetting: ℳbase, which sees D_SHQ only in SFT, is the lowest performer in all categories, while ℳSHQ, which sees D_SHQ in pretraining and SFT, scores higher (App. C, Figure 2). The authors interpret the second exposure as reinforcing rather than overwriting (Interpretation).
- Distillation: D_SHQ answers were generated by strong teacher models (App. C), and SFT on D_SHQ was the best SFT corpus in Table 5.

## Connections
- [[open-thoughts]]: the D_SHQ corpus (Guha et al. 2025) used here in pretraining and SFT.
- [[interplay-pretraining-midtraining-rl]]: a separate study of pretraining, mid-training, and RL for reasoning; compare stage definitions before combining conclusions.
- [[transferability-of-llm-reasoning]]: cited here (Huan et al. 2025) for small science gains from reasoning-centric post-training (§4).
- [[echo-chamber-rl-post-training]]: a controlled small-model study of RL amplifying pretraining behaviors.
- [[grpo]]: the GRPO algorithm (Shao et al. 2024) used in the RL stage.
- [[lima]]: cited for the view that SFT data quality matters (§5).
- [[rlp-reinforcement-as-pretraining-objective]], [[quiet-star]]: other methods that add reasoning signal before post-training.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2510.03264 (arXiv v1, 26 Sep 2025, full PDF including Tables 7–9 and App. C)
- Corrections to the previous card version: "front-loading gives a reported 19% average gain and sets a higher final ceiling" → 19% is the abstract figure; §1 ties it to "expert-level benchmarks" after post-training and §4 prints an 18.57% average lead for ℳLMQ vs ℳbase after GRPO (Table 3). Core insight and guideline now state the tested scope (8B, one architecture, 1T tokens). deps [[dolma]], [[self-instruct]] removed (not related to this paper); [[open-thoughts]] added.
- Removed as unsupported by the source: "It connects directly to the 2025 RL mechanism debate: if RL mostly amplifies existing priors, then what priors you install during pretraining becomes crucial"; "If compute is limited, do not spend all reasoning budget at the SFT stage"; "use later post-training to refine style, correctness, and task alignment"; the "stage-by-stage performance growth" figure (no such figure); "[[dolma]] and [[olmo-3]] are strong open examples of stage-aware data curricula".
- Not reported by the source: pretraining warmup and decay shape; SFT epochs, loss masking, packing; how 4.8M SFT samples relate to D_SHQ's 1.2M examples; GRPO clip ε and max response length; seeds or confidence intervals; contamination checks; which comparison yields the abstract's 11% and 15%.
- Internal inconsistencies in the source: ℳbase + SFT averages 26.62 in Tables 2 and 4 (Table 4 states SFT on D_SHQ) but ℳbase + SFT[D_SHQ] is 29.92 in Table 5 and in the Table 8 category averages; the catch-up margin depends on which value is used. Table 2's ℳres + SFT value (35.92) equals the mean of the three ℳres rows of Table 5 (derived: (31.21 + 31.54 + 44.99)/3), so Table 2 appears to average over SFT corpora, which the paper does not state.

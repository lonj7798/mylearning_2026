<!-- scope: Yang et al. (arXiv 2505.17315): controlled study of long-context ability (RoPE-theta scaling, model merging) before long-CoT reasoning SFT; "extend context first" recipe
     deps: [[ruler]], [[hf-open-r1-update-2-math220k]]
     see-also: [[longred]], [[longpo]], [[long-more-important-than-difficult]], [[demystifying-long-cot]]
-->

# Longer Context, Deeper Thinking: Uncovering the Role of Long-Context Ability in Reasoning
- **Core Insight:** With identical reasoning-SFT data, LLaMA3-8B-Instruct variants with a higher 32K Needle-in-a-Haystack score reach higher MATH500 accuracy after SFT (short/long-SFT average 54.80 at RoPE θ×1 vs 59.36 at θ×16), and applying θ×16 plus a 0.3 merge with Qwen2.5-7B-Instruct-1M to Qwen2.5-Math-7B-Instruct raises post-SFT MATH500 from 85.04 to 88.70 and AIME22-24 from 15.00 to 28.00 (Tables 3, 7).
- **Guideline:** When the effective context length of the model to be fine-tuned is shorter than the long-CoT SFT responses (here up to 16K tokens), raise its effective context (RoPE θ scaling or merging with a long-context sibling) and measure it on NIAH before reasoning SFT, because in the LLaMA3-8B-Instruct θ sweep the factor with the best 32K NIAH score (×16) also gave the best post-SFT MATH500, AIME22-24, and GSM8K averages (Table 8); do not select the largest θ factor or the longest nominal window by default, because θ×32/×64 and the pure 1M merge scored lower (Tables 3, 9).
- **Authors:** Wang Yang, Zirui Liu, Hongye Jin, Qingyu Yin, Vipin Chaudhary, Xiaotian Han (Case Western Reserve University; University of Minnesota - Twin Cities; Texas A&M University)
- **Year:** 2025 (arXiv v1 2025-05; v2 2026-06; the v2 PDF footer reads "39th Conference on Neural Information Processing Systems (NeurIPS 2025)")
- **URL:** https://arxiv.org/abs/2505.17315
- **Source type:** paper
- **Relevant topics:** long-context ability, reasoning SFT, long chain-of-thought, RoPE θ scaling, model merging, effective context length, short-input generalization

## Abstract
The authors hypothesize that current reasoning limits come partly from insufficient long-context capacity. Two observations motivate this: models with longer context windows tend to reason better, and failed reasoning outputs show the same symptoms as failed long-context cases (repetition, wrong references to earlier content). They compare models with the same architecture and the same fine-tuning data but different long-context capacity, then apply SFT. Models with stronger long-context capacity reach higher accuracy on reasoning benchmarks after SFT, including tasks with short inputs. The authors recommend treating long-context capacity as a first-class objective and extending context before reasoning SFT. Code: github.com/uservan/LCTMerge.

## Key Contributions
- Behavioral motivation: 32K vs 128K public 7B reasoning models (Table 1), failure case studies (Figs. 2-3), and length distributions of long-CoT datasets (Fig. 4) (§2).
- Controlled study: six RoPE θ scale factors on LLaMA3-8B-Instruct, each fine-tuned on short and on long reasoning data (§3.2, Tables 2-3).
- Generality checks: GPQA and Livecode after science/code SFT, Phi-4 (14B), and short-input MMLU-STEM (§3.3, Tables 4-6, 10).
- Extreme-length test: linear merges of Qwen2.5-7B-Instruct and Qwen2.5-7B-Instruct-1M at ratios 0 to 1.0 (§3.4, App. B Table 9).
- Proposed recipe validated on Qwen2.5-Math-7B-Instruct (§3.5, Table 7).

## Key Figures/Tables to Study
- Table 2: effective context length, 32K NIAH, LongBench, and RULER for each θ factor.
- Table 3 / App. A Table 8: pre-SFT and post-SFT accuracy and output length on MATH500, AIME22-24, GSM8K.
- Tables 4-6 and 10: cross-domain, cross-family, and short-input results. Table 7: recipe validation.
- App. C Table 11: counts and lengths of correct and wrong generations.

## Technical Details
- **Public-model comparison.** 32K group (OpenR1-Qwen-7B, OpenThinker-7B, OpenThinker2-7B, OpenThinker3-7B) averages MATH500 90.37 and AIME 43.28; 128K group (DeepSeek-R1-Distill-Qwen-7B, OpenMath-Nemotron-7B, DeepMath-Zero-7B, AceReason-Nemotron-7B) averages 90.75 and 49.45 (§2.1, Table 1). The groups differ in training data and method.
- **Failure lengths.** For DeepSeek-R1-Distill-Qwen-1.5B/7B/14B, wrong answers have longer average outputs than correct ones on AIME22-24 and MATH500 (§2.2, Fig. 4).
- **Dataset lengths.** NuminaMath-CoT samples are mostly under 1K tokens; OpenR1-Math-220K and DeepMath-103K (both from DeepSeek-R1) have many samples over 4K and some over 10K (§2.3, Fig. 4).
- **Extension methods.** (1) Multiply RoPE θ by a factor; the paper cites NTK-aware scaling [7] and describes no long-context continued pretraining. (2) Linear merge with a longer-context model, with the ratio chosen so that base performance stays nearly unchanged (§3.1, §3.4).
- **θ sweep, LLaMA3-8B-Instruct (8K native).** Factors ×1/×4/×8/×16/×32/×64: estimated effective length 9k/21k/29k/32k/21k/17k; 32K NIAH 0.00/3.75/58.30/77.05/58.86/35.00; RULER 56.13/69.57/79.62/94.24/88.07/84.98 (Table 2). How effective length is estimated is not described.
- **NIAH scoring.** Single-haystack accuracy: +1 correct, −1 repetitive or degenerate, 0 incorrect but non-degenerate (§3.1).
- **MATH500 after SFT (Table 3).** Short/long/avg: ×1 50.68/58.92/54.80; ×8 53.12/62.24/57.68; ×16 54.40/64.32/59.36; ×64 53.28/62.36/57.82. Pre-SFT accuracy falls with θ: 24.40 (×1) to 14.20 (×64).
- **Other benchmarks (App. A Table 8).** AIME22-24 avg 2.89 (×1) vs 5.45 (×16); GSM8K avg 84.73 (×1) vs 86.01 (×16).
- **Output length.** Average MATH500 output after SFT falls from 12,089 (×1) to 9,059 (×16) tokens (Table 3). At ×1 short SFT, correct answers average 3,189 tokens and wrong answers 23,145 (App. C Table 11). The authors attribute the drop to a higher share of correct answers (Interpretation, App. C).
- **Merge sweep.** Qwen2.5-7B-Instruct + 1M model at ratio 0/0.1/0.7/1.0: 32K long-context score 78.1/79.1/79.5/77.7; MATH500 avg 84.16/84.92/84.38/82.88; AIME22-24 avg 20.56/21.00/21.56/18.11 (App. B Table 9). Because NIAH saturates, §3.4 also uses 32K Value Tracking and QA tasks.
- **Recipe validation (Table 7).** Qwen2.5-Math-7B-Instruct (4K context, §3.5): θ×1 gives long-SFT MATH500 83.80, below short-SFT 86.28; after θ×16 + 0.3 merge, long-SFT is 89.12 and short-SFT 88.28.
- **Inconsistencies in the paper.** §3.2 names LLaMA3-8B-Instruct, while the Fig. 7-8 captions name LLaMA-3.1-8B-Instruct. §5 limits the analysis to 7B-8B models, while §3.3 includes Phi-4 (14B). §1 describes the comparison as "varying degrees of long-context pretraining", while §3.1 describes only θ scaling and merging.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LLaMA3-8B-Instruct θ variants; Phi-4; Qwen2.5 variants ("All models") | 7B-14B | SFT | Library; hardware | LLaMAFactory; 4× NVIDIA H200 | arXiv:2505.17315v2 §3.1 | verified 2026-09-14 | no ablation reported |
| same | 7B-14B | SFT | Batch size (unit not stated); LR; epochs | 32; 1.0e-5; 3 | §3.1 | verified 2026-09-14 | no ablation reported |
| same (math runs) | 7B-14B | distill-SFT | Data | OpenR1-Math-220K split by response length: short ≤8K tokens, long 8K-16K; 20K sampled per split, then correctness-filtered; splits used separately | §3.1 | verified 2026-09-14 | Table 3: long split > short split at every θ factor on MATH500 |
| LLaMA3-8B-Instruct θ ×1/×4/×16 | 8B | distill-SFT | Cross-domain data | 20k samples, science and code domains of OpenThoughts3-1.2M | §3.3 | verified 2026-09-14 | Table 4 |
| LLaMA3-8B-Instruct | 8B | long-context | RoPE θ scale factors tested | {1, 4, 8, 16, 32, 64} | §3.2 | verified 2026-09-14 | Tables 2-3: ×16 best NIAH (77.05) and best MATH500 avg (59.36) |
| Qwen2.5-Math-7B-Instruct (recipe) | 7B | long-context | Extension | θ×16, then linear merge 0.7 × (θ×16 model) + 0.3 × Qwen2.5-7B-Instruct-1M | §3.5, Table 7 | verified 2026-09-14 | Table 7: MATH500 avg 88.70 vs 88.20 (θ×16 only) vs 85.04 (θ×1); number of runs not stated |
| all evaluated models | all | eval-gate | Metric | pass@1(5): 5 responses per question, accuracy over all responses | §3.1 | verified 2026-09-14 | no ablation reported |

Not reported (checked body, App. A-C, NeurIPS checklist): SFT maximum sequence length, LR schedule and warmup, optimizer, packing, loss masking, original RoPE θ values, evaluation sampling temperature and max generation length, number of seeds or error bars, merge ratio search procedure.

## Findings relevant to generality, long context, distillation
- **Generality (cross-domain).** GPQA after science SFT: 37.27 (×1), 39.19 (×4), 41.92 (×16); Livecode after code SFT: 25.34, 27.95, 32.27. Pre-SFT accuracy falls with θ (GPQA 31.82 to 30.81; Livecode 12.30 to 7.45) (§3.3, Table 4).
- **Generality (short input).** MMLU-STEM after SFT: 71.06 (×1), 73.04 (×4), 74.27 (×16); pre-SFT 54.36, 53.85, 51.44 (§3.3, Table 6). The authors state that long-context training can reinforce reasoning on short-input tasks (Interpretation).
- **Other family.** Phi-4 MATH500 after SFT 88.62/89.14/89.90 and AIME 47.56/49.45/50.17 for ×1/×4/×16 (Tables 5, 10).
- **Long context.** Nominal length is not sufficient: the pure 1M model (ratio 1.0) has a lower 32K score (77.7) and lower post-SFT reasoning than the 0.1 and 0.7 merges (App. B Table 9). θ beyond ×16 lowers NIAH and RULER (Table 2).
- **Distillation.** Stage: reasoning SFT on teacher traces. Prompt types: math (OpenR1-Math-220K); science and code (OpenThoughts3). Teacher: DeepSeek-R1 for OpenR1-Math-220K (§2.3). Quality control: correctness filtering (§3.1). Teacher sampling settings are not reported.
- **Scope limits stated by the authors.** SFT only; interaction with RL is left open (§5).

## Connections
- [[hf-open-r1-update-2-math220k]] — OpenR1-Math-220K, the math SFT data split by response length.
- [[open-thoughts]] — OpenThoughts3-1.2M science/code subsets; OpenThinker models form the 32K group in Table 1.
- [[deepmath-103k]] — length distribution in Fig. 4; DeepMath-Zero-7B in Table 1.
- [[deepseek-r1]] — generator of the long-CoT datasets analyzed.
- [[localllama-ntk-aware-rope]] — reference [7] for the RoPE θ scaling used as the extension method.
- [[rope-base-bounds-context-length]], [[string-effective-context]] — other cards on RoPE base and on effective vs nominal context length (not cited by this paper).
- [[ruler]], [[needle-in-haystack-data]] — long-context evaluations used for the θ sweep.
- [[longred]] — also measures lost short-text ability after RoPE changes; compare with the pre-SFT accuracy drop in Table 3.
- [[longpo]] — long-context alignment that keeps short-context scores.
- [[long-more-important-than-difficult]], [[demystifying-long-cot]] — other studies of long reasoning data in SFT.
- [[ultralong-128k-to-4m]] — cited as an example of a model with a context window beyond 1M tokens.
- [[model-merging-at-scale]] — merging study (not cited by this paper); here linear merging is used as a context-extension tool.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2505.17315 (arXiv v2, 2 Jun 2026; full PDF including App. A-C; abs page for version dates).
- Audit claims not found in the source: "contradicts ch-32 §6" is a course-level interpretation, not a statement in the paper. The venue "NeurIPS 2025" is not on the arXiv abs page (no comment field) but is printed in the v2 PDF footer (p. 1).

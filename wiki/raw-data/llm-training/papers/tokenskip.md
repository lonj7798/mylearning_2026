<!-- scope: TokenSkip (PolyU, EMNLP 2025): prune a model's own correct CoTs with LLMLingua-2 token importance at a target ratio γ, LoRA-SFT the model with γ in the input, and control CoT length at inference; GSM8K and MATH-500 on LLaMA-3.1-8B-Instruct and Qwen2.5-3B/7B/14B-Instruct, with out-of-domain checks
     deps: [[rejection-sampling-finetuning]]
     see-also: [[c3ot-compressed-cot]], [[l1-lcpo]], [[overthinking-o1-like-llms]], [[kimi-k1-5]], [[demystifying-long-cot]]
-->

# TokenSkip: Controllable Chain-of-Thought Compression in LLMs
- **Core Insight:** LoRA fine-tuning Qwen2.5-14B-Instruct on its own correct GSM8K CoTs, pruned by LLMLingua-2 token importance, reduces average CoT length from 313.11 to 180.68 tokens at γ = 0.6 while GSM8K accuracy moves from 93.1 to 92.7 (Table 2; Abstract: "40% (from 313 to 181)"); on MATH-500 with LLaMA-3.1-8B-Instruct, γ = 0.7 gives 349.13 vs 502.60 tokens, 46.7 vs 48.6 accuracy, and 1.4× lower latency (Table 1).
- **Guideline:** When a reasoning model's CoT must be shortened to a chosen ratio at low training cost, fine-tune it on its own correct CoTs pruned by a token-importance model with the ratio placed in the input, because on LLaMA-3.1-8B-Instruct this kept actual ratios within 0.08 of the target for γ ≥ 0.5 on GSM8K and MATH-500, while token-efficient prompts reached actual ratios of 0.94-0.97 on MATH-500 (Table 1). Below γ = 0.5 ratio adherence degraded (Fig. 6), and models above 14B and long-CoT models such as QwQ-32B-Preview were not tested (Limitations).
- **Authors:** Heming Xia, Chak Tou Leong, Wenjie Wang, Yongqi Li, Wenjie Li (The Hong Kong Polytechnic University; University of Science and Technology of China)
- **Year:** 2025 (arXiv v1 2025-02; v3 2025-09; EMNLP 2025 long paper per the arXiv comment)
- **URL:** https://arxiv.org/abs/2502.12067 (code and checkpoints: github.com/hemingkx/TokenSkip)
- **Source type:** paper
- **Relevant topics:** CoT compression, long-to-short reasoning, token pruning, LLMLingua-2, self-generated SFT data, LoRA, controllable output length, reasoning efficiency, out-of-domain retention

## Abstract
Longer chain-of-thought (CoT) outputs improve reasoning but raise inference latency linearly with length, which the authors describe as a problem once CoTs exceed 10,000 tokens. The authors measure the semantic importance of tokens inside CoT outputs and find that tokens contribute unequally to reasoning. TokenSkip lets an LLM skip less important tokens so that CoT compression is controllable. Across models and tasks it reduces CoT tokens while keeping reasoning performance; on Qwen2.5-14B-Instruct it cuts reasoning tokens on GSM8K by 40% (313 to 181) with less than a 0.4% performance drop (Abstract).

## Key Contributions
- A token-importance analysis of CoT outputs: equations and numbers score higher than connectors such as "so" and "since" under LLMLingua-2 (§2.1, Fig. 2).
- A CoT recovery check: LLaMA-3.1-8B-Instruct and GPT-4o can restore full CoTs from compressed ones (§2.2, Fig. 3, App. A).
- The TokenSkip method: quantile-threshold pruning (Eq. 3-4), training format `Q [EOS] γ [EOS] Compressed CoT A` (§3.2), and inference with a chosen γ (§3.3).
- Comparisons with token-efficient prompts, a length-control prompt, and hard truncation (Table 1), scaling across Qwen2.5-Instruct sizes (Fig. 5, Table 2), and out-of-domain tests (App. C, Tables 4-5).

## Key Figures/Tables to Study
- **Table 1** — LLaMA-3.1-8B-Instruct on GSM8K and MATH-500: accuracy, tokens, latency, actual ratio for all methods.
- **Table 2 / Figure 5** — Qwen2.5-3B/7B/14B-Instruct on GSM8K across γ.
- **Figure 6** — target vs actual ratio, including the "More Ratio" variant with γ = 0.3 and 0.4.
- **Figure 8** — importance metrics: LLMLingua-2, Selective Context, GPT-4o trimming.
- **Figure 9** — MATH-500 with the original length budget instead of max_len×γ.
- **Tables 4-5** — MATH-trained model on GSM8K and MMLU-STEM; CommonsenseQA.

## Technical Details
**Token importance (§2.1).** Selective Context: I_1(x_i) = −log P(x_i | x_<i; θ_ML), with ML a causal LLM (Eq. 1). LLMLingua-2: I_2(x_i) = P(x_i | x_≤n; θ_MB), with MB a bidirectional BERT-like model trained on GPT-4 token labels (Eq. 2). TokenSkip uses LLMLingua-2 (§4.1).

**Pruning (§3.1).** For a CoT c = {c_i}, i = 1..m, from target model M and ratio γ ∈ [0, 1], importance scores are ranked in descending order and the γ-quantile I_γ = Q_γ(I(c_1), ..., I(c_m)) is the threshold (Eq. 3). Tokens with I(c_i) ≥ I_γ are kept: c̃ = {c_i | I(c_i) ≥ I_γ} (Eq. 4). γ is therefore the target fraction of CoT tokens kept; Table 1 reports the realized value as ActRatio.

**Training (§3.2).**
1. Generate N CoTs with the target model M on the training set.
2. Remove trajectories with incorrect answers.
3. Prune each remaining CoT with γ sampled from {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}; γ = 1.0 keeps original CoTs in the mix (§3.2, App. B.1).
4. Format each sample as `Q [EOS] γ [EOS] Compressed CoT A`; the answer A is not compressed.
5. Train on the log-likelihood of the output y = (c̃_1..c̃_m′, a_1..a_t) given x and γ (Eq. 5, printed without a minus sign although the text says "minimizing").

**Inference and evaluation (§3.3, §4.1, App. B.1).** The prompt is `Q [EOS] γ [EOS]`. Greedy decoding; accuracy via DeepSeek-Math scripts; latency on one RTX 3090 at batch size 1. max_len is 512 for GSM8K and 1024 for MATH; for TokenSkip on MATH-500 the budget is max_len×γ because many samples reached the limit (App. B.1 footnote 4). MATH-500 is the Lightman et al. test subset.

**Results, LLaMA-3.1-8B-Instruct (Table 1).**
- GSM8K original: 86.2 accuracy, 213.17 tokens, 5.96 s. TokenSkip γ = 0.9: 86.1, 198.01 tokens, ActRatio 0.93; γ = 0.7: 82.5, 150.12, 0.70, 1.4×; γ = 0.5: 78.2, 113.05, 0.53, 1.8×.
- MATH-500 original: 48.6, 502.60 tokens, 16.37 s. TokenSkip γ = 0.7: 46.7, 349.13, 0.69, 1.4×; γ = 0.5: 40.2, 292.17, 0.58, 1.7×.
- Prompt baselines: BeConcise, OnlyNumbers, AbbreWords reach ActRatio 0.94-0.97 on MATH-500; LC-Prompt with target 0.5 reaches 0.89 on GSM8K and 0.94 on MATH-500.
- Truncation at 0.5: GSM8K 7.0 (79.2-point drop), MATH-500 27.4 (21.2-point drop).

**Results, Qwen2.5-Instruct on GSM8K (Table 2).** 3B: original 83.7 / 314.87 tokens; γ = 0.5: 74.4 / 170.55. 7B: 91.4 / 297.83; γ = 0.5: 86.0 / 151.44. 14B: 93.1 / 313.11; γ = 0.7: 93.4 / 218.62; γ = 0.5: 91.4 / 156.85. The authors conclude that larger models lose less accuracy at the same ratio (§4.2; Interpretation).

**Analyses (§4.3).**
- Training with extra ratios 0.3 and 0.4 ("More Ratio") gives worse adherence at those ratios and worse overall adherence than the default set (Fig. 6).
- Tokens skipped at γ = 0.7 (Qwen2.5-14B-Instruct, GSM8K test) have lower LLMLingua-2 importance than retained tokens (Fig. 7).
- LLMLingua-2 outperforms Selective Context; GPT-4o (gpt-4o-2024-08-06) trimming does better, but its API cost is impractical at dataset scale (Fig. 8).
- With the original 1024-token budget on MATH-500, γ = 0.7, 0.8, and 0.9 exceed the original model by 1.3 to 2.6 points (Fig. 9).
- In case studies, TokenSkip keeps the number of reasoning steps and removes tokens within steps; numbers and equations are usually retained (Fig. 10).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| TokenSkip LoRA on LLaMA-3.1-8B-Instruct and Qwen2.5-3B/7B/14B-Instruct | 3B-14B | distill-SFT (self-generated, pruned) | training data | target model's own CoTs on GSM8K train (7,473) or MATH train (7,500); incorrect-answer CoTs removed | arXiv:2502.12067v3 §1, §3.2 | verified 2026-09-14 | no ablation reported |
| TokenSkip LoRA, all four models | 3B-14B | distill-SFT | importance model | LLMLingua-2 | §4.1; App. B.1 | verified 2026-09-14 | Fig. 8: above Selective Context (LLaMA-3.1-8B, GSM8K) |
| TokenSkip LoRA, all four models | 3B-14B | distill-SFT | ratio set (sampled per example) | {0.5, 0.6, 0.7, 0.8, 0.9, 1.0} | §4.1; App. B.1 | verified 2026-09-14 | Fig. 6: adding 0.3 and 0.4 lowers adherence |
| TokenSkip LoRA, all four models | 3B-14B | distill-SFT | LoRA rank; α; trainable share | 8; 16; 0.2% of parameters (Qwen2.5-14B-Instruct) | App. B.1; §1 | verified 2026-09-14 | no ablation reported |
| TokenSkip LoRA, all four models | 3B-14B | distill-SFT | epochs; peak LR; schedule; warmup; optimizer | 3; 5e-5; cosine decay; warmup ratio 0.1; AdamW | App. B.1 | verified 2026-09-14 | no ablation reported |
| TokenSkip LoRA, all four models | 3B-14B | distill-SFT | framework; hardware; wall time | LLaMA-Factory; 2× RTX 3090 (24GB); ~2 h (7B), ~2.5 h (14B) | §1; §4.1; App. B.1 | verified 2026-09-14 | not applicable |
| TokenSkip LoRA, all four models | 3B-14B | eval-gate | max_len (maximum tokens at inference) | 512 (GSM8K); 1024 (MATH), ×γ for TokenSkip on MATH-500 | App. B.1, footnote 4 | verified 2026-09-14 | Fig. 9: original budget scores higher |
| TokenSkip LoRA, all four models | 3B-14B | distill-SFT | batch size, max sequence length, decoding used to generate training CoTs, loss masking | not reported | checked §3, §4.1, App. B | not reported | — |

## Findings relevant to generality, negative feedback, and distillation
- **Distillation setup:** stage = SFT on self-generated, pruned CoTs; source model = the target model itself; compressor = LLMLingua-2 (trained on GPT-4 labels, §2.1); prompts = GSM8K or MATH training questions; quality control = answer-correctness filter (§3.2). Sampling settings for data generation are not reported.
- **Negative samples (negative marginal value):** CoTs with incorrect answers are discarded, not trained on (§3.2).
- **Out-of-domain retention (Table 4, LLaMA-3.1-8B-Instruct trained on MATH):** GSM8K 86.2 / 213.17 tokens → γ = 0.5: 76.6 / 122.55; MMLU-STEM 58.5 / 356.31 → γ = 0.9: 59.4 / 327.18, γ = 0.5: 58.1 / 188.87.
- **Beyond math (Table 5, CommonsenseQA, 9,700 training samples, validation set):** Qwen2.5-7B-Instruct 80.3 / 272.13 → γ = 0.5: 80.6 / 128.43; Qwen2.5-14B-Instruct 82.1 / 247.81 → γ = 0.5: 82.1 / 121.03.
- **Long CoT:** the motivation cites CoTs above 10,000 tokens (Abstract), but evaluated CoTs average 213-503 tokens (Table 1), and long-CoT models were excluded for compute reasons (Limitations). The authors also note that LLMLingua-2 was not trained on mathematical data (Limitations).

## Connections
- [[c3ot-compressed-cot]] — GPT-4-shortened CoTs used for fine-tuning, cited as Kang et al. 2024 (§5).
- [[l1-lcpo]] — RL-based length control, an alternative to TokenSkip's SFT-based ratio control.
- [[overthinking-o1-like-llms]], [[sky-t1-flash-overthinking]], [[kimi-k1-5]] — long-to-short methods applied to long-CoT models, the setting TokenSkip did not test.
- [[demystifying-long-cot]] — CoT length behavior in long-CoT training.
- [[rejection-sampling-finetuning]], [[star]] — fine-tuning on self-generated answers filtered by correctness.
- [[distilling-step-by-step]] — rationale-based distillation into smaller models.
- [[lora-learns-less-forgets-less]] — LoRA fine-tuning trade-offs; TokenSkip uses rank-8 LoRA.
- [[training-verifiers-to-solve-math-word-problems]], [[let-verify]] — sources of GSM8K and the MATH-500 subset.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2502.12067 (arXiv v3, 2025-09-16; v1 2025-02-17).
- Audit claims not found in the source: none. The audit's "<0.4 drop" is printed as "less than 0.4%" while Table 2 shows 0.4 points (93.1 → 92.7); the audit's "<4% drop" for MATH-500 matches the text, and Table 1 shows 1.9 points at γ = 0.7.
- Not reported by the source: batch size, sequence length, data-generation decoding settings, results above 14B or on long-CoT models, variance across seeds.

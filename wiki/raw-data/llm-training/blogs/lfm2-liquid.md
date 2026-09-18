<!-- scope: Liquid AI LFM2 launch blog (2025-07-10): 350M/700M/1.2B hybrid conv+GQA on-device models; pre-training on 10T tokens with LFM1-7B as knowledge-distillation teacher for the whole run; large-scale SFT; length-normalized DPO on offline + semi-online pairs; checkpoint merging.
     deps: [[pretraining-distillation-design-space]], [[dpo]]
     see-also: [[distillation-scaling-laws]], [[minitron-approach]], [[gemma-3]], [[llama-3-2]], [[mix-data-or-merge-models]], [[wildchat]]
-->

# Introducing LFM2: The Fastest On-Device Foundation Models on the Market
- **Core Insight:** All three LFM2 checkpoints (350M, 700M, 1.2B) were pre-trained on 10T tokens with the cross-entropy between their outputs and the outputs of LFM1-7B as the primary training signal for the entire run, and LFM2-1.2B scores 55.23 MMLU and 74.89 IFEval against 59.11 and 73.98 for Qwen3-1.7B ("Training LFM2"; "Automated benchmarks" table).
- **Guideline:** When a small on-device model is pre-trained and a larger in-house model of the same lineage exists, Liquid AI's practice is to use that model as a distillation teacher throughout pre-training; the blog gives no ablation against training without the teacher, so the size of the benefit is not reported ("Training LFM2").
- **Authors:** Liquid AI (no individual authors listed)
- **Year:** 2025 (blog published 2025-07-10; no arXiv version)
- **URL:** https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models
- **Source type:** official blog
- **Relevant topics:** pre-training knowledge distillation, small models, hybrid convolution-attention architecture, neural architecture search, SFT on downstream tasks, length-normalized DPO, semi-online preference data, LLM-judge scoring, model merging, on-device inference

## Summary
The post releases LFM2, three dense checkpoints with 0.35B, 0.7B, and 1.2B parameters built for on-device generation ("Quick picks on LFM2"). The architecture has 16 blocks: 10 double-gated short-range convolution blocks and 6 grouped query attention (GQA) blocks, and was selected with Liquid AI's STAR architecture search engine using over 50 internal evaluations and measured memory and speed on Qualcomm Snapdragon CPUs ("LFM2 Architecture"). Pre-training uses 10T tokens and LFM1-7B as a knowledge-distillation (KD) teacher; post-training uses large-scale SFT, a custom length-normalized DPO on offline and semi-online data, and merging of selected checkpoints ("Training LFM2"). The post reports seven automated benchmarks, a pairwise LLM-jury evaluation on 1,000 WildChat conversations, and CPU throughput in ExecuTorch and llama.cpp ("Benchmarks"; "Inference"). A banner states that LFM2.5 was released in 2026 (page header).

## Key Contributions
- Pre-training KD from LFM1-7B over the full 10T-token run ("Training LFM2").
- A hybrid block layout: 10 gated short-convolution blocks and 6 GQA blocks; the sentence on the GQA blocks adds that "each block contains a SwiGLU and an RMSNorm layer" ("LFM2").
- Architecture search scored by over 50 capability evaluations and on-device peak memory and prefill+decode speed, instead of validation loss and KV-cache size ("LFM2").
- A post-training pipeline: SFT that includes downstream tasks such as RAG and function calling, length-normalized DPO on offline and semi-online pairs, and merging of candidate checkpoints ("Training LFM2").
- Open weights under a license based on Apache 2.0, with free commercial use for companies under $10m revenue ("Build with LFM2").

## Key Figures/Tables to Study
- Fig. 1: average of MMLU, IFEval, IFBench, GSM8K, MMMLU against model size ("Automated benchmarks").
- Benchmark table: seven benchmarks for three LFM2 checkpoints and four baselines ("Automated benchmarks").
- LLM-as-a-Judge preference charts on WildChat conversations ("LLM-as-a-Judge").
- Throughput plots on CPU in ExecuTorch and llama.cpp ("Inference").
- `lfm2_conv` pseudocode for the gated convolution block ("LFM2").

## Technical Details
**Architecture.** A linear input-varying (LIV) operator is a linear operator whose weights are generated from its input ("Systematic Neural Architecture Search in LIV Operators"). The LFM2 convolution block computes `B, C, x = linear(x)`, then `x = B*x`, a short convolution, `x = C*x`, and an output linear layer ("LFM2" pseudocode). The model has 16 blocks: 10 double-gated short-range LIV convolutions and 6 GQA blocks ("LFM2"). The authors state that the use of short convolutions follows from the target device class (embedded SoC CPUs) and the kernel libraries optimized for it ("LFM2").

**Pre-training data and KD.** All three sizes were trained on 10T tokens from a corpus of approximately 75% English, 20% multilingual, and 5% code, sourced from the web and licensed materials ("Training LFM2"). The multilingual focus is Japanese, Arabic, Korean, Spanish, French, and German ("Training LFM2"). LFM1-7B is the teacher; "the cross-entropy between LFM2's student outputs and the LFM1-7B teacher outputs" is "the primary training signal throughout the entire 10T token training process" ("Training LFM2"). The blog does not state whether the teacher distribution is full or truncated (top-k), the temperature, or whether a next-token hard-label term is added. The context length was extended to 32k during pre-training ("Training LFM2").

**SFT.** Post-training starts with a "very large-scale" SFT stage on a diverse mixture ("Training LFM2"). For these small models the authors "found it beneficial to directly train on a representative set of downstream tasks, such as RAG or function calling" ("Training LFM2"). Data are open-source, licensed, and targeted synthetic data, filtered by "quantitative sample scoring and qualitative heuristics" ("Training LFM2").

**Preference optimization and merging.** A custom DPO algorithm with length normalization is trained on offline data and semi-online data ("Training LFM2"). Semi-online data: (1) sample multiple completions from the model for prompts from a seed SFT dataset; (2) score all responses with LLM judges; (3) pair the highest- and lowest-scored completions among the SFT and on-policy samples; (4) filter both offline and semi-online sets by a score threshold ("Training LFM2"). Multiple candidate checkpoints are trained with different hyperparameters and mixtures, and a selection is combined "via different model merging techniques" ("Training LFM2").

**Evaluation protocol.** Benchmarks: 5-shot MMLU, 0-shot GPQA, IFEval, IFBench, 0-shot GSM8K, 5-shot MGSM, 5-shot OpenAI MMMLU in seven languages ("Automated benchmarks"). Scores come from Liquid AI's internal suite, which differs from lm-evaluation-harness in stripping whitespace from the most likely logit, consolidating math answer extraction (which "particularly improved results for Gemma 3 1B IT"), and running Qwen3 in non-reasoning mode, which "consistently improved scores" because reasoning traces tend to be longer than output budgets for edge deployment (<4,096 tokens) ("Automated benchmarks"). The LLM-as-a-Judge test used 1,000 real-world WildChat conversations and a jury of five LLMs giving pairwise preferences ("LLM-as-a-Judge").

**Reported scores (LFM2-350M / 700M / 1.2B).** MMLU 43.43 / 49.9 / 55.23; GPQA 27.46 / 28.48 / 31.47; IFEval 65.12 / 72.23 / 74.89; IFBench 16.41 / 20.56 / 20.7; GSM8K 30.1 / 46.4 / 58.3; MGSM 29.52 / 45.36 / 55.04; MMMLU 37.99 / 43.28 / 46.73 ("Automated benchmarks" table). Qwen3-1.7B: MMLU 59.11, GSM8K 51.4, MGSM 66.56; gemma-3-1b-it GSM8K 59.59 (same table). The blog describes LFM2-1.2B as competitive with Qwen3-1.7B, which has a 47% larger parameter count ("Automated benchmarks").

**Efficiency.** The blog reports 3x faster training than the previous LFM generation and up to 2x faster decode and prefill than Qwen3 on CPU ("Quick picks on LFM2"). On-device tests used 8da4w quantization (ExecuTorch) and Q4_0 (llama.cpp) on a Samsung Galaxy S24 Ultra and an AMD Ryzen HX370; LFM2-700M is "consistently faster than Qwen-0.6B" (model name as printed) on decode and prefill in both frameworks while being 16% larger ("Inference").

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LFM2-350M, LFM2-700M, LFM2-1.2B (all three) | 0.35B / 0.7B / 1.2B | pretrain-stable | tokens seen | 10T | blog "Training LFM2" | verified 2026-09-14 | no ablation reported |
| same | same | pretrain-stable | corpus mixture (share type not stated) | ~75% English, 20% multilingual, 5% code | blog "Training LFM2" | verified 2026-09-14 | no ablation reported |
| same | same | pretrain-stable | KD teacher; loss | LFM1-7B; cross-entropy between student and teacher outputs, primary signal for all 10T tokens | blog "Training LFM2" | verified 2026-09-14 | no ablation reported |
| same | same | pretrain-stable | KD top-k, temperature, hard-label weight | not reported | blog (all sections) | not reported | — |
| same | same | long-context | context length | extended to 32k during pre-training | blog "Training LFM2" | verified 2026-09-14 | no ablation reported |
| same | same | pretrain / SFT / preference | LR, batch, optimizer, epochs | not reported | blog (all sections) | not reported | — |
| same | same | SFT | data | open-source, licensed, targeted synthetic; includes RAG and function calling; size not reported | blog "Training LFM2" | verified 2026-09-14 | "found it beneficial" (no numbers) |
| same | same | preference | loss; β | custom DPO with length normalization; β not reported | blog "Training LFM2" | verified 2026-09-14 | no ablation reported |
| same | same | preference | pair construction | highest vs lowest LLM-judge score among SFT and on-policy samples; score-threshold filter | blog "Training LFM2" | verified 2026-09-14 | no ablation reported |
| same | same | merge | method and weights | "different model merging techniques"; weights not reported | blog "Training LFM2" | verified 2026-09-14 | no ablation reported |

## Findings relevant to generality and distillation
- **Distillation.** Stage: pre-training, all 10T tokens. Prompt types: the pre-training corpus itself (75/20/5 English/multilingual/code). Teacher: LFM1-7B; sampling, top-k, and temperature not reported. Quality control for KD targets: not reported ("Training LFM2").
- **Narrow vs broad SFT (Result, single study, no numbers).** The authors report that training small models directly on downstream tasks such as RAG and function calling was beneficial; no comparison, metric, or effect size is given ("Training LFM2").
- **Semi-online negatives (negative as gradient).** The lowest-judge-scored completion is the rejected side of a DPO pair; the judge models, the threshold value, and false-negative rates are not reported ("Training LFM2").
- **Measurement caveat.** Baselines were scored with a modified harness and Qwen3 in non-reasoning mode, so numbers are not directly comparable to other reports ("Automated benchmarks").

## Connections
- [[pretraining-distillation-design-space]]: design choices for KD during pre-training, the stage LFM2 applies it to.
- [[distillation-scaling-laws]]; [[capacity-gap-law-distillation]]: teacher-student size effects relevant to a 7B teacher for 0.35B-1.2B students.
- [[sparse-logit-sampling-kd]]: top-k teacher logits, a setting the blog does not report.
- [[minitron-approach]]; [[gemma-3]]; [[llama-3-2]]: other small models trained with distillation (Llama 3.2 1B is also a baseline here).
- [[dpo]]; [[simpo]]: the preference loss and a length-normalized variant.
- [[online-offline-alignment-gap]]: offline vs on-policy preference data, the axis of the semi-online set.
- [[mix-data-or-merge-models]]: merging checkpoints trained on different mixtures.
- [[wildchat]]: source of the 1,000 conversations for the LLM-jury evaluation; [[ifbench]]: one of the benchmarks.
- [[judge-llm-bias]]: judge biases that affect both the pair scoring and the jury evaluation.

## Verification
- Created on 2026-09-14 from https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models (page dated JUL 10, 2025, with a 2026 banner noting LFM2.5).
- Audit claims not found in the source: none.
- Not reported by the source: KD top-k, temperature, loss weighting; LR, batch, optimizer; SFT dataset size; DPO β; judge models; merge weights. The separate Hugging Face card for LFM2-1.2B (huggingface.co/LiquidAI/LFM2-1.2B, "Training approach") states an SFT split of 50% downstream tasks and 50% general domains; that split is not in the blog.

---
chapter: ch-35a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/model-reports/llama-nemotron.md on 2026-09-15)
source_url: https://arxiv.org/abs/2505.00949
source_version: arXiv v5 (2025-09-09); v1 2025-05
created_at: "2026-09-15"
---

# Excerpt: Llama-Nemotron: Efficient Reasoning Models (NVIDIA)

Facts used by [[read]], read in the arXiv v5 PDF text on 2026-09-15. Models: LN-Nano (8B), LN-Super (49B), LN-Ultra (253B), derived from Llama 3.1 and Llama 3.3.

## Mode control and data (§3, Table 2)
- Reasoning samples carry the system instruction "detailed thinking on"; non-reasoning samples carry "detailed thinking off".
- Math: the OpenMathReasoning AoPS pipeline; proof, multiple-choice, binary, and invalid problems are removed; LLM-based benchmark decontamination; DeepSeek-R1 produces reasoning solutions (16 per problem) and Qwen2.5-Math-7B-Instruct non-reasoning solutions (64 per problem); Qwen2.5-32B-Instruct judges answer equivalence; the most common answer serves as ground truth when none can be extracted (§3.1.1).
- Code: 28,904 questions from OpenCodeReasoning; DeepSeek-R1 at temperature 0.6, top-p 0.95; about 488K Python samples after post-processing (§3.1.2).
- Science: StackOverflow and synthetic multiple-choice questions; decontamination against GPQA, MMLU, MMLU-Pro; DeepSeek-R1 traces; majority vote where ground truth is missing (§3.1.3).
- General: DeepSeek-R1 generations with rejection sampling by the Llama-3.1-Nemotron-70B reward model (§3.1.4).
- Reasoning-off pairs: prompts sampled from the reasoning set receive non-reasoning responses from Llama-3.1-Nemotron-70B-Instruct (general) or Llama-3.3-70B-Instruct (others), filtered by ground truth or reward models (§3.2).
- Table 2 (samples, share): Math 22,066,397 (66.8%; on 2,225,427, off 19,840,970); Code 10,108,883 (30.6%; on 991,706, off 9,117,177); Science 708,920 (2.1%, all on); Chat 39,792 (0.12%); Instruction following 56,339 (0.17%); Safety 31,426 (0.10%); total 33,011,757.

## SFT (§4)
- Token-level cross-entropy; batches mix reasoning and non-reasoning data; "models require higher learning rates to effectively learn from long reasoning traces, especially due to sequence-length-dependent token loss averaging"; multiple epochs help, "particularly for smaller models"; Adam; cosine decay with linear warmup to about 10% of steps (§4.1).
- LN-Nano: global batch 256, packing to 32k tokens; stage 1 reasoning-only data at LR 1e-4 for four epochs ("prevents failure modes such as repetitive completions"); stage 2 adds non-reasoning data; stage 3 a smaller chat, instruction-following, and tool-calling blend (§4.2).
- LN-Super: full SFT set, one epoch, fixed LR 5e-6, sequence length 16k, global batch 256; smaller runs "suggested that performance improves up to 3-4 epochs with larger learning rates (5e-5)"; rejection fine-tuning gave no gain (§4.2).
- LN-Ultra: packing to 24k, global batch 256; warmup to 1e-5 then cosine to 1e-6, warmup ratio 10%; gradient explosions after the first epoch required resuming with reinitialized optimizer states (§4.2).

## Distillation ceiling and RL (§5, §7.4)
- "distillation inherently sets an upper bound on the student's performance ... Using supervised fine-tuning, LN-Ultra can approach the performance of DeepSeek-R1 but not exceed it"; preliminary experiments found RL "suboptimal" for smaller models compared with distillation, so reasoning RL was applied only to LN-Ultra.
- LN-Ultra RL data: 8 LN-Super responses per question; prompts with pass rate 0.75 or higher are discarded (§5.1).
- "Although we had access to SFT checkpoints with higher benchmark scores, we initialized RL from an earlier checkpoint to improve final RL outcomes" (§7.4).
- Table 5 (reasoning on): GPQA-Diamond LN-Ultra-SFT 66.4, LN-Ultra 76.0, DeepSeek-R1 71.5; AIME25 60.4, 72.5, 70.0; IFEval 83.2, 88.9, 88.8.

## Generality measurements (§7)
- Table 3 LN-Nano (on | off): AIME24 61.3 | 3.0; IFEval 79.29 | 82.1; BFCL V2 Live 63.9 | 63.6.
- Table 4: "reasoning-focused SFT causes a noticeable drop in IFEval scores"; LN-Super-SFT IFEval 81.9 (on) vs Llama-3.3-70B-Instruct 92.1; the released LN-Super, after a dedicated IFEval RL run (§6.1) and the later alignment stages, scores 89.2.
- Training context 16k (LN-Super) and 24k (LN-Ultra); all evaluations at 32k; up to 16 completions per prompt.

## Not reported
Teacher temperature for math and science generations; SFT epochs for LN-Ultra; decontamination n-gram lengths.

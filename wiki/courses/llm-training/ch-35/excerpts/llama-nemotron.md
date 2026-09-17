---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: arXiv:2505.00949v5 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2505.00949
created_at: "2026-09-15"
---

# Excerpt: Llama-Nemotron: Efficient Reasoning Models

- **Authors:** NVIDIA (byline)
- **Year:** 2025 (arXiv v1 2025-05; v5 2025-09-09)
- **Source type:** official technical report
- **Used in:** ch-35 §6.1, Recipe, Generalization lens
- **Not the same report as:** the Nemotron 3 card `nemotron-ultra`.

## Models and post-NAS distillation (§1, §2.2)
- LN-Nano (8B), LN-Super (49B), LN-Ultra (253B), derived from Llama 3.1 and Llama 3.3 by neural architecture search.
- LN-Super: knowledge distillation for 40B tokens on the Distillation Mix dataset. LN-Ultra: 65B tokens of distillation, then 88B tokens of continued training on Nemotron-H phase 4 pretraining data (§2.2). Table 1: LN-Ultra after CPT scores MMLU 88.1 vs Llama-3.1-405B-Instruct 88.6.

## Teacher data by domain (§3)
- Reasoning samples carry the system instruction "detailed thinking on"; non-reasoning samples carry "detailed thinking off" (§3).
- Math (§3.1.1): problems from AoPS; proof, multiple-choice, binary, and invalid problems removed; LLM-based benchmark decontamination. DeepSeek-R1 produces reasoning solutions (16 generations per problem) and Qwen2.5-Math-7B-Instruct produces non-reasoning solutions (64 per problem). Qwen2.5-32B-Instruct judges answer equivalence; if no answer can be extracted, the most common answer is the ground truth.
- Code (§3.1.2): 28,904 unique competitive-programming questions; R1 solutions at temperature 0.6, top-p 0.95; post-processing yields about 488K Python samples. Scaling from 25k to 736k samples "showed continuous improvement".
- Science (§3.1.3): R1 traces; majority voting where no ground truth exists.
- General (§3.1.4): R1 generations with rejection sampling by the Llama-3.1-Nemotron-70B reward model.
- Reasoning off (§3.2): prompts sampled from the reasoning set receive non-reasoning responses from Llama-3.1-Nemotron-70B-Instruct (general) or Llama-3.3-70B-Instruct (other domains), filtered by ground truth or reward models.
- Table 2: 33,011,757 samples in total; math 22,066,397 (2,225,427 reasoning on); code 10,108,883 (991,706 on); science 708,920 (all on); chat 39,792; instruction following 56,339; safety 31,426.

## SFT settings (§4.1, §4.2)
- Token-level cross-entropy; batches mix reasoning and non-reasoning data. "Models require higher learning rates to effectively learn from long reasoning traces."
- LN-Nano: global batch 256, packing to 32k; stage 1 reasoning-only data at LR 1e-4 for four epochs; stage 2 adds non-reasoning data; stage 3 chat, instruction following, tool calling.
- LN-Super: one epoch, fixed LR 5e-6, sequence length 16k, global batch 256; smaller runs improved up to 3–4 epochs at 5e-5; rejection fine-tuning gave no gain.
- LN-Ultra: packing to 24k, global batch 256; linear warmup to 1e-5, cosine to 1e-6, warmup ratio 10%; gradient explosions after the first epoch required a restart with reinitialized optimizer states.

## SFT versus RL (§5, §7)
- "Using supervised fine-tuning, LN-Ultra can approach the performance of DeepSeek-R1 but not exceed it." RL "yields suboptimal results for smaller models compared to distillation", so reasoning RL is applied only to LN-Ultra (§5).
- LN-Ultra RL (§5.1): GRPO, 72 prompts × 16 responses, temperature 1, top-p 1, global batch 576, 2 gradient updates per rollout, about 140k H100 hours; prompts with pass rate ≥ 0.75 (8 responses from LN-Super) removed.
- Table 5 GPQA-Diamond: LN-Ultra-SFT 66.4 (reasoning on); DeepSeek-R1 71.5; LN-Ultra 76.0. "Although we had access to SFT checkpoints with higher benchmark scores, we initialized RL from an earlier checkpoint to improve final RL outcomes" (§7.4).
- Table 4 IFEval: LN-Super-SFT 81.9 (on) / 83.0 (off); LN-Super 89.2 / 89.0; DeepSeek-R1-Distilled-Llama-70B 85.1; Llama-3.3-70B-Instruct 92.1. LN-Nano-SFT (Table 3) IFEval 69.9 / 69.9. "Reasoning-focused SFT causes a noticeable drop in IFEval scores", recovered with an IFEval RL run (§7.3).
- Table 3 (the report does not state whether baseline rows were evaluated by NVIDIA or copied; §7.1 states that all evaluations use 32k context): DeepSeek-R1-Distilled-Llama-8B IFEval 73.4, BFCL V2 Live 37.8; Llama-3.1-8B-Instruct 81.8, 44.3; LN-Nano (on) 79.29, 63.9.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2505.00949 (v5, 2025-09-09): §2.2, §3, Table 2, §4, §5.1, §7, Tables 3–5.
- Not reported by the source: token counts of the SFT set; per-domain sampling temperatures other than code; RL step count for LN-Ultra.

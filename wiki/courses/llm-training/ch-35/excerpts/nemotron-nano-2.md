---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: arXiv:2508.14444 (NVIDIA PDF dated 2025-09-02; no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2508.14444
created_at: "2026-09-15"
---

# Excerpt: NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model

- **Authors:** NVIDIA (byline)
- **Year:** 2025 (arXiv v1 2025-08; text checked is dated 2025-09-02)
- **Source type:** official technical report
- **Used in:** ch-35 §6.2, Recipe, Generalization lens

## Pipeline (Abstract, §3, §4)
Nemotron-Nano-12B-v2-Base is pre-trained on 20T tokens, aligned (SFT 1–3, IFEval RL, DPO, GRPO-based RLHF, merge), then compressed to Nemotron-Nano-9B-v2 with the Minitron pruning-and-distillation strategy.

## SFT teacher data (§3.1, Table 7)
- About 80 billion tokens of prompt-response pairs in total (§3.1).
- Math, science, code: responses from DeepSeek-R1-0528 on the prompts used for Nemotron-H reasoning models.
- Tool calling: responses from Qwen3-235B-A22B; multi-turn tool data simulated with Qwen3-235B-A22B as user, assistant, and API server; only successful trajectories kept.
- Conversation: Qwen3-235B-A22B reasoning responses on LMSYS, HelpSteer2/3, and about 550k WildChat prompts; multi-turn R1 responses.
- Samples (Table 7): math 1.5M, coding 1.1M, science 2.0M, tool calling 400K, conversational 1.5M, safety 2K, multilingual 5.0M.

## Trace truncation and mode control (§1, §3.2, §3.4)
- Stage 1 SFT: full dataset plus "a subsample of roughly 10% of prompts paired with outputs stripped of reasoning traces", packed to about 128k tokens.
- Stage 2 SFT: tool calling without concatenation, because Stage 1 degraded tool-calling accuracy.
- Stage 3 SFT: long-context data plus "examples across domains where reasoning traces were abruptly truncated to 1–2k tokens while preserving the final answer".
- §1: "About 5% of the data contained deliberately truncated reasoning traces, enabling fine-grained thinking budget control at inference time."
- §3.4, Figure 5: without truncated examples the model lengthens the final answer to compensate for a short thinking budget, and well-formed responses drop sharply at short budgets; with truncation training both effects are absent (values shown only in the figure).

## Compression with distillation (§4.3)
- "We adopt logit-based distillation for continued training, employing forward KL divergence loss exclusively during the accuracy recovery phase."
- Reasoning model stages: (1) depth pruning to 56 layers, KD ~60B tokens at 8,192; (2) width pruning, KD ~50B tokens at 8,192, ~25B at 49,152, ~1B at 262,144; (3) DPO; (4) GRPO; (5) KD ~0.4B tokens at 262,144 "to recover post-RL drops"; (6) RLHF; (7) merge of steps 5 and 6 at 0.5.
- Figure 6: GRPO improves instruction following but "temporarily degrades multi-task understanding (MMLU-Pro), which is recovered in the next step (post-GRPO KD)"; RLHF drops are recovered by merging (values shown only in the figure).
- Table 11 (math accuracy average after ~6B tokens of KD): 50% reasoning-SFT / 50% pretraining data 57.5; 70/30 58.5; 90/10 57.2.

## Verification
- Checked on 2026-09-15 against the arXiv:2508.14444 PDF text (document dated 2025-09-02): Abstract, §1, §3.1–3.4, Table 7, §4.3, Table 11, §5.
- Not reported by the source: SFT learning rates and epochs; the token share of truncated examples (only the sample-level "about 5%"); KD learning rates.

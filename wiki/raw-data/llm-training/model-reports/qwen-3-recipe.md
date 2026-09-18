<!-- scope: recipe ledger for the Qwen3 technical report, split out of [[qwen-3]]
     deps: [[qwen-3]]
     see-also: [[qwen-2.5-recipe]], [[phi-4-recipe]]
-->

# Qwen3 Technical Report — recipe ledger
- **Core Insight:** The report discloses stage-level token counts, sequence lengths and the Table 21
  distillation-versus-RL comparison, but not the RL optimizer settings.
- **Guideline:** When reusing a value here, carry the Stage column with it, because Qwen3's pretraining,
  reasoning-stage and long-context values differ and the RL numbers apply to named checkpoints only.
- **Authors:** Qwen Team (Alibaba)
- **Year:** 2025 (arXiv v1 2025-05)
- **URL:** https://arxiv.org/abs/2505.09388
- **Source type:** official technical report
- **Relevant topics:** recipe ledger, pretraining stages, reasoning RL, on-policy distillation, GPU hours

## Summary
This card holds the `## Recipe ledger` for [[qwen-3]] and adds no claims of its own. Every row is read at
the locus in its Source location column.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen3 (all) | 0.6B-235B-A22B | pretrain-stable (S1) | tokens seen; sequence length | over 30T; 4,096 | arXiv:2505.09388 §3.2 | verified 2026-09-18 | no ablation reported |
| Qwen3 (all) | 0.6B-235B-A22B | mid-train (S2, reasoning) | tokens seen; sequence length; LR | about 5T; 4,096; accelerated decay | arXiv:2505.09388 §3.2 | verified 2026-09-18 | no ablation reported |
| Qwen3 (all) | 0.6B-235B-A22B | long-context (S3) | tokens; sequence length; RoPE base; corpus | hundreds of billions; 32,768; 10,000 → 1,000,000 (ABF); 75% of texts 16,384-32,768 tokens, 25% 4,096-16,384 | arXiv:2505.09388 §3.2 | verified 2026-09-18 | no ablation reported |
| Qwen3 (all) | — | pretrain total | tokens; languages and dialects | 36T; 119 | arXiv:2505.09388 §3.1 | verified 2026-09-18 | no ablation reported |
| Qwen3 (all) | — | pretrain | LR schedule and batch size | set per model from scaling-law predictions; no numbers printed | arXiv:2505.09388 §3.2 | not reported (checked §3.2, §2, tables 1-2) | — |
| Qwen3-235B-A22B | 235B total / 22B activated | RL (Reasoning RL) | prompts; algorithm; steps | 3,995 query-verifier pairs; GRPO; 170 steps | arXiv:2505.09388 §4.2 | verified 2026-09-18 | §4.2: AIME'24 70.1 → 85.1 over a single run with no manual hyperparameter intervention |
| Qwen3 flagship | 32B, 235B-A22B | RL (Reasoning RL) | batch size; rollouts per query; KL coefficient; clip ε; LR | described only as "a large batch size" and "a high number of rollouts per query" | arXiv:2505.09388 §4.2 | not reported (checked §4.2, §4.4, §5.3; no appendix hyperparameter table) | — |
| Qwen3 lightweight | 0.6B, 1.7B, 4B, 8B, 14B, 30B-A3B | distill-SFT (off-policy) | data | teacher outputs generated in both `/think` and `/no think` modes | arXiv:2505.09388 §4.5 | verified 2026-09-18 | no ablation reported |
| Qwen3 lightweight | 0.6B, 1.7B, 4B, 8B, 14B, 30B-A3B | distill-SFT (on-policy) | objective; teachers | KL divergence between student and teacher logits; Qwen3-32B or Qwen3-235B-A22B | arXiv:2505.09388 §4.5 | verified 2026-09-18 | §4: higher pass@1 and improved pass@64 versus the four-stage process, at ~1/10 of the GPU hours |
| Qwen3-8B | 8B | distill-SFT (off-policy; the shared starting checkpoint) | AIME'24 pass@1 (pass@64); AIME'25; MATH500; LiveCodeBench v5; MMLU-Redux; GPQA-Diamond | 55.0 (90.0); 42.8 (83.3); 92.4; 42.0; 86.4; 55.6 | arXiv:2505.09388 Table 21 | verified 2026-09-18 | Table 21 baseline row; GPU hours not given for this row |
| Qwen3-8B | 8B | RL (from that checkpoint) | GPU hours; AIME'24 pass@1 (pass@64); AIME'25; MATH500; LiveCodeBench v5; MMLU-Redux; GPQA-Diamond | 17,920; 67.6 (90.0); 55.5 (83.3); 94.8; 52.9; 86.9; 61.3 | arXiv:2505.09388 Table 21 | verified 2026-09-18 | §5.3: pass@64 unchanged from the starting checkpoint; math and code queries only |
| Qwen3-8B | 8B | distill-SFT (on-policy, from that checkpoint) | GPU hours; AIME'24 pass@1 (pass@64); AIME'25; MATH500; LiveCodeBench v5; MMLU-Redux; GPQA-Diamond | 1,800; 74.4 (93.3); 65.5 (86.7); 97.0; 60.3; 88.3; 63.3 | arXiv:2505.09388 Table 21 | verified 2026-09-18 | §5.3: higher on every column at about 1/10 of the RL GPU hours; math and code queries only |
| Qwen3 (all) | — | eval-gate | sampling, thinking mode | temperature 0.6, top-p 0.95, top-k 20 | arXiv:2505.09388 §4.6 | verified 2026-09-18 | no ablation reported |
| Qwen3 (all) | — | eval-gate | sampling, non-thinking mode | temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5 | arXiv:2505.09388 §4.6 | verified 2026-09-18 | no ablation reported |
| Qwen3 (all) | — | eval-gate | max output length | 32,768 tokens; 38,912 for AIME'24 and AIME'25 | arXiv:2505.09388 §4.6 | verified 2026-09-18 | §5.3: performance is expected to improve beyond 32K, left as future work |

Units: "tokens seen" is training tokens, not unique tokens. GPU hours in Table 21 are for the stage applied
on top of the off-policy-distilled 8B checkpoint, not cumulative from pretraining. pass@64 values are the
numbers printed in parentheses in Table 21.

## Connections
- [[qwen-3]] — the card this ledger belongs to; it carries the abstract, contributions and findings.
- [[phi-4-recipe]] — the comparable 2025 ledger with a short GRPO stage disclosed in full.
- [[grpo]] — the algorithm named in §4.2.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2505.09388 (arXiv v1, 2025-05-14).
- Corrections to the previous card version: this card is new; it holds the ledger [[qwen-3]] previously
  lacked. Corrections to values that were in [[qwen-3]] are recorded in that card's Verification section.
- Removed as unsupported by the source: none.
- Not reported by the source: pretraining learning rate, batch size and optimizer values; RL batch size,
  rollouts per query, KL coefficient, clip ε and learning rate; the number of long-CoT cold-start samples
  or steps; the size of the Thinking Mode Fusion SFT set; total pretraining compute or GPU hours.

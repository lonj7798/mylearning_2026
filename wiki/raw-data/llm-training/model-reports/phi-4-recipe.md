<!-- scope: recipe ledger for the Phi-4 and Phi-4-reasoning technical reports, split out of [[phi-4]]
     deps: [[phi-4]]
     see-also: [[qwen-3]], [[deepseek-r1]]
-->

# Phi-4 and Phi-4-reasoning — recipe ledger
- **Core Insight:** Every training setting the two reports disclose is listed here with its locus; the
  reports disclose the full pretraining, midtraining, SFT and GRPO schedules but no compute totals.
- **Guideline:** When reusing a value from this table, copy the row's Model and Stage as well, because the
  Phi-4 and Phi-4-reasoning runs use different learning rates, batch sizes and context lengths.
- **Authors:** Microsoft Research (Phi-4); Microsoft (Phi-4-reasoning)
- **Year:** 2024 (arXiv v1 2024-12) and 2025 (arXiv v1 2025-04)
- **URL:** https://arxiv.org/abs/2412.08905 — https://arxiv.org/abs/2504.21318
- **Source type:** official technical report (two reports)
- **Relevant topics:** recipe ledger, pretraining schedule, midtraining, SFT hyperparameters, GRPO hyperparameters

## Summary
This card holds the `## Recipe ledger` for [[phi-4]]. It adds no claims of its own. Each row is read at the
locus given in the Source location column.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| phi-4 | 14B | pretrain-stable | tokens seen | ~10T | arXiv:2412.08905 §3 | verified 2026-09-18 | no ablation reported |
| phi-4 | 14B | pretrain-stable | sequence length | 4096 | arXiv:2412.08905 §3 | verified 2026-09-18 | no ablation reported |
| phi-4 | 14B | pretrain-stable | LR schedule | linear warm-up and decay, peak LR 0.0003 | arXiv:2412.08905 §3 | verified 2026-09-18 | tuned by interpolation from shorter-horizon runs and warm-up stress tests (§3) |
| phi-4 | 14B | pretrain-stable | weight decay | 0.1, constant | arXiv:2412.08905 §3 | verified 2026-09-18 | no ablation reported |
| phi-4 | 14B | pretrain-stable | global batch (sequences) | 5760 | arXiv:2412.08905 §3 | verified 2026-09-18 | no ablation reported |
| phi-4 | 14B | pretrain-stable | mixture, share of training tokens | web 15%, web rewrites 15%, synthetic 40%, code 20%, acquired sources 10% | arXiv:2412.08905 Table 5 | verified 2026-09-18 | Table 4: ablations over allocating 75% of tokens among synthetic, filtered web and web rewrites; relative to the chosen mixture, uniform allocation is worse on 7 of 9 benchmarks |
| phi-4 | 14B | pretrain-stable | unique tokens per source | web 1.3T, web rewrites 290B, synthetic 290B, code 820B, acquired 580B | arXiv:2412.08905 Table 5 | verified 2026-09-18 | no ablation reported |
| phi-4 | 14B | pretrain-stable | epochs per source | web 1.2, web rewrites 5.2, synthetic 13.8, code 2.4, acquired 1.7 | arXiv:2412.08905 Table 5 | verified 2026-09-18 | derived in the report from allocated tokens / unique tokens (§3.2) |
| phi-4 | 14B | pretrain-stable | synthetic corpus size | ~400B unweighted tokens across 50 dataset types | arXiv:2412.08905 §2.2 | verified 2026-09-18 | no ablation reported |
| phi-4 | 14B | long-context (midtrain) | context length | 4K → 16K | arXiv:2412.08905 §3.3 | verified 2026-09-18 | §3.3: inherently long sources beat padded-concatenated samples on long-context tasks |
| phi-4 | 14B | long-context (midtrain) | tokens | 250B | arXiv:2412.08905 §3.3 | verified 2026-09-18 | no ablation reported |
| phi-4 | 14B | long-context (midtrain) | max LR | pretraining peak / 10 | arXiv:2412.08905 §3.3 | verified 2026-09-18 | no ablation reported |
| phi-4 | 14B | long-context (midtrain) | RoPE base | 250,000 | arXiv:2412.08905 §3.3 | verified 2026-09-18 | no ablation reported |
| phi-4 | 14B | long-context (midtrain) | mixture | 30% newly curated long-context data, 70% recall tokens from pretraining | arXiv:2412.08905 §3.3 | verified 2026-09-18 | no ablation reported |
| phi-4 | 14B | SFT | learning rate | 1e-6 | arXiv:2412.08905 §4.1 | verified 2026-09-18 | no ablation reported |
| phi-4 | 14B | SFT | data volume | ~8B tokens, chatml format, incl. 40 languages | arXiv:2412.08905 §4.1 | verified 2026-09-18 | no ablation reported |
| phi-4 | 14B | preference (DPO round 1) | pair construction | Pivotal Token Search pairs; questions filtered to 0.2 ≤ p(success) ≤ 0.8 | arXiv:2412.08905 §4.3 | verified 2026-09-18 | Table 9: scores after DPO stage 1 vs SFT; §4.4 compares skipping pivotal-token DPO |
| phi-4 | 14B | preference (DPO round 2) | pairs | ~850K desired/undesired pairs, judge-guided | arXiv:2412.08905 §4.2 | verified 2026-09-18 | Table 9: scores after DPO stage 2 |
| Phi-4-reasoning | 14B | distill-SFT | teacher | o3-mini | arXiv:2504.21318 §2, §3 | verified 2026-09-18 | §3.2: o3-mini medium ≈ DeepSeek-R1 but more token-efficient; o3-mini high > medium |
| Phi-4-reasoning | 14B | distill-SFT | examples | >1.4M prompt-response pairs | arXiv:2504.21318 §3 | verified 2026-09-18 | §3.1: exploration-stage mixture search on math then code, aggregated |
| Phi-4-reasoning | 14B | distill-SFT | unique tokens | 8.3B | arXiv:2504.21318 §3 | verified 2026-09-18 | no ablation reported |
| Phi-4-reasoning | 14B | distill-SFT | tokens consumed by the final run | 16B | arXiv:2504.21318 §3.2 | verified 2026-09-18 | no ablation reported |
| Phi-4-reasoning | 14B | distill-SFT | steps | ~16K | arXiv:2504.21318 §3 | verified 2026-09-18 | Figure 4a: AIME-24 and GPQA-diamond pass@1 across SFT iterations |
| Phi-4-reasoning | 14B | distill-SFT | global batch (sequences) | 32 | arXiv:2504.21318 §3 | verified 2026-09-18 | no ablation reported |
| Phi-4-reasoning | 14B | distill-SFT | context length | 32K (RoPE base doubled from the 16K Phi-4 base) | arXiv:2504.21318 §3 | verified 2026-09-18 | no ablation reported |
| Phi-4-reasoning | 14B | distill-SFT | optimizer, LR, warmup, weight decay | AdamW, 1e-5, linear warm-up over 450 steps, wd 1e-4 | arXiv:2504.21318 §3 | verified 2026-09-18 | §3.1: grid search over [1e-6, 2e-5]; higher LRs gave lower training loss but saturation or degradation downstream |
| Phi-4-reasoning | 14B | distill-SFT | base model choice | Phi-4 (not the pre-post-training mid-trained checkpoint) | arXiv:2504.21318 §3.1 | verified 2026-09-18 | §3.1: both similar on reasoning; Phi-4 better on Responsible AI metrics |
| Phi-4-reasoning-plus | 14B | RL | algorithm | GRPO, verl framework | arXiv:2504.21318 §4.2 | verified 2026-09-18 | no ablation reported |
| Phi-4-reasoning-plus | 14B | RL | prompt pool | 72,401 math problems; 64 seeds per iteration; ~6K examples used | arXiv:2504.21318 §4, §4.2 | verified 2026-09-18 | §4.2: AIME improves >10 points in 90 steps, no gain after |
| Phi-4-reasoning-plus | 14B | RL | samples per prompt (group size G) | 8 | arXiv:2504.21318 §4.2 | verified 2026-09-18 | no ablation reported |
| Phi-4-reasoning-plus | 14B | RL | global batch | 64, across 32 NVIDIA H100 GPUs | arXiv:2504.21318 §4.2 | verified 2026-09-18 | no ablation reported |
| Phi-4-reasoning-plus | 14B | RL | optimizer, LR, warmup | Adam, 5e-8, cosine warm-up over the first 10 steps | arXiv:2504.21318 §4.2 | verified 2026-09-18 | no ablation reported |
| Phi-4-reasoning-plus | 14B | RL | KL coefficient β | 0.001, applied against π_θold in the printed objective | arXiv:2504.21318 §4.2 | verified 2026-09-18 | no ablation reported |
| Phi-4-reasoning-plus | 14B | RL | entropy coefficient γ | 0.001 | arXiv:2504.21318 §4.2 | verified 2026-09-18 | no ablation reported |
| Phi-4-reasoning-plus | 14B | RL | clip ε | not printed as a number (appears as ε in the objective) | arXiv:2504.21318 §4.2 | not reported (checked §4.2, §4.1, abstract; no appendix hyperparameter table) | — |
| Phi-4-reasoning-plus | 14B | RL | max response length | 32K training maximum; responses beyond 31K clipped, 1K reserved for the prompt | arXiv:2504.21318 §4.2 and footnote 2 | verified 2026-09-18 | §4.2 names the 31K clip as a limit on GRPO's effect |
| Phi-4-reasoning-plus | 14B | RL | reward, correct answers | cosine-scaled in [0.5, 1.0]; L_pos_control = 25,600; L_max = 31,744 | arXiv:2504.21318 §4.1 | verified 2026-09-18 | Figure 6 plots the reward against response length |
| Phi-4-reasoning-plus | 14B | RL | reward, incorrect answers | cosine-scaled in [-1.0, -0.5]; L_neg_control = 3,702 | arXiv:2504.21318 §4.1 | verified 2026-09-18 | Figure 6 |
| Phi-4-reasoning-plus | 14B | RL | reward, format overrides | missing `<|im_end|>` → -0.5; missing or invalid `<think>` block → -1.0 | arXiv:2504.21318 §4.1 | verified 2026-09-18 | no ablation reported |
| Phi-4-reasoning-plus | 14B | RL | reward, repetition | 5-gram repetition penalty R_rep; R_final = (8/13)·R_acc_scaled + (1/13)·R_rep | arXiv:2504.21318 §4.1 | verified 2026-09-18 | no ablation reported |
| Phi-4-reasoning-plus | 14B | eval-gate | checkpoint selection | step 90, chosen by best observed AIME 2024 score | arXiv:2504.21318 §4.2 | verified 2026-09-18 | Figure 7: GRPO dynamics over the first 125 updates |

## Connections
- [[phi-4]] — the card this ledger belongs to; it carries the abstract, contributions and findings.
- [[qwen-3]] — the comparable 2025 report with a GRPO stage on a curated query-verifier set.
- [[grpo]] — the algorithm whose objective the Phi-4-reasoning-plus rows parameterize.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2412.08905 (arXiv v1, 2024-12-12) and
  https://arxiv.org/abs/2504.21318 (arXiv v1, 2025-04-30).
- Corrections to the previous card version: this card is new; it holds the ledger that [[phi-4]] previously
  lacked. Corrections to the values that were in [[phi-4]] are recorded in that card's Verification section.
- Removed as unsupported by the source: none.
- Not reported by the source: GRPO clip ε; pretraining and SFT compute or GPU-hours; the tokenizer-level
  packing scheme for SFT; DPO β for either round; the number of SFT epochs stated as a number (the report
  says "2+ passes over reasoning data sources", §3).

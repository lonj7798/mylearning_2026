<!-- scope: Recipe ledger for the DeepSeek-V3 technical report (arXiv:2412.19437v2); companion to [[deepseek-v3]]
     deps: [[deepseek-v3]]
     see-also: [[deepseek-v3.1-recipe]]
-->

# DeepSeek-V3 Technical Report — Recipe ledger
Companion to [[deepseek-v3]]. All rows refer to the released DeepSeek-V3 (671B total, 37B activated per token; arXiv:2412.19437v2 §4.2) unless the Model column names an ablation model. Status dates refer to reading the arXiv v2 PDF on 2026-09-14.

Units: the report prints "batch size" without a unit for pre-training and context extension; values are copied as printed. "Tokens" are training tokens seen.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Architecture | 61 layers; hidden 7168; 128 heads × 128 dim; d_c 512; d_c' 1536; d_h^R 64; first 3 FFNs dense | arXiv:2412.19437v2 §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | MoE routing | 1 shared + 256 routed experts (intermediate dim 2048); 8 routed active; at most 4 nodes per token (M = 4) | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Initialization | std 0.006 for all learnable parameters | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Tokens seen | 14.8T | §4.2, Abstract | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Tokenizer | byte-level BPE, 128K vocabulary; combined punctuation+line-break tokens randomly split for a proportion of cases (proportion not printed) | §4.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Max sequence length | 4K | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Packing | document packing, no cross-sample attention masking | §4.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Fill-in-Middle | PSM format at document level before packing, rate 0.1 | §4.1 | verified 2026-09-14 | refers to DeepSeekCoder-V2 observation that FIM does not hurt next-token prediction; no V3 ablation |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Mixture | math and programming ratio increased vs DeepSeek-V2; multilingual coverage extended; percentages not reported | §4.1 | not reported (body and appendix checked) | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Optimizer | AdamW, β1 = 0.9, β2 = 0.95, weight decay 0.1 | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Gradient clipping | norm 1.0 | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | LR warmup | linear 0 → 2.2e-4 over first 2K steps | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Constant LR | 2.2e-4 until 10T tokens | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Batch size schedule | 3072 → 15360 over the first 469B tokens, then 15360 (unit not printed) | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-decay/anneal | Cosine decay | 2.2e-4 → 2.2e-5 over 4.3T tokens | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-decay/anneal | Final 500B tokens | constant 2.2e-5 for 333B tokens, then constant 7.3e-6 for 167B tokens | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | MTP | depth D = 1; loss weight λ = 0.3 for first 10T tokens, 0.1 for remaining 4.8T | §4.2 | verified 2026-09-14 | Table 4: 1-depth MTP vs no MTP at 15.7B (1.33T tokens) and 228.7B (540B tokens), better on most benchmarks; λ values not ablated |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Aux-loss-free bias update speed γ | 0.001 for first 14.3T tokens; 0.0 for remaining 500B | §4.2 | verified 2026-09-14 | Table 5: aux-loss-free vs aux-loss-based at 15.7B (1.33T) and 228.7B (578B), better on most benchmarks; γ not ablated |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Sequence-wise balance loss α | 0.0001 | §4.2 | verified 2026-09-14 | §4.5.3: 1B MoE val. loss 2.258 (sequence-wise aux) vs 2.253 (aux-loss-free) vs 2.253 (batch-wise aux); 3B: 2.085 vs 2.080 vs 2.080 |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Parallelism | 16-way PP; 64-way EP across 8 nodes; ZeRO-1 DP; no TP | §3.2, §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Precision | FP8 GEMMs (Fprop, Dgrad, Wgrad); E4M3 on all tensors; activations 1x128 tiles, weights 128x128 blocks; embedding, output head, gating, norms, attention in BF16/FP32 | §3.3.1, §3.3.2 | verified 2026-09-14 | App. B.1: relative loss error vs BF16 below 0.25% at ~16B (1.33T tokens) and ~230B (~0.9T tokens) |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | Compute | 2664K H800 GPU hours; 180K per trillion tokens; cluster of 2048 H800 | Table 1, §1, §3.1 | verified 2026-09-14 | n/a |
| DeepSeek-V3 | 671B / 37B act. | long-context | Method | YaRN applied only to decoupled shared key k^R; s = 40, α = 1, β = 32, √t = 0.1 ln s + 1; same in both phases | §4.3 | verified 2026-09-14 | same configuration as DeepSeek-V2; no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | long-context | Phase 1 | 4K → 32K; batch 1920; 1000 steps | §4.3 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | long-context | Phase 2 | 32K → 128K; batch 480; 1000 steps | §4.3 | verified 2026-09-14 | Figure 8: NIAH robust up to 128K (measured after SFT) |
| DeepSeek-V3 | 671B / 37B act. | long-context | LR | 7.3e-6 in both phases (final pre-training LR) | §4.3 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | long-context | Tokens | not printed | §4.3 | not reported (body checked) | n/a |
| DeepSeek-V3 | 671B / 37B act. | long-context | Compute | 119K H800 GPU hours | Table 1 | verified 2026-09-14 | n/a |
| DeepSeek-V3 | 671B / 37B act. | distill-SFT | Dataset size | 1.5M instances, multiple domains | §5.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | distill-SFT | Reasoning data | per-domain expert models (SFT + RL) trained on <problem, original response> and <system prompt, problem, R1 response>; high-temperature RL sampling; "hundreds of RL steps"; rejection sampling from experts | §5.1 | verified 2026-09-14 | Table 9 (on DeepSeek-V2.5): LCB-CoT 31.1 → 37.4, MATH-500 74.6 → 83.2 vs short-CoT baseline |
| DeepSeek-V3 | 671B / 37B act. | distill-SFT | Non-reasoning data | responses from DeepSeek-V2.5, verified by human annotators | §5.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | SFT | Epochs | 2 | §5.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | SFT | LR schedule | cosine, 5e-6 → 1e-6 | §5.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | SFT | Packing / masking | multiple samples per sequence; sample masking keeps them mutually invisible | §5.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | SFT | Batch size, warmup, loss masking | not printed | §5.1 | not reported (body checked) | n/a |
| DeepSeek-V3 | 671B / 37B act. | reward-model | Reward types | rule-based RM (boxed math answers, compiler test cases); model-based RM trained from V3 SFT checkpoints, preference data includes CoT | §5.2.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | RL | Algorithm | GRPO; advantage (r_i − mean)/std over group; KL penalty β·D_KL(π_θ‖π_ref) with k3-form estimator in the loss | §5.2.2, Eq. 26-28 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | RL | Prompt domains | coding, math, writing, role-playing, question answering | §5.2.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | RL | β, ε, G, LR, prompts per step, steps, max response length, temperature | not printed | §5.2 | not reported (body and appendix checked) | n/a |
| DeepSeek-V3 | 671B / 37B act. | RL | Open-ended feedback | V3 voting judgments as feedback (constitutional AI) | §5.3.4, §5.4.2 | verified 2026-09-14 | Table 8: RewardBench 87.0 single, 89.6 maj@6 |
| DeepSeek-V3 | 671B / 37B act. | SFT + RL | Compute | 5K H800 GPU hours (post-training) | Table 1 | verified 2026-09-14 | n/a |
| DeepSeek-V3 | 671B / 37B act. | all | Total compute | 2788K H800 GPU hours; $5.576M at $2/GPU hour; excludes prior research and ablations | Table 1, §1 | verified 2026-09-14 | n/a |
| DeepSeek-V3 | 671B / 37B act. | SFT + RL | Post-training share of pre-training GPU hours | 5K / 2664K = 0.19% | Table 1 | derived | n/a |
| DeepSeek-V3 (chat) | 671B / 37B act. | eval-gate | Evaluation settings | maximum output 8192 tokens; AIME and CNMO 2024 at temperature 0.7 averaged over 16 runs; MATH-500 greedy | §5.3.1 | verified 2026-09-14 | n/a |

Checked sources: arXiv:2412.19437v2 body §2-§6 and Appendices B-C. No released training config is cited by the report.

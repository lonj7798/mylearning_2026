<!-- scope: Recipe ledger for the MiniMax-01 technical report (arXiv:2501.08313v1); companion to [[minimax-01]]
     deps: [[minimax-01]]
     see-also: [[deepseek-v3-recipe]]
-->

# MiniMax-01: Scaling Foundation Models with Lightning Attention — Recipe ledger
Companion to [[minimax-01]]. Rows refer to MiniMax-Text-01 (456B total, 45.9B activated; arXiv:2501.08313v1 §2) unless the Model column names an ablation model. Status dates refer to reading the arXiv v1 PDF on 2026-09-14. MiniMax-M1 values (arXiv:2506.13585) are not in this ledger.

Units: the report prints pre-training batch sizes as "16M" ... "128M" without a unit; values are copied as printed. "Tokens" are training tokens seen. Post-training "Batch Size" in Table 7 is printed without a unit.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Architecture | 80 layers; 1 softmax-attention block after every 7 lightning-attention blocks; 64 heads × 128 dim; hidden 6144 | arXiv:2501.08313v1 §2 | verified 2026-09-14 | §2.2.2-2.2.3 Tables 3-4 (hybrid every 8 layers vs hybrid-cosFormer2, hybrid-HGRN2, hybrid-window at 1B/3B); Table 5 (28B/5B-act. MoE, 1T tokens) |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Softmax-layer attention | GQA, group size 8 | §2 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Positional encoding | RoPE on half of the attention head dimension, base 10,000 | §2 | verified 2026-09-14 | §2.4: half-dimension RoPE "enables length extrapolation without performance degradation" in small-scale experiments; no numbers printed |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | MoE | 32 experts, top-2, expert FFN hidden 9216; token-drop with capacity limit; global router | §2, §2.1 | verified 2026-09-14 | Figure 4: 24B-total/2B-act. MoE vs 7B dense on 1T tokens (plots only); size chosen by Eq. 13-14 fits on 44M-1.2B act. models, 500B tokens, 16/32/64 experts (§2.4) |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | MoE auxiliary loss coefficient | 0.01 | §4.2 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Normalization / init | DeepNorm (PostNorm) with α = (2N)^0.25, β = (8N)^−0.25, N = number of layers; Xavier initialization | §2, §4.2 | verified 2026-09-14 | Table 5: PostNorm > PreNorm on 7 of 8 benchmarks (60B/9.3B act., 48 layers, 500B tokens) |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Optimizer | AdamW, β1 = 0.9, β2 = 0.95, weight decay 0.1; ε not printed | §4.2 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Sequence length | 8192 | §4.2 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Batch size schedule | 16M → 32M at 69B tokens → 64M at 790B tokens → 128M at 4.7T tokens, then constant | §4.2 | verified 2026-09-14 | Figure 13: power-law fit of loss vs critical batch size on 50M-600M act. models (loss 2.22 → 16M, 1.98 → 32M, 1.77 → 64M, 1.58 → 128M); batch doubled when that loss is reached |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | LR warmup and peak | linear over 500 iterations to 2 × 10⁻⁴ | §4.2 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Constant LR phase | 2 × 10⁻⁴ "for 7.2T tokens" | §4.2 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | LR reduction | 1.3 × 10⁻⁴ "for the remaining 3.2T tokens"; whether these 3.2T are part of the 7.2T is not stated | §4.2 | verified 2026-09-14 | anomalous gradient-norm values attributed to an excessively high LR; no ablation |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-decay/anneal | Fast decay | 1T tokens, exponential decay to 3 × 10⁻⁵ | §4.2 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Total tokens | not printed as a single figure | §4.2 | not reported (body and appendices checked) | n/a |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Tokenizer | byte-level BPE with pre-tokenizer; multilingual content up-sampled; vocabulary 200K | §4.1.2 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Mixture | sampling starts uniform over the base corpus, then weights favor high-quality content while keeping category coverage; percentages not printed | §4.1.1 | not reported (percentages) | §4.1.1: removing all low-scoring content lowered downstream performance (data experiments, no numbers) |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Repetition limits | low-quality data degrades after > 2 epochs; high-quality data up to 4 epochs | §4.1.3.2 | verified 2026-09-14 | repetition-aware data experiments (§4.1.3.2); no table |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Gradient clipping, dropout | not printed | §4.2 | not reported (body checked) | n/a |
| MiniMax-Text-01 | 456B / 45.9B act. | pretrain-stable | Compute | cluster of 1,500-2,500 H800 GPUs (changing over the project); GPU hours not printed | §3 | verified 2026-09-14 (cluster); not reported (GPU hours) | n/a |
| MiniMax-Text-01 | 456B / 45.9B act. | long-context | Stage 1 | length 128K; RoPE base 5M; 300B tokens; Short/Medium/Long = 30/70/0% (share basis not stated) | Table 6 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | long-context | Stage 2 | length 512K; RoPE base 10M; 32B tokens; 35/35/30% | Table 6 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | long-context | Stage 3 | length 1M; RoPE base 10M; 26B tokens; 30/30/40% | Table 6 | verified 2026-09-14 | Figure 14: vanilla NIAH up to 4M tokens; Table 9: RULER 0.910 at 1M |
| MiniMax-Text-01 | 456B / 45.9B act. | long-context | Length buckets | Short < 32K; Medium 32K-128K; Long > 128K tokens | Table 6 caption | verified 2026-09-14 | n/a |
| MiniMax-Text-01 | 456B / 45.9B act. | long-context | Long-context QA | 10% high-quality long-context QA data during the last 20% of each stage; linear interpolation of source weights between stages | §4.2 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | long-context | LR, batch size | not printed | §4.2 | not reported (body checked) | n/a |
| MiniMax-Text-01 | 456B / 45.9B act. | SFT | Stage I | seq 8,192; 2 epochs; batch 128; LR 1e-5 → 1e-6 cosine; prompts longer than 8,192 removed | Table 7, §5.6 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | long-context | Stage II (post-training; objective not named) | seq 1,032,192; 2 epochs; batch 80; LR 3e-6 constant; 50% long-context prompts | Table 7, §5.6 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | preference | Stage III DPO (short) | seq 8,192; 1 epoch; batch 64; LR 5e-7 → 5e-8 cosine | Table 7, §5.6 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | preference | Stage IV DPO (long) | seq 1,032,192; 1 epoch; batch 64; LR 5e-7 constant; entirely long-context data | Table 7, §5.6 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | RL | Stage V online RL | seq 8,192; 1 epoch; batch 512; LR 1e-6 → 1e-7 cosine | Table 7, §5.6 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | SFT | RoPE base in post-training | 10M in all stages | §5.6 | verified 2026-09-14 | n/a |
| MiniMax-Text-01 | 456B / 45.9B act. | distill-SFT | SFT data construction | responses from domain expert models (iterative SFT + RL); rejection sampling over several temperatures; n-gram and semantic-similarity filters; dataset size not printed | §5.3 | verified 2026-09-14 (method); not reported (size) | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | preference | DPO pairs | best and worst responses by the §5.2 reward models, sampled at varying temperatures; SFT-trained prompts; β not printed | §5.4.1 | verified 2026-09-14 (pairs); not reported (β) | §5.4.1: SFT-trained vs SFT-untrained homologous prompts gave "negligible performance variations" (no numbers) |
| MiniMax-Text-01 | 456B / 45.9B act. | RL | Algorithm | modified GRPO: extra clipping that drops tokens with large policy ratio and negative advantage; KL term 𝔼_t[SG(π_θ − π_ref) log π_θ]; balanced advantage estimation | §5.4.2 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | RL | Prompts | SFT-untrained prompts; moderate success rates prioritized; focus on mathematical reasoning | §5.4.2 | verified 2026-09-14 | §5.4.2: reusing earlier-phase prompts led to saturation with lower response perplexity (no numbers) |
| MiniMax-Text-01 | 456B / 45.9B act. | RL | Clip ε, KL coefficient, group size, prompts per step, steps, max response length, temperature | not printed | §5.4.2 | not reported (body and appendices checked) | n/a |
| MiniMax-Text-01 | 456B / 45.9B act. | reward-model | Reward dimensions | correctness (early MiniMax-Text-01 binary answer consistency for math/reasoning; sandbox test-case success rate for code); truthfulness; helpfulness (weighted rule-based and human signals); harmlessness (early MiniMax-Text-01 with calibrated prompts) | §5.2 | verified 2026-09-14 | no ablation reported |
| MiniMax-Text-01 | 456B / 45.9B act. | eval-gate | Evaluation settings | greedy decoding; zero-shot chain-of-thought | §5.7.1 | verified 2026-09-14 | n/a |
| Scaling-law models | 70M-7B | pretrain-stable | Attention comparison runs | up to 300B tokens; context 8192; global batch 4M tokens; Adam, LR 3e-4, weight decay 0.1; fixed LR schedule | §2.2.2.1 | verified 2026-09-14 | Table 2 fits |
| Data-experiment MoE | 8B / 1B act. | pretrain-stable | Data ablation runs | 40B tokens = 20B web + 20B candidate data; 95% confidence, 80% power | §4.1.3.1 | verified 2026-09-14 | n/a |

Checked sources: arXiv:2501.08313v1 body §1-§7 and Appendices A-B. The report cites no released training configuration.

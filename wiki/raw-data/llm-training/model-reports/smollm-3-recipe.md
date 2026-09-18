<!-- scope: recipe ledger split out of [[smollm-3]] — every training setting the SmolLM3 blog discloses, with its locus
     deps: [[smollm-3]]
     see-also: [[tulu-3]], [[olmo-3]], [[qwen-3]]
-->

# SmolLM3 recipe ledger (companion to [[smollm-3]])

Source for every row unless stated otherwise: https://huggingface.co/blog/smollm3 (published 2025-07-08).
Section names in the "Source location" column are the blog's own headings. All rows read on 2026-09-18.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| SmolLM3-3B-Base | 3B | pretrain-stable | total training tokens | 11.2T | Data mixture and training stages | verified 2026-09-18 | no ablation reported |
| SmolLM3-3B-Base | 3B | pretrain-stable | stage 1 span and mixture | 0T→8T; web 85% (12% multilingual), code 12%, math 3% | Data mixture and training stages | verified 2026-09-18 | mixture ratios set by ablations on 3B models trained on 50B–100B tokens (same section) |
| SmolLM3-3B-Base | 3B | pretrain-stable | stage 2 span and mixture | 8T→10T; web 75% (12% multilingual), code 15%, math 10% | Data mixture and training stages | verified 2026-09-18 | same ablation note |
| SmolLM3-3B-Base | 3B | pretrain-decay/anneal | stage 3 span and mixture | 10T→11.1T; web 63% (12% multilingual), code 24%, math 13% | Data mixture and training stages | verified 2026-09-18 | same ablation note |
| SmolLM3-3B-Base | 3B | pretrain-stable | global batch | 2.36M tokens | Architecture and training details | verified 2026-09-18 | no ablation reported |
| SmolLM3-3B-Base | 3B | pretrain-stable | sequence length | 4096 | Architecture and training details | verified 2026-09-18 | no ablation reported |
| SmolLM3-3B-Base | 3B | pretrain-stable | peak learning rate | 2e-4 | Architecture and training details | verified 2026-09-18 | no ablation reported |
| SmolLM3-3B-Base | 3B | pretrain-stable | optimizer | AdamW, beta1 0.9, beta2 0.95, weight decay 0.1, gradient clipping 1 | Architecture and training details | verified 2026-09-18 | weight decay removed from embedding layers, following OLMo 2, for stability (same section) |
| SmolLM3-3B-Base | 3B | pretrain-stable | schedule | WSD; 2000 warmup steps; linear decay to 0 over the final 10% of steps | Architecture and training details | verified 2026-09-18 | no ablation reported |
| SmolLM3-3B-Base | 3B | pretrain-stable | frameworks | nanotron (training), datatrove (data), lighteval (eval) | Architecture and training details | verified 2026-09-18 | — |
| SmolLM3-3B-Base | 3B | pretrain-stable | compute | 384 H100 GPUs for 24 days | Architecture and training details | verified 2026-09-18 | — |
| SmolLM3-3B-Base | 3B | pretrain-stable | attention | GQA with 4 groups | Architecture and training details | verified 2026-09-18 | ablation on 3B / 100B FineWeb-Edu tokens: GQA matches multi-head attention while reducing KV cache size |
| SmolLM3-3B-Base | 3B | pretrain-stable | positional encoding | NoPE: rotary embeddings removed from every 4th layer | Architecture and training details | verified 2026-09-18 | ablations reported to improve long context without affecting short context (same section) |
| SmolLM3-3B-Base | 3B | pretrain-stable | sequence composition | intra-document attention masking | Architecture and training details | verified 2026-09-18 | reported as faster and more stable long-context training, following Llama 3 |
| SmolLM3-3B-Base | 3B | long-context | total tokens | 100B, in two sequential 50B-token stages | Long Context extension | verified 2026-09-18 | upsampling code repositories, books, and long web pages did not further improve RULER or HELMET |
| SmolLM3-3B-Base | 3B | long-context | stage A | 4k → 32k context, RoPE theta 1.5M, upsampled math/code/reasoning | Long Context extension | verified 2026-09-18 | same |
| SmolLM3-3B-Base | 3B | long-context | stage B | 32k → 64k context, RoPE theta 5M, upsampled math/code/reasoning | Long Context extension | verified 2026-09-18 | same |
| SmolLM3-3B | 3B | long-context | inference extrapolation | YaRN to 128k (2x the 64k training length), following Qwen2.5 | Long Context extension | verified 2026-09-18 | — |
| SmolLM3-3B | 3B | mid-train | reasoning data | 35B tokens: OpenThoughts3-1.2M plus an R1-trace subset of Llama-Nemotron-Post-Training-Dataset-v1.1 | Reasoning Mid-training | verified 2026-09-18 | no ablation reported |
| SmolLM3-3B | 3B | mid-train | epochs and tokens seen | 4 epochs, about 140B tokens | Reasoning Mid-training | verified 2026-09-18 | no ablation reported |
| SmolLM3-3B | 3B | mid-train | template and packing | ChatML chat template, wrapped packing | Reasoning Mid-training | verified 2026-09-18 | chosen to avoid providing too much structure to the model (same section) |
| SmolLM3-3B | 3B | SFT | dataset size and split | 1.8B tokens: 1B non-reasoning (12 datasets) + 0.8B reasoning (10 datasets) | Supervised Finetuning | verified 2026-09-18 | mixture selected by ablations on reasoning-to-non-reasoning ratio and within-mode composition (same section) |
| SmolLM3-3B | 3B | SFT | epochs and tokens seen | 4 epochs, about 8B tokens | Supervised Finetuning | verified 2026-09-18 | no ablation reported |
| SmolLM3-3B | 3B | SFT | packing | BFD (best-fit decreasing) | Supervised Finetuning | verified 2026-09-18 | no ablation reported |
| SmolLM3-3B | 3B | SFT | loss masking | loss masked on user turns and on tool-call results | Supervised Finetuning | verified 2026-09-18 | no ablation reported |
| SmolLM3-3B | 3B | SFT | synthetic gap filling | Qwen3-32B prompted in reasoning mode with prompts from existing non-reasoning datasets, for multi-turn, multilingual, and everyday-conversation domains | Supervised Finetuning | verified 2026-09-18 | added because reasoning traces were scarce in those domains (same section) |
| SmolLM3-3B | 3B | SFT | template | SmolLM3 final chat template (mid-training used ChatML) | author comment, Jul 9 2025, comments thread | verified 2026-09-18 | comment by the article author; lower reliability than the body text |
| SmolLM3-3B | 3B | preference | algorithm | Anchored Preference Optimization (APO), a DPO variant | Off-policy model alignment | verified 2026-09-18 | reported as more stable and higher downstream performance in internal ablations; no numbers given |
| SmolLM3-3B | 3B | preference | non-reasoning data | Tulu3 preference dataset | Off-policy model alignment | verified 2026-09-18 | no ablation reported |
| SmolLM3-3B | 3B | preference | reasoning data | synthetic pairs: Qwen3-32B generations as chosen, Qwen3-0.6B as rejected | Off-policy model alignment | verified 2026-09-18 | generated to cover the domains present in the non-thinking dataset (same section) |
| SmolLM3-3B | 3B | preference | maximum sequence length of preference data | 24k tokens | Off-policy model alignment | verified 2026-09-18 | named as a cause of the RULER regression (same section) |
| SmolLM3-3B | 3B | preference | β, learning rate, batch size, epochs | not reported | body of Off-policy model alignment; no config quoted in the post | not reported | — |
| SmolLM3-3B | 3B | merge | method and library | model soup of APO checkpoints, then linear merge with a mid-training checkpoint; MergeKit | Model Merging | verified 2026-09-18 | reported as the best-performing configuration explored |
| SmolLM3-3B | 3B | merge | weights | 0.9 APO soup / 0.1 mid-training checkpoint | Model Merging | verified 2026-09-18 | recovered the base model's RULER score on contexts up to 128k |
| SmolLM3-3B | 3B | eval-gate | recommended sampling | temperature 0.6, top_p 0.95 | Extending thinking evaluation | verified 2026-09-18 | — |
| SmolLM3-3B | 3B | RL | any RL stage | none described | whole post | not reported | the post describes pretraining, mid-training, SFT, APO, and merging only |

## Units and scope notes
- Pretraining token counts are tokens seen. The stage spans are cumulative positions in the 11.2T schedule; stage 3 is written as 10T→11.1T in the post while the headline total is 11.2T, and the post does not reconcile the difference.
- Reasoning mid-training is quoted both ways in the post: 35B unique tokens and about 140B tokens over 4 epochs. Use the unique figure when comparing dataset sizes and the 140B figure when comparing compute.
- The SFT figure is likewise 1.8B unique tokens and about 8B tokens over 4 epochs.
- Every row describes the 3B release. The post reports no other model size.

## Verification
- Checked on 2026-09-18 against: https://huggingface.co/blog/smollm3 (published 2025-07-08).
- Corrections to the previous card version: this file is new; it carries the ledger split out of [[smollm-3]] to keep that card under the 120-line limit.
- Removed as unsupported by the source: none.
- Not reported by the source: SFT and APO optimizer settings, the APO β, per-stage compute outside pretraining, and the full dataset lists for both SFT modes.

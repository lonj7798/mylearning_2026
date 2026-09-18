<!-- scope: recipe ledger for [[physics-of-lm-3]] (Allen-Zhu & Li 2024, arXiv:2404.05405v1): training settings of the synthetic knowledge-capacity experiments
     see-also: [[physics-of-lm-3]]
-->

# Recipe ledger: Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws
- **Source card:** [[physics-of-lm-3]]
- **URL:** https://arxiv.org/abs/2404.05405 (arXiv v1)
- **Source type:** paper (controlled experiments on synthetic biography data; these are not settings of a released model)

## Notes on units
- "Batch size" is printed without a unit. Training data is cut into 512-token windows (§2.3, App. A). Remark A.3 lists 64 GPUs × 24, 48 GPUs × 32, and 1536 GPUs × 1 as equivalent setups, which implies a global batch of 1536 (derived); the paper does not name the unit.
- "Exposure" is one occurrence of a knowledge piece in training (§1, §2.2). It is not a pass over the data.
- Slash-separated learning rates are printed as options. For Parameters 6, 7(a), and 8 the text says the best option is used (App. B); for Parameters 2, 9, and 10 the selection rule is not stated beyond tuning or "explored" options (App. A.1, D, E).
- Every pretraining run is one cosine schedule from random initialization. The stage label "pretrain-stable" covers the whole run.

## Shared protocol
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GPT2-ℓ-h (rotary embedding, no dropout, 64 dims per head) | 1M–0.5B | pretrain-stable | initialization; objective | random initialization; standard autoregressive loss | §2.3 | verified 2026-09-14 | — |
| GPT2-ℓ-h | 1M–0.5B | pretrain-stable | context; packing | 512-token windows; biographies randomly concatenated with `<EOS>` and randomly segmented | §2.3; App. A | verified 2026-09-14 | — |
| GPT2-ℓ-h | 1M–0.5B | pretrain-stable | optimizer; schedule | AdamW; cosine, 1K warmup steps, decay from 1 to 0.1 times the reference rate | App. A "Training parameters" | verified 2026-09-14 | no ablation reported |
| GPT2-ℓ-h | 1M–0.5B | pretrain-stable | precision | mixed-precision fp16 | §2.3; Remark A.1 | verified 2026-09-14 | Remark A.1: bf16 gave "nearly identical" results |
| LLaMA-ℓ-h, Mistral-ℓ-h | not reported as a range | pretrain-stable | precision | bf16 where needed | Remark B.1; App. B.2 footnote 27 | verified 2026-09-14 | fp16 "can sometimes fail" for LLaMA/Mistral below 100M |
| GPT2-ℓ-h | 1M–0.5B | pretrain-stable | tuning rules | larger models use smaller LR; at least 50K steps; lower wd for longer runs | Remark A.2 | verified 2026-09-14 | authors' reasoning; no ablation reported |

## GPT2 on bioS(N), 1000 exposures (Parameter 1; also used for bioS_simple and bioR, Parameter 4)
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GPT2-ℓ-h, bioS(10K) | Figure 1(a) models | pretrain-stable | wd; lr; batch; steps | 0.02; 0.001; 24; about 140K | App. A.1 Parameter 1 | verified 2026-09-14 | no ablation reported; final result "not very sensitive" to LR at 1000 exposures (App. A.1) |
| GPT2-ℓ-h, bioS(20K) | Figure 1(a) models | pretrain-stable | wd; lr; batch; steps | 0.02; 0.001; 48; about 140K | App. A.1 Parameter 1 | verified 2026-09-14 | as above |
| GPT2-ℓ-h, bioS(50K) | Figure 1(a) models | pretrain-stable | wd; lr; batch; steps | 0.02; 0.001; 96; about 175K | App. A.1 Parameter 1 | verified 2026-09-14 | as above |
| GPT2-ℓ-h, bioS(100K), bioS(200K) | Figure 1(a) models | pretrain-stable | wd; lr; batch; steps | 0.02; 0.001; 192; about 175K, 349K | App. A.1 Parameter 1 | verified 2026-09-14 | as above |
| GPT2-ℓ-h, bioS(500K), bioS(1M) | Figure 1(a) models | pretrain-stable | wd; lr; batch; steps | 0.01; 0.0005; 192; about 435K, 870K | App. A.1 Parameter 1 | verified 2026-09-14 | as above |
| GPT2-16-8, bioS(2M) | one size per N ≥ 2M (App. A) | pretrain-stable | wd; lr; batch; steps | 0.005; 0.0003; 1536; about 220K | App. A.1 Parameter 1 | verified 2026-09-14 | as above |
| GPT2-6-20, bioS(5M) | one size per N ≥ 2M | pretrain-stable | wd; lr; batch; steps | 0.002; 0.0003; 1536; about 540K | App. A.1 Parameter 1 | verified 2026-09-14 | as above |
| GPT2-20-16, bioS(10M) | one size per N ≥ 2M | pretrain-stable | wd; lr; batch; steps | 0.001; 0.0003; 1536; about 1M | App. A.1 Parameter 1 | verified 2026-09-14 | as above |
| GPT2-20-16, bioS(10M) | one size per N ≥ 2M | pretrain-stable | compute | 8.5 days on 64 A100s | Figure 1 remarks | verified 2026-09-14 | — |

Parameter 1 does not name models. The model names for N = 2M, 5M, 10M follow the list order in App. A ("N = 2M, 5M, 10M, 20M ... GPT2-16-8, GPT2-6-20, GPT2-20-16, GPT2-25-20"); the 10M pairing is confirmed by the Figure 1 remarks. Parameter 1 gives no N = 20M row.

## GPT2 on bioS(N), 100 exposures (Parameter 2)
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GPT2-ℓ-h, bioS(50K) | Figure 1(b) models | pretrain-stable | wd; lr; batch | 0.01; 0.001; 12 | App. A.1 Parameter 2 | verified 2026-09-14 | "careful tuning of learning rates is required" (App. A.1); no ablation table |
| GPT2-ℓ-h, bioS(100K) | Figure 1(b) models | pretrain-stable | wd; lr; batch | 0.01; 0.001; 24 | App. A.1 Parameter 2 | verified 2026-09-14 | as above |
| GPT2-ℓ-h, bioS(200K) | Figure 1(b) models | pretrain-stable | wd; lr; batch | 0.01; 0.001 (GPT2-2-20: 0.0005); 48 | App. A.1 Parameter 2 | verified 2026-09-14 | as above |
| GPT2-ℓ-h, bioS(500K) | Figure 1(b) models | pretrain-stable | wd; lr; batch | 0.01; 0.0005; 96 | App. A.1 Parameter 2 | verified 2026-09-14 | as above |
| GPT2-ℓ-h, bioS(1M) | Figure 1(b) models | pretrain-stable | wd; lr; batch | 0.01; 0.0005; 192 | App. A.1 Parameter 2 | verified 2026-09-14 | as above |
| GPT2-ℓ-h, bioS(2M) | Figure 1(b) models | pretrain-stable | wd; lr; batch | 0.01; 0.0003/0.0005/0.001; 384 | App. A.1 Parameter 2 | verified 2026-09-14 | as above |
| GPT2-ℓ-h, bioS(5M) | Figure 1(b) models | pretrain-stable | wd; lr; batch | 0.01; 0.0003/0.0005; 768 | App. A.1 Parameter 2 | verified 2026-09-14 | as above |
| GPT2-ℓ-h, bioS(10M) | Figure 1(b) models | pretrain-stable | wd; lr; batch | 0.01; 0.0002/0.0003/0.0005; 1024 | App. A.1 Parameter 2 | verified 2026-09-14 | as above |
| GPT2-ℓ-h, bioS(20M) | Figure 1(b) models | pretrain-stable | wd; lr; batch | 0.002; 0.0002/0.0003/0.0005; 1536 (GPT2-28-20: 1280, footnote 22) | App. A.1 Parameter 2 | verified 2026-09-14 | as above |
| GPT2-12-32, bioS(20M) | not reported | pretrain-stable | compute | 2.4 days (GPU count not stated in this sentence) | Figure 1 remarks | verified 2026-09-14 | — |

N = 10K and 20K are not run at 100 exposures "due to the excessively short training process" (App. A.1).

## Other experiments
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GPT2-ℓ-h, bioD(N, K, C, D, L, T), 1000 exposures | Figure 2 models | pretrain-stable | wd; lr; batch | 0.01; 0.0005; 192 | App. A.4 Parameter 5 | verified 2026-09-14 | Remark A.4: narrower size range, so settings are not varied |
| GPT2 pretrained models (Figure 10) | Figure 10 models | SFT (LoRA extraction test) | data split | QA finetuning on half of the people, test on the other half | App. A.2 | verified 2026-09-14 | — |
| GPT2 pretrained models (Figure 10) | Figure 10 models | SFT (LoRA extraction test) | LoRA ranks (r′ embedding, r query/value) | (8,2), (16,2), (8,4), (32,4), (8,8), (32,8), (128,8), (32,16), (128,16); best result reported | App. A.2 Parameter 3 | verified 2026-09-14 | footnote 25: best option chosen because the aim is maximum extractable bits |
| GPT2 pretrained models (Figure 10) | Figure 10 models | SFT (LoRA extraction test) | warmup; batch; lr; wd; steps | none; 96; 0.001 with linear decay to 0; 0.1; 75,000 | App. A.2 | verified 2026-09-14 | no ablation reported |
| GPT2-MoE, 32 experts per layer (each d→d→d) | Figure 16 models | pretrain-stable | routing | topk = 1, cap_factor = 2 in training; cap_factor = 8 at test; tutel package | App. D; Remark 9.1 | verified 2026-09-14 | Figure 17: (1,2), (2,1), (2,2) differ minimally at 100 exposures |
| GPT2 on useful + junk mixture | Figure 8 models | pretrain-stable | mixture (share of training tokens) | 1/8 useful bioS(N); 7/8 junk bioS(N′), N′ = 100M (random) or 1K (repetitive) | §10; App. E | verified 2026-09-14 | Figure 8(a)–(h) |
| GPT2 on useful + junk mixture | Figure 8 models | pretrain-stable | packing | each 512-token window holds only useful or only junk data | App. E | verified 2026-09-14 | "outcomes are similar" when mixed in one window (App. E) |
| GPT2 on useful + junk mixture | Figure 8 models | pretrain-stable | tag | special token at the start of each useful data piece (case c) | App. E; Result 12 | verified 2026-09-14 | Figure 8(g)–(h) |

Per-N optimizer values for LLaMA/Mistral (Parameters 6–8, App. B), MoE (Parameter 9, App. D), and the junk-data runs (Parameter 10, App. E) are printed at those loci and are not copied here. Parameter 10 states that negative results received wider learning-rate searches than positive results (App. E).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2404.05405 (arXiv v1), App. A–E.
- New file created in the 2026-09 repair; no earlier version.

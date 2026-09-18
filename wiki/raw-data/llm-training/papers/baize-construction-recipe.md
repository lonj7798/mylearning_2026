<!-- scope: recipe ledger for the Baize models (v1 7B/13B/30B, Healthcare, v1.5, v2 SDF) reported in arXiv:2304.01196
     deps: [[baize-construction]]
     see-also: [[alpaca]]
-->

# Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data — Recipe ledger
- **Parent card:** [[baize-construction]]
- **URL:** https://arxiv.org/abs/2304.01196 (arXiv v4, 2023-12-02; EMNLP 2023)
- **Source type:** paper
- **Scope note:** §5 gives optimizer, batch size, and learning rates per model size without naming a version (v1, v1.5, or v2). Those rows are recorded per size with that scope stated; they are not assigned to one version.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Self-chat corpus for Baize v1 | — | distill-SFT | teacher / generation cost | ChatGPT (gpt-3.5-turbo); approximately $100 for 111.5k dialogues | arXiv:2304.01196v4 §3 | verified 2026-09-14 | not applicable |
| Baize-v1-7B / 13B / 30B | 7B, 13B, 30B | distill-SFT | training data | Quora, Stack Overflow, Alpaca (54,456 / 57,046 / 51,942 dialogues; average turns 3.9 / 3.6 / 1.0) | arXiv:2304.01196v4 Table 2; Table 3 | verified 2026-09-14 | no ablation reported |
| Baize-v1-7B | 7B | distill-SFT | base model / trainable params | LLaMA-7B / 17.9M | arXiv:2304.01196v4 Table 3 | verified 2026-09-14 | no ablation reported |
| Baize-v1-13B | 13B | distill-SFT | base model / trainable params | LLaMA-13B / 28.0M | arXiv:2304.01196v4 Table 3 | verified 2026-09-14 | no ablation reported |
| Baize-v1-30B | 30B | distill-SFT | base model / trainable params | LLaMA-30B / 54.6M | arXiv:2304.01196v4 Table 3 | verified 2026-09-14 | no ablation reported |
| Baize-v1-7B / 13B / 30B | 7B, 13B, 30B | distill-SFT | compute | 9 / 16 / 36 GPU hours, NVIDIA A100-80G, single GPU | arXiv:2304.01196v4 Table 3 and caption | verified 2026-09-14 | not applicable |
| Baize v1 (all sizes) | 7B, 13B, 30B | distill-SFT | maximum input sequence length | 512 | arXiv:2304.01196v4 §5 | verified 2026-09-14 | no ablation reported |
| Baize (all versions) | all | distill-SFT / SFT | LoRA placement / rank | all linear layers / 8 | arXiv:2304.01196v4 §4; §5 | verified 2026-09-14 | stated motivation: more trainable parameters and adaptation capability (§4); no ablation reported |
| Baize (all versions) | all | distill-SFT / SFT | base weights / LoRA init | LLaMA int8 checkpoints, frozen / A Gaussian, B zero | arXiv:2304.01196v4 §5 | verified 2026-09-14 | no ablation reported |
| Baize, version not stated | 7B, 13B, 30B | distill-SFT / SFT | optimizer / batch size | Adam / 64 (unit not stated) | arXiv:2304.01196v4 §5 | verified 2026-09-14 | no ablation reported |
| Baize, version not stated | 7B | distill-SFT / SFT | learning rate | 2e-4 | arXiv:2304.01196v4 §5 | verified 2026-09-14 | no ablation reported |
| Baize, version not stated | 13B | distill-SFT / SFT | learning rate | 1e-4 | arXiv:2304.01196v4 §5 | verified 2026-09-14 | no ablation reported |
| Baize, version not stated | 30B | distill-SFT / SFT | learning rate | 5e-5 | arXiv:2304.01196v4 §5 | verified 2026-09-14 | no ablation reported |
| Baize-Healthcare | 7B | distill-SFT | base / data / trainable params / compute | LLaMA-7B / Quora, MedQuAD (46,867 dialogues) / 17.9M / 5 GPU hours, single GPU | arXiv:2304.01196v4 Table 2; Table 3 | verified 2026-09-14 | qualitative check by one healthcare practitioner (§6, Table 9); no quantitative evaluation |
| Baize-v1.5-7B / 13B | 7B, 13B | distill-SFT | training data | Quora v2 (55,770 dialogues, 3.0 turns), Stack Overflow v2 (112,343, 3.9 turns); AI turns regenerated one call at a time | arXiv:2304.01196v4 §3; Table 2; Table 3 | verified 2026-09-14 | no ablation against v1 data reported |
| Baize-v1.5-7B / 13B | 7B, 13B | distill-SFT | loss masking | loss on AI responses only, following Vicuna | arXiv:2304.01196v4 §4 | verified 2026-09-14 | no ablation reported |
| Baize-v1.5-7B / 13B | 7B, 13B | distill-SFT | trainable params / compute | 17.9M, 28.0M / 32, 64 GPU hours on 8 A100-80G GPUs | arXiv:2304.01196v4 Table 3 and caption | verified 2026-09-14 | not applicable |
| Baize-v1.5-7B / 13B | 7B, 13B | distill-SFT | maximum input sequence length | not reported (§5 states lengths for v1 and v2 only) | — | not reported (checked §3-§5, Table 3) | — |
| Baize-v2-7B / 13B | 7B, 13B | SFT (SDF) | initialization / data | Baize-v1.5-7B, Baize-v1.5-13B / instructions from the Quora dataset (number used not stated) | arXiv:2304.01196v4 §4; Table 3 | verified 2026-09-14 | Fig. 3: GPT-4 score 87% → 90% (7B), 89% → 92% (13B) |
| Baize-v2-7B / 13B | 7B, 13B | SFT (SDF) | candidates / scorer / target | 4 responses per instruction / ChatGPT overall score 1-100 / best-ranked response | arXiv:2304.01196v4 §4; Fig. 2; App. C | verified 2026-09-14 | Fig. 3 (as above); no ablation of candidate count |
| Baize-v2-7B / 13B | 7B, 13B | SFT (SDF) | trainable modules | new LoRA modules on all linear layers; v1.5 LoRA fixed | arXiv:2304.01196v4 §4, Eq. 2 | verified 2026-09-14 | no ablation reported |
| Baize-v2-7B / 13B | 7B, 13B | SFT (SDF) | maximum input sequence length | 1024 | arXiv:2304.01196v4 §5 | verified 2026-09-14 | no ablation reported |
| Baize-v2-7B / 13B | 7B, 13B | SFT (SDF) | compute | 38 / 76 GPU hours, single A100-80G | arXiv:2304.01196v4 Table 3 and caption | verified 2026-09-14 | not applicable |
| Baize (all) | all | all | epochs / LR schedule / warmup | not reported | — | not reported (checked §3-§6, Table 3, App. A-C) | — |
| Baize v1 / v2 | 7B, 13B, 30B | eval-gate | inference decoding | nucleus sampling, temperature 1, top-p 0.95 (default) | arXiv:2304.01196v4 §5 | verified 2026-09-14 | not applicable |
| Baize (all) | all | eval-gate | checkpoint selection rule | not reported | — | not reported (checked §4-§6) | — |

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2304.01196 (arXiv v4), §3-§6, Table 2, Table 3, App. A-C.
- Corrections to the previous card version: none (new file; recipe values previously appeared only in [[baize-construction]] and [[baize]], where they are corrected).
- Removed as unsupported by the source: none.
- Not reported by the source: epochs, LR schedule, warmup, v1.5 sequence length, number of SDF prompts, SDF sampling temperature.

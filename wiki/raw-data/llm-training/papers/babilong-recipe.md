<!-- scope: recipe ledger for the task-specific fine-tuning runs reported in the BABILong paper (RMT, ARMT, Mamba-130m, GPT-3.5-Turbo, Mistral-7B)
     deps: [[babilong]]
     see-also: [[ruler]]
-->

# BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack — Recipe ledger
- **Parent card:** [[babilong]]
- **URL:** https://arxiv.org/abs/2406.10149 (arXiv v2, 2024-11-06)
- **Source type:** paper
- **Scope:** these runs fine-tune models on individual BABILong tasks to test whether the tasks are solvable. They are not general-purpose post-training runs (§3.3).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| RMT, GPT-2 backbone | 137M | SFT | training data per task | trained on each task individually; 10,000 train and 1,000 eval samples per task; 2-320 facts per sample depending on task | arXiv:2406.10149v2 §3.3 | verified 2026-09-14 | no ablation reported |
| RMT, GPT-2 backbone | 137M | SFT | segment size / memory tokens | 512 tokens / 16 memory tokens | arXiv:2406.10149v2 §3.3; App. C | verified 2026-09-14 | no ablation reported |
| RMT, GPT-2 backbone | 137M | SFT | curriculum (segments per stage) | 1, 2, 4, 6, 8, 16, 32; at stage n the number of segments per batch is sampled from 1 to n | arXiv:2406.10149v2 App. C | verified 2026-09-14 | stated purpose "to prevent overfitting to a certain context size" (App. C); no ablation reported |
| RMT, GPT-2 backbone | 137M | SFT | max training length | 32 segments, "totalling in 16K tokens" | arXiv:2406.10149v2 §3.3 | verified 2026-09-14 | no ablation reported |
| RMT, GPT-2 backbone | 137M | SFT | batch size | 64 (unit not stated) | arXiv:2406.10149v2 App. C | verified 2026-09-14 | no ablation reported |
| RMT, GPT-2 backbone | 137M | SFT | optimizer / LR / schedule / warmup | AdamW / learning rate in range {5e-05, 3e-05} / linear / 1000 warmup steps | arXiv:2406.10149v2 App. C | verified 2026-09-14 | no ablation reported |
| RMT, GPT-2 backbone | 137M | SFT | weight decay / gradient stopping | 0.01 / "no gradient stopping was used" | arXiv:2406.10149v2 App. C | verified 2026-09-14 | no ablation reported |
| RMT, GPT-2 backbone | 137M | SFT | steps per stage / stopping rule | maximum 10,000 steps per curriculum stage, early stopping if metrics stop increasing | arXiv:2406.10149v2 App. C | verified 2026-09-14 | App. C Fig. 5: accuracy beyond training length varies across 3 runs; authors state short-context early stopping "may not be optimal" |
| RMT, GPT-2 backbone | 137M | SFT | runs | 3 runs with different memory initializations and dataset shuffles | arXiv:2406.10149v2 App. C | verified 2026-09-14 | App. C Fig. 5 (mean and std over 3 runs) |
| RMT / ARMT | 137M | SFT | compute | 1-4 NVIDIA A100 or H100; 40 minutes to 20 hours per curriculum stage | arXiv:2406.10149v2 App. C | verified 2026-09-14 | not applicable |
| ARMT, GPT-2 backbone | 137M | SFT | memory tokens | 10 | arXiv:2406.10149v2 §3.3; App. C | verified 2026-09-14 | no ablation reported |
| ARMT, GPT-2 backbone | 137M | SFT | curriculum (segments per stage) | 2-3-5-8-16-32 | arXiv:2406.10149v2 App. C | verified 2026-09-14 | no ablation reported |
| ARMT, GPT-2 backbone | 137M | SFT | learning rate | 1e-04 | arXiv:2406.10149v2 App. C | verified 2026-09-14 | no ablation reported |
| ARMT, GPT-2 backbone | 137M | SFT | memory dimension / non-linearity | 64 / DPFP-3 | arXiv:2406.10149v2 App. C | verified 2026-09-14 | no ablation reported |
| ARMT, GPT-2 backbone | 137M | SFT | other optimizer settings | not reported separately from RMT (App. C gives only the ARMT differences above) | arXiv:2406.10149v2 App. C | not reported (checked §3.3, App. C) | — |
| mamba-130m | 130M | SFT | curriculum | "exact same curriculum approach" as RMT, with randomly selected segment counts | arXiv:2406.10149v2 App. C | verified 2026-09-14 | no ablation reported |
| mamba-130m | 130M | SFT | batch size | 128 (constant across curriculum steps; unit not stated) | arXiv:2406.10149v2 App. C | verified 2026-09-14 | no ablation reported |
| mamba-130m | 130M | SFT | optimizer / LR / schedule / warmup | AdamW / 3e-4 / linear / 10% of total training steps | arXiv:2406.10149v2 App. C | verified 2026-09-14 | no ablation reported |
| mamba-130m | 130M | SFT | weight decay / gradient clipping | 2.0 / 1.0 | arXiv:2406.10149v2 App. C | verified 2026-09-14 | no ablation reported |
| mamba-130m | 130M | SFT | steps per stage | 10,000 per stage; 15,000 for the last stage (32 segments) | arXiv:2406.10149v2 App. C | verified 2026-09-14 | no ablation reported |
| mamba-130m | 130M | SFT | compute | 4 NVIDIA H100; 2 to 3 days per BABILong task | arXiv:2406.10149v2 App. C | verified 2026-09-14 | not applicable |
| GPT-3.5-Turbo | not reported | SFT | data / epochs | 1000 samples from QA1 / 3 epochs | arXiv:2406.10149v2 §3.3 | verified 2026-09-14 | App. I Fig. 9a caption: 90%+ accuracy on QA1 across context lengths after fine-tuning; no ablation of epochs |
| GPT-3.5-Turbo | not reported | SFT | LR, batch, max length | not reported (checked §3.3, App. C, App. I) | — | not reported | — |
| Mistral-7B-Instruct-v0.2 | 7B | SFT | data / epochs / method | 1000 samples from QA1 / 3 epochs / full fine-tuning | arXiv:2406.10149v2 §3.3; App. I Fig. 9c caption | verified 2026-09-14 | App. I Fig. 9c: QA2-QA5 scores degrade after QA1 fine-tuning (0K, no distractor text) |
| Mistral-7B-Instruct-v0.2 | 7B | SFT | LR, batch, max length | not reported (checked §3.3, App. C, App. I) | — | not reported | — |
| RMT, GPT-2 backbone | 137M | eval-gate | evaluation sample count | full test set up to 1M tokens; average over 100 samples at 10M | arXiv:2406.10149v2 App. C | verified 2026-09-14 | not applicable |

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2406.10149 (arXiv v2), §3.3, App. C, App. I.
- Corrections to the previous card version: none (new file; the previous [[babilong]] card had no recipe values).
- Removed as unsupported by the source: none.
- Not reported by the source: GPT-3.5-Turbo and Mistral-7B learning rate, batch size, and sequence length; epochs or total tokens for RMT, ARMT, and Mamba beyond the per-stage step limits.

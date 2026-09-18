<!-- scope: WRAP (Rephrasing the Web) pretraining and data-generation settings, split out of [[rephrasing-the-web]] for length
     deps: [[rephrasing-the-web]]
-->

# WRAP — Recipe ledger

Split out of [[rephrasing-the-web]] under the 120-line card limit. All rows read from
arXiv:2401.16380v1 (2024-01-29) on 2026-09-18.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| WRAP GPT-XL (paper model) | 1.3B | pretrain | architecture | 24 layers, 16 heads, hidden 2048 | arXiv:2401.16380v1 §3.2 | verified (2026-09-18) | no ablation reported |
| WRAP GPT-medium (paper model) | 350M | pretrain | architecture | 24 layers, 16 heads, hidden 1024 | arXiv:2401.16380v1 §3.2 | verified (2026-09-18) | no ablation reported |
| WRAP GPT-small (paper model) | 128M | pretrain | architecture | 12 layers, 12 heads, hidden 768 | arXiv:2401.16380v1 §3.2 | verified (2026-09-18) | no ablation reported |
| WRAP GPT-XL (paper model) | 1.3B | pretrain | max sequence length | 1024 | arXiv:2401.16380v1 §3.2 | verified (2026-09-18) | no ablation reported |
| WRAP GPT-XL (paper model) | 1.3B | pretrain | training steps; batch (tokens) | 300k steps; 1M tokens per step | arXiv:2401.16380v1 §3.2 | verified (2026-09-18) | no ablation reported |
| WRAP GPT-XL (paper model) | 1.3B | pretrain | peak LR | 2e-4 | arXiv:2401.16380v1 §3.2 | verified (2026-09-18) | no ablation reported |
| WRAP GPT-small / medium | 128M, 350M | pretrain | peak LR | 3e-4 | arXiv:2401.16380v1 §3.2 | verified (2026-09-18) | no ablation reported |
| WRAP (all sizes) | — | pretrain | dropout | none | arXiv:2401.16380v1 §3.2 | verified (2026-09-18) | no ablation reported |
| WRAP (all sizes) | — | pretrain | real:synthetic sampling ratio | 1:1 | arXiv:2401.16380v1 §3.1 | verified (2026-09-18) | §5 Table 4: synthetic-only loses accuracy on specialized-knowledge tasks |
| WRAP (data generation) | Mistral-7B-Instruct | data-gen | max rephrase chunk | 300 tokens | arXiv:2401.16380v1 §3.1 | verified (2026-09-18) | authors' empirical observation of information loss above 300 tokens (§3.1) |
| WRAP (implementation) | — | pretrain | codebase | NVIDIA Megatron-LM | arXiv:2401.16380v1 §3.2 | verified (2026-09-18) | no ablation reported |


## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2401.16380 (v1).
- Corrections to the previous card version: none (new file; rows moved unchanged from [[rephrasing-the-web]]).
- Removed as unsupported by the source: none.
- Not reported by the source: optimizer and betas, weight decay, warmup and decay shape, gradient clipping,
  total GPU-hours, synthetic token counts per style.

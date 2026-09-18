<!-- scope: recipe ledger split out of [[adam]] (§5.2 table format)
     deps: [[adam]]
-->

# Adam / AdamW recipe ledger

Split from [[adam]] to keep that card under 120 lines. Every row was read at the stated locus on
2026-09-18.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| (paper default, no model) | n/a | n/a | α, β₁, β₂, ε | 0.001, 0.9, 0.999, 1e-8 | arXiv:1412.6980 Algorithm 1 caption | verified 2026-09-18 | "Good default settings for the tested machine learning problems" (same caption); no ablation table |
| (paper default, no model) | n/a | n/a | AdamW α, β₁, β₂, ε | 0.001, 0.9, 0.999, 1e-8 | arXiv:1711.05101 Algorithm 2 line 1 | verified 2026-09-18 | grid search over (α, λ), Figure 2 |
| AdamW ResNet 26 2x64d | 11.6M | image classification | batch size | 128 | arXiv:1711.05101 §4 | verified 2026-09-18 | fixed across all runs in §4 |
| GPT-3 (all sizes) | 125M-175B | pretrain | Adam β₁, β₂, ε | 0.9, 0.95, 1e-8 | arXiv:2005.14165 App. B | verified 2026-09-18 | no ablation reported |
| GPT-3 (all sizes) | 125M-175B | pretrain | weight decay; grad-norm clip | 0.1; 1.0 | arXiv:2005.14165 App. B | verified 2026-09-18 | paper states weight decay was used "to provide a small amount of regularization"; no ablation reported |

## Verification
- Checked on 2026-09-18 against https://arxiv.org/abs/1412.6980 (v9), https://arxiv.org/abs/1711.05101
  (v3), and https://arxiv.org/abs/2005.14165 App. B.
- Corrections to the previous card version: new file; no previous version.
- Removed as unsupported by the source: none.
- Not reported by the source: Adam and AdamW give no language-model training settings; the GPT-3 rows
  give no per-size learning rates here (those are in the GPT-3 card).

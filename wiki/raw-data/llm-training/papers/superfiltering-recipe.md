<!-- scope: training and filtering settings disclosed by the Superfiltering paper, split out of [[superfiltering]]
     deps: [[superfiltering]]
-->

# Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning — recipe ledger
- **Core Insight:** All Superfiltering fine-tuning runs share one configuration except the learning rate, which is 2e-5 at 7B and 1e-5 at 13B (arXiv:2402.00530v2 §4.2).
- **Guideline:** When reproducing Superfiltering, take the values below as printed in §4.2 and note that the paper reports no ablation that selected any of them; the only selected quantity with evidence is the data budget.
- **Authors:** Ming Li, Yong Zhang, Shwai He, Zhitao Li, Hongyu Zhao, Jianzong Wang, et al.
- **Year:** 2024 (arXiv v1 2024-02; ACL 2024 main)
- **URL:** https://arxiv.org/abs/2402.00530
- **Source type:** paper
- **Relevant topics:** SFT configuration, data selection, IFD

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LLaMA2-7B Superfiltering SFT | 7B | SFT | learning rate | 2×10⁻⁵ | arXiv:2402.00530v2 §4.2 | verified 2026-09-18 | no ablation reported |
| LLaMA2-13B Superfiltering SFT | 13B | SFT | learning rate | 1×10⁻⁵ | §4.2 | verified 2026-09-18 | no ablation reported |
| Both | 7B, 13B | SFT | optimizer | Adam | §4.2 | verified 2026-09-18 | no ablation reported |
| Both | 7B, 13B | SFT | batch size | 128 | §4.2 | verified 2026-09-18 | no ablation reported |
| Both | 7B, 13B | SFT | epochs | 3 | §4.2 | verified 2026-09-18 | no ablation reported |
| Both | 7B, 13B | SFT | max sequence length | 2048 | §4.2 | verified 2026-09-18 | no ablation reported |
| Both | 7B, 13B | SFT | warmup rate | 0.03 | §4.2 | verified 2026-09-18 | no ablation reported |
| Both | 7B, 13B | SFT | selected data | 5% (2,600), 10% (5,200), 15% (7,800) of Alpaca | §5.1, Table 2 | verified 2026-09-18 | Table 2: all three budgets beat the 100% baseline on the Open LLM Leaderboard average |
| Superfilter | 124M | data selection | filter model | GPT-2, no fine-tuning | §3.3, §6.1 | verified 2026-09-18 | Table 3: GPT-2 IFD 1.133 against random 0.936 at 5% |


## Connections
- [[superfiltering]] — the card this ledger was split from.
- [[cherry-llm-recipe]] — the selection script for the IFD method Superfiltering reuses.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2402.00530 (arXiv v2, 7 Jun 2024, full PDF text)
- Corrections to the previous card version: none (new file, split out of [[superfiltering]])
- Removed as unsupported by the source: none
- Not reported by the source: GPU type and count for fine-tuning, weight decay, gradient clipping, LR schedule shape, loss masking, packing, and the hardware behind the filtering times in Table 4 beyond the statement that Superfiltering fits in 6 GB.

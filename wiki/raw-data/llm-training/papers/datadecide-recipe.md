<!-- scope: per-size pretraining configurations of the DataDecide suite (14 OLMo-ladder sizes, 4M-1B), companion ledger to the DataDecide card
     deps: [[datadecide]]
     see-also: [[task-scaling-model-ladders]], [[resolving-scaling-discrepancies]]
-->

# DataDecide: How to Predict Best Pretraining Data with Small Experiments — Recipe ledger
- **Core Insight:** The 14 DataDecide sizes share one token-to-parameter ratio (100), sequence length, and MLP ratio; batch size and learning rate change with size according to ladder heuristics, from batch 32 and LR 1.4e-02 at 4M to batch 704 and LR 2.1e-03 at 1B (App. A Table 2).
- **Guideline:** When reusing DataDecide checkpoints as small-scale baselines, compare against the size-specific configuration below, because every size was trained on all 25 recipes with the same configuration (Table 2 caption).
- **Authors:** Ian Magnusson, Nguyen Tai, Ben Bogin, David Heineman, Jena Hwang, Luca Soldaini, et al. (Allen Institute for AI; University of Washington; University of Pennsylvania)
- **Year:** 2025 (arXiv v1 2025-04; ICML 2025)
- **URL:** https://arxiv.org/abs/2504.11393
- **Source type:** paper
- **Relevant topics:** pretraining configurations, model ladder, batch size and learning-rate scaling

## Summary
Values are copied from arXiv:2504.11393v2 Appendix A, Table 2. Model size is the number of non-embedding parameters. Batch size is the number of sequences per batch. All sizes use sequence length "2024" (as printed in the caption) and MLP ratio 8. Each configuration is trained on the 25 data recipes with 3 seeds; for sizes below 1B, all seeds except the default stop at 25% of the compute used for the 1B model (Table 2 caption; §2.1). Hyperparameters are set by heuristics from Porian et al. (2024) through the OLMo model ladder (§2.1).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DataDecide 4M | 3.7M | pretrain-stable | batch (seq); hidden dim; peak LR; heads; layers; steps; tokens | 32; 64; 1.4e-02; 8; 8; 5,725; 0.4B | arXiv:2504.11393v2 App. A Table 2 | verified 2026-09-14 | ladder heuristics; no ablation reported |
| DataDecide 6M | 6.0M | pretrain-stable | same fields | 32; 96; 1.2e-02; 8; 8; 9,182; 0.6B | Table 2 | verified 2026-09-14 | same |
| DataDecide 8M | 8.5M | pretrain-stable | same fields | 32; 128; 1.1e-02; 8; 8; 13,039; 0.9B | Table 2 | verified 2026-09-14 | same |
| DataDecide 10M | 9.9M | pretrain-stable | same fields | 32; 144; 1.0e-02; 8; 8; 15,117; 1.0B | Table 2 | verified 2026-09-14 | same |
| DataDecide 14M | 14.4M | pretrain-stable | same fields | 32; 192; 9.2e-03; 8; 8; 21,953; 1.4B | Table 2 | verified 2026-09-14 | same |
| DataDecide 16M | 16.0M | pretrain-stable | same fields | 32; 208; 8.9e-03; 8; 8; 24,432; 1.6B | Table 2 | verified 2026-09-14 | same |
| DataDecide 20M | 19.1M | pretrain-stable | same fields | 64; 192; 8.4e-03; 8; 16; 14,584; 1.9B | Table 2 | verified 2026-09-14 | same |
| DataDecide 60M | 57.1M | pretrain-stable | same fields | 96; 384; 5.8e-03; 12; 16; 29,042; 5.7B | Table 2 | verified 2026-09-14 | same |
| DataDecide 90M | 97.9M | pretrain-stable | same fields | 160; 528; 4.9e-03; 12; 16; 29,901; 9.8B | Table 2 | verified 2026-09-14 | same |
| DataDecide 150M | 151.9M | pretrain-stable | same fields | 192; 768; 4.2e-03; 12; 12; 38,157; 15.0B | Table 2 | verified 2026-09-14 | same |
| DataDecide 300M | 320.0M | pretrain-stable | same fields | 320; 1,024; 3.3e-03; 16; 16; 45,787; 30.0B | Table 2 | verified 2026-09-14 | same |
| DataDecide 530M | 530.1M | pretrain-stable | same fields | 448; 1,344; 2.8e-03; 16; 16; 57,786; 53.0B | Table 2 | verified 2026-09-14 | same |
| DataDecide 750M | 681.3M | pretrain-stable | same fields | 576; 1,536; 2.5e-03; 16; 16; 63,589; 75.0B | Table 2 | verified 2026-09-14 | same |
| DataDecide 1B | 1176.8M | pretrain-stable | same fields | 704; 2,048; 2.1e-03; 16; 16; 69,369; 100.0B | Table 2 | verified 2026-09-14 | same |
| DataDecide, all sizes | 4M-1B | pretrain-stable | sequence length; MLP ratio; tokens per parameter | 2024 (as printed); 8; 100 | Table 2 caption; §2.1 | verified 2026-09-14 | ratio 100 chosen as typical overtraining (§2.1, §5) |
| DataDecide, all sizes | 4M-1B | pretrain-stable | seeds | 3 per recipe and size; non-default seeds below 1B stopped at 25% of 1B compute | Table 2 caption; §2.1 | verified 2026-09-14 | 1B reruns completed to capture run-to-run variance in the targets (§2.1) |
| DataDecide, all sizes | 4M-1B | pretrain-stable | optimizer; betas; weight decay; warmup; LR schedule shape; tokenizer | not reported | checked §2, App. A-C, blog | not reported | none |
| DataDecide suite | 4M-1B | pretrain-stable | total compute | about 820K H100 GPU hours | Impact Statement | verified 2026-09-14 | not applicable |

## Connections
- [[datadecide]]: main card with the method, results, and limitations.
- [[resolving-scaling-discrepancies]]: source of the batch-size and learning-rate heuristics.
- [[task-scaling-model-ladders]]: the OLMo model-ladder method these configurations follow.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2504.11393 (v2, 2025-07-13; v1 2025-04-15), Appendix A Table 2 read in both layout and non-layout text extractions to confirm row alignment.
- Audit claims not found in the source: none.
- Note: the printed sequence length "2024" is kept as printed; the paper gives no other value.

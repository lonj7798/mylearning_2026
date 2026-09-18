<!-- scope: recipe ledger extracted from [[mixed-precision]] (arXiv:1710.03740v3)
     deps: [[mixed-precision]]
-->

# Mixed Precision Training — recipe ledger

Split out of [[mixed-precision]] to keep that card under 120 lines. Every row is read at the stated locus
in arXiv:1710.03740v3 (ICLR 2018).

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Multibox SSD (detection) | not reported | train | constant loss scale | 8 | arXiv:1710.03740v3 §3.2, §4.2 | verified 2026-09-18 | §4.2 Table 2: unscaled FP16 fails to train; scaled matches baseline mAP |
| bigLSTM (1B-word LM) | 2 layers x 8192 LSTM cells, 1024 projection | train | constant loss scale | 128 | arXiv:1710.03740v3 §4.5 | verified 2026-09-18 | §4.5, Figure 5: unscaled FP16 perplexity diverges after 300K iterations |
| bigLSTM (1B-word LM) | as above | train | optimizer / epochs / batch | Adagrad; 50 epochs; batch 1024 aggregated over 4 GPUs; 8K sampled-softmax negatives; 793K vocab | arXiv:1710.03740v3 §4.5 | verified 2026-09-18 | no ablation reported |
| ResNet-50 and five other CNNs | not reported | train (ILSVRC12) | loss scaling | none required | arXiv:1710.03740v3 §4.1 | verified 2026-09-18 | §4.1 Table 1: mixed precision within 0.3 points of FP32 top-1 |
| DCGAN (128x128 faces) | 7 generator + 8 discriminator layers | train | optimizer / iterations / loss scaling | Adam; 100K iterations; no loss scaling | arXiv:1710.03740v3 §4.6 | verified 2026-09-18 | no quantitative metric reported; comparison is qualitative |
| All experiments | — | train | arithmetic | FP16 storage, FP32 master weights, Tensor Core accumulation into FP32 | arXiv:1710.03740v3 §4 preamble | verified 2026-09-18 | §3.3: FP16 accumulation did not match baseline for some models |

Peak learning rates, warmup, and weight decay are not reported; §4 states that public training schedules
were reused and that hyper-parameters were unchanged from the FP32 baselines.


## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/1710.03740 (arXiv v3).
- Corrections to the previous card version: none (new file).
- Removed as unsupported by the source: none.

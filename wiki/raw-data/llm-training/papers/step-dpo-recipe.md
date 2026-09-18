<!-- scope: training settings disclosed by the Step-DPO paper, split out of [[step-dpo]]
     deps: [[step-dpo]]
-->

# Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs — recipe ledger
- **Core Insight:** The SFT and Step-DPO runs use one shared configuration per stage, with only the number of epochs and the preference strength beta varying by model size (arXiv:2406.18629v1 §4.1).
- **Guideline:** When reproducing Step-DPO, take the values below from §4.1 as printed and do not transfer a value across stages; the paper reports no ablation that selected any of them.
- **Authors:** Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, Jiaya Jia
- **Year:** 2024 (arXiv v1 2024-06)
- **URL:** https://arxiv.org/abs/2406.18629
- **Source type:** paper
- **Relevant topics:** preference optimization, training configuration, mathematical reasoning

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Step-DPO SFT models (7B) | 7B | SFT | epochs | 3 | arXiv:2406.18629v1 §4.1 Implementation Details | verified 2026-09-18 | no ablation reported |
| Step-DPO SFT models (>30B) | 32B-72B | SFT | epochs | 2 | §4.1 | verified 2026-09-18 | no ablation reported |
| Step-DPO SFT models | all | SFT | global batch size | 256 | §4.1 | verified 2026-09-18 | no ablation reported |
| Step-DPO SFT models | all | SFT | learning rate / scheduler / warmup ratio | 5e-6, linear decay, 0.03 | §4.1 | verified 2026-09-18 | no ablation reported |
| Step-DPO SFT models | all | SFT | optimizer / memory | AdamW; DeepSpeed ZeRO3 with CPU offload | §4.1 | verified 2026-09-18 | no ablation reported |
| Step-DPO models (7B) | 7B | preference | epochs | 8 | §4.1 | verified 2026-09-18 | no ablation reported |
| Step-DPO models (>30B) | 32B-72B | preference | epochs | 4 | §4.1 | verified 2026-09-18 | no ablation reported |
| Step-DPO models | all | preference | global batch size | 128 | §4.1 | verified 2026-09-18 | no ablation reported |
| Step-DPO models | all | preference | learning rate / scheduler / warmup ratio | 5e-7, cosine, 0.1 | §4.1 | verified 2026-09-18 | no ablation reported |
| Step-DPO models (72B) | 72B | preference | β | 0.5 | §4.1 | verified 2026-09-18 | no ablation reported |
| Step-DPO models (other sizes) | 7B-57B | preference | β | 0.4 | §4.1 | verified 2026-09-18 | no ablation reported |
| Step-DPO models | all | preference | preference pairs | 10K (5K in the Table 3 ablation) | §4.1; §4.3 | verified 2026-09-18 | Table 3 compares DPO and Step-DPO at 5K |
| Step-DPO models | all | preference | training steps | fewer than 500 | Abstract | verified 2026-09-18 | no ablation reported |


## Connections
- [[step-dpo]] — the card this ledger was split from.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2406.18629 (arXiv v1, 26 Jun 2024, full PDF text)
- Corrections to the previous card version: none (new file, split out of [[step-dpo]])
- Removed as unsupported by the source: none
- Not reported by the source: GPU count and type, wall-clock training time, sequence length, weight decay, gradient clipping, loss masking, and the reference model used for each Step-DPO run beyond "the SFT model".

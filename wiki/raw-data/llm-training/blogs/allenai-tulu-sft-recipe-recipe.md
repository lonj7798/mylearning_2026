<!-- scope: recipe ledger split out of [[allenai-tulu-sft-recipe]] (§5.2 table format)
     deps: [[allenai-tulu-sft-recipe]]
-->

# Tülu 3 SFT recipe ledger

Split from [[allenai-tulu-sft-recipe]] to keep that card under 120 lines. Every row was read at the
stated locus on 2026-09-18 in arXiv:2411.15124v5.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | peak LR; schedule; warmup ratio | 5e-6; linear; 0.03 | arXiv:2411.15124v5 Table 11 | verified 2026-09-18 | §4.3.2 Figure 5: sum loss at 5e-6 beat 2e-6, 1e-5, 2e-5 on average performance (Llama 3.0 on the Tülu 2 mix, 1 seed) |
| Llama-3.1-Tulu-3-70B-SFT | 70B | SFT | peak LR | 2e-6 | Table 11; §4.3 text ("found after a hyperparameter search") | verified 2026-09-18 | search reported, grid not printed |
| Llama-3.1-Tulu-3-8B-SFT / 70B-SFT | 8B, 70B | SFT | effective batch size (sequences); max token length; epochs | 128; 4,096; 2 | Table 11 | verified 2026-09-18 | §4.3.2 Figure 6: 2 epochs best over 2-7 |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | prompts; loss aggregation | 939,344; sum over tokens | Table 7; §4.3.2 | verified 2026-09-18 | Figure 5 compares sum vs mean loss |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | compute | 32 GPUs, 6 hours (H100 nodes) | §4.3 | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3-70B-SFT | 70B | SFT | compute | 64 GPUs, 50 hours | §4.3 | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3-405B-SFT | 405B | SFT | peak LR; effective batch; max length; epochs | 2e-6; 256; 4,096; 2 | Table 34 | verified 2026-09-18 | §8.1 states the larger batch follows from more GPUs and the LR was lowered |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | preference mix size | 271,409 | Table 15 | verified 2026-09-18 | §5.2 mix selection by average score |
| all sizes | — | SFT | optimizer and betas; packing; prompt-token masking; NEFTune; precision; sharding | not reported | body, App. A-E, Tables 11 and 34 all checked | not reported | — |

## Verification
- Checked on 2026-09-18 against https://arxiv.org/abs/2411.15124 (arXiv v5, 2025-04-14).
- Corrections to the previous card version: new file; no previous version.
- Removed as unsupported by the source: none.
- Not reported by the source: optimizer name and betas, weight decay, precision, sharding strategy,
  packing, prompt-token masking, NEFTune.

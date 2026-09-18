<!-- scope: Recipe ledger for the Yi family (arXiv:2403.04652v3), split out of [[yi]] to keep that card under 120 lines
     deps: [[yi]]
     see-also: [[qwen-2.5]], [[glm-4]]
-->

# Yi: Open Foundation Models by 01.AI — Recipe ledger

Every row below is read from arXiv:2403.04652v3 (checked 2026-09-18). Table format follows the course
authoring standard §5.2. The parent card is [[yi]].


| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Yi-6B / Yi-34B base | 6B, 34B | pretrain-stable | pretraining tokens (bilingual EN+ZH) | 3.1T tokens | arXiv:2403.04652v3 Abstract, §1 | verified (2026-09-18) | §1: 34B chosen below the 70B convention, tokens raised to 3.1T to compensate for lower FLOPs; §8 states the model had not saturated at 3.1T. No token-count ablation reported. |
| Yi-6B base | 6B | pretrain-stable | sequence length; max LR | 4096; 3 × 10⁻⁴ | §2.3 Table 1 | verified (2026-09-18) | no ablation reported |
| Yi-34B base | 34B | pretrain-stable | sequence length; max LR | 4096; 1.5 × 10⁻⁴ | §2.3 Table 1 | verified (2026-09-18) | no ablation reported |
| Yi-6B / Yi-34B base | 6B, 34B | pretrain-stable | tokenizer vocabulary | 64,000 (BPE, SentencePiece) | §2.2 | verified (2026-09-18) | §2.2: stated as a balance of computational efficiency and word comprehension; no ablation reported |
| Yi-6B-200K / Yi-34B-200K | 6B, 34B | long-context | continual-pretraining tokens | 10B tokens from the pretraining mixture with upsampled long sequences; "only 1-2B tokens is enough for the model to converge to low loss on 4K-200K length" | §2.3 | conflict (2026-09-18) | §2.3 prose; conflicts with the §7.1 figure below, and the report does not say which number produced the released 200K checkpoints |
| Yi-6B-200K / Yi-34B-200K | 6B, 34B | long-context | continual-pretraining tokens; batch; steps | 5B tokens; 4M batch; 100 optimization steps | §7.1, Figure 6 caption | conflict (2026-09-18) | Figure 6 caption ties the 5B-token run to the reported Needle-in-a-Haystack heatmap, so this is the row tied to a released result |
| Yi-6B-Chat / Yi-34B-Chat | 6B, 34B | SFT | dataset size | fewer than 10K multi-turn instruction-response dialogs | Abstract, §3.1 | verified (2026-09-18) | §3.1: preliminary experiments found a smaller manually annotated set superior to open-source sets of several hundred thousand entries; the comparison is described, not tabulated |
| Yi-6B-Chat / Yi-34B-Chat | 6B, 34B | SFT | optimizer; seq len; batch; steps; LR; weight decay; grad clip | AdamW β₁ 0.9, β₂ 0.999, ε 10⁻⁸; 4096; 64; 300; constant 1 × 10⁻⁵; 0.1; 1.0 | §3.2 | verified (2026-09-18) | no ablation reported |
| Yi-34B-Chat | 34B | SFT | NEFTune noise scale | 45 | §3.2 | verified (2026-09-18) | no ablation reported |
| Yi-6B-Chat | 6B | SFT | NEFTune noise scale | 5 | §3.2 | verified (2026-09-18) | no ablation reported |
| Yi-9B base | 9B | mid-train (depth upscale + continual pretraining) | layer duplication | Yi-6B's 32 layers → 48 layers by duplicating layers 12-28 | §7.3 | verified (2026-09-18) | §7.3 Figure 8: input/output cosine similarity of the duplicated layers is near 1; Table 8 shows Yi-9B-Init within ~1 point of Yi-6B on MMLU before further training |
| Yi-9B base | 9B | mid-train | continual-training tokens; LR; batch | approximately 800B tokens across two stages, around 70% recently collected; constant 3e-5; increased from 4M tokens whenever loss plateaued | §7.3 | verified (2026-09-18) | no ablation reported |


## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2403.04652 (arXiv v3, 2025-01-21).
- Corrections to the previous card version: none (this file is new; it holds rows moved out of [[yi]]).
- Removed as unsupported by the source: none.
- Not reported by the source: pretraining batch size, LR schedule shape, warmup, optimizer settings, total
  compute or GPU-hours; pretraining and long-context mixture proportions; SFT epoch count (only 300 steps);
  which of the §2.3 (10B) and §7.1 (5B) continual-pretraining token counts produced the released 200K checkpoints.

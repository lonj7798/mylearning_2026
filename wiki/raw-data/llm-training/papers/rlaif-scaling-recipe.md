<!-- scope: recipe ledger for [[rlaif-scaling]]; values read at the loci given
     deps: [[rlaif-scaling]]
-->

# Recipe ledger — RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback

Split from `rlaif-scaling.md` to keep that card at or under 120 lines. Table format follows §5.2 of the
llm-training authoring standard.


| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| PaLM 2 XS (Reddit TL;DR SFT) | XS | SFT | batch size; epochs; optimizer; LR; max input/output length | 128; 1 epoch; Adafactor; 1e-5; 1024 / 128 tokens | arXiv:2309.00267v3 App. F | verified 2026-09-18 | no ablation reported |
| PaLM 2 XS reward models | XS | reward-model | epochs; optimizer; LR; batch size; max input length | trained to plateau, 2–3 epochs; Adafactor; 1e-5; 128 (summarization) / 32 (other tasks); 1152 tokens | arXiv:2309.00267v3 App. F | verified 2026-09-18 | RM accuracies in Table 5 (App. G) |
| PaLM 2 XS summarization RM | XS | reward-model | initialization checkpoint | SFT model for AI-feedback RM; PaLM 2 XS for human-feedback RM | arXiv:2309.00267v3 App. F, Table 6 | verified 2026-09-18 | Table 6 (App. G): SFT init gives 74.2% vs 73.0% on AI-labeled data |
| RLAIF / RLHF policies | XS | RL | algorithm | REINFORCE with a learned value baseline; γ = 1; terminal reward only | arXiv:2309.00267v3 App. E | verified 2026-09-18 | no ablation reported |
| RLAIF / RLHF policies | XS | RL | sampling temperature; batch size; LR; epochs; KL coefficient β | T = 0.9; 128; 1e-5; 8 epochs; β = 0.05 in J(θ) = E[(1−β)r_φ − β·D_KL(π^RL ‖ π^SFT)] | arXiv:2309.00267v3 App. F, App. A.3 | verified 2026-09-18 | no ablation reported |
| AI labeler (PaLM 2 L / S / XS) | — | preference | max input context; CoT decode length; decoding temperature | 4096 tokens; 512 tokens; T = 0.0 | arXiv:2309.00267v3 App. D | verified 2026-09-18 | self-consistency at T = 0.3–1.0 degraded alignment (App. M) |
| Checkpoint selection | XS | RL | rule | select among checkpoints with high validation reward, then rank by LLM-judged win rate and manual inspection | arXiv:2309.00267v3 App. F | verified 2026-09-18 | no ablation reported |

## Verification
- Checked on 2026-09-18 against the primary source cited in each Source location cell.
- Corrections to the previous card version: new file, split out of `rlaif-scaling.md`.
- Removed as unsupported by the source: none.

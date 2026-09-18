<!-- scope: recipe ledger for model spec midtraining (MSM) experiments — data sizes, LoRA/optimizer settings, evaluation sampling — from the paper linked by the Anthropic MSM blog post
     deps: [[anthropic-model-spec-midtraining]]
     see-also: [[openai-alignment-midtraining-generalization]], [[anthropic-teaching-claude-why]]
-->

# Model Spec Midtraining: Improving How Alignment Training Generalizes — recipe ledger
- **Core Insight:** All MSM experiments are LoRA fine-tunes (rank 64, alpha 128, 1 epoch, AdamW lr 1e-4) of 8B–32B open models; MSM budgets are ∼8M tokens (Llama-3.1-8B) and 27M–41M tokens (Qwen), and the Qwen runs use 4M–8M AFT tokens plus 2M instruction-tuning tokens (arXiv:2605.02087v2 §3.1, §4, §5.1, §5.2, App. B.4).
- **Guideline:** When comparing MSM with other midtraining studies, compare token budgets and the stage that follows midtraining, because MSM was followed only by SFT on already post-trained Qwen checkpoints (§4 "Training"; §7).
- **Authors:** Chloe Li, Nevan Wichers, Sara Price, Samuel Marks, Jon Kutasov
- **Year:** 2026 (arXiv v1 2026-05-03; v2 2026-05-22)
- **URL:** https://arxiv.org/abs/2605.02087 (linked from https://alignment.anthropic.com/2026/msm/)
- **Source type:** paper (recipe companion to an official blog card)
- **Relevant topics:** mid-training data budget, LoRA SFT, alignment SFT data, evaluation sampling

## Recipe ledger
All loci are arXiv:2605.02087v2. All rows verified 2026-09-14 unless marked otherwise.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B base (pro-affordability / pro-America runs) | 8B | mid-train | MSM tokens per spec | ∼8M tokens | §3.1 "Training" | verified | Figure 2: each model generalizes to its spec's value; 4 seeds |
| same | 8B | SFT | AFT data | 165k tokens (5k samples) of cheese-preference chat | §3.1 "Training" | verified | no ablation reported |
| same | 8B | SFT | instruction-tuning data | 2M tokens (13.5k samples) | §3.1 "Training" | verified | no ablation reported |
| same | 8B | SFT | instruction-tuning composition | No Robots + 4,000 formatted MMLU variants; plus 2,500 synthetic identity samples | App. B.3 | verified | no ablation reported |
| Llama-3.1-8B base (6 further values) | 8B | SFT | AFT data | 150-162k tokens (5k samples) | §3.2 "Setup" | verified | Figure 3 |
| Qwen2.5-32B-Instruct; Qwen3-32B (AM spec) | 32B | mid-train | MSM tokens | 41M tokens | §4 "Training"; §4.2 | conflict | App. E Figure 12 caption prints 40M tokens for the scaling sweep (next row) |
| Qwen2.5-32B-Instruct (scaling sweep) | 32B | mid-train | MSM tokens | 40M tokens | App. E Figure 12 caption | conflict | main text §4.2 prints 41M for the same sweep |
| Qwen2.5-32B-Instruct; Qwen3-32B | 32B | SFT | AFT data | 8M tokens (with CoT) or 5M tokens (no CoT) | §4 "Training" | verified | Figure 4 right: MSM + AFT vs AFT, 4 seeds |
| same | 32B | SFT | instruction-tuning data | 2M tokens (10k samples), Table 2 mix, samples ≤ 8192 tokens, spec-misaligned samples filtered with Claude Sonnet 4.6 | App. B.3, Table 2 | verified | stated purpose: fix incoherence from midtraining Instruct models; no ablation |
| same | 32B | SFT | AFT scaling sweep | 1,250 to 80k samples; random subsamples of 80k generated samples; 1 seed | §4.2; Figure 5 caption | verified | Figure 5 |
| Qwen3-32B | 32B | SFT | thinking control for AFT (no CoT) | `\no_think` appended to system and user prompt | App. B.2 | verified | no ablation reported |
| Qwen2.5-14B-Instruct, Qwen2.5-32B-Instruct, Qwen3-14B, Qwen3-32B (rules vs values) | 14B, 32B | mid-train | MSM tokens per spec | 27M tokens | §5.1 "Training" | verified | Figure 7; 4 seeds |
| same | 14B, 32B | SFT | AFT data | 7M tokens (with CoT), 5M tokens (no CoT) | §5.1 "Training" | verified | Figure 7 |
| Qwen2.5-32B-Instruct; Qwen3-32B (general vs specific spec) | 32B | mid-train / SFT | MSM; AFT; instruction tuning | 41M; 7M (DA) or 4M (non-CoT); 2M tokens | §5.2 "Training" | verified | Figure 8 |
| all runs | 8B–32B | mid-train and SFT | adapter | LoRA rank 64, alpha 128, all attention and MLP projection layers | App. B.4 | verified | no ablation reported |
| all runs | 8B–32B | mid-train and SFT | optimizer and schedule | AdamW, lr 1e-4, cosine, 5% warmup, weight decay 0.01, 1 epoch | App. B.4 | verified | no ablation reported |
| all runs | 8B–32B | mid-train and SFT | max sequence length | 8192 (§4–5, Table 2 mix); 4096 (§3) | App. B.4 | verified | 8192 chosen for long-context samples |
| all runs | 8B / 14B / 32B | compute | GPUs | 1 / 2 / 4 H200 (141 GB) | App. B.4 | verified | — |
| all runs | — | data generation | generator model | Claude Opus 4.6 for MSM and AFT data | §2; App. B.1 | verified | — |
| Qwen AM runs | 32B | SFT data filter | judge and criteria | Claude Opus 4.6; spec alignment and no continuation desires (PASS on all criteria) | App. B.2 | verified | no ablation reported |
| Qwen AM runs | 32B | eval-gate | AM sampling | 27 scenarios × n_repeat=300, temperature 0.7, judge Claude Sonnet 4.6 | App. D.3 | verified | — |
| all runs | — | mid-train and SFT | batch size; number of MSM documents; GPU-hours | not reported | checked §2–5, §7, App. B.1–B.4, App. D.3, App. E | not reported | — |

## Connections
- [[anthropic-model-spec-midtraining]] — main card for the blog post and results.
- [[openai-alignment-midtraining-generalization]] — midtraining budget of 230k documents (~340M tokens) followed by SFT + RLVR, for comparison with the 41M-token MSM budget.
- [[anthropic-teaching-claude-why]] — Anthropic-internal SDF corpora of 14M to over 300M tokens.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2605.02087 (v2, 2026-05-22), the paper linked from https://alignment.anthropic.com/2026/msm/.
- Audit claims not found in the source: none (no audit claims were supplied for the recipe).

<!-- scope: Tülu 3 fully-open post-training report — prompt curation, SFT, length-normalized DPO, RLVR, and the development/unseen evaluation split
     deps: [[README]]
     see-also: [[olmo-2]], [[llama-3]], [[rlvr-tulu3]], [[dpo]], [[tulu-3.1]]
-->

# Tülu 3: Pushing Frontiers in Open Language Model Post-Training
- **Core Insight:** A four-stage open post-training recipe (prompt curation → SFT → length-normalized DPO → RLVR) raises the 8B average on the development suite from 64.9 (SFT) to 68.8 (final) and on a separately designed unseen suite from 29.9 to 32.4, so the gains are measured against evaluations that were not inspected during development (§7.4, Table 31).
- **Guideline:** When a post-training pipeline is tuned against a fixed evaluation set, hold out a second suite that is designed independently and never read during development, because the report's own unseen-suite numbers show large format-specific gaps that the development suite hides (IFEval 82.4 vs IFEval-OOD 24.3 for the 8B final model, Table 31).
- **Authors:** Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, et al. (Allen Institute for AI and University of Washington)
- **Year:** 2024 (arXiv v1 2024-11; v5 2025-04)
- **URL:** https://arxiv.org/abs/2411.15124
- **Source type:** official technical report
- **Relevant topics:** open post-training recipe, SFT data mixing, length-normalized DPO, RLVR / verifiable rewards, PPO for LLMs, decontamination, held-out evaluation

## Abstract
Tülu 3 is a family of fully-open post-trained models built on Llama 3.1 base models, released with data, code, and training recipes. The report states that Tülu 3 achieves results surpassing the instruct versions of Llama 3.1, Qwen 2.5 and Mistral, and closed models including GPT-4o-mini and Claude 3.5-Haiku. The training algorithms are supervised finetuning (SFT), Direct Preference Optimization (DPO), and a method the authors name Reinforcement Learning with Verifiable Rewards (RLVR). The report also introduces a multi-task evaluation scheme with separate development and unseen evaluations, standard benchmark implementations, and decontamination of existing open datasets against those benchmarks. It closes with analysis of training methods that did not reliably improve performance (Abstract).

## Key Contributions
- RLVR: PPO whose reward is a deterministic verifier output, `v(x,y) = α` if the extracted answer is correct and 0 otherwise, with `α = 10` set from pilot experiments and not tuned further (§6, Eq. 7–8).
- A curated prompt pool of 23,327,961 instances, of which 939,344 prompts are used in SFT and 425,145 in DPO (Table 7).
- An evaluation regime split into a development suite and an unseen suite, where unseen scores were not examined during development (§2.2, §7.3, Table 3).
- A decontamination toolkit and the resulting removals: training sets overlapping the unseen evaluations were removed entirely; those overlapping development evaluations were removed when this did not significantly harm performance (§3.2, Table 8).
- Released 8B, 70B and 405B checkpoints, the evaluation suite (olmes) and the training code (open-instruct).

## Key Figures/Tables to Study
- **Table 3:** the development/unseen split per core skill.
- **Table 7:** prompt pool by dataset, with SFT and DPO prompt counts.
- **Table 11, Table 20, Table 21:** SFT, DPO and PPO/RLVR hyperparameters.
- **Table 23:** 8B and 70B final RLVR scores against their DPO starting points and Llama 3.1 Instruct.
- **Table 31:** development vs unseen scores for SFT, DPO and final checkpoints — the overfitting measurement.

## Technical Details — post-training pipeline

### SFT
- 939,344 prompts used in SFT, drawn from the pool in Table 7; the largest contributors are Tülu 3 Persona MATH (149,960), Evol CodeAlpaca (107,276), WildChat GPT-4 subset (100,000) and Aya (100,000) (Table 7).
- 2 epochs, effective batch size 128, max sequence length 4,096, linear schedule, warmup ratio 0.03; LR 5e-6 for 8B and 2e-6 for 70B (Table 11).
- 8B SFT trained on 32 GPUs for 6 hours; 70B on 64 GPUs for 50 hours (§4.3).
- Ablations (Table 10, 8B): removing WildChat drops the average from 60.1 to 58.9; removing safety data drops the safety score from 93.1 to 74.7 while the rest is roughly unchanged; removing Persona data drops IFEval from 72.8 to 53.6; removing math data drops GSM8K from 76.2 to 64.1 and MATH from 31.5 to 23.5.

### DPO
- Objective: length-normalized DPO, selected over standard DPO and SimPO in Table 18 (§5.4).
- 425,145 prompts are used for DPO across the 8B and 70B mixes; the final preference mixes contain more than 270k data points for the 8B model and more than 330k for the 70B model (Table 7, §5.3, Table 15).
- 1 epoch, effective batch size 128, max token length 2,048, linear schedule, warmup ratio 0.1, β = 5; LR 5e-7 for 8B and 2e-7 for 70B (Table 20, §5.4).

### RLVR
RLVR replaces the learned reward model with a verifier: `max E[v(x,y) − β·KL(π_θ ‖ π_ref)]` (Eq. 7). Verifiers cover three evaluations — GSM8K (extract the final number, compare to the label), MATH (extract the answer under the "flex" logic) and IFEval (one verification function per constraint template). The combined verifiable prompt mixture is roughly 30,000 prompts (§6.1, Table 22). PPO is the optimizer; no code-execution verifier is used, and the authors leave more complex verifiers to future work (§6.1).
- 8B final RLVR run: 65 hours on 8 H100 GPUs; 70B: 60 hours on 48 GPUs; 405B: 46 hours on 256 GPUs. For all of these an earlier-than-final checkpoint was taken (§6.3).
- Checkpoint selection: evaluated every 100 training steps (40 for 70B); the 8B final model is the checkpoint with best overall MATH and IFEval performance (§6.4).
- Measured 8B gain over the DPO starting point (Table 23): GSM8K 84.3 → 87.6, MATH 42.0 → 43.7, IFEval 81.1 → 82.4, average 64.4 → 64.8. At 70B the report describes more modest IFEval and MATH improvements and no GSM8K improvement (93.5 → 93.5), attributed to saturation (§6.4).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | LR / schedule / warmup | 5e-6, linear, 0.03 | arXiv:2411.15124v5 Table 11 | verified 2026-09-18 | §4.3: "found after a hyperparameter search"; no table given |
| Llama-3.1-Tulu-3-70B-SFT | 70B | SFT | LR | 2e-6 | Table 11 | verified 2026-09-18 | same search, not tabulated |
| Tülu 3 SFT (both sizes) | 8B/70B | SFT | Epochs / effective batch / max length | 2 / 128 / 4,096 tokens | Table 11 | verified 2026-09-18 | no epoch ablation reported |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | Length-normalized DPO β / LR / epochs / batch / max length | 5 / 5e-7 / 1 / 128 / 2,048 | Table 20 | verified 2026-09-18 | Table 18: length-normalized DPO is the only variant beating the base checkpoint |
| Llama-3.1-Tulu-3-70B-DPO | 70B | preference | LR | 2e-7 | Table 20, §5.4 | verified 2026-09-18 | Table 19 LR ablation on two 70B mixes (5e-7 best on Mix 1, 2e-7 best on Mix 2) |
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | PPO LR / effective batch / K / ε / GAE λ / γ | 3e-7 / 224 / 4 / 0.2 / 0.95 / 1.0 | Table 21 | verified 2026-09-18 | no ablation reported for these rows |
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | KL coefficient β / warmup ω | 0.05 / 0.0 | Table 21 caption | verified 2026-09-18 | §6.2 sweep over β ∈ [0.1, 0.05, 0.03, 0.01]; §6.4 also tested up to 0.15 |
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | Total episodes / response length / temperature | 100,000 / 2,048 (1,024 for GSM8K only) / 1.0 | Table 21 | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | Verifier reward α | 10 | §6, after Eq. 8 | verified 2026-09-18 | "based on pilot experiments and did not tune it further" |
| Llama-3.1-Tulu-3-70B | 70B | RL (RLVR) | LR / warmup / response length / episodes / batch / β | 1e-7 / 0.1 / 2,048 / 400,000 / 640 / 0.7 | §6.4 | verified 2026-09-18 | "based on previous 70B RL development runs" |
| Tülu 3 8B RM (for PPO-with-RM ablation) | 8B | reward-model | PPO-against-RM total episodes / batch / K / response length | 300,000 / 224 / 1 / 1,024 | Table 21 | verified 2026-09-18 | Figure 16 PPO-vs-DPO ablation |

Note the β conflict: Table 21 caption states the final 70B RLVR model used β = 0.07 and ω = 0.07, while §6.4 states β = 0.7 for the 70B run. Both rows are from the same document version; the report does not reconcile them. **conflict**.

## Findings relevant to generality
- **Development vs unseen split.** The development suite is MMLU, PopQA, TruthfulQA, BigBenchHard, DROP, MATH, GSM8K, HumanEval, HumanEval+, IFEval, AlpacaEval 2 and Tülu 3 Safety; the unseen suite is MMLU-Pro, GPQA, AGIEval English, Deepmind Mathematics, BigCodeBench, IFEval-OOD and HREF (Table 3). The report states that unseen scores were not examined while developing the models (§2.2).
- **Overfitting measured, not assumed.** Table 31 (8B final): IFEval 82.4 on the development side against IFEval-OOD 24.3 on the unseen side; coding moves the opposite way to the development score, with HumanEval 86.2 → 83.9 from SFT to final while BigCodeBench falls 11.5 → 7.4. The overall 8B unseen average still rises 29.9 → 32.4 across the pipeline, which the authors read as the pipeline generalizing (§7.4). **Result (single study)**.
- **Overoptimization in RLVR.** §6.2 reports that lowering the KL penalty β increases the KL from the initial model and that more KL divergence typically results in lower average scores (Figure 21), with one exception in Figure 22. Appendix B.4 shows RLVR overoptimization on prompts with constraints. The claim that verifiable rewards are immune to reward hacking is contradicted by the report itself.
- **LM-as-judge agreement.** The composite HREF evaluation procedure reached 69.4% agreement with human majority judgments, compared to 67% inter-human agreement, on the subset where human judgments were collected (§7.3.2).

## Connections
- [[rlvr-tulu3]] — dedicated page for the RLVR component.
- [[tulu-3.1]] — the 8B follow-up checkpoint that changes only the final RL stage from PPO to GRPO.
- [[olmo-2]] — applies this recipe to Allen AI's own base models.
- [[llama-3]] — contrast: Llama 3 post-training uses rejection sampling and DPO without a verifiable-reward RL stage.
- [[deepseekmath]] — GRPO as an alternative optimizer for the same verifiable-reward setting.
- [[dpo]] — the preference objective this report length-normalizes.
- [[wildchat]] — 100,000 prompts of the SFT mix (Table 7).

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2411.15124 (arXiv v5, 14 Apr 2025)
- Corrections to the previous card version:
  - "Total episodes: 10,000,000" → 100,000 for the RLVR runs and 300,000 for the PPO-against-RM runs (Table 21).
  - "Beta (KL coeff): 0.05" stated without scope → 0.05 applies to the final 8B RLVR model only; the sweep was β ∈ [0.1, 0.05, 0.03, 0.01] and the 70B value is reported inconsistently as 0.07 (Table 21 caption) and 0.7 (§6.4).
  - "Local mini batch size: 32; local rollout batch size: 32" → not in Table 21; the reported figure is effective batch size 224 (8B) and 640 (70B).
  - "939,344 prompts (57% public, 43% synthetic/in-house)" → 939,344 is the count of prompts used in SFT; the report gives no 57/43 split (Table 7).
  - "Code tasks: unit-test execution" as an RLVR verifier → the verifiers are GSM8K, MATH and IFEval only; code-execution verifiers are named as future work (§6.1).
  - "Measured gains relative to DPO-only checkpoint: +5–10pp on GSM8K, +~4pp on IFEval" → 8B GSM8K +3.3 (84.3 → 87.6) and IFEval +1.3 (81.1 → 82.4); at 70B GSM8K is unchanged (Table 23).
  - "Train for 2 epochs; standard completion-masked loss" → 2 epochs is correct (Table 11); the report does not describe the loss masking in that table, so the masking claim is dropped.
  - "DPO preference data: hundreds of thousands of pairs ... curated from on-policy sampling of the SFT model + reward model ranking" → the final mixes are over 270k (8B) and over 330k (70B) data points (§5.3); preference labels come from LLM-judge pairwise comparisons over completions from Tülu 3-SFT and other models (§2.1, §5.2.1, Table 17), not from a reward model.
  - "Authors: ... (Allen AI)" → the author list spans the Allen Institute for AI and the University of Washington.
- Removed as unsupported by the source: "No reward hacking observed because the verifier is ground-truth" (contradicted by §6.2 "Overoptimization Happens" and App. B.4); "The report is the most detailed open disclosure of a modern post-training pipeline to date"; "Tulu 3 8B post-training fits on a single 8xH100 node" as a general statement (the report gives 32 GPUs for 8B SFT and 8 GPUs for the 8B RL run, §4.3, §6.3); "405B requires FSDP + sequence parallel" (the report describes ZeRO Stage 3 and adapted distributed-RLHF allocation, §6.3); the Figure/Table descriptions that did not match the report.
- Not reported by the source: per-dataset token counts for the SFT mix; SFT loss-masking details in the hyperparameter table; a reconciliation of the two 70B RLVR β values.

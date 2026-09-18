<!-- scope: SPIN — iterative self-play fine-tuning that trains the model to separate human SFT responses from its own previous-iteration samples
     deps: [[dpo]]
     see-also: [[self-rewarding-lm]], [[self-play-preference]], [[trl-online-dpo]], [[rejection-sampling-finetuning]], [[west-of-n]]
-->

# Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models
- **Core Insight:** Training a model to distinguish human SFT responses from its own previous-iteration samples, with no preference labels and no new data, raises zephyr-7b-sft-full from 58.14 to 63.16 average on the HuggingFace Open LLM Leaderboard and from 5.94 to 6.78 on MT-Bench (§1, §6.2, App. B Tables 4 and 6).
- **Guideline:** When an SFT dataset is already exhausted by supervised training and no preference data is available, run SPIN, because SPIN at iteration 0 matched DPO trained on ~62k UltraFeedback Binarized pairs from the same checkpoint and surpassed it from iteration 1 on the leaderboard average (§6.2). When gains per iteration have fallen toward zero, stop: the target distribution is fixed and caps the achievable performance (§7 Limitation).
- **Authors:** Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu (UCLA; Chen and Deng contributed equally)
- **Year:** 2024 (arXiv v1 2024-01; ICML 2024)
- **URL:** https://arxiv.org/abs/2401.01335
- **Source type:** paper
- **Relevant topics:** self-play, preference optimization without preference labels, iterative training, distribution matching, self-generated negatives

## Abstract
SPIN (Self-Play fIne-tuNing) starts from a supervised fine-tuned model and improves it without additional human-annotated data. In each iteration the model generates responses for prompts drawn from the existing SFT dataset; the next iterate is trained to assign a higher value to the human-written response than to the self-generated one, where the value function is the log-ratio of the new policy to the previous iterate. The authors prove that the global optimum of this objective is attained if and only if the policy distribution equals the target data distribution. Empirically, SPIN improves zephyr-7b-sft-full across the HuggingFace Open LLM Leaderboard, MT-Bench, and Big-Bench tasks, and outperforms a DPO model trained from the same checkpoint on additional GPT-4-labeled preference data.

## Key Contributions
- Casts SFT-only post-training as a two-player game between the current policy (main player) and the previous iterate (opponent), formulated as maximizing an integral probability metric rather than a Bradley-Terry likelihood (§4.1, §4.2).
- Derives a closed-form opponent update: the ideal value function is `f_{t+1}(x,y) = λ·log(p_{θ_{t+1}}(y|x) / p_{θ_t}(y|x))`, which collapses the two-step game into one end-to-end loss (Eq. 4.6, Eq. 4.7).
- Proves convergence conditions: Theorem 5.2 (global optimum iff `p_θ = p_data`) and Theorem 5.4 (the logistic-loss opponent update satisfies `p_{θ_{t+1}}(y|x) ∝ p_{θ_t}(y|x)·(p_data(y|x)/p_{θ_t}(y|x))^{1/λ}`).
- Shows SPIN matches DPO at iteration 0 and exceeds it from iteration 1 on the leaderboard average, without new data (§6.2).
- Shows iteration is not interchangeable with longer training: extending iteration 0 over more epochs does not reach iteration 1's performance (§6.3).

## Key Figures/Tables to Study
- **Algorithm 1**: the per-iteration generate-then-update loop.
- **Equation 4.7**: the end-to-end SPIN objective.
- **Figure 2**: leaderboard average by iteration — 58.14 (SFT), 60.80, 62.12, 62.97, 63.16.
- **Figure 3 / App. B Table 4**: per-task breakdown against zephyr-7b-sft-full and against DPO.
- **Theorem 5.4 and Remark 5.5**: the direction of the probability update relative to `p_data`.

## Technical Details
- **Objective (Eq. 4.7):** `L_SPIN(θ, θ_t) = E[ ℓ( λ·log(p_θ(y|x)/p_{θ_t}(y|x)) − λ·log(p_θ(y'|x)/p_{θ_t}(y'|x)) ) ]`, with `x ~ q(·)` the prompt distribution, `y ~ p_data(·|x)` the human SFT response, `y' ~ p_{θ_t}(·|x)` the response sampled from the previous iterate, `λ` the regularization weight on the KL deviation from the opponent, and `ℓ` any convex non-increasing loss (Assumption 5.1 admits correlation, hinge, exponential, and logistic losses). Only the logistic choice makes the objective resemble DPO (§4.2, item 3).
- **Opponent update:** the next opponent is the new policy parameters copied directly, `θ_{t+1}` (§4.1, end-to-end paragraph; Algorithm 1). There is no separately maintained reference model.
- **Base model:** zephyr-7b-sft-full, which is Mistral-7B fine-tuned on UltraChat200k for one epoch (§6.1, §6.3).
- **Data:** 50k prompts randomly sampled from UltraChat200k; for multi-round conversations only the first round is used as the prompt and gold completion (App. B.1). Synthetic data accumulates: 50k at iteration 0, 100k at iterations 1, 2, and 3 (§6.1).
- **Prompt template:** `### Instruction: {prompt}\n\n### Response: ` (App. B.1).
- **Results:** leaderboard average 58.14 → 60.80 (iter 0, +2.66) → 62.12 (iter 1, +1.32) → 62.97 → 63.16 (§6.2, Figure 2). Iteration-0 gains exceed 5% on TruthfulQA and 10% on GSM8k (§6.2). MT-Bench 5.94 (SFT) → 6.46 (iter 0) → 6.65 (iter 1) → 6.78 (iter 2), above vicuna-13b-v1.5 at 6.57; Table 6 does not report iteration 3 (App. B.4, Table 6). SPIN iteration 3 followed by DPO reaches 64.05 (App. B Table 4).
- **Training-size ablation:** iteration-0 training sizes of 14k, 26k, and 50k, with the larger sets containing the smaller (§6.3).
- **Epoch ablation:** most of the gain occurs in the first two epochs of an iteration; longer training does not degrade performance but does not reach the next iteration's level (§6.3).
- **Compute:** 8×A100 (80G). Per 64 examples, generation 6.69s and training 10s. Per iteration: 1.45h generation for all four iterations; training 4.32h at iteration 0 and 8.64h at iterations 1–3, the doubling following the doubled dataset (App. B.2, Table 2).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| zephyr-7b-sft-full + SPIN | 7B | preference (self-play) | Codebase | Alignment Handbook, DeepSpeed ZeRO-3, FlashAttention-2 | arXiv:2401.01335 App. B.1 | verified (2026-09-18) | no ablation reported |
| zephyr-7b-sft-full + SPIN | 7B | preference (self-play) | Optimizer | RMSProp, no weight decay | arXiv:2401.01335 App. B.1 | verified (2026-09-18) | stated as common practice in alignment fine-tuning; no ablation reported |
| zephyr-7b-sft-full + SPIN | 7B | preference (self-play) | Global batch (sequences) | 64 | arXiv:2401.01335 App. B.1 | verified (2026-09-18) | no ablation reported |
| zephyr-7b-sft-full + SPIN | 7B | preference (self-play) | Peak LR, iterations 0–1 | 5e-7, 10% warmup steps | arXiv:2401.01335 App. B.1 | verified (2026-09-18) | no ablation reported |
| zephyr-7b-sft-full + SPIN | 7B | preference (self-play) | Peak LR, iterations 2–3 | 1e-7 | arXiv:2401.01335 App. B.1 | verified (2026-09-18) | stated as decayed "approaching the end of self-play fine-tuning"; no ablation reported |
| zephyr-7b-sft-full + SPIN | 7B | preference (self-play) | β (loss temperature), iterations 0–2 | 0.1 | arXiv:2401.01335 App. B.1 | verified (2026-09-18) | taken from the Alignment Handbook setting |
| zephyr-7b-sft-full + SPIN | 7B | preference (self-play) | β, iteration 3 | 5.0 | arXiv:2401.01335 App. B.1 | verified (2026-09-18) | stated as raised because the model is close to convergence; consistent with Remark 5.5, where larger λ means a smaller opponent update |
| zephyr-7b-sft-full + SPIN | 7B | preference (self-play) | Max sequence length | 2048 tokens | arXiv:2401.01335 App. B.1 | verified (2026-09-18) | taken from the Alignment Handbook setting |
| zephyr-7b-sft-full + SPIN | 7B | preference (self-play) | Precision | bfloat16 | arXiv:2401.01335 App. B.1 | verified (2026-09-18) | no ablation reported |
| zephyr-7b-sft-full + SPIN | 7B | preference (self-play) | Epochs per iteration | 2 | arXiv:2401.01335 §6.1 | verified (2026-09-18) | §6.3 epoch ablation: most gain in the first two epochs |
| zephyr-7b-sft-full + SPIN | 7B | preference (self-play) | Generation batch | global batch size 64 via Accelerate distributed inference | arXiv:2401.01335 App. B.1 | verified (2026-09-18) | no ablation reported |
| zephyr-7b-sft-full + SPIN | 7B | preference (self-play) | Iterations run | 4 (iter 0–3) | arXiv:2401.01335 §6.2 | verified (2026-09-18) | gains approach zero by the last iteration |

Sampling temperature and top-p for synthetic generation are not reported.

## Findings relevant to negative feedback
- **Which sense of "negative" this is:** negative as gradient. The second term of Eq. 4.7 decreases `λ·log(p_θ(y'|x)/p_{θ_t}(y'|x))` for the self-generated response `y'`, so probability mass is actively removed from the model's own previous-iteration samples, not merely withheld.
- **Where negatives come from:** the policy itself at iteration t. There is no verifier, judge, or reward model, and no label of correctness, so what is treated as "rejected" is simply "self-generated", regardless of quality (§4.1, Algorithm 1).
- **Mechanism the paper gives:** Remark 5.5 states that the update increases `p_{θ_{t+1}}(y|x)` where `p_{θ_t}(y|x) < p_data(y|x)` and decreases it where `p_{θ_t}(y|x) > p_data(y|x)`, with the size of the move controlled by `1/λ`. The push-down therefore targets over-represented regions of the policy rather than incorrect content.
- **Control on the negative term:** λ (implemented as β) bounds the opponent update; the authors raise β from 0.1 to 5.0 at iteration 3 to shrink it near convergence (App. B.1).
- **What the paper does not report:** it does not isolate the contribution of the negative term, does not log chosen versus rejected log-probabilities, and does not measure pass@k, entropy, diversity, or likelihood displacement. Claims about how much of SPIN's gain comes from the push-down have no support in this source.
- **Stated ceiling:** because `p_data` is fixed and human-written, the objective converges when the policy matches it, which the authors name as an inherent performance ceiling (§7 Limitation and Future Work).

## Connections
- [[dpo]] — the logistic-loss instance of Eq. 4.7 resembles the DPO loss; §4.2 lists three differences, including that SPIN needs only `(x, y)` pairs and that SPIN is inherently iterative.
- [[self-rewarding-lm]] — concurrent work using the model as its own reward model with iterative DPO; §4.2 notes SPIN's self-assessment is implicit, with no intermediate reward or preference feedback.
- [[self-play-preference]] — also frames alignment as a two-player game; SPIN's equilibrium is distribution matching against fixed human data.
- [[trl-online-dpo]] — framework implementation of on-policy preference training with a similar generate-then-update loop.
- [[rejection-sampling-finetuning]] — course-level contrast, not drawn by this paper: rejection sampling trains on selected self-samples as positives only, while SPIN uses self-samples exclusively as the pushed-down term.
- [[west-of-n]] — course-level contrast, not drawn by this paper: both construct preference pairs without human labels.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2401.01335 (arXiv PDF, ICML 2024 camera-ready text)
- Corrections to the previous card version:
  - "monotone improvement across 3 SPIN iterations (MT-Bench 6.39 → 7.12)" → MT-Bench moves 5.94 → 6.46 → 6.65 → 6.78 over iterations 0 to 2 (§1; App. B.4, Table 6).
  - "Theorem 4.1: Nash equilibrium characterization" → the convergence results are Theorem 5.2 and Theorem 5.4; the word "Nash" does not appear in the paper, and the characterization is global optimality iff `p_θ = p_data`, not a Nash equilibrium.
  - "Table 2 (HF Open LLM Leaderboard): Zephyr-7B-SFT → +SPIN matches Zephyr-7B-DPO" → the leaderboard breakdown is Table 4 in Appendix B and Figures 2–3; Table 2 reports generation and training times.
  - "DPO training ... used 60K GPT-4 preferences" → approximately 62k pairs from UltraFeedback Binarized (§6.2).
  - "3 epochs" per iteration → 2 epochs per iteration (§6.1).
  - "Sample 50K (prompt, response) pairs from π_{t−1} at T=1.0" → 50k prompts are sampled from UltraChat200k; the synthetic set is 50k at iteration 0 and 100k at iterations 1–3 by accumulation, and no sampling temperature is reported.
  - "Reset reference to π_{t−1} for the next iteration" → the opponent for iteration t+2 is `θ_{t+1}` copied directly; the formulation has no separate frozen reference model.
  - "L_SPIN = −logσ(β·(...))" → the paper's Eq. 4.7 is stated for a general convex decreasing `ℓ` with weight `λ`; the logistic case is one instance (§4.2, item 3).
  - "Base: mistral-7B SFT'd on UltraChat-200K" → the exact checkpoint is zephyr-7b-sft-full, one epoch of UltraChat200k on Mistral-7B (§6.1, §6.3).
- Removed as unsupported by the source:
  - "Budget: ~8× the SFT compute — one full SFT round per iteration" — App. B.2 reports wall-clock times on 8×A100 and states the fine-tuning cost is computationally equal to SFT and DPO; no 8× multiplier appears.
  - "the 1:1 human:generated pair ratio (off-ratio hurts)" — no ratio ablation exists; the synthetic set doubles from iteration 1 onward.
  - "higher β collapses to SFT" — the paper raises β to 5.0 at the last iteration for stability and reports no collapse.
  - "iteration count (monotone gain for 3, minor gain after)" as a hyperparameter finding — the paper runs four iterates and reports gains approaching zero at the last one, without testing beyond iteration 3.
  - "SPIN's 'judge' is the human-written text itself — far lower variance than an LLM-judge" — variance is not measured or compared in this paper.
- Not reported by the source: generation temperature and top-p; results at model sizes other than 7B; pass@k, entropy, or diversity measurements; an ablation isolating the negative (self-generated) term.

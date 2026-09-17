---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: "primary source (no library card at the time of writing)"
source_url: https://arxiv.org/abs/2406.09279
created_at: "2026-09-15"
verified_against: "arXiv:2406.09279v2 (7 Oct 2024), cached plain text"
---

# Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback

Hamish Ivison, Yizhong Wang, Jiacheng Liu, Zeqiu Wu, Valentina Pyatkin, Nathan Lambert, et al. (Ai2, UW).

## Setting
Tülu 2 13B SFT as the starting policy; 14 preference datasets; 11 benchmarks grouped into factuality
(MMLU), reasoning (GSM8k, BBH), coding (HumanEval+, MBPP+), truthfulness (TruthfulQA), safety
(ToxiGen, XSTest) and instruction following (AlpacaEval 1 and 2, IFEval). DPO hyperparameters: LR 5e-7,
β = 0.01, 3 epochs, chosen in a pilot sweep over β ∈ {0.1, 0.01, 0.001} and LR ∈ {5e-6, 5e-7, 5e-8}
(App. F.1). PPO: KL coefficient β = 0.05 (0.0325 with smaller reward models), 1 epoch (App. F.2).

## Claims the chapter uses
- **Where DPO helps (§3.1, Table 1).** Relative to the SFT model (factuality 55.4, reasoning 47.8,
  coding 45.1, truthfulness 56.6, safety 91.8, instruction following 44.2), the best DPO runs improve
  instruction following and truthfulness by over 8 points (UltraFeedback fine-grained: 52.8 and 69.3),
  while all 14 datasets leave factuality within about 1 point of each other (54.7–55.7). The paper states
  that preference-based learning with existing datasets "does not aid factuality".
- **Reasoning.** Reasoning scores stay close to the SFT value across datasets (45.8–50.9 in Table 1).
- **DPO versus PPO (§3.2, Table 2).** Averaged over five datasets, PPO minus DPO is +1.3 reasoning,
  +2.9 coding, +2.3 safety, +0.1 instruction following, −0.1 factuality, −2.5 truthfulness, +0.7 overall.
- **Data quality dominates (Abstract, §3.1).** "Better preference data leading to the largest
  improvements, followed by the choice of learning algorithm, the use of improved reward models, and
  finally the use of additional unlabeled prompts"; synthetic per-aspect UltraFeedback is the best of the
  14 datasets at an average of 61.0.
- **Reward-model scaling (§3.3, Table 3).** Scaling the reward model to 70B gives up to 5 points on the
  mathematical evaluation but "marginal improvements in other categories".

## Conditions and limits
- One base model (Tülu 2 13B) and one SFT mixture; all datasets downsampled to 60,908 examples for the
  DPO-versus-PPO comparison (Table 2 caption).

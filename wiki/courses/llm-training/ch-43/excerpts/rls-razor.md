---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2509.04259
primary_version: arXiv:2509.04259v1 (4 Sep 2025)
created_at: "2026-09-15"
---

# Excerpt: RL's Razor — Why Online Reinforcement Learning Forgets Less

Shenfeld, Pari, Agrawal (Improbable AI Lab, MIT), 2025. This chapter uses the source for one fact: which KL
quantity predicts forgetting. The full treatment is in ch-38a.

## The forgetting law
- Claim (§1, §3): "When fine-tuning a model π on a new task τ, the degree of forgetting is accurately
  predicted by `E_{x~τ} KL(π_0 || π)`, the KL divergence between the fine-tuned and base policy evaluated on
  the new task." The KL is measured on the new task's own prompts, not on the prior tasks, which is what makes
  it usable during training.
- The direction is the forward KL `KL(π_0 ‖ π)` with the base model first. Table 1 (ParityMNIST) compares
  candidate predictors by the `R²` of a second-degree polynomial fit: forward KL 0.96 ± 0.01, reverse KL
  0.93 ± 0.01; the other candidates are lower.
- Fits: `R² = 0.96` in the ParityMNIST/FashionMNIST setting (§3, Fig. 3) and `R² = 0.71` for the LLM
  experiments (Fig. 11).
- RL's Razor (§1): among the many policies that solve the new task, on-policy RL converges to ones that are
  closer in KL to the starting policy, while SFT can converge to distant ones. An SFT run on an oracle
  KL-minimal labeling retained more than RL, and an SFT run distilled from an RL-trained model matched RL's
  trade-off (§3), so the authors attribute the difference to the distribution reached, not to the optimizer.

## Setting
- LLM tasks: Qwen2.5-3B-Instruct trained on Open-Reasoner-Zero math, the Chemistry L-3 subset of
  SciKnowEval, and ToolAlpaca; retention measured on HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande and
  HumanEval. Robotics: OpenVLA-7B in SimplerEnv.
- The RL runs used "only a binary success indicator as the reward, without explicit KL regularization"
  (§3.1); App. B.3 lists a GRPO variant with a KL coefficient of 0.1 among the compared configurations.

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2509.04259v1 (scratchpad `sources/rls-razor.txt`).
- Not reported: policy entropy, output diversity, or pass@k; model sizes above 14B for the LLM sweep.

---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: arXiv:2509.04259v1 (no library card for slug rls-razor on 2026-09-15; chapter-local verified extract)
source_url: https://arxiv.org/abs/2509.04259
created_at: "2026-09-15"
---

# Excerpt: RL's Razor: Why Online Reinforcement Learning Forgets Less

- **Authors:** Idan Shenfeld, Jyothish Pari, Pulkit Agrawal (Improbable AI Lab, MIT)
- **Year:** 2025 (arXiv v1 2025-09-04; preprint)
- **Source type:** paper
- **Used in:** ch-37 §5 (on-policy updates and forgetting), Negative samples and negative feedback, Recipe, Generalization lens. The full treatment is in ch-38a.

## Claim (Abstract, §1)
- "the degree of forgetting is determined by the distributional shift, measured as the KL-divergence between the fine-tuned and base policy evaluated on the new task."
- The empirical predictor is E_{x∼τ}[KL(π_0 ‖ π)], computed on inputs x from the new task τ (§1).
- "on-policy RL is implicitly biased towards KL-minimal solutions among the many that solve the new task, whereas SFT can converge to distributions arbitrarily far from the base model" (Abstract).

## Setup (§3.1, App. B.1, Table 2)
- LLM tasks: Qwen 2.5 3B-Instruct trained on Open-Reasoner-Zero math questions, on the Chemistry L-3 subset of SciKnowEval, and on ToolAlpaca. Robotics: OpenVLA 7B in SimplerEnv.
- RL: GRPO with a binary success reward and no explicit KL regularization (§3.1). Table 2 GRPO settings: KL reg. 0; group size 64; prompts per generation 8; num iterations μ ∈ {1, 2}; loss type Dr. GRPO; LR sweep {1e-5, 2e-5, 3e-5, 4e-5, 5e-5}; constant with warmup (50 steps); 1 epoch.
- SFT: LR sweep {1e-5, 3e-5, 5e-5, 7e-5, 9e-5}; epochs {1, 2}; batch {16, 32, 64, 128}. Math SFT targets were DeepSeek-R1 responses that matched the correct answer (up to 16 samples per prompt; 96% of prompts annotated); Science Q&A targets came from GPT-4o (App. B.1).
- Prior-task benchmarks: HellaSwag, TruthfulQA, MMLU, IFEval, WinoGrande, HumanEval (§3.1).
- Trade-off curves: dozens of hyperparameter settings per method; models within 2 accuracy points of the Pareto frontier kept; exponential fit (App. B.1).

## Results
- **RL forgets less at matched new-task accuracy (Fig. 2).** For RL, prior-task scores stay nearly unchanged as new-task accuracy rises; SFT gains come with forgetting, most strongly on Math (§3.1). Values are given as plots.
- **KL predicts forgetting (§4).** ParityMNIST: quadratic fit R² = 0.96. LLM experiments: "a quadratic fit achieving R2 = 0.71 (Figure 11)".
- **Oracle SFT (§4, Fig. 3, ParityMNIST only).** In the toy ParityMNIST setting the KL-minimal fully correct labeling can be identified analytically (App. B.3); SFT on it forgot less than RL. The construction was not applied to the LLM tasks.
- **On-policy data versus negative gradients (§5.1, Fig. 4, Science Q&A).** Four objectives: GRPO (on-policy, negatives), 1–0 Reinforce (on-policy, A = 1 for correct and 0 for incorrect, no negatives), SFT (offline, no negatives), SimPO (offline, negatives from an external model). "1–0 Reinforce behaves similarly to GRPO, while SimPO resembles SFT. Thus, the critical factor is not the presence of negative gradients but the use of on-policy data."
- **Theory (§5.2, App. A).** Rejection sampling from p restricted to reward 1 is the I-projection argmin_q KL(q ‖ p) subject to E_q[R] = 1 (Lemma A.1). Theorem 5.2: with a finite output set, a convex policy family, and binary reward, policy gradient converges (under regularity conditions) to argmin_{π ∈ P* ∩ Π} KL(π ‖ π_0).

## Limits stated by the authors
- "we still lack a mechanistic account of why larger KL shifts on the new task disrupt prior knowledge" (§7).
- The LLM experiments use one 3B model; seeds and confidence intervals for the Pareto curves are not reported.
- A concurrent study cited in §2 (Lai et al. 2025) attributes RL's lower forgetting to negative examples; this paper argues against that explanation (§2, §5.1). ch-30a records a further report in which the KL–forgetting relation does not always hold.

## Verification
- Read on 2026-09-15 against the arXiv:2509.04259v1 PDF text (Abstract, §1–§5, §7, App. A–B, Table 2).

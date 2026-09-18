<!-- chapter excerpt for ch-54. Primary-source extract, read 2026-09-17.
     If a library card of this slug exists under wiki/raw-data/llm-training/, prefer the card. -->

# RL's Razor: Why Online Reinforcement Learning Forgets Less
- **Artifact:** Idan Shenfeld, Jyothish Pari, Pulkit Agrawal (MIT). arXiv:2509.04259 (v1 2025-09). Source type: paper. Project page: jyopari.github.io/posts/rl_razor.
- **Core Insight:** The degree of catastrophic forgetting after fine-tuning is predicted by `E_{x∼τ}[KL(π₀ ‖ π)]`, the KL divergence between the fine-tuned and base policy measured on the **new** task distribution, not on the old tasks; on-policy RL forgets less than SFT at matched new-task performance because policy-gradient updates are biased toward low-KL solutions (§1, §3-§4).
- **Guideline:** When a post-training stage must not erode existing ability, log the KL to the base model on a fixed prompt set from the new task during training and use it as the early signal for forgetting, because it is measurable without any access to prior-task data, unlike a direct measurement of distribution shift on past tasks (§1).

## Technical details (with loci)
- **Forgetting law (§1, §4).** "When fine-tuning a model π on a new task τ, the degree of forgetting is accurately predicted by `E_{x∼τ}[KL(π₀‖π)]`." A quadratic fit gives R² = 0.96 in the ParityMNIST setting (§4, Fig. 3 middle) and R² = 0.71 in the LLM experiments (Fig. 11); the authors attribute the weaker LLM fit to noise in approximate KL and accuracy estimation.
- **Setting (§3).** Qwen 2.5 3B-Instruct trained on three tasks — math reasoning (Open-Reasoner-Zero), science Q&A (SciKnowEval Chemistry L-3), and tool use (ToolAlpaca) — plus OpenVLA 7B on a SimplerEnv pick-and-place task. Prior capability is measured on HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, and HumanEval.
- **Main comparison (§1, §3, Fig. 2).** RL fine-tuning forgets less than SFT even when both reach the same new-task performance; the comparison is made as a Pareto frontier over hyperparameters, where each point is one trained model.
- **Oracle SFT test (§4).** In ParityMNIST the labeling that minimizes KL to the base model subject to 100% accuracy can be derived analytically (§B.3). SFT on that oracle distribution retained more prior knowledge than RL. The authors conclude that RL's advantage is not intrinsic to the algorithm but follows from its implicit KL minimization.
- **Mechanism claim (§1, §5).** On-policy sampling reweights the model's own distribution rather than pulling it toward an arbitrary label distribution, so among the many high-reward solutions to a new task, policy gradient converges to KL-minimal ones; a theoretical argument is given in a simplified setting.

## Not reported
The paper does not study staleness or asynchronous rollouts, so the law is not verified for off-policy rollout data at a given staleness level.

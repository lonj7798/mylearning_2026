<!-- scope: Shenfeld, Pari, Agrawal (2025): forgetting after fine-tuning is predicted by the forward KL divergence to the base policy measured on the new task; on-policy RL is biased toward KL-minimal solutions, SFT is not
     deps: [[grpo]], [[ppo]]
     see-also: [[rlvr-beyond-base-model]], [[likelihood-displacement]]
-->

# RL's Razor: Why Online Reinforcement Learning Forgets Less
- **Core Insight:** Across three LLM fine-tuning tasks on Qwen2.5-3B-Instruct and a robotic task on OpenVLA-7B, models on the RL Pareto frontier keep prior-benchmark performance nearly unchanged as new-task accuracy rises, while SFT models reach the same new-task accuracy only with a large drop on prior benchmarks; forgetting collapses onto one curve when plotted against the forward KL divergence E_{x∼τ}[KL(π₀‖π)] on the new task, with a quadratic fit of R² = 0.96 in the controlled MNIST setting and R² = 0.71 in the LLM experiments (§3, §4, Figure 2, Figure 3, Figure 11).
- **Guideline:** When comparing two fine-tuning runs for capability retention, measure the forward KL to the base policy on the new-task distribution and plot retention against it, because in this paper KL predicted forgetting independently of the algorithm and outperformed reverse KL (R² 0.93), total variation (0.80), and every weight- or activation-space distance tested (≤ 0.58) (§6, Table 1).
- **Authors:** Idan Shenfeld, Jyothish Pari, Pulkit Agrawal (Improbable AI Lab, MIT)
- **Year:** 2025 (arXiv v1 2025-09-04)
- **URL:** https://arxiv.org/abs/2509.04259
- **Source type:** paper
- **Relevant topics:** catastrophic forgetting, KL divergence, on-policy RL versus SFT, Pareto frontier, capability retention

## Abstract
RL and SFT reach similar new-task performance, but RL preserves prior capabilities better. The paper finds that the degree of forgetting is determined by the distributional shift, measured as the KL divergence between the fine-tuned and base policy evaluated on the new task. On-policy RL is implicitly biased toward KL-minimal solutions among those that solve the new task, while SFT can converge to distributions arbitrarily far from the base model. The authors validate this with language models and robotic foundation models and give a theoretical argument for why on-policy updates produce smaller KL change. They name the principle RL's Razor.

## Key Contributions
- An empirical forgetting law: forward KL to the base policy, measured on the new task, predicts forgetting across objectives and hyperparameters (§1, §4).
- Pareto-frontier comparison of RL and SFT on three LLM tasks and one robotics task (§3, Figure 2).
- A controlled ParityMNIST setting where the KL-minimal labeling can be computed analytically; SFT on that oracle distribution beats RL, which shows KL and not the algorithm is the operative variable (§4).
- A theoretical account of why policy-gradient updates select low-KL solutions (§5).
- A comparison of candidate predictors of forgetting (§6, Table 1).

## Key Figures/Tables to Study
- **Figure 2:** Pareto frontiers of new-task accuracy against prior-task score for RL and SFT on the four tasks.
- **Figure 3:** learning–forgetting trade-off, forgetting against KL, and accuracy against KL in ParityMNIST.
- **Figure 11:** the same KL–forgetting relation in the LLM experiments (quadratic fit R² = 0.71).
- **Table 1:** R² of candidate predictors of forgetting.

## Technical Details
- **Tasks (§3).** LLM math reasoning: Qwen2.5-3B-Instruct on Open-Reasoner-Zero math questions. LLM science Q&A: Qwen2.5-3B-Instruct on the Chemistry L-3 subset of SciKnowEval. LLM tool use: Qwen2.5-3B-Instruct on ToolAlpaca. Robotics: OpenVLA-7B in SimplerEnv on a pick-a-can task.
- **Prior-capability suite (§3).** HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval for the LLMs; for the robot policy, SimplerEnv open/close drawer tasks excluding the fine-tuned one.
- **Protocol (§3, App. B).** Each point is a model trained with a different hyperparameter setting; the Pareto frontier in the (new-task, prior-task) plane is used for the comparison, and only models within 2 accuracy points of the frontier are retained (App. B, step 4).
- **Result (§3).** For RL, prior-benchmark performance stays nearly unchanged as new-task accuracy rises. For SFT, gains on the new task come with substantial forgetting, most sharply in the math task.
- **Predictor (§4).** Forgetting plotted against E_{x∼τ}[KL(π₀‖π)], the forward KL between base policy π₀ and fine-tuned policy π evaluated on new-task inputs, falls on a single curve for both RL and SFT. Quadratic fit R² = 0.96 (ParityMNIST) and R² = 0.71 (LLM experiments, Figure 11). Two different arbitrary SFT labelings gave different Pareto frontiers but the same forgetting–KL curve.
- **Oracle SFT (§4).** In ParityMNIST the labeling q* = argmin D_KL(π₀‖q) subject to 100% accuracy can be computed; SFT on it retained more prior knowledge than RL. The authors conclude that RL performs well because on-policy updates bias the solution toward low-KL regions, not because the RL objective is special.
- **Predictor comparison (§6, Table 1, R² of a second-degree polynomial fit, MNIST task):** forward KL 0.96 ± 0.01; reverse KL 0.93 ± 0.01; total variation 0.80 ± 0.01; distribution change L2 0.56 ± 0.02; weight change L1 0.34 ± 0.02; Fisher-weighted L2 0.58 ± 0.02; spectral norm 0.58 ± 0.02; activation change L1 0.52 ± 0.02, L2 0.55 ± 0.02.

## Findings relevant to generality
- **What causes narrowing.** Distance from the base policy on the new-task distribution, not the fine-tuning objective, is what the paper identifies as the operative variable for loss of prior capability (§4, §7).
- **Measurement.** The paper's measurement procedure — many hyperparameter cells, a new-task axis, a prior-capability axis, and forward KL as the third coordinate — is directly usable as a retention protocol.
- **Limits (§7).** No mechanistic account of why larger KL on the new task disrupts prior knowledge. The LLM evidence is at 3B (with 7B and 14B SFT-only checks in App.); the LLM fit is weaker (R² = 0.71) than the controlled setting.

## Connections
- [[rlvr-beyond-base-model]] — a different narrowing measurement (coverage at large k) for the same RL stage.
- [[grpo]], [[ppo]] — the on-policy algorithms whose implicit KL bias the paper analyses.
- [[likelihood-displacement]] — an off-policy preference-loss failure that is not covered by this KL argument.

## Verification
- Checked on 2026-09-15 against the cached primary text of https://arxiv.org/abs/2509.04259 (arXiv v1, 2025-09-04): Abstract, §1, §3, §4, §6 Table 1, §7, App. B.
- Corrections to the previous card version: no previous card existed in the library for this slug.
- Not reported by the source: absolute prior-benchmark scores per cell in a table (Figure 2 is a plot), LLM-scale KL values, effect at model sizes above 14B for RL.

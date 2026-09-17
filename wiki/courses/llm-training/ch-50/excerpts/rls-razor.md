---
chapter: ch-50
course: llm-training
phase: read
excerpt_of: primary source (no library card exists for this slug as of 2026-09-15)
source_url: https://arxiv.org/abs/2509.04259
created_at: "2026-09-15"
---

# Excerpt: RL's Razor — a measurable predictor for the forgetting slice

**Artifact:** RL's Razor: Why Online Reinforcement Learning Forgets Less. Idan Shenfeld, Jyothish Pari,
Pulkit Agrawal (Improbable AI Lab, MIT). arXiv:2509.04259, v1 2025-09-04.
**Checked on 2026-09-15** against the arXiv v1 PDF.
**Note:** `wiki/raw-data/llm-training/` has no card for this slug. When a card is created, this excerpt
should be replaced by a link to it.

---

## Why ch-50 uses this source

ch-50 §5 needs a quantity that can be logged during training and read as an early indicator for the
forgetting slices. This paper proposes one and measures how well it predicts.

---

## Setup (§3, App. B)

- Base model for the LLM experiments: **Qwen 2.5 3B-Instruct**.
- New tasks: math questions from the Open-Reasoner-Zero dataset; the Chemistry L-3 subset of SciKnowEval;
  the ToolAlpaca dataset. A robotics task (OpenVLA 7B on SimplerEnv pick-and-place) is run separately.
- Prior-capability benchmarks used to measure forgetting: **HellaSwag, TruthfulQA, MMLU, IFEval,
  Winogrande, HumanEval** (§3).
- Each point in the reported trade-off curves is a separately trained model with a different hyperparameter
  setting; the curve is the Pareto frontier over those points (§3).

## The claim, stated as an equation (§4)

> When fine-tuning π on a new task τ , the degree of forgetting is accurately predicted by
> `E_{x∼τ} KL(π₀ ‖ π)`, the KL divergence between the fine-tuned and base policy evaluated on the new task.

Symbols: `π` is the fine-tuned policy, `π₀` the base policy, `τ` the new task's input distribution.
The divergence is measured **on the new task's inputs only**, which is what makes it cheap: the prior-task
benchmarks do not need to be run to compute it.

## Measured fit

| Setting | Fit | Locus |
|---|---|---|
| ParityMNIST controlled setting (3-layer MLP pretrained jointly on ParityMNIST and FashionMNIST, fine-tuned on ParityMNIST, forgetting measured on FashionMNIST) | quadratic fit R² = 0.96 | §4, Fig. 3 (middle) |
| LLM experiments | quadratic fit R² = 0.71; residuals mean-zero, attributed to approximate KL and accuracy estimation | §4, Fig. 11 |

Both RL and SFT points fall on the same forgetting-against-KL curve, which is the authors' argument that
the algorithm is not the operative variable (§4). Repeating the experiment with two different arbitrary SFT
labellings gave different Pareto frontiers and coinciding forgetting-KL curves (§4).

## The control that tests the claim (§4, "Optimal SFT Distribution")

In ParityMNIST the KL-minimal labelling among those achieving 100% accuracy can be identified analytically.
SFT trained on that oracle distribution **retained more prior knowledge than RL**, giving the best
accuracy-forgetting trade-off observed. The authors read this as evidence that RL's advantage comes from
its implicit bias toward low-KL solutions rather than from a property of the RL objective itself.

## Limits

- The LLM experiments are one base model at 3B, three new tasks, one prior-benchmark set; Appendix C
  repeats the SFT experiments at 3B, 7B, and 14B.
- R² = 0.71 for the LLM case is a correlation across runs, not a decision rule for a single run.
- The paper does not evaluate held-out or perturbation slices; its forgetting measure is the prior-benchmark
  average.

## Used by

ch-50 §5 (what to log alongside the target score), Generalization lens (a) and (c).

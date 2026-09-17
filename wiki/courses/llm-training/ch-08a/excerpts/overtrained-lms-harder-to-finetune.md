---
chapter: ch-08a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2503.19206v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2503.19206
created_at: "2026-09-15"
---

# Excerpt: Overtrained Language Models Are Harder to Fine-Tune

**Paper:** Jacob Mitchell Springer, Sachin Goyal, Kaiyue Wen, Tanishq Kumar, Xiang Yue, Sadhika Malladi, Graham Neubig, Aditi Raghunathan (Carnegie Mellon University; Stanford; Harvard; Princeton). arXiv v1 2025-03; v2 2025-03-28 read. Source type: paper.

## Claim (Abstract, §1)
- "Catastrophic overtraining": extending pre-training makes a model harder to fine-tune, so the post-trained model can be worse. The instruction-tuned OLMo-1B pre-trained on 3T tokens is over 2% worse on multiple standard benchmarks than its 2.3T-token counterpart (Abstract).
- Proposed mechanism: "progressive sensitivity", a systematic increase during pre-training in how much a parameter modification of fixed size degrades the model (Abstract, §3).

## Real-model experiments (§2, §3.1, App. B.1)
- Base models with public intermediate checkpoints: OLMo-1B (0.04T to 3.1T tokens), OLMo-2-7B stage-1 (0.08T to 3.9T), LLM360-Amber 7B (0.12T to 1.3T) (App. B.1 Table 1).
- Post-training: instruction tuning on Anthropic-HH (chosen responses as targets, 180k examples) and TULU; multimodal tuning with LLaVA; learning rate tuned per checkpoint for best in-distribution (ID) performance; out-of-distribution (OOD) performance on ten common benchmarks (§2, App. B.2).
- OLMo-1B after Anthropic-HH: the 3T-token base gives up to 3% lower AlpacaEval response rate and 2% lower ARC than the 2.3T-token base; after instruction tuning the 3T model drops to the level of the 1.5T model (§1, §2, Fig. 2).
- Base models improve monotonically with more tokens on all evaluated tasks (§2, Fig. 2 dashed line).
- OLMo-1B shows the effect beyond 2.5T tokens for instruction tuning and for multimodal tuning; for multimodal tuning the ID VLM score does not degrade, while some OOD tasks do (§3.1).
- "Catastrophic overtraining is not observed on OLMo-7B models for pre-training token budgets up to 3T tokens" under the same setups (§3.1, App. E).
- Confound stated by the authors: the public checkpoints come from single runs, so each token budget has a different final learning rate from the annealing schedule; §3.2 removes this confound (§3.2).

## Controlled experiments (§3.2–3.4)
- Models of 15M to 90M parameters pre-trained on C4 with budgets from 4B to 128B tokens, each cosine-annealed to zero; main text uses 30M (§3.2).
- Gaussian perturbation θ̃ = θ + ε, ε ∼ N(0, γ²Σ), with Σ the initialization covariance: for fixed γ, the perplexity increase from the perturbation grows monotonically with pre-training tokens; the perturbed model's C4 perplexity is U-shaped in tokens (§3.3, Fig. 3).
- Fine-tuning with fixed learning rate (GSM8k, Starcoder-Python, SIQA, MR, RTE, TREC; LRs from 4e-6 to a dataset-specific maximum): C4-perplexity degradation grows monotonically with pre-training tokens; larger learning rates reach the inflection point at fewer tokens (§3.4.1, Fig. 4–5).
- With the learning rate tuned for ID performance: ID perplexity degrades with extensive over-training on RTE and TREC; C4 perplexity degrades on GSM8k, Starcoder-Python, MR, and RTE (§3.4.2, Fig. 6). A smaller-than-optimal learning rate can delay the inflection point at a cost in ID performance (§3.4.2).

## Theory (§4)
- Two-layer linear network pre-trained by gradient flow learns singular values incrementally (Saxe et al.; Gidel et al.). Without regularization and with sufficiently misaligned pre-training and fine-tuning tasks, pre-training loss after fine-tuning increases monotonically beyond an inflection point; regularization toward the pre-trained weights delays the inflection point and raises fine-tuning loss (Theorem 4.7).

## Stated limits and open questions (§6)
- Which pre-training settings control severity (optimizer, data distribution, objective) is left to future work; replay, LP-FT, and WiSE-FT are suggested but not tested (§6).
- RL and pruning as post-training modifications are left to future work (§3).

## Verification
- Read on 2026-09-15 against arXiv:2503.19206v2 PDF text (Abstract, §1–§4.3, §6, App. B.1–B.2).
- Source-internal inconsistency: the Figure 1 caption says "five common LLM benchmarks" and lists four.
- Not reported by the source: a fitted relation between tokens per parameter and fine-tuning degradation; results above 7B parameters.

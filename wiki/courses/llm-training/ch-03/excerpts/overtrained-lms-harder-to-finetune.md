---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: primary source arXiv:2503.19206v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2503.19206
created_at: "2026-09-15"
---

# Excerpt: Overtrained Language Models Are Harder to Fine-Tune

**Paper:** Jacob Mitchell Springer, Sachin Goyal, Kaiyue Wen, Tanishq Kumar, Xiang Yue, Sadhika Malladi, Graham
Neubig, Aditi Raghunathan (Carnegie Mellon University; Stanford; Harvard; Princeton). arXiv v1 2025-03; v2
2025-03-28 read. Source type: paper. Consistent with the ch-08a excerpt of the same paper; ch-03 uses the
learning-rate and sensitivity results.

## Claim (Abstract)
"Extended pre-training can make models harder to fine-tune, leading to degraded final performance. We term this
phenomenon catastrophic overtraining. For example, the instruction-tuned OLMo-1B model pre-trained on 3T tokens
leads to over 2% worse performance on multiple standard LLM benchmarks than its 2.3T token counterpart."

## Public checkpoints (§2, §3.1)
- Base models: OLMo-1B, OLMo-2-7B, LLM360-Amber-7B intermediate checkpoints; instruction tuning on Anthropic-HH and
  TULU; multimodal tuning with LLaVA; LR tuned per checkpoint for best in-distribution (ID) score; OOD score on ten
  benchmarks (§2.1).
- OLMo-1B after Anthropic-HH: the 3T base shows "up to 3% lower response rate (AlpacaEval score)" than the 2.3T base,
  and after instruction tuning the 3T models drop "to the level of models pre-trained with just 1.5T tokens" (§2.2,
  Fig. 2). Base models improve monotonically with tokens (§2.2).
- Onset for OLMo-1B: token budgets "exceeding 2.5T tokens" (§3.1). "Catastrophic overtraining is not observed on
  OLMo-7B models for pre-training token budgets up to 3T tokens" (§3.1, App. E).
- Confound stated by the authors: checkpoints from one run have different final LRs "due to the annealing
  schedule"; §3.2 removes this confound.

## Controlled runs (§3.2–§3.4)
- 15M–90M models, 4B–128B tokens of C4, each cosine-annealed to zero; main text 30M (§3.2).
- Gaussian perturbation θ̃ = θ + ε, ε ~ N(0, γ²Σ): for fixed γ, the perplexity increase grows monotonically with
  pre-training tokens ("progressive sensitivity"); perturbed C4 perplexity is U-shaped in tokens (§3.3, Fig. 3).
- Fine-tuning with fixed LR (4e-6 up to a dataset maximum; GSM8k, Starcoder-Python, SIQA, MR, RTE, TREC): "For a
  fixed learning rate, the change in perplexity increases monotonically with the number of pre-training tokens";
  larger LRs reach the inflection point at lower token budgets (§3.4.1, Fig. 4–5).
- With LR tuned for ID: degradation persists for some tasks (Fig. 6); a smaller-than-optimal LR "would delay the
  inflection point" but "would also result in a lower ID performance" (§3.4.2).

## Verification
- Read on 2026-09-15 against arXiv:2503.19206v2 PDF text (Abstract, §1–§3.4).
- Not reported: a tokens-per-parameter threshold; models above 7B; RL or pruning as the post-training modification.

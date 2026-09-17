---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: arXiv:2310.06452v3 (no library card for slug rlhf-generalisation-diversity on 2026-09-15; chapter-local verified extract)
source_url: https://arxiv.org/abs/2310.06452
created_at: "2026-09-15"
---

# Excerpt: Understanding the Effects of RLHF on LLM Generalisation and Diversity

- **Authors:** Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, et al. (UCL, Meta, Oxford)
- **Year:** 2024 (arXiv v1 2023-10; v3 2024-02-19; ICLR 2024)
- **Source type:** paper
- **Used in:** ch-37 §6 (reverse-KL objective and diversity), Generalization lens, Recipe.

## Setup (§4–§5)
- Base model LLaMA 7B; OPT at five sizes (125M–6.7B) in App. J.
- RLHF: PPO with reward R(x, y) = RM(x, y) − β_KL · D_KL(π_RL(y|x) ‖ π_SFT(y|x)), β_KL = 0.05 throughout (§4, Eq. 1).
- Best-of-N (BoN) from the SFT model: N = 16, temperature 0.7 (§4).
- Summarisation: TL;DR in distribution, CNN/DailyMail out of distribution. Instruction following: AlpacaFarm splits in distribution, AlpacaEval and Sequential Instructions out of distribution (§4).
- Diversity (§5.2): expectation-adjusted distinct n-grams (EAD), Sentence-BERT diversity, NLI diversity. K = 16 outputs per input for N = 500 inputs at temperature 1. Per-input diversity uses several outputs for one input; across-input diversity uses one output per input.

## Results
- RLHF generalises better than SFT out of distribution, especially on the larger distribution shift (Sequential Instructions) (§6.1).
- "RLHF significantly reduces output diversity compared to SFT across a variety of measures" (Abstract). The per-input reduction is larger than the across-input reduction; NLI diversity shows no meaningful difference on LLaMA summarisation (§6.2).
- OPT summarisation, per-input diversity (App. J.4, Table 11), RLHF vs SFT: EAD 0.15 vs 0.81 (125M), 0.07 vs 0.79 (6.7B); Sentence-BERT 0.06 vs 0.45 (6.7B).
- OPT summarisation, across-input EAD (Table 12) at 6.7B: RLHF 0.87, SFT 0.87, BoN 0.86.
- KL penalty sweep (§6.3, App. I, LLaMA summarisation): "increasing the KL penalty coefficient leads to a drop in performance as expected, but also to a drop in per-input diversity, rather than a gain."
- Instruction-following diversity was not reported because the metrics did not separate models on long outputs (§6.2).

## Limits stated by the authors
- "It is unclear whether this tradeoff is a fundamental one in fine-tuning LLMs with RLHF or just demonstrates a deficiency in current methods" (§7).

## Verification
- Read on 2026-09-15 against the arXiv:2310.06452v3 PDF text (Abstract, §4–§7, App. I, App. J.4 Tables 11–12).

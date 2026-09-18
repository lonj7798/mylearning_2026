---
chapter: ch-07
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2308.08747
created_at: "2026-09-17"
---

# Excerpt: "An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning"

**Artifact.** Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, Yue Zhang (Westlake University; WeChat
AI, Tencent). arXiv:2308.08747 (v1 2023-08; v5 2025-01-05). Read on 2026-09-17. No library card existed
when ch-07 was revised.

## Setting

- Models continually instruction-tuned in a fixed task order: Simp → Emdg → InqQG → Exp → HGen, 100,000
  samples, with a shared instruction template (§3.1).
- Models: BLOOMZ at 1.1b, 1.7b, 3b, 7.1b; mT0 at 1.2b and 3.7b (encoder-decoder, same instruction data);
  LLAMA-7b and ALPACA-7b (§4.1).
- General-ability evaluation sets (§3.2): MMLU for domain knowledge (5-shot); HellaSwag, BoolQ, Winogrande,
  PIQA, MathQA, MuTual for reasoning; RACE for reading comprehension; CrowS-Pairs for bias. Measured with
  lm-evaluation-harness (§3.3).

## Forgetting metric (Eq. 1, §3.3)

`FG_i = (1/|E_i|) Σ_{e∈E_i} (1/N) Σ_{m=1..N} (R_e^0 − R_e^m) / R_e^0 × 100%`

`E_i` is one evaluation set; `e` one dataset or split in it; `R_e^m` the score after `m` continually trained
tasks; `R_e^0` the score of the initial model. FG is an average **relative** decrease, so it is comparable
across sets with different score scales.

## Results quoted in ch-07

- Table 4 (FG, %): BLOOMZ-1.1b 9.54 (domain knowledge) / 6.73 (reasoning) / 18.04 (reading comprehension);
  BLOOMZ-1.7b 10.72 / 6.48 / 24.29; BLOOMZ-3b 14.63 / 11.09 / 27.56; BLOOMZ-7.1b 18.37 / 13.62 / 26.75.
  Forgetting increases with scale over the 1.1b–7.1b range tested.
- Per-dataset illustration (§5, Fig. 3): MMLU-Other drops 36.18% → 26.35% for BLOOMZ-7.1b, and
  30.58% → 25.97% for BLOOMZ-1.1b. The authors note the larger model starts higher.
- Table 6 (FG, %, domain knowledge / reasoning / reading comprehension): LLAMA-7b 34.57 / 31.33 / 31.72
  versus ALPACA-7b 18.14 / 7.56 / 10.31. The pair differs only in that ALPACA was general-instruction-tuned
  from LLAMA before the continual tuning, which the authors read as evidence that general instruction
  tuning reduces later forgetting.
- Table 3 shows the target tasks themselves improve over the same runs (for example BLOOMZ-7.1b on Exp,
  51.47 → 68.71 BLEU), so the forgetting is not a failed-training artifact.

## Limits

One task order, one instruction-data size, models up to 7.1b, and no pretraining-data replay condition.

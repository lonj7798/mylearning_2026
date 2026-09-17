---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Shi, Yang, Wu, Aitchison, Yilmaz, Lipani — "Instruction Tuning With Loss Over Instructions" (NeurIPS 2024)
source_url: https://arxiv.org/abs/2405.14394
created_at: "2026-04-23"
revised_at: "2026-09-15 (generality revision; rewritten from the primary PDF, arXiv v2 2024-10-02)"
---

# Excerpt: Shi et al. 2024 — Instruction Tuning With Loss Over Instructions

**Artifact:** arXiv:2405.14394 (v1 2024-05; read as v2). Authors: Zhengyan Shi, Adam X. Yang, Bin Wu, Laurence Aitchison, Emine Yilmaz, Aldo Lipani (UCL, University of Bristol). Code: github.com/ZhengxiangShi/InstructionModelling.

**Correction to the library card.** `papers/loss-masking-prompt.md` (2026-07 version) states that response-only loss is "strictly better" and dominates across dataset sizes, and that the paper prescribes masking prior assistant turns. The paper reports the opposite direction for many settings and prescribes no multi-turn rule. This excerpt follows the paper.

## Definitions (§3)

- Instruction tuning (IT): `L = − Σ_{j=1..n} log P(C_j | I_1..I_m, C_1..C_{j−1})` (Eq. 2), where `I` is the instruction (which may include template tokens such as `<|user|>`) and `C` the completion.
- Instruction modelling (IM): `L = − Σ_{t=1..m+n} log P(x_t | x_1..x_{t−1}) · 1(x_t ∉ T)` (Eq. 4), where `T` is the set of prompt-template tokens. Loss covers instruction and completion tokens but not template tokens.

## Main findings

- Abstract: across 21 benchmarks, IM "in many scenarios" improves NLP tasks (e.g. MMLU, TruthfulQA, HumanEval) and open-ended benchmarks (MT-Bench, AlpacaEval); in the most favourable case AlpacaEval 1.0 improves by over 100%. Two factors: the ratio of instruction length to output length, and the number of training examples. "We are not proposing IM as a replacement for current fine-tuning processes."
- Table 1 (LLaMA-2-7B; IT → IM):

  | Dataset (examples) | NLP mean (18 tasks) | AlpacaEval 1.0 | AlpacaEval 2.0 |
  |---|---|---|---|
  | Alpagasus Alpaca 5k (5,305) | 45.29 → 47.47 | 16.29 → 19.52 | 2.46 → 3.29 |
  | Alpagasus Dolly 3k (2,996) | 46.58 → 48.95 | 13.42 → 15.11 | 2.00 → 2.44 |
  | Alpagasus Dolly 9k (9,229) | 45.54 → 48.00 | 21.54 → 30.77 | 2.28 → 2.67 |
  | Less Tydiqa (13,533) | 48.21 → 48.70 | 5.12 → 10.10 | 1.88 → 2.88 |
  | Less MMLU Chat (13,533) | 47.18 → 47.84 | 4.42 → 9.78 | 1.20 → 1.93 |
  | Less BBH ICL (13,533) | 48.28 → 49.15 | 36.20 → 44.15 | 2.36 → 3.56 |
  | LIMA (1,030) | 48.79 → 49.60 | 33.06 → 32.94 | 2.58 → 2.47 |

  NEFTune rows in the same table: e.g. Less Tydiqa NLP mean 48.21 → 47.47 and AlpacaEval 1.0 5.12 → 8.35.
- Fig. 2 and §4.2: gains are largest for datasets with long instructions and short outputs (e.g. Less MMLU Chat, Code Alpaca); Tulu V2 (326,181 examples, instruction/output length ratio about 0.5) benefits less than Science Literature (ratio 24.7). On Tulu V2 subsets with ratio fixed near 10, the AlpacaEval 1.0 gain grows as the number of examples shrinks (1K to 35K subsets; App. C).

## Overfitting analysis (§4.3)

- Loss computed on outputs only: LIMA mean training loss 1.37 (IT) vs 1.45 (IM); test loss on a 10% Tulu V2 sample 1.32 (IT) vs 1.17 (IM) (Fig. 3).
- BLEU of greedy outputs on training prompts against training responses is lower for IM on all 7 datasets, e.g. Less BBH ICL 60.96 → 53.94 (Table 2).
- Mean NLP performance over epochs degrades less with IM (Fig. 4).
- KL regularization to the base model: on LIMA, NLP mean 48.79 → 49.26 but AlpacaEval 2.0 2.58 → 0.06; on Alpagasus Dolly 9k, NLP mean 45.54 → 49.31 and AlpacaEval 2.0 2.28 → 0.04 (Table 3).

## Further analysis (§4.4)

- IM's advantage holds for OPT-6.7B and LLaMA-2-13B on the same 7 datasets (Fig. 5).
- Output lengths of IM and IT are similar across Tulu V2 data fractions (Fig. 6).
- IM + NEFTune raises AlpacaEval 1.0 on 6 of 7 datasets (e.g. Less Tydiqa 10.10 → 23.41) but lowers the NLP mean on 3 of 7: LIMA 49.60 → 49.47, Less MMLU Chat 47.84 → 47.73, Less BBH ICL 49.15 → 48.62 (Table 4).

## Training settings (App. C, Table 6)

LR 2e-5, AdamW (β 0.9, 0.98; ε 1e-6), linear schedule with warmup ratio 0.03, weight decay 0, total batch 128, epochs 2, 3, or 10 (typically 2), maximum length 2,048, bf16, DeepSpeed stage 3, FlashAttention, implemented with Open-Instruct.

## Limits stated by the authors

Effectiveness depends on the quality and diversity of the instructions; harmful or biased instructions become training targets (§5).

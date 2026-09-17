---
chapter: ch-36
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/loss-masking-prompt.md
source_url: https://arxiv.org/abs/2405.14394
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from arXiv:2405.14394v2)"
---

# Excerpt: Instruction Tuning With Loss Over Instructions (Shi et al.)

**Paper:** Zhengyan Shi, Adam X. Yang, Bin Wu, Laurence Aitchison, Emine Yilmaz, Aldo Lipani (UCL; University of Bristol). arXiv v1 2024-05; read at v2 (2024-10-02), NeurIPS 2024. Source type: paper.

**Why this excerpt was rewritten.** The earlier version stated that response-only loss "strictly dominates" full-sequence loss and that the paper's Table 2 shows an MT-Bench gain for response-only loss. The paper reports the opposite direction in most of its settings: adding loss on instruction tokens (IM) improved the 18-task NLP mean over response-only loss (IT) on all seven datasets in Table 1, and Table 2 is a BLEU table. The library card [[loss-masking-prompt]] repeats the "strictly better" wording; the statements below are taken from the paper.

## Objectives (§3, Eqs. 2 and 4)

```
IT:  L = − Σ_{j=1..n} log P(C_j | I_1..I_m, C_1..C_{j−1})
IM:  L = − Σ_{t=1..m+n} log P(x_t | x_1..x_{t−1}) · 1(x_t ∉ T)
```

I is the instruction (m tokens), C the completion (n tokens), x their concatenation, and T the set of prompt-template tokens such as `<|user|>` and `<|assistant|>`. IM trains on instruction and completion tokens and excludes template tokens.

## Setting (§4.1, App. C Table 6)

LLaMA-2-7B base (also LLaMA-2-13B and OPT-6.7B). Total batch 128; epochs "2, 3, or 10" (text: "Training typically proceeds for 2 epochs"); maximum sequence length 2048; learning rate 2×10⁻⁵; AdamW, β = (0.9, 0.98), ε = 1e-6; linear schedule, warmup 0.03; weight decay 0; bf16; DeepSpeed stage 3; open-instruct code. Evaluation: 18 NLP tasks in six categories (MMLU, PIQA, OpenbookQA, HellaSwag, LAMBADA; LAMBADA multilingual, WMT 2014, WMT 2016; WSC, WinoGrande, ARC, CoQA; GSM8K, HumanEval; TruthfulQA, ToxiGen, Hendrycks Ethics; BBH), MT-Bench, AlpacaEval 1.0 and 2.0. No seed count or run-to-run variance is reported.

## Table 1 (LLaMA-2-7B; NLP mean of 18 tasks, MT-Bench, AlpacaEval 1.0)

| Training data (examples) | NLP mean IT → IM | MT-Bench IT → IM | AlpacaEval 1.0 IT → IM |
|---|---|---|---|
| LLaMA-2-7B base (no SFT) | 49.32 | 1.16 | 0.01 |
| Alpagasus Alpaca 5k (5,305) | 45.29 → 47.47 | 3.62 → 3.48 | 16.29 → 19.52 |
| Alpagasus Dolly 3k (2,996) | 46.58 → 48.95 | 4.23 → 4.06 | 13.42 → 15.11 |
| Alpagasus Dolly 9k (9,229) | 45.54 → 48.00 | 4.33 → 4.55 | 21.54 → 30.77 |
| Less Tydiqa (13,533) | 48.21 → 48.70 | 4.08 → 4.36 | 5.12 → 10.10 |
| Less MMLU Chat (13,533) | 47.18 → 47.84 | 3.86 → 4.54 | 4.42 → 9.78 |
| Less BBH ICL (13,533) | 48.28 → 49.15 | 4.78 → 5.03 | 36.20 → 44.15 |
| LIMA (1,030) | 48.79 → 49.60 | 4.77 → 4.83 | 33.06 → 32.94 |

Every IT row has an NLP mean below the base model's 49.32. Category detail for Alpagasus Alpaca 5k IT: Commonsense Reasoning 75.86 (base) → 66.06; Multilinguality 61.99 → 57.24; BBH 38.80 → 26.80. The authors call this "instruction tuning tax" (§4.3 #3).

## Conditions reported for the IM effect (§4.2, Fig. 2, Table 5)

- IM helps more when the instruction/output length ratio is large (Fig. 2 left); Tülu V2 (ratio about 0.5) benefits less than Science Literature (ratio 24.7).
- IM helps more with fewer examples: Tülu V2 subsets from 1,000 to 35,000 examples at a fixed instruction/output ratio near 10 (Fig. 2 right; App. C). The figure has no per-point numeric labels.
- Table 5 average lengths (unit not stated): LIMA total 484.47, output 442.75, instruction 41.72, instruction/output 0.0942; Less MMLU Chat total 225.19, output 8.24, instruction 216.95, instruction/output 26.3316.
- Abstract: "we are not proposing IM as a replacement for current fine-tuning processes."

## Overfitting evidence (§4.3)

- LIMA training loss on output tokens: mean 1.37 (IT) versus 1.45 (IM); test loss on a 10% sample of Tülu V2: 1.32 (IT) versus 1.17 (IM) (Fig. 3).
- BLEU between greedy outputs and training targets (Table 2), IT → IM: LIMA 18.15 → 17.30; Less BBH ICL 60.96 → 53.94; Less MMLU Chat 72.43 → 69.20.
- Fig. 4: NLP mean over epochs 2-10 on five datasets; "IM generally has a lower instruction tuning tax compared to IT."
- Table 3: a KL-divergence loss to the base model reduced NLP-task degradation but lowered open-ended generation scores.

## Used in

ch-36 §3 (narrow data and the instruction tuning tax), §4.2 (IM versus response-only axis), Recipe row.

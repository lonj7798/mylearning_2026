---
chapter: ch-08
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2405.14394
source_url: https://arxiv.org/abs/2405.14394
created_at: "2026-04-23"
revised: 2026-09 (generality revision)
---

# Excerpt: Shi et al. 2024, "Instruction Tuning With Loss Over Instructions"

**Artifact:** Zhengyan Shi, Adam X. Yang, Bin Wu, Laurence Aitchison, Emine Yilmaz, Aldo Lipani
(UCL, University of Bristol), arXiv:2405.14394 (v1 2024-05; v2 2024-10-02), NeurIPS 2024.
**Read on:** 2026-09-17 from the PDF text cached at `scratchpad/sources/loss-masking-prompt.txt`.
**Source type:** paper.

> **Correction to the 2026-04 version of this excerpt and to the library card [[loss-masking-prompt]].**
> Both stated that this paper shows "response-only loss is strictly better … for typical instruction datasets"
> and that its abstract says so. The paper proposes the opposite method. INSTRUCTION MODELLING (IM) applies
> the loss to the instruction and prompt part as well as the output, and the paper reports that IM improves
> over instruction tuning (IT, response-only loss) in the tested settings, with the largest gains when
> instructions are long relative to outputs and when the training set is small (abstract; §4.2).
> Numbers below are from the paper, not from the card.

---

## Definitions used by the paper

- **IT (INSTRUCTION TUNING)** — cross-entropy on the response tokens only; instruction and prompt tokens are
  excluded from the loss. This is the `assistant_only_loss` / `completion_only_loss` behaviour in
  [[trl-sft-trainer]].
- **IM (INSTRUCTION MODELLING)** — cross-entropy over instruction and prompt tokens as well.

## Setting

LLaMA-2-7B-BASE (also LLaMA-2-13B and OPT-6.7B), 7 instruction datasets, evaluated on 18 traditional NLP tasks
in 6 categories plus MT-Bench, AlpacaEval 1.0 and AlpacaEval 2.0 (§4, Table 1). Hyperparameters (App. C,
Table 6): total batch 128 sequences, per-GPU batch 1, max sequence length 2048, LR 2e-5, AdamW
(ε = 1e-6, β = 0.9, 0.98), linear schedule with warm-up ratio 0.03, weight decay 0, bf16, DeepSpeed ZeRO-3,
2 epochs by default (2, 3, or 10 across experiments), 2-4 A100-80G GPUs.

## Table 1 rows ch-08 uses (LLaMA-2-7B; "Mean" is over the 18 NLP tasks)

| Training set (examples) | Method | Mean, 18 NLP tasks | MT-Bench | AlpacaEval 1.0 | AlpacaEval 2.0 |
|---|---|---|---|---|---|
| — | LLaMA-2-BASE | 49.32 | 1.16 | 0.01 | 0.01 |
| — | LLaMA-2-CHAT | 49.15 | 6.63 | 79.04 | 6.48 |
| Alpagasus Alpaca 5k (5,305) | IT | 45.29 | 3.62 | 16.29 | 2.46 |
| Alpagasus Alpaca 5k | NEFTUNE | 45.62 | 3.50 | 21.37 | 2.37 |
| Alpagasus Alpaca 5k | IM | 47.47 | 3.48 | 19.52 | 3.29 |
| Alpagasus Dolly 9k (9,229) | IT | 45.54 | 4.33 | 21.54 | 2.28 |
| Alpagasus Dolly 9k | IM | 48.00 | 4.55 | 30.77 | 2.67 |
| Less MMLU Chat (13,533) | IT | 47.18 | 3.86 | 4.42 | 1.20 |
| Less MMLU Chat | IM | 47.84 | 4.54 | 9.78 | 1.93 |
| LIMA (1,030) | IT | 48.79 | 4.77 | 33.06 | 2.58 |
| LIMA | IM | 49.60 | 4.83 | 32.94 | 2.47 |

Two readings ch-08 takes from this table:

1. **SFT on a narrow target distribution can lower the broad-suite score while raising the target score.**
   On Alpagasus Alpaca 5k, IT moves the 18-task mean from 49.32 (base) to 45.29, a drop of 4.03 points, while
   AlpacaEval 1.0 rises from 0.01 to 16.29. The paper calls the drop an *instruction tuning tax*; Fig. 4 plots
   the 18-task mean against epochs 2 to 10 on five datasets, and the paper's stated reading is that IM has a
   lower tax than IT (§4.3 "#3", Fig. 4).
2. **The loss-masking choice is one of the levers that moves that trade-off.** IM recovers 2.18 points of the
   18-task mean on the same data (45.29 → 47.47) and also raises AlpacaEval 1.0 (16.29 → 19.52).

## Conditions the paper states

- Gains depend on the instruction-to-output length ratio and on the number of training examples; the
  Figure 2 analysis varies both on Tülu V2 subsets of ~3,000 examples with ratios ≈5, 10, 15, and subsets from
  1,000 to 35,000 examples at a fixed ratio ≈10 (App. C).
- The authors state they are "not proposing IM as a replacement for current fine-tuning processes", and frame
  the guidance as for low-resource settings (abstract).
- On LIMA the AlpacaEval change is within noise in both directions (33.06 → 32.94), so the benefit is not
  uniform across datasets.
- A KL-divergence regulariser reduces the NLP-task degradation but harms open-ended generation (§4.3 "#4",
  Table 3). One seed per cell; no variance is reported.

## Mechanism recorded for the lab's masking test

Response-only loss sets `labels = -100` on prompt positions. In a multi-turn conversation the prompt for
assistant turn k includes every earlier user turn and every earlier assistant turn, so those positions are
masked too when training on turn k; TRL's `assistant_only_loss` instead keeps every assistant turn in the loss
in a single pass ([[trl-sft-trainer]], `build_labels` L1606-1632).

## Connections

- [[neftune]] — the third method in Table 1; its own card carries the NEFTune numbers.
- [[trl-sft-trainer]] — the configuration fields that select between these losses.

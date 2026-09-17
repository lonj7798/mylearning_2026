---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Huerta-Enochian, Ko — "Instruction Fine-Tuning: Does Prompt Loss Matter?"
source_url: https://arxiv.org/abs/2401.13586
created_at: "2026-09-15"
note: "No library card for this artifact on 2026-09-15; written from the primary PDF (arXiv v4, 2024-10-14)."
---

# Excerpt: Huerta-Enochian & Ko 2024 — Prompt loss weight in supervised instruction fine-tuning

**Artifact:** arXiv:2401.13586 (v1 2024-01; read as v4). Authors: Mathew Huerta-Enochian, Seung Yong Ko (EQ4ALL).

## Definitions (§2.1, §4.3)

- Prompt = instruction plus optional input; completion = target output.
- Generation ratio R_g = completion length / prompt length. Data with R_g < 1 is short-completion; R_g ≥ 1 is long-completion.
- Prompt loss weight (PLW) scales the next-token loss on prompt tokens; PLW = 0 is prompt masking and PLW = 1 is unmasked training. Tested values: 0, 5×10⁻⁴, 2.236×10⁻³, 10⁻², 2.463×10⁻², 5×10⁻², 10⁻¹, 2.463×10⁻¹, 5×10⁻¹, 1. For analysis PLW is transformed by w_p = PLW^0.30103, so that 0.1 maps to 0.5.
- Background: OpenAI's former `prompt_loss_weight` default was 0.01 and was removed with the v1 fine-tune API deprecation in January 2024 (§2.2).

## Setup (§4, App. E.1)

- 10 PLW values × 2 models (LLaMA 1 7B, LLaMA 2 7B) × 3 datasets = 60 runs, trained with the Stanford Alpaca code (commit 761dc5b) and its recommended hyperparameters, on one 4×A100-80GB node.
- Datasets (Table 1): AlpacaData (R_g 3.27), AlpacaDataCleaned (7.83), AlpacaDataShort (0.08; long-completion items rewritten as prompt-prediction tasks). AlpacaDataShort mean tokens: instruction 16.93, input 162.34, completion 14.62.
- 13 benchmarks: 4 multiple choice (ARC Challenge, PIQA, TruthfulQA-MC2, WinoGrande), 7 short generation (TruthfulQA-Gen, 6 WMT translation directions), 2 long generation (AlpacaEval 1 with a Mixtral 8x7B judge, PandaLM) (Table 2).

## Results (§5)

- For AlpacaDataShort, ARC Challenge, PIQA, TruthfulQA-Gen, and WinoGrande show a negative quadratic relation with an optimum strictly between 0 and 1; the long-generation benchmarks increase steadily toward PLW = 1; PLW = 0 almost always gave the worst scores on these seven (§5.1).
- Maximum PLW-based gain: about 20 percentage points on long-generation benchmarks, under 2 points on short-generation and multiple-choice benchmarks (§5.1).
- Regression (beta GLMM): significant only for AlpacaDataShort (p < 0.001); AlpacaData p = 0.237 and AlpacaDataCleaned p = 0.0861 with convergence warnings (Table 3). Fitted score = −4.284(w_p − 0.652)² + 0.781, giving a critical PLW λ = 0.242 (§5.2).
- Abstract summary: small PLW (0.01-0.5) better on multiple-choice and short-generation benchmarks; PLW near 1.0 better on long-generation benchmarks; PLW "can be safely ignored" for long-completion data.

## Mechanism analysis (§5.3)

- Training-loss relative standard deviation rises sharply for small non-zero PLW and is lowest at PLW = 1, so stability does not explain the pattern.
- Fine-tuned weights stay closer to the pre-trained weights for small non-zero PLW.
- Corpus BLEU of generations on 10,000 training prompts stays near 80 for PLW from 0 to about 0.1 and decreases above 0.1 (memorization).
- Interpretation by the authors: small PLW preserves pre-trained weights; larger PLW reduces overfitting and increases generation length.

## Supplemental (§6)

Weight decay, a Minkowski distance penalty, dropout, and label smoothing applied at PLW 0 or 1 did not match the relative aggregate of tuned PLW (0.855 vs at most 0.812 for LLaMA 1; 0.894 vs at most 0.837 for LLaMA 2). Effects extended to prompt-inverted UltraFeedback and Dolly data and to LLaMA 3 8B.

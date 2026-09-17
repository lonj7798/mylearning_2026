---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: arXiv:2310.05914v2 (Jain et al., "NEFTune: Noisy Embeddings Improve Instruction Finetuning")
source_url: https://arxiv.org/abs/2310.05914
created_at: "2026-04-23"
revised: "2026-09-15 (superseded by the generality revision of read.md)"
---

# Excerpt (superseded): NEFTune

This file is kept so that the path listed in `crawl-manifest.json` resolves. Its earlier body quoted the pre-revision library card and contained claims that the paper does not support. The NEFTune material used by [[read]] is in read.md §5, the NEFTune recipe row, and the library card [[neftune]] (verified 2026-09-14).

## Claims removed from the earlier version

- A dataset-size rule for NEFTune ("on when |D| ≤ 100K, off at |D| ≥ 500K") and the statement that Tülu 3 found NEFTune neutral at 939K prompts. No source gives the rule, and the Tülu 3 report (arXiv:2411.15124v5) does not mention NEFTune.
- "α = 5 default … works on 7B and 70B without re-tuning". The paper chooses α per dataset and model from {5, 10, 15} on AlpacaEval (App. A.1, Tables 7-8) and uses α = 15 at 70B (App. A.1).
- "Training loss drops slightly" as the NEFTune signature. Figure 4 shows higher training loss and slightly lower held-out loss with NEFT.
- "Compatible with gradient checkpointing, packing, FSDP", "uniform noise keeps tail behavior bounded", and "positional embeddings left untouched". None of these statements is in the paper.
- "Train loss up ~2–5%, eval win-rate up 5–30 pts". The paper reports no such range.

## What the paper reports (for orientation; loci in the card)

- Noise: X′ = X + (α/√(L·d))·ε with ε ~ Uniform(−1, 1), added to input embeddings during training only (Algorithm 1).
- LLaMA-2 7B AlpacaEval win rate against Text-Davinci-003 with a GPT-4 judge: Alpaca 29.79 → 64.69, Evol-Instruct 70.34 → 79.60, ShareGPT 68.74 → 76.28, OpenPlatypus 62.00 → 70.61 (Table 1).
- Settings for 7B runs: LR 5e-5 (Adam), 3 epochs, effective batch 128, sequence length 512 (App. A.1).

## Connections

- [[read]] §5; [[neftune]]; [[loss-masking-regimes]] for the NEFTune baseline in Shi et al. Table 1.

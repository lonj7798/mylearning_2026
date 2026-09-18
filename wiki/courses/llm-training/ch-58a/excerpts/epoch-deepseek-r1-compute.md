---
chapter: ch-58a
course: llm-training
phase: read
excerpt_of: "What went into training DeepSeek-R1? (Epoch AI, Gradient Updates newsletter, Jan. 31, 2025), by Ege Erdil"
source_url: https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1
created_at: "2026-09-17"
note: "No library card exists for this slug as of 2026-09-17. Read on 2026-09-17 in the cached page text (scratchpad sources/epoch-deepseek-r1-compute.txt)."
---

# Excerpt: Epoch AI's outside estimate of the DeepSeek-R1 stage budgets

Source type: practitioner evidence (an independent analysis with an explicit calculation, published as an opinionated
newsletter issue: "Gradient Updates shares more opinionated or informal takes ... These posts solely represent the views
of the authors"). Published 2025-01-31, eleven months before the arXiv v2 of the R1 paper disclosed GPU hours.

## The pre-training side (restating the DeepSeek-V3 report)
- "processing each trillion tokens of training data took them 3.7 days on this cluster, so about 180,000 H800 hours ...
  their total training dataset size was 14.8 trillion tokens, implying a training cost of around 14.8 * 180,000 = 2.66
  million H800 hours or around $5.3M if we price the cost of an H800 hour at $2."
- Derived model FLOP utilization: "the pretraining of this model would have required 6 * (37 billion) * (14.8 trillion) =
  3e24 FLOP", and with H800 PCIe cards at 1.5e15 FP8 per second "the implied model FLOP utilization (MFU) of DeepSeek
  v3's 55 day training run ends up being around 23%."

## The RL side (an estimate, not a disclosure)
- Cost model: `N·B·G·L·(37B)·(8 FLOP/param/token) + N·(K−1)·B·G·L·(37B)·(6 FLOP/param/token)`, where N is the number of
  exploration phases, B questions per batch, G samples per question, L the mean completion length, and K the gradient
  steps per phase.
- Inputs used: "N*K is the total number of gradient steps throughout RL training, and we know this is equal to around
  8000 from various figures in the paper"; L about 4000 tokens; "B and G are not explicitly stated in this paper, but in
  prior work DeepSeek used B = 1024 and G = 64. This is the biggest source of uncertainty in my calculation. For example,
  it's quite plausible DeepSeek used B = 512 for R1, in which case all the cost estimates here will be too high by a
  factor of 2."
- Result: "a final cost estimate for the initial reasoning RL phase of 6.1e23 FLOP, or around $1M if the RL phase had a
  similar MFU to pretraining."
- Whole-pipeline conclusion: "a reasonable ballpark estimate for the dollar cost of the GPU time that went into training
  R1 starting from V3 is around $1M, on top of the $5M that went into the pretraining of V3 itself", and "all of these
  numbers ignore the compute costs of experiments, personnel costs, administrative overhead".
- On the SFT stage: "If the average length per sample is around 8K ... this SFT dataset has a total of 800K * 8K = 6.4B
  tokens and so the fine-tuning itself has a negligible cost."

## What the later primary source printed
The arXiv v2 of the R1 paper (2026-01-04) prints, in Supplementary B.4.4 and Table 7: 101K H800 GPU hours for R1-Zero
(64 × 8 GPUs, about 198 hours), 41K for R1 (about 80 hours on the same GPUs), and 5K for SFT data creation, which sum to
147K ([[deepseek-r1-recipe]] rows at those loci). The paper does not print a dollar figure for R1 at that locus in the
card; 147K hours at the $2 per H800 hour that this newsletter itself assumes for V3 is $294K (derived here, not quoted).
Comparing the two: the estimate ("around $1M") is about 3.4 times the disclosed figure under that shared price
assumption, and the estimate's stated dominant uncertainty (B and G, assumed 1024 and 64, i.e. 65,536 sampled outputs
per step) is the quantity the paper later printed as 32 questions per step and 16 outputs per question, i.e. 512 (§2.1).
Because 65,536 / 512 = 128 is far larger than the 3.4× cost gap, the estimate's other inputs (8,000 gradient steps
against the paper's 10,400 for R1-Zero alone, a 4,000-token mean completion, pre-training MFU) offset most of it; the
split is not recoverable from either document.

## Verification
- Read on 2026-09-17 in the cached page text. Sections used: "Pre-training", "RL training for R1-Zero", "Subsequent
  training for R1".
- Status of every number above: an outside estimate under stated assumptions, not a disclosure. It is used in ch-58a as
  evidence about what an outside reader could and could not recover, never as a training setting.

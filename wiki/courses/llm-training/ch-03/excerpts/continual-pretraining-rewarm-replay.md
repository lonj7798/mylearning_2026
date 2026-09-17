---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: primary source arXiv:2403.08763v4 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2403.08763
created_at: "2026-09-15"
---

# Excerpt: Simple and Scalable Strategies to Continually Pre-train Large Language Models

**Paper:** Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, Mats L. Richter, Quentin Anthony, Timothée Lesort, et al.
(Université de Montréal; Concordia University; Mila; EleutherAI). arXiv v1 2024-03; v4 2024-09-04 read; TMLR
06/2024. Source type: paper. ch-03 uses the LR re-warming and replay results; ch-32a covers the topic in full.

## Claim (Abstract)
"a simple and scalable combination of learning rate (LR) re-warming, LR re-decaying, and replay of previous data is
sufficient to match the performance of fully re-training from scratch on all available data, as measured by the
final loss and the average score on several language model (LM) evaluation benchmarks", shown for a weak shift
(Pile → SlimPajama, English → English) and a strong shift (English → German) at 405M parameters, and for the weak
shift at 10B parameters.

## Re-warming and re-decaying (§6.1.2, Fig. 4; 405M)
- Pre-train on 300B Pile tokens with ηmax 3·10^-4 and ηmin 3·10^-5; continue on 300B SlimPajama or 200B German tokens.
- Baselines: constant at ηmin (no re-warm); re-warm to ηmax without re-decay. Re-warm and re-decay runs: ηmax ∈
  {1.5·10^-4, 3·10^-4, 6·10^-4}, linear warmup for 1% of iterations, cosine to 0.1·ηmax.
- Results: "the constant ηmin learning rate model achieves the least forgetting on D0"; re-warm and re-decay runs
  "adapt better to the new dataset by a significant margin for both distribution shifts"; "higher values of ηmax
  lead to more forgetting and more adaptation while the opposite is true for lower values."

## Replay (§6.2–§6.3, Fig. 1)
- Replay fractions compared include 1%, 5%, 10%, 50% for both shifts, plus 0.5% (weak) and 25% (strong) (§6.2).
- Selected settings: 5% replay for Pile → SlimPajama and 25% for Pile → German, with re-warming to the pre-training
  ηmax (3·10^-4) and cosine re-decay (§6.3; Fig. 1 caption).

## Verification
- Read on 2026-09-15 against arXiv:2403.08763v4 PDF text (Abstract, §1, §6.1.2, §6.2–§6.3 opening, Fig. 1 and Fig. 4
  captions).
- Not reported in the parts read: models above 10B; instruction-tuned starting checkpoints; downstream evaluations
  beyond the LM evaluation average used in Fig. 1.

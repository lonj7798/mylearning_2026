---
chapter: ch-23
course: llm-training
phase: read
excerpt_of: primary source arXiv:2310.06452v3 (no library card as of 2026-09-15; card planned under slug rlhf-generalisation-diversity)
source_url: https://arxiv.org/abs/2310.06452
created_at: "2026-09-15"
---

# Excerpt: Understanding the Effects of RLHF on LLM Generalisation and Diversity

**Paper:** Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, Roberta Raileanu (UCL, Meta, Oxford). arXiv v1 2023-10; v3 2024-02-19 read (ICLR 2024). Source type: paper.

## Setup (§4–5)
- Base model LLaMA 7B; OPT at five sizes in App. J.
- SFT with cross-entropy on demonstrations; reward model with a scalar head; PPO with reward R(x, y) = RM(x, y) − β_KL · D_KL(π_RL(y|x) ‖ π_SFT(y|x)), β_KL = 0.05 (§4, Eq. 1).
- Best-of-N from the SFT model with the RM: N = 16, temperature 0.7 (§4).
- Summarisation: TL;DR in distribution, CNN/DailyMail out of distribution. Instruction following: AlpacaFarm SFT/RM/RLHF models; AlpacaFarm Self-Instruct in distribution, AlpacaEval and Sequential Instructions out of distribution (§4).
- Generalisation metric: GPT-4 preference versus reference outputs (PvR) and head-to-head win rates (§5.1).
- Diversity metrics: expectation-adjusted distinct n-grams (EAD), 1 − mean Sentence-BERT cosine similarity, NLI diversity. K = 16 outputs per input for N = 500 inputs at temperature 1. Per-input diversity averages D over each input's outputs; cross-input diversity takes one output per input (§5.2, Eq. 2–3).

## Results (§6)
- Summarisation: Bo16 > RLHF > SFT in and out of distribution; generalisation gaps similar across methods (Fig. 2).
- Instruction following: RLHF and Bo16 beat SFT; on the harder Sequential Instructions shift RLHF generalises better; win rates of RLHF and Bo16 versus SFT rise by about 3.5% from in distribution to AlpacaEval (Figs. 3–4).
- Diversity (summarisation): RLHF has much lower per-input diversity than SFT on EAD and Sentence-BERT; cross-input diversity is also lower but by a smaller margin; NLI shows no meaningful difference (Figs. 5–6). The same trends hold for OPT across sizes (App. J.4).
- The authors call the cross-input result the first rigorous empirical demonstration of across-input mode collapse from RLHF (§6.2).
- Instruction-following diversity was not reported because the metrics did not separate models on long outputs and length confounds them (§6.2).

## Verification
- Read on 2026-09-15 against arXiv:2310.06452v3 PDF text (Abstract, §1, §4–6).

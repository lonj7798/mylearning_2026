---
chapter: ch-38a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlhf-generalisation-diversity.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2310.06452
created_at: "2026-09-15"
---

# Excerpt: Understanding the Effects of RLHF on LLM Generalisation and Diversity

**Authors:** Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, Roberta Raileanu (UCL, Meta, Oxford).
**Version read:** arXiv:2310.06452v3 (19 Feb 2024), ICLR 2024.
**Status:** no library card existed for this slug on 2026-09-15; numbers and quotes read at the stated loci in the v3 PDF. Most results are plotted in figures without printed values; only values the paper states in text are quoted here.

## What is compared (§1, §5)
Three fine-tuning methods on one base model: SFT, best-of-N sampling against the reward model (BoN, N = 16), and RLHF (PPO with a per-token KL penalty, β_KL = 0.05 in Eq. 1). Two tasks: summarisation (TL;DR in distribution, CNN/DailyMail out of distribution) with LLaMa 7B, and instruction following with the AlpacaFarm LLaMa 7B models (AlpacaFarm Self-Instruct in distribution; AlpacaEval and a new Sequential Instructions set out of distribution). OPT models of several sizes repeat the summarisation comparison (App. J). The judge is GPT-4 ("GPT-4-PvR" and head-to-head win rate). The generalisation gap is in-distribution minus out-of-distribution performance (§5.1).

## Generalisation results (§6.1)
Summarisation: "Bo16 outperforms RLHF which outperforms SFT, both ID and OOD", while "the generalisation gap is fairly similar between methods" (Fig. 2). The reward model's preference accuracy is 75.8% in distribution and 71.6% on the CNN/DailyMail preference data, which the authors use to argue the RM is not the source of the OOD drop. Instruction following: RLHF and Bo16 both beat SFT; "on AlpacaEval (the easier OOD generalisation task), models all generalise equally well, but on the harder Sequential Instructions OOD task, RLHF generalises much better" (Fig. 3). In head-to-head comparisons, "both RLHF and Bo16 winrates improves vs SFT by approximately 3.5% from ID to AlpacaEval OOD" (Fig. 4).

## Diversity metrics (§5.2)
For each of N = 500 test inputs, K = 16 outputs are sampled at temperature 1. Three metrics are applied to a set of outputs: expectation-adjusted distinct N-grams (EAD, n = 1…5), 1 minus average cosine similarity of Sentence-BERT embeddings, and NLI diversity (more contradictions and fewer entailments count as more diverse). Per-input diversity is the average over inputs of the diversity of the K outputs for one input; across-input diversity is the diversity of a set formed by taking one output per input (Eqs. 2–3).

## Diversity results (§6.2)
Diversity is reported for summarisation only; the instruction-following models showed no meaningful differences, which the authors attribute to metric design for short outputs and to RLHF producing longer outputs. "across the first two metrics, RLHF has much lower output diversity than SFT" in the per-input setting (Fig. 5). Across inputs the gap is smaller but present, and BoN keeps similar or higher across-input diversity than SFT on the first two metrics, so the reward model alone does not explain the drop (Fig. 6). NLI shows no meaningful difference. The authors call the across-input result "the first rigorous empirical demonstration of across-input mode collapse emerging from RLHF training specifically".

## KL coefficient sweep (§6.3, App. I)
"increasing the KL penalty coefficient leads to a drop in performance as expected, but also to a drop in per-input diversity, rather than a gain" (Figs. 11–13). The swept values are not listed in the text.

## Hyperparameters (App. E.3–E.4, Tables 2–4)
Summarisation, learning rates swept and selected on an in-distribution validation set: SFT {3e-4, 1e-4, 3e-5} with 3e-5 selected; RM {3e-4, 1e-4, 3e-5, 1e-5, 3e-6} with 3e-5 selected; RLHF {1.5e-6, 3e-6, 6e-6, 1.5e-5, 3e-5} with 1.5e-5 selected. SFT: batch 128, 1 epoch, 80% of layers frozen. RM: batch 64, 1 epoch, 80% frozen. RLHF: batch 256, 4 PPO epochs per batch, 750 PPO steps, PPO minibatch 256, KL penalty coefficient 0.05, advantage normalization on, 80% frozen. A single model carries the policy and a randomly initialized value head, rather than a separate value model initialized from the RM (App. F). Instruction-following models are taken from AlpacaFarm.

## How ch-38a uses it
§6 (generalisation gap and per-input versus across-input diversity; the KL-coefficient result), §9 (diversity measurement protocol), Recipe, Generalization lens.

---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: arXiv:2312.09244v3 (Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking)
source_url: https://arxiv.org/abs/2312.09244
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because no library card exists yet)"
---

# Excerpt: pretrain versus fine-tune reward-model ensembles

Used by [[read]] Guideline, §1.4, §3.2, the Generalization lens, and Common mistakes. Authors: Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D'Amour, DJ Dvijotham, et al. (Google DeepMind; Google Research). arXiv v1 2023-12; COLM 2024; checked against v3 (2024-08-16) on 2026-09-15.

This paper is the source of the shared-error finding that an earlier version of the chapter attributed to Coste et al. ([[reward-ensembling]]).

## Abstract findings
1. RMs are underspecified: RMs with similar in-distribution performance give very different rewards when used in alignment, because of distribution shift.
2. Aligning to one RM does not improve reward as measured by another RM trained on the same data.
3. Ensembles reduce over-optimization; ensembles that differ in pretraining seed generalize better than ensembles that differ only in fine-tuning seed; both beat individual RMs.
4. Pretrain ensembles do not eliminate reward hacking when all members share error patterns.

## Setup (§2)
- BT objective with an added term that regularizes the sum of rewards per preference pair towards zero, `J_reg(r) = J(r) + η · E[(r(x,y+) + r(x,y−))²]` with small η > 0, because the BT model is underdetermined by a prompt-dependent constant and order statistics such as median and minimum are otherwise meaningless (§2.1, Eq. 2).
- Five T5 models pretrained from scratch on C4 at base (220M), large (770M), and XL (3B), differing only in seed; each fine-tuned with five seeds; 25 RMs per task and scale (§2.3).
- Tasks: TL;DR, Anthropic Helpfulness (base 44K examples, split half for RM and half for policy), XSum/NLI (pointwise NLI RMs on ANLI) (§2.3).
- Policies: T5-large for summarization; instruction-tuned PaLM-2-XXS for Helpfulness; best-of-n and PPO with a λ sweep (§2.3).
- Evaluation: a T5-XXL RM trained on the same data, and a prompted PaLM-2-Large judge run on both response orders with 8 samples each (§2.3).
- Mean in-distribution RM accuracy (Table 1): TL;DR 65.8 / 69.3 / 71.4 (base / large / XL); Helpfulness 66.7 / 68.5 / 69.2; XSum/NLI 86.7 / 88.3 / 91.3.

## Shared reward hacks (§1)
- Summarization models produce outputs that are too short when tuned for factuality and too verbose when tuned for summarization quality.
- Assistant models overuse formulaic answer formats when tuned for helpfulness.
- Figure 1 (right): members of an insufficiently diverse ensemble unanimously rate an overly verbose, non-responsive reply as positive.

## Relation to other work (§1)
The authors differ from Coste et al. by studying pretrain and fine-tune ensembles, using human-annotated preferences instead of synthetic labels, and analysing real reward hacks not prevented by ensembles; they note that weight averaging (WARM) and LoRA ensembles cover fine-tune ensembles only.

---
chapter: ch-29e
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/opt-iml.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2212.12017
primary_version: arXiv:2212.12017v3 (v1 2022-12)
created_at: "2026-09-15"
---

# Excerpt: OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization

Verbatim quotations and table values used by ch-29e `read.md`, with loci. Checked against the v3 PDF on 2026-09-15.

## Generalization levels and leakage control (§2.3, §4.1)
> "a) tasks from clusters not included in training (Fully Held-out), b) tasks unseen during training but from seen clusters (Partially Supervised), and c) tasks seen during training (Fully Supervised)."

> "For each pair of train and eval tasks, we compute the fraction of examples that have any 13-gram overlap between the instantiated sequences from those examples. We manually examine every pair where more than 1% of the eval set overlaps with the training set (∼14,000 pairs)"

Table 1: OPT-IML Bench (train) 1,545 tasks, 17.9M examples; (dev) 35 tasks, 145K; (test) 87 tasks, 321K.

## Objective, packing, hyperparameters (§3.1-3.3, Table 3)
> "we separate the training sequence into a source context sequence and a target sequence and only include loss terms from the tokens in the target sequence (label-loss)."

> "we use document attention masking i.e. we modify the token attention mask in causal LMs to attend only to the tokens that are part of the same example [...] This changes the attention mask from a triangular to a block triangular mask and improves both stability and performance in our experiments."

> "We use Adam (Kingma and Ba, 2014) with 32-bit state with (β1, β2) = (0.9, 0.95), linearly warming up the learning rate for 60 steps to the maximum, followed by linearly decaying it to 0. We conduct preliminary experiments to select learning rates from {1e−5, 3e−5, 5e−5, 6e−5} and per-GPU batch sizes from {2, 4, 8} using our validation split"

> "We use a dropout of 0.1 (including embedding dropout) and clip gradient norms to 1.0"

| Model | # GPUs | Batch Size | Learning Rate | Steps | Warm-up Steps | FT Time (h) | # Tokens |
|---|---|---|---|---|---|---|---|
| OPT-IML 30B | 64 | 256 | 5e-05 | 4000 | 60 | 19 | 2B |
| OPT-IML 175B | 128 | 128 | 5e-05 | 8000 | 60 | 72 | 2B |

## Mixture factors (§4.2-4.7)
> "on average all models that use EPS outperform the model without it, after a certain threshold i.e. less than 4096 in our case, there is minimal variation in performance across all generalization levels."

> "in our benchmark, 71% of training examples would come from SuperNatInst, with 18% from PromptSource, and only 5% from FLAN."

> "setting Crossfit, Exmix, T5 and Unified-SKG proportions to 0 results in the worst model" ; final proportion "4/2/20/25/45/2/2" (Crossfit/Exmix/Flan/NIV2/PS/T5/U-SKG).

> "the model improves while adding pre-training data for up to 10% and then starts deteriorating after that. [...] we choose to include 5% pre-training data"

> "We see a substantial performance improvement on the 2/14 held-out validation reasoning tasks (Rouge-L from 12.2% to 31.6%) [...] we use 1% reasoning data for our final OPT-IML models."

> "adding even just 0.5% of the aforementioned dialogue data lowers 0-shot performance while 5-shot performance remains unchanged." (Table 7 average: baseline 46.0/57.6; + 0.5% BB3 44.8/57.5)

> "Training with BB3 data weakened the model's ability to conform to the required format."

## Task scaling (§4.4)
> "We observe that both fully held-out and partially supervised tasks get the most improvements with the increase in the number of training tasks. Interestingly, fully supervised tasks' performance remains unchanged"

## Final models (Table 9, 0-shot/32-shot; Table 14)
- OPT 30B average 59.2/60.5 → OPT-IML 30B 66.3/64.4; OPT 175B 61.4/69.9 → OPT-IML 175B 68.2/70.3.
- OPT 30B → OPT-IML 30B: PIQA 77.5/78.8 → 77.3/69.2; OpenBookQA 57.2/60.1 → 50.6/55.2.
- Table 14: OPT-IML-Max 175B BBH 35.7, MMLU 49.1/47.1; FLAN-T5 11B BBH 45.3, MMLU 53.7/54.9.

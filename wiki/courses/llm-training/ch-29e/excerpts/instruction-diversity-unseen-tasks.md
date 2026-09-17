---
chapter: ch-29e
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/instruction-diversity-unseen-tasks.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2402.10891
primary_version: arXiv:2402.10891v1 (2024-02)
created_at: "2026-09-15"
---

# Excerpt: Instruction Diversity Drives Generalization To Unseen Tasks

Verbatim quotations and table values used by ch-29e `read.md`, with loci. Authors: Dylan Zhang, Justin Wang, Francois Charton. Checked against the v1 PDF on 2026-09-15.

## Task (§3)
> "Model inputs are triplets of strings, (x,y,z), representing the rule x→y and the input sequence z. Model outputs are obtained by replacing the leftmost instance of x by y, in the sequence z. If x does not appear in z, the model output is z (input sentence is copied, Figure 1a)."

## Instruction count (§4.1)
> "we train GPT-2 models with 6 layers, 4 heads, and hidden dimension 256 [...] on a generated sample of S×I input sequences, corresponding to I different replacement rules (instructions) applied to S different sequences. The trained model is then tested on a dataset of 10^5 examples with unseen instructions."

> "We note that models trained on less than 300 instructions never generalize, even when the model only has a few rules to learn and is provided with a very large number of examples per rule. On the other hand, models trained on 1,000 instructions or more always generalize, even when the number of instructions becomes very large, and each rule is only featured in a handful of examples. A very sharp phase transition happens around 400 instructions."

(Figure 2a reports models trained on 10^6 examples.)

## Semantic spread (§4.3)
> "models trained on one set of constrained (repeated, periodic, or mirror) do not generalize to lower k [...] Models trained on large k (for all three constraints) do generalize to small k (and other unconstrained instructions)."

## Uneven distributions (§4.4)
> "For models trained on 1000 instructions [...] performance drops steeply once the example distribution becomes too uneven. Models trained on larger instruction sets, on the other hand, suffer little penalty."

## Input occurrence variety (§4.5, Table 1; rows for 1, 10, 20, and mixed occurrences shown)
| Train occurrences \ instructions | 2000 | 1000 | 500 | 200 |
|---|---|---|---|---|
| 1 | 0.71 | 0.41 | 0.00 | 0.00 |
| 10 | 0.88 | 0.71 | 0.00 | 0.00 |
| 20 | 0.73 | 0.28 | 0.07 | 0.00 |
| 1,5,10,15,20 | 0.94 | 0.94 | 0.62 | 0.00 |

## Pretrained model (§5, Figure 5)
> "We generate training sets of 40,000 sequences and test them on sets of 5,000 instances [...] Both sets contain 40% no-ops cases. We fine-tuned the pre-trained language model (Llama2-7b) (Touvron et al., 2023) with LoRA (Hu et al., 2021) with rank 512 and α of 1024 till convergence."

> "Uniform sub-sampling does not harm performance whereas non-uniform subsampling impacts generalization." (Figure 5 caption; subsampling to half the sample size at 9000 instructions)

## Limitations
> "To gain full control over different factors and ablate the effect of each, we adopted a synthetic setup instead of experimenting with real-world instruction-following datasets."

---
chapter: ch-29e
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/false-promise-imitating-proprietary-llms.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2305.15717
primary_version: arXiv:2305.15717v1 (2023-05)
created_at: "2026-09-15"
---

# Excerpt: The False Promise of Imitating Proprietary LLMs

Verbatim quotations and table values used by ch-29e `read.md`, with loci. Authors: Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, et al. Checked against the v1 PDF on 2026-09-15.

## Setup (Abstract, §3, §4.1)
> "We first finetune a series of LMs that imitate ChatGPT using varying base model sizes (1.5B–13B), data sources, and imitation data amounts (0.3M–150M tokens)."

> "we first curated a seed set of ten QA pairs from the validation dataset. We then iteratively generated 6,000 additional examples by prompting ChatGPT with five random QA pairs and asking it to generate similar but distinct examples." (NQ-synthetic, §3)

> "we train for one epoch using the AdamW optimizer with gradients re-scaled by the magnitude of each weight. We use a learning rate of 2e-3 with 1000 steps of linear warm-up from 0, and we train with batch size 32." (§4.1)

## Crowdworker ratings (Figure 1 caption)
> "Crowdworkers initially rate the quality of our imitation models highly, as ∼70% of their outputs are rated as equal or better than those of ChatGPT (left)."

## Broad versus targeted imitation (Table 1, NQ accuracy)
| Model | Imitation data | NQ |
|---|---|---|
| 7B | – | 17 |
| 7B | ShareGPT-Mix | 10 |
| 7B | NQ-Synthetic | 22 |
| 13B | – | 20 |
| 13B | ShareGPT-Mix | 15 |
| 13B | NQ-Synthetic | 27 |
| ChatGPT | – | 31 |

(§4.1 describes the NQ evaluation as 3-shot; the Table 1 caption says zero-shot.)

## Style convergence (Table 2)
| Metric | LLaMA | 20M | 80M | 150M | ChatGPT #2 |
|---|---|---|---|---|---|
| If ChatGPT outputs a list, do we? | 13% | 50% | 67% | 81% | 83% |
| Outputs are in authoritative tone according to GPT-4 | 57% | 99% | 98% | 98% | 98% |

## Base model quality (§4.3)
> "Rather than increasing imitation data size, we find that using better base LMs (by increasing base model size) does lead to substantial accuracy improvements (Figure 4, bottom)."

## Interpretation stated by the authors (§5)
> "Our results show that a modest amount of finetuning provides little to no improvements on an LM's knowledge or capabilities."

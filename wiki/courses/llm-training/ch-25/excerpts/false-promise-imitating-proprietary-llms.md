---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/false-promise-imitating-proprietary-llms.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2305.15717
primary_version: arXiv:2305.15717v1 (2023-05)
created_at: "2026-09-15"
---

# Excerpt: The False Promise of Imitating Proprietary LLMs

Authors (v1 title page): Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, Dawn Song (UC Berkeley). Checked against the v1 PDF on 2026-09-15. Used by ch-25 `read.md` §8 and the Generalization lens. A separate excerpt with the same slug exists in ch-29e.

## Setup (Abstract, §3)
> "We first finetune a series of LMs that imitate ChatGPT using varying base model sizes (1.5B–13B), data sources, and imitation data amounts (0.3M–150M tokens)."

Broad-coverage imitation data, "ShareGPT-Mix" (§3):
> "ShareGPT: we use approximately 90K dialogues shared by users on the website ShareGPT. To maintain data quality, we deduplicated on the query level and removed any non-English conversations using a language detector. This leaves approximately 50K examples, each of which consist of multiple turns of dialogue."

plus about 27K HC3 ChatGPT responses and 10k Discord ChatGPT-bot examples. Targeted data, "NQ-synthetic": 6,000 single-turn QA examples generated from 10 seed pairs.

## Crowd ratings vs benchmarks
> "Crowdworkers initially rate the quality of our imitation models highly, as ∼70% of their outputs are rated as equal or better than those of ChatGPT" (Fig. 1 caption)

> "imitation models close little to none of the gap from the base LM to ChatGPT on tasks that are not heavily supported in the imitation data." (Abstract)

Table 1 (Natural Questions accuracy): 7B base 17, ShareGPT-Mix 10, NQ-Synthetic 22; 13B base 20, ShareGPT-Mix 15, NQ-Synthetic 27; ChatGPT 31.

## Style convergence (Table 2; LLaMA → 20M → 80M → 150M imitation tokens → ChatGPT #2)
| Metric | LLaMA | 20M | 80M | 150M | ChatGPT #2 |
|---|---|---|---|---|---|
| If ChatGPT outputs a list, do we? | 13% | 50% | 67% | 81% | 83% |
| Outputs are in authoritative tone according to GPT-4 | 57% | 99% | 98% | 98% | 98% |

## Interpretation stated by the authors (§4.3–4.4)
> "training on more ShareGPT-Mix data hurts performance as compared to the base model on some of our evaluations ... We believe that these performance regressions arise from a distribution shift and tension between the conversational-style fine-tuning data and the downstream benchmarks."

> "crowd workers without domain expertise or significant time investments can easily be deceived by stylistic components—answers that sound confident and correct are often spuriously chosen more often."

> "Surprisingly, our GPT-4 evaluations also showed the same trends as our crowdworker evaluations"

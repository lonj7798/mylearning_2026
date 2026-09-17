---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/pretrainers-guide-training-data.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2305.13169
primary_version: arXiv:2305.13169v2 (2023-11-13; v1 2023-05)
created_at: "2026-09-15"
---

# Excerpt: A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity

Verbatim quotations and table values used by ch-09 `read.md`, with loci. Authors: Shayne Longpre, Gregory Yauney, Emily Reif, Katherine Lee, Adam Roberts, Barret Zoph, et al. (MIT, Cornell, Google Research, OpenAI). Checked against the v2 PDF on 2026-09-15.

## Scope (Abstract)
> "we pretrain 28 1.5B parameter decoder-only models, training on data curated (1) at different times, (2) with varying toxicity and quality filters, and (3) with different domain compositions."

## Datasets and models (§2.1, §2.4, App. C Table 5)
> "We further deduplicate both datasets using the approximate deduplication method described in Lee et al. (2022)."

> "Our main experiments use LM-XL, a 1.5B parameter decoder-only model similar to the t5.1.1-XL architecture configuration [...] For experiments that measure scaling effects, we use LM-Small, a 20M parameter decoder-only model"

Table 5 (LM-XL and LM-Small): batch size 4096; sequence length 512; training steps 88,064; dropout 0.0; base learning rate 0.5; decay factor 0.5; warmup steps 1000; steps per decay 20000. Derived: 4,096 × 512 × 88,064 = 184,683,593,728 tokens seen.

## Domain removal (§6)
> "We then pretrain LM-XL with the full dataset minus each category, yielding nine models, then finetune each for QA using Natural Questions. Finally, we evaluate the model on 27 unique datasets from MRQA (Fisch et al., 2019) and UnifiedQA (Khashabi et al., 2020) that have also been partitioned into domains."

Figure 8 values (relative to the Full Dataset model; % data remaining in parentheses):
| Removed | Wiki | Web | Books | Biomed | Academic | Common Sense | Contrast Sets | Average |
|---|---|---|---|---|---|---|---|---|
| No Social (99%) | −0.8 | −3.7 | 2.6 | 0.1 | 3.5 | −3.5 | 3.5 | 0.4 |
| No Wiki (98%) | −1.3 | −5.3 | 3.0 | 0.2 | 0.9 | −4.4 | 7.2 | −0.3 |
| No Books (93%) | −3.5 | −6.3 | 1.0 | 0.0 | −1.6 | −6.5 | −4.4 | −2.7 |
| No OpenWeb (93%) | −2.0 | −4.1 | 0.1 | −1.0 | 0.6 | −5.8 | −2.9 | −1.4 |
| No Legal (91%) | −2.7 | −2.9 | 3.8 | 0.4 | 0.8 | −2.6 | −0.4 | −0.6 |
| No Academic (87%) | −0.3 | −2.5 | 0.3 | −0.9 | 2.2 | −1.1 | 4.3 | 0.2 |
| No Pubmed (85%) | −0.3 | −3.0 | 3.9 | −5.8 | −1.5 | −5.9 | 3.9 | −1.2 |
| No Code (81%) | −0.5 | −3.1 | 2.9 | −1.2 | 1.2 | −5.8 | 4.4 | −0.1 |
| No CC (73%) | −3.2 | −6.2 | −2.9 | −4.6 | −5.9 | −8.0 | −5.2 | −4.8 |

> "removing CC from the pretraining dataset reduces performance on downstream Academic QA tasks to a much greater extent than removing the Academic domain. Our hypothesis is that CC, OpenWeb and Books contain extensive coverage of many topics"

> "Despite the importance of data heterogeneity, the best mean performance still comes from models that train on all, or nearly all, the data."

Table 3 (% Data; Toxicity Identification score; Toxic Generation score): Full 100.0, 0.0, 0.0; No Social 98.8, +0.1, +0.4; No Wiki 97.9, −0.4, +4.2; No Books 93.1, −1.3, −6.2; No OpenWeb 93.1, −1.5, −5.2; No Legal 91.0, −0.4, +0.8; No Academic 87.1, +0.0, −1.2; No Pubmed 85.1, −0.2, −0.2; No Code 80.9, +0.2, +0.6; No CC 73.1, −1.9, −2.1.

## Data age (§4, Table 2)
> "We pretrain four autoregressive language models on versions of C4: 2013, 2016, 2019, and 2022. For each version we begin with Common Crawl data and remove all data that was scraped after the cutoff year."

Table 2, pretraining-to-evaluation temporal degradation (TD) and Pearson r, mean over PubCLS, NewSum, PoliAff, TwiERC, AIC: LM-Small TD 0.08, r 0.07; LM-XL TD 0.41, r 0.61. Finetuning-to-evaluation mean TD: LM-Small 2.36, LM-XL 2.84.

> "This suggests that even substantial finetuning cannot overcome pretraining data that is temporally misaligned."

> "we do not find the same temporal degradation effects of pretraining were significant for LM-Small models."

## Quality and toxicity filters (§5, Fig. 6, Fig. 7)
Fig. 6 (quality filter on C4; average column): Inverse T=0.5 (73%) −3.1; T=0.975 (91%) 2.5; T=0.95 (84%) 1.0; T=0.9 (73%) 1.2; T=0.7 (46%) 0.7. Books column: −2.2 at T=0.975, −6.7 at T=0.7.

Fig. 7 (toxicity filter on C4; average column): Inverse T=0.06 (92%) 1.7; T=0.95 (98%) 0.2; T=0.9 (95%) −0.7; T=0.7 (86%) −1.2; T=0.5 (76%) −2.0; T=0.3 (61%) −2.7.

> "Quality filtering effects are not easily predicted by dataset characteristics."

> "Most interesting of all, the strongest performance on toxicity identification for every dataset comes from the inverse toxicity filter."

## Recommendation on toxicity (§7)
> "suggests practitioners should prioritize toxic identification rather than curbing toxic generation abilities during pretraining."

## Limitations (§8)
> "We carefully curated the choice of experiments in advance, without the luxury of multiple rounds of reflection and repetition"

> "Our analysis was limited to two English datasets."

> "Our experiments focus on finetuned settings rather than zero- or few-shot prompting."

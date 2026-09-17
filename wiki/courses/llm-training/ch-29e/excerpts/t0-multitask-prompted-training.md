---
chapter: ch-29e
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/t0-multitask-prompted-training.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2110.08207
primary_version: arXiv:2110.08207v3 (v1 2021-10; ICLR 2022)
created_at: "2026-09-15"
---

# Excerpt: Multitask Prompted Training Enables Zero-Shot Task Generalization (T0)

Verbatim quotations used by ch-29e `read.md`, with loci. Checked against the v3 PDF on 2026-09-15.

## Held-out tasks (§3)
> "To test zero-shot generalization, we hold out all constituent datasets of four tasks: natural language inference (NLI), coreference resolution, sentence completion, and word sense disambiguation."

> "Additionally, we do not train our main model on any datasets that Brown et al. (2020) used for evaluation, so that our main results will be a fair zero-shot comparison."

## Prompt collection (§4)
> "As of writing, P3 contains 2073 prompts for 177 datasets (11.7 prompts per dataset on average)."

## Training and selection (§5)
> "we use Lester et al. (2021)'s LM-adapted T5 model (referred to as T5+LM), produced by training T5 on 100B additional tokens from C4 on a standard language modeling objective."

> "We perform checkpoint selection by choosing the checkpoint that yields the highest score on the validation splits of our training datasets. This still satisfies the true zero-shot (Perez et al., 2021) setting as we do not use any examples from any of the held-out tasks to select the best checkpoint."

> "treat any dataset with over 500'000 examples as having 500'000 / num templates examples for the purposes of sampling"

> "We use a batch size of 1024 sequences (corresponding to 2^20 total input tokens per batch) and the Adafactor optimizer [...] we use a learning rate of 1e-3 and a dropout rate of 0.1." (the exponent is a superscript in the PDF)

> "We do not perform prompt selection by comparing the performance of different prompts on the validation split [...] For a given dataset, we report the median performance across all prompts for this dataset along with their interquartile range (Q3 - Q1)"

## Results (§6.1-6.2)
> "We find that T0 matches or exceeds the performance of all GPT-3 models on 9 out of 11 held-out datasets."

> "further increasing p from 1 to an average of 5.7 does yield additional improvement in both median (increases for 8/11 datasets) and spread (decreases for 7/11 datasets). [...] T0's inclusion all prompts (including those that do not correspond to the dataset's original task) further improves the median (increases for 9/11 datasets) and spread (decreases for 8/11 datasets)"

> "the median performance of all 5 held-out datasets increases as d increases from 39 to 49. However, the spread only decreases for 1 out of 5 datasets. [...] As d increases from 49 to 55, the median performance of all datasets again increases, but the spread only decreases for 2 out of 5 datasets."

> "One of these templates is identical to Brown et al. (2020, p. 59)'s reported prompt, which scores an accuracy of 58.8%, lower than the 63.5% reported in Brown et al. (2020). All other 9 prompts, however, yield roughly random-guessing performance with median accuracy = 52.96% and interquartile range = 1.28%."

## Discussion (§7)
> "we reevaluate these two datasets without instructions as done by Wei et al. (2021) and Brown et al. (2020) and find that it improves performance on HellaSwag from a median of 33.65% to 57.93%"

> "we find that multitask prompted training improves the performance of models at least as small as 3B parameters (Figure 8). We identify two key differences between the models that could explain this discrepancy"

---
chapter: ch-19
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/false-promise-imitating-proprietary-llms.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2305.15717
primary_version: arXiv:2305.15717v1 (2023-05)
created_at: "2026-09-15"
---

# Excerpt: The False Promise of Imitating Proprietary LLMs

Authors: Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, et al. (UC Berkeley). Checked against the v1 PDF on 2026-09-15. Used by ch-19 `read.md` §9. The same paper is excerpted for ch-29e.

## Claim (Abstract)
> "An emerging method to cheaply improve a weaker language model is to finetune it on outputs from a stronger model, such as a proprietary system like ChatGPT (e.g., Alpaca, Self-Instruct, and others)."

> "when conducting more targeted automatic evaluations, we find that imitation models close little to none of the gap from the base LM to ChatGPT on tasks that are not heavily supported in the imitation data. We show that these performance discrepancies may slip past human raters because imitation models are adept at mimicking ChatGPT's style but not its factuality."

## Data (§3)
- Task-specific (NQ-synthetic): "we first curated a seed set of ten QA pairs from the validation dataset. We then iteratively generated 6,000 additional examples by prompting ChatGPT with five random QA pairs and asking it to generate similar but distinct examples."
- Broad-coverage (ShareGPT-Mix): about 50K ShareGPT examples after query-level deduplication and removal of non-English conversations (from about 90K dialogues); about 27K HC3 ChatGPT responses for about 24K questions; 10k Discord ChatGPT-bot examples. "for each user query in the dataset, the most similar other user query has an average BLEU score similarity of just 8%", versus 61% for Super-NaturalInstructions.

## Training and evaluation (§4.1)
- Base models: GPT-2 1.5B, LLaMA 7B, LLaMA 13B; imitation data 0.3M to 150M tokens.
- "we chunk the conversations into 2048 tokens blocks ... We fine-tune using standard LM losses on only the model outputs ... we train for one epoch using the AdamW optimizer with gradients re-scaled by the magnitude of each weight. We use a learning rate of 2e-3 with 1000 steps of linear warm-up from 0, and we train with batch size 32."
- Automatic evaluation: 5-shot MMLU, 3-shot Natural Questions, 0-shot HumanEval (§4.1). Human evaluation by crowdworkers and GPT-4 against ChatGPT.

## Results
- Fig. 1 caption: "∼70% of their outputs are rated as equal or better than those of ChatGPT".
- §4.2: "human ratings quickly saturate as we scale up the amount of imitation data".
- §4.3: "across every benchmark that we measured, ShareGPT-mix imitation models do not improve (or even decline) in accuracy as compared to the base model, even when adding additional imitation data (Figure 4, top)."
- §4.3: "using better base LMs (by increasing base model size) does lead to substantial accuracy improvements (Figure 4, bottom)."

Table 1 (NQ accuracy; §4.1 describes NQ as 3-shot):

| Model | Imitation data | NQ |
|---|---|---|
| 7B | – | 17 |
| 7B | ShareGPT-Mix | 10 |
| 7B | NQ-Synthetic | 22 |
| 13B | – | 20 |
| 13B | ShareGPT-Mix | 15 |
| 13B | NQ-Synthetic | 27 |
| ChatGPT | – | 31 |

Table 2 (style similarity to a random ChatGPT response; columns are imitation-data amounts; the base model size is not stated in the caption):

| Metric | LLaMA | 20M | 80M | 150M | ChatGPT #2 |
|---|---|---|---|---|---|
| If ChatGPT outputs a list, do we? | 13% | 50% | 67% | 81% | 83% |
| If ChatGPT outputs a summary paragraph, do we? | 2% | 40% | 42% | 48% | 55% |
| Unigram intersection w/ ChatGPT's output | 19.5 | 40.4 | 41.9 | 42.5 | 49.2 |
| Pearson correlation in length w/ ChatGPT's output | -0.11 | 0.51 | 0.62 | 0.62 | 0.67 |
| Outputs are in authoritative tone according to GPT-4 | 57% | 99% | 98% | 98% | 98% |

## Stated interpretations
- §4.3: "these performance regressions arise from a distribution shift and tension between the conversational-style fine-tuning data and the downstream benchmarks. An open problem is whether these performance regressions can be mitigated using regularization or by mixing in pre-training data during fine-tuning."
- §4.4: "our GPT-4 evaluations also showed the same trends as our crowdworker evaluations (albet with a slightly larger absolute preference for ChatGPT's outputs)."
- §4.4: imitation models "inherit the safety and toxicity style of the teacher model" (RealToxicityPrompts, Fig. 5).

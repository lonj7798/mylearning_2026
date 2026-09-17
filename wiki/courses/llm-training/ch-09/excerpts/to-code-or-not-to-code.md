---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/to-code-or-not-to-code.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2408.10914
primary_version: arXiv:2408.10914v1 (2024-08-20)
created_at: "2026-09-15"
---

# Excerpt: To Code, or Not To Code? Exploring Impact of Code in Pre-training

Verbatim quotations and table values used by ch-09 `read.md`, with loci. Authors: Viraat Aryabumi, Yixuan Su, Raymond Ma, Adrien Morisot, Ivan Zhang, Acyr Locatelli, et al. (Cohere For AI, Cohere). Checked against the v1 PDF on 2026-09-15.

## Data (§2.1)
> "We filter out all documents from GitHub and StackExchange to remove code and code-adjacent data sources and ensure this is a text-only source. SlimPajama has a total of 627B tokens. After removing all code sources, this results in our text pre-training corpus with a total of 503B tokens."

Code sources: web-based code from the Stack, top 25 languages, 139B tokens; markup-style languages 180B tokens; a proprietary synthetic set of formally verified Python problems, 3.2B tokens; code-adjacent data (commits, notebooks, StackExchange) 21.4B tokens.

## Evaluation (§2.2, Table 1)
World knowledge: TriviaQA and NaturalQuestionsOpen (0-shot). Natural-language reasoning: 11 tasks (BoolQ, PiQA, SciQ, SocialIQA, QUAC, SuperGLUE-CB, SuperGLUE-COPA, StoryCloze, HellaSwag, Winogrande, ARC-Easy). Code: HumanEval-Python and MBPP pass@1. Generation: Dolly-200 English win-rates with Command-R+ as judge.

## Models (§2.3)
> "we use 470M and 2.8B parameters decoder-only auto-regressive Transformer models [...] All models are pre-trained using AdamW optimizer with a batch size of 512 and a cosine learning rate scheduler with a warmup of 1325 steps. We use a maximum sequence length of 8192."

> "we pre-train 64 models in total."

## Code proportion (§3.3)
> "We train six models for 200B tokens with increasing code proportions: 0%, 25%, 50%, 75%, 90%, and 100%."

> "The best performance is from a model with 25% code and 75% text, with a 3.4% relative improvement over a model with 0% code. While performance is maintained up to 75% code, it starts to rapidly erode at higher proportions with a sharp relative drop of 18.3% when the model is trained on 100% code compared to a model with no code."

> "For World Knowledge tasks, we see an inverse relationship with increasing the amount of code. [...] there is a slight relative drop of 3.4% at 25% code and this relative drop worsens to 31% at 75% code compared to the no-code model. The fully code model (100% code) is unable to perform in world knowledge task (86% drop relative to text-only)"

> "the 100% code leads to a 2.6x increase in the code benchmarks compared to the 25% code model."

## Scale (§3.2)
> "Overall our experiments scaling to a larger size shows that our results hold and are consistent with the trends we observe at 470M parameter ablations."

## Cooldown (§3.5)
> "We change the learning rate schedule from cosine-based to linear annealing with a final learning rate of 1e−6. We evaluate the impact of including code in cooldown by comparing 3 models: a pre-trained model before cooldown, cooldown without code data, and cooldown with 20% code data. [...] 40B tokens which is 10% of the token budget of the pre-trained model."

> "a cooldown with code data is most beneficial with 3.6%, 10.1%, and 20% in NL reasoning, world knowledge, and code relative to the model without cooldown. In contrast, we find that cooldown without code does not provide any increases for both NL reasoning and Code, while providing a relative improvement of 3.1% in World Knowledge tasks"

## Table 2 (recipes; NL Reason., Know., Avg., Code, Total Avg.)
| Variant | Recipe | Text | Code | Reason. | Know. | NL Avg. | Code | Total Avg. |
|---|---|---|---|---|---|---|---|---|
| Text-only | Pre-training | 400B | - | 49.0 | 9.5 | 29.2 | 0.4 | 19.6 |
| Text-only | Cooldown | +32B | +8B | 54.1 | 11.1 | 32.6 | 4.4 | 23.2 |
| Balanced-only | Pre-training | 200B | 200B | 51.8 | 8.1 | 30.0 | 9.0 | 23.0 |
| Balanced-only | Cooldown | +32B | +8B | 53.2 | 11.1 | 32.1 | 8.4 | 24.2 |
| Balanced → Text | Pre-training Init. | 100B | 100B | 52.0 | 7.4 | 29.6 | 7.8 | 22.4 |
| Balanced → Text | Continue Pre-train. | +180B | +20B | 53.0 | 9.9 | 31.5 | 4.8 | 22.6 |
| Balanced → Text | Cooldown | +32B | +8B | 54.9 | 10.9 | 32.9 | 5.8 | 23.9 |
| Code → Text | Pre-training Init. | - | 200B | 44.7 | 1.5 | 23.1 | 15.5 | 20.6 |
| Code → Text | Continue Pre-train. | +180B | +20B | 53.3 | 9.5 | 31.4 | 4.1 | 22.3 |
| Code → Text | Cooldown | +32B | +8B | 52.1 | 10.3 | 31.2 | 7.5 | 23.3 |

The caption does not state the model size. The §3.6 summary ("relative increase of 8.2% in natural language (NL) reasoning, 4.2% in world knowledge [...] and a 12x boost in code performance") matches the Text-only pre-training and Balanced → Text continue-pre-training rows and the 470M results of §3.1.

## Recommendation (§3.6)
> "Our recommendation for the best overall general downstream performance would be to include a balanced mixture of code and text data during pre-training from scratch (Section 3.3), use a relatively lower code percentage during continual pre-training (Section 3.1), and include code data into cooldown mixture."

## Limitations (§6)
> "we do not study its impact on safety. Additionally, given the nature of pre-training and the number of ablations we have conducted we were limited by the scale of larger model sizes due to prohibitive compute costs."

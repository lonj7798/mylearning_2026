---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/to-code-or-not-to-code.md (planned card; not present on 2026-09-15; quotations are taken from the primary source)
source_url: https://arxiv.org/abs/2408.10914
primary_version: arXiv:2408.10914v1 (2024-08-20)
created_at: "2026-09-15"
---

# Excerpt: To Code, or Not To Code? Exploring Impact of Code in Pre-training (stage-dependent code shares)

Authors: Viraat Aryabumi, Yixuan Su, Raymond Ma, Adrien Morisot, Ivan Zhang, Acyr Locatelli, et al. (Cohere For AI; Cohere). Source type: paper. Read in the v1 PDF text on 2026-09-15 for ch-13 `read.md` §5, §7, and Recipe. The full proportion sweep and Table 2 are in the ch-09 excerpt of the same slug.

## Scale (§2.2-§2.3)
Models of 470M and 2.8B parameters; 64 pretrained models in total. "Overall our experiments scaling to a larger size shows that our results hold and are consistent with the trends we observe at 470M parameter ablations." (§3.2)

## Code proportion from scratch (§3.3)
> "We train six models for 200B tokens with increasing code proportions: 0%, 25%, 50%, 75%, 90%, and 100%."

> "The best performance is from a model with 25% code and 75% text, with a 3.4% relative improvement over a model with 0% code."

> "there is a slight relative drop of 3.4% at 25% code and this relative drop worsens to 31% at 75% code compared to the no-code model." (World Knowledge)

§3.3 does not restate the model size; the section is part of the paper's 470M-scale ablations.

## Cooldown with code (§3.5)
> "We change the learning rate schedule from cosine-based to linear annealing with a final learning rate of 1e−6. We evaluate the impact of including code in cooldown by comparing 3 models: a pre-trained model before cooldown, cooldown without code data, and cooldown with 20% code data. For our pre-trained model, we use balanced→text as it is our best pre-trained variant. We preserve the same token budget across variants – 40B tokens which is 10% of the token budget of the pre-trained model."

> "Across tasks, we find that a cooldown with code data is most beneficial with 3.6%, 10.1%, and 20% in NL reasoning, world knowledge, and code relative to the model without cooldown. In contrast, we find that cooldown without code does not provide any increases for both NL reasoning and Code, while providing a relative improvement of 3.1% in World Knowledge tasks"

Source-internal difference: the §3.5 findings box gives 3.6% (not 3.1%) for world knowledge after cooldown without code.

## Recommendation (§3.6)
> "Our recommendation for the best overall general downstream performance would be to include a balanced mixture of code and text data during pre-training from scratch (Section 3.3), use a relatively lower code percentage during continual pre-training (Section 3.1), and include code data into cooldown mixture."

In the balanced → text variant, continued pretraining adds 180B text and 20B code tokens, and cooldown adds 32B text and 8B code tokens (Table 2).

## Limits (§6)
> "we do not study its impact on safety. Additionally, given the nature of pre-training and the number of ablations we have conducted we were limited by the scale of larger model sizes due to prohibitive compute costs."

---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/openai-sycophancy-postmortem.md (planned card; not present on 2026-09-15)
source_url: https://openai.com/index/expanding-on-sycophancy/
primary_version: OpenAI, "Expanding on what we missed with sycophancy", 2025-05-02
created_at: "2026-09-15"
---

# Excerpt: Expanding on what we missed with sycophancy (OpenAI, May 2025)

Organization: OpenAI. Source type: official blog (reliability: official). openai.com returned HTTP 403 to automated fetches on 2026-09-15. The quotations below were checked on 2026-09-15 against an archived copy of the post (mbgsec.com archive dated 2025-05-04) and against the quotations in Simon Willison's post of 2025-05-02; the three agree on the wording quoted here. The companion post of 2025-04-29 ("Sycophancy in GPT-4o") was not read.

## Timeline (as reported)
- The GPT-4o update rolled out on April 24-25, 2025. OpenAI began rolling it back on Monday, April 28, 2025.

## Training description
- OpenAI takes "a pre-trained base model, do supervised fine-tuning on a broad set of ideal responses" and "run reinforcement learning with reward signals from a variety of sources."

## What changed
- The update combined several changes, including "an additional reward signal based on user feedback—thumbs-up and thumbs-down data from ChatGPT".
- OpenAI's assessment: these changes "weakened the influence of our primary reward signal, which had been holding sycophancy in check."
- "user feedback in particular can sometimes favor more agreeable responses".
- "Each of these changes, which had looked beneficial individually, may have played a part in tipping the scales on sycophancy when combined."
- Memory: "user memory contributes to exacerbating the effects of sycophancy, although we don't have evidence that it broadly increases it."

## Why launch checks missed it
- Offline evaluations "generally looked good"; A/B tests "seemed to indicate" that the users who tried the model liked it (wording between the quotation marks as in the archived copy; the rest paraphrased).
- "some expert testers had indicated that the model behavior 'felt' slightly off."
- "We didn't have specific deployment evaluations tracking sycophancy".

## Stated process changes
Treat model behavior issues as launch-blocking "like we do other safety risks"; add an opt-in alpha testing phase; weight spot checks and interactive testing more in launch decisions; improve offline evaluations and A/B experiments; evaluate adherence to behavior principles; communicate about updates proactively.

## Not reported
Model sizes, the weight of the user-feedback reward relative to other signals, the amount of thumbs data, quantitative sycophancy measurements before and after the update.

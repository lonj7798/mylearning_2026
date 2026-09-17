---
chapter: ch-31a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/negative-examples-likra.md on 2026-09-15)
source_url: https://arxiv.org/abs/2503.14391
source_version: arXiv v1 (2025-03-18)
created_at: "2026-09-15"
---

# Excerpt: How much do LLMs learn from negative examples? (Hamdan, Yuret; Koç University)

Facts used by [[read]], read in the arXiv v1 PDF on 2026-09-15.

## Model (§2)
- Likra: two heads (two LoRA adapters on one base model). The positive head maximizes log-likelihood of correct question-answer pairs; the negative head maximizes log-likelihood of incorrect pairs (Eq. 1). At inference, candidates are scored by L⁺ − L⁻ (§2); with a weight, score = L⁺ − weight·L⁻ (§4.2).
- "The downside is that the resulting Likra model can not be easily used for generation, so all our testing is done on multiple-choice benchmarks" (§1).

## Setting (§3)
- Mistral-7B-v0.1 on ARC-Challenge: 6,615 training questions (1,172 used by lm-evaluation-harness excluded); evaluation with 25 few-shot examples. LoRA, 1 epoch ("more epochs did not help"), batch size 8, Adam, LR 10⁻⁴ (§3.1).
- Negatives for the main result: each question paired with an incorrect option chosen randomly from its multiple-choice options (§3.2).

## Results
- SFT with 0 to 6,615 positives raises accuracy from 60% to 66% (§3.1).
- "Increasing the number of negative training examples from 64 to 128 (only 8 extra updates with a batch size of 8) adds nearly 15% accuracy, whereas the SFT model averages less than 1% improvement per doubling of positive examples" (§3.2). The introduction states that during this critical phase each additional negative "can improve the accuracy of a model 10× more than each additional positive example."
- Table 1 (ARC / HellaSwag): Mistral-7B-v0.1 base .5998 / .8323; + SFT .6630 / .8468; Likra .8123 / .9633. Mistral-7B-Instruct-v0.3: .6365 / .8463; .6408 / .8360; .8063 / .9569. Llama-3.2-3B-Instruct: .5222 / .7312; .5486 / .7254; .7321 / .9071.
- Base-Likra (base model as positive head, no positives trained) has a learning curve similar to SFT-Likra (§4.1, Fig. 3).
- Negative-head weight: accuracy increases with the weight and peaks around 0.9-1.0 (§4.2, Fig. 4).
- Near-miss negatives (§4.3, Fig. 5): incorrect options of the same question work best; irrelevant answers from other ARC questions and unrelated text from other benchmarks are also beneficial (Likra reaches 70%, above SFT).
- Probability mass (§4.4, Fig. 2, Fig. 6): with positive-only SFT, "the incorrect answers do not seem to be sharply distinguished"; the negative head separates plausible incorrect answers from correct ones.
- The authors interpret the jump as negatives unlocking "latent knowledge that already exists in the pretrained model" (§3.2) (Interpretation).

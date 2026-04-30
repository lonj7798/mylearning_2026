<!-- scope: ChatGPT-rated filtering of Alpaca data to 9K samples; 5.7x faster, better quality
     deps: [[alpaca]]
     see-also: [[deita]], [[cherry-llm]], [[ifd]]
-->

# AlpaGasus: Training a Better Alpaca with Fewer Data
- **Core Insight:** Alpaca's 52K Self-Instruct set is riddled with incorrect or irrelevant responses; asking ChatGPT to rate each `(instruction, input, response)` tuple 0–5 and keeping only those ≥4.5 leaves ~9K samples that train a *better* model 5.7× faster.
- **Guideline:** When cleaning noisy synthetic SFT data, use a strong LLM (GPT-3.5-Turbo or later) as a one-shot scorer with a carefully worded rubric prompt; discard below a tight threshold (4.5/5); avoid marginal-quality samples — they drag the model down.
- **Authors:** Lichang Chen, Shiyang Li, Jun Yan, Hai Zhang, Fuxiao Liu, Chen Zhu, Tianyi Zhou, Tomas Pfister, Rajarshi Roy, Robert Zaigrajew, Heng Huang, Hongxia Jin (UMD + Samsung)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2307.08701
- **Relevant topics:** LLM-rated data filtering, Alpaca, quality scoring

## Abstract
AlpaGasus takes the 52K Alpaca instruction-tuning dataset, asks ChatGPT to score each instance on a 0–5 rubric for accuracy + helpfulness, and keeps only the 9K samples rated ≥4.5. Fine-tuning LLaMA-7B/13B on this filtered set yields a model that *beats* Alpaca on multiple test sets as judged by GPT-4 and in a controlled human eval. The 13B version matches >90% of its teacher (text-davinci-003) on test tasks. Training time drops 5.7× (80 min → 14 min for 7B). The paper is the canonical reference for "just ask GPT to rate" data filtering.

## Key Contributions
- Demonstrated that **a significant fraction of Self-Instruct-generated data is low quality** — the 52K → 9K shrinkage isn't an edge case.
- Provided a prompt-level rubric for ChatGPT rating that became a standard template.
- Showed **threshold sensitivity matters** — 4.5/5 is better than 4.0/5; marginal samples hurt.
- Triggered the data-filter research thread that includes Cherry-LLM, DEITA, Superfiltering, LESS.

## Key Figures/Tables to Study
- **Rating histogram** — long tail of low-quality samples in Alpaca.
- **Performance-vs-threshold sweep** — 4.5 > 4.0 > 3.5.
- **Table: AlpaGasus-7B/13B vs Alpaca** on GPT-4-judged win rates.

## Scoring prompt (paraphrased)
ChatGPT is given the `(instruction, input, response)` triple and asked to rate *accuracy* and *helpfulness* on a 5-point scale (0 = very bad, 5 = excellent). The specific prompt emphasizes:
- relevance of response to instruction,
- correctness of any factual claims,
- completeness,
- format appropriateness.
The final score is the average of the rated dimensions.

## Pipeline
1. For each Alpaca sample, send `(instruction, input, response)` + rubric prompt to ChatGPT.
2. Parse returned score.
3. Keep samples with score ≥ 4.5.
4. SFT on the kept subset.

## Quality / diversity evaluation
- 9K-AlpaGasus-7B beats 52K-Alpaca-7B on Vicuna-benchmark, Koala, Self-Instruct eval, WizardLM eval (all GPT-4 judged).
- AlpaGasus-13B achieves ≥90% of text-davinci-003 on test tasks.
- Training 5.7× faster at 7B.

## Risks + gotchas
- **Teacher's rubric bias** — ChatGPT's quality model is not aligned with all downstream tasks; scores for math/reasoning are less reliable.
- **Threshold tuning**: 4.5 is dataset-specific; higher thresholds can over-prune.
- **Single-axis scoring** — newer methods ([[deita]]) decompose into complexity × quality × diversity; AlpaGasus conflates them.
- **Evaluation via GPT-4-judge** inherits known judge biases (length bias, style bias); the controlled human eval partially mitigates but doesn't fully address.

## Connections
- Baseline filter against which [[cherry-llm]] / [[ifd]] / [[superfiltering]] / [[deita]] / [[less]] compare themselves.
- Conceptually simplest of the family: no gradients, no custom scorer, just API.
- Feeds into the later LLM-as-judge research thread for preference data — [[ultrafeedback]] scales this pattern up 10×.

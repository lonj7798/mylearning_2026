<!-- scope: synthetic preference data — UltraFeedback dataset, judge rubric, and downstream reward/critique models
     deps: [[ultrafeedback-construction]]
     see-also: [[rlcd]], [[west-of-n]], [[direct-judgement-preference]]
-->

# UltraFeedback: Boosting Language Models with Scaled AI Feedback
- **Core Insight:** Large-scale AI-generated preference data becomes genuinely useful when three things are diversified at once: prompts, candidate responses, and the judging rubric. UltraFeedback turned that into an open dataset and made synthetic preference learning practical for the open-model ecosystem.
- **Guideline:** If you want reusable preference data, do not just produce chosen/rejected pairs. Keep the multi-aspect scores and natural-language critiques so the same corpus can support DPO pairs, reward-model training, critique-model training, and best-of-N selection.
- **Authors:** Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, Maosong Sun
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2310.01377
- **Relevant topics:** synthetic preferences, GPT-4 as judge, reward models, critique models, open alignment data

## Abstract
UltraFeedback studies whether high-quality AI feedback can replace or augment scarce human preference labels for alignment. The paper releases a large open corpus where diverse prompts are answered by multiple models and then rated by GPT-4 across several dimensions, with both numeric scores and natural-language explanations. Those annotations are then used to train aligned chat models, reward models, and critique models.

## Key Contributions
- Releases a large open **AI-feedback preference corpus** built from diverse prompts and heterogeneous model outputs.
- Uses a **multi-aspect rubric** rather than a single scalar preference, preserving richer supervision.
- Trains both **UltraRM** and **UltraCM**, showing the same dataset can support reward and critique modeling.
- Demonstrates that open models fine-tuned with the resulting data become competitive with the strongest open chat models of the period.

## Key Figures/Tables to Study
- **Dataset statistics table:** prompt count, response count, and aspect-label count clarify why the dataset became a default preference source.
- **Alignment pipeline figure:** follow the path from raw AI feedback to RM / CM / PPO or best-of-N.
- **Benchmark table for aligned chat models:** this shows the practical payoff of synthetic feedback, not just the dataset scale.

## Technical Details
- **Prompt side:** the dataset draws on heterogeneous instruction sources so the preference signal is not tied to a single benchmark style.
- **Response side:** multiple different generator models answer each prompt, creating separable quality gaps for the judge.
- **Judge side:** GPT-4 provides aspect-wise ratings plus written feedback; the language feedback is as important as the scalar label because it supports critique modeling.
- **Artifacts released:** raw scored responses, binarized preference subsets, reward-model training targets, and critique-model supervision.
- **Downstream use:** the paper trains:
  - a chat model aligned with AI feedback
  - a reward model for ranking
  - a critique model that explains why one response is better
- **Practical lesson:** UltraFeedback is not just one dataset; it is a pattern for manufacturing reusable alignment supervision from a strong judge.

## Risks + gotchas
- Judge bias becomes part of the dataset and can propagate into every downstream model trained on it.
- Binarizing the corpus throws away a lot of useful structure; many later projects use only the pairwise subset and lose the richer critique signal.
- If the model pool is too homogeneous, the preference gap shrinks and the labels become less informative.

## Connections
- Construction details live in [[ultrafeedback-construction]]; this page is the dataset-and-downstream-model view.
- Contrasts with [[rlcd]], which creates preference pairs without an external judge.
- Connects to [[direct-judgement-preference]] and later judge-distillation work that internalizes the GPT-4 evaluator.

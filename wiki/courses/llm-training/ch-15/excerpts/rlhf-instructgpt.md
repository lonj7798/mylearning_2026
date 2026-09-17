---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlhf-instructgpt.md
source_url: https://arxiv.org/abs/2203.02155
primary_version: arXiv:2203.02155v1 (2022-03)
created_at: "2026-04-23"
revised_at: "2026-09-15 (rewritten from the primary text; the earlier version of this excerpt contained unsupported claims)"
---

# Excerpt: Training language models to follow instructions with human feedback (InstructGPT), human-data sections

Authors: Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, et al. (OpenAI). Source type: paper. Read in the v1 PDF text on 2026-09-15 for ch-15 §1, §2, §4, §7 and Recipe. Training hyperparameters for PPO are covered in ch-38, not here.

## Prompt source and splits (§3.2, App. A)
- Prompts were submitted to an earlier InstructGPT model on the API Playground; production API data was not used (§3.2).
- Deduplication by long common prefix; at most 200 prompts per user ID; train, validation and test splits made by user ID; PII filtered in the training split (§3.2).
- "Our dataset is over 96% English" (§3.3).
- Table 6 (number of prompts):

| Split | SFT labeler | SFT customer | RM labeler | RM customer | PPO customer |
|---|---|---|---|---|---|
| train | 11,295 | 1,430 | 6,623 | 26,584 | 31,144 |
| valid | 1,550 | 103 | 3,488 | 14,399 | 16,185 |

- Table 1 (use-case categories of the API prompt dataset): Generation 45.6%, Open QA 12.4%, Brainstorming 11.2%, Chat 8.4%, Rewrite 6.6%, Summarization 4.2%, Classification 3.5%, Other 3.5%, Closed QA 2.6%, Extract 1.9%.
- §4.1: classification and QA are "about 18%" of use; open-ended generation and brainstorming "about 57%".

## Labelers (§3.4, App. B.1)
- "a team of about 40 contractors on Upwork and through ScaleAI" (§3.4).
- Screening criteria (App. B.1): (1) agreement with researchers on sensitive speech flagging; (2) agreement with researcher rankings of model completions; (3) sensitive demonstration writing rated on a 1-7 Likert scale; (4) self-assessed ability to identify sensitive speech for different groups. "soft cutoffs at 75% agreement on sensitive speech flagging and comparisons, and a 6/7 demonstration score."
- Onboarding, "detailed instructions for each task", and "answer labeler questions in a shared chat room" (§3.4).
- Instructions "evolved over the course of the project" and were amended "when they were confusing or inconsistent" (App. B.2).
- Priority order: during training data labeling, helpfulness to the user was the most important criterion; in final evaluations, truthfulness and harmlessness were prioritized (§3.4, App. B.2). App. B.2 notes the risk that models "could over-generalize and refuse innocuous instructions".

## Agreement (§3.4, §4.1, §5.2, §5.3)
- "training labelers agree with each-other 72.6 ± 1.5% of the time, while for held-out labelers this number is 77.3 ± 1.3%. For comparison, in the summarization work of Stiennon et al. (2020) researcher-researcher agreement was 73 ± 4%." (§3.4)
- Held-out labelers "do not undergo a screening test" (§3.4).
- RM generalization across labeler groups: 5 labeler groups, 5-fold cross-validation, 3 seeds: "accuracy of 69.6 ± 0.9% on predicting the preferences of labelers in the held-out group, a small decrease from their 72.4 ± 0.4% accuracy on ... their training set" (§4.1).
- §5.2: labelers are "mostly English-speaking people living in the United States or Southeast Asia"; "we found the inter-labeler agreement to be about 73%".
- §5.3: "most comparisons are only labeled by 1 contractor for cost reasons"; "In cases of disagreement, aligning to the average labeler preference may not be desirable."

## Comparison collection and RM loss (§3.5)
- Labelers rank K = 4 to K = 9 responses per prompt, giving C(K,2) comparisons.
- "if we simply shuffle the comparisons into one dataset, a single pass over the dataset caused the reward model to overfit"; all C(K,2) comparisons from a prompt are trained as a single batch element.
- Eq. 1: loss(θ) = −(1/C(K,2)) E_(x,y_w,y_l)~D [log σ(r_θ(x,y_w) − r_θ(x,y_l))].
- Only 6B RMs were used (§3.5).

## Coverage result (§4.1)
- 175B InstructGPT outputs were preferred over the FLAN-finetuned 175B model 78 ± 4% of the time and over the T0-finetuned model 79 ± 4%. The authors attribute this to public NLP datasets covering tasks that are easy to evaluate automatically and having low input diversity.

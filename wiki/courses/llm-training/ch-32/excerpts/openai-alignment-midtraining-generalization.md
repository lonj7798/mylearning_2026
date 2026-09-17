---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: official blog post (planned library card blogs/openai-alignment-midtraining-generalization.md; not present on 2026-09-15)
source_url: https://alignment.openai.com/how-far-does-alignment-midtraining-generalize/
created_at: "2026-09-15"
---

# Excerpt: How far does alignment midtraining generalize?

**Authors:** Tomek Korbak, Cameron Raymond, Micah Carroll, Marcus Williams, Mikita Balesni, Alan Guo, et al. (OpenAI). Published 2026-03-27. Source type: official blog; reliability: official. The post calls its results "preliminary" and "early negative results".

## Setup ("Setup")

- An o4-mini-sized model after standard pretraining is fine-tuned on the midtraining subset of the Tice et al. (2026) dataset: papers, news articles, lecture transcripts, textbook chapters, science-fiction passages and movie plot summaries in which AIs choose an aligned or a misaligned action, "including why the AI avoided the alternative".
- Three runs: alignment midtraining on 230k documents (~340M tokens) from the positive split; misalignment midtraining on 230k documents (~340M tokens) from the negative split; no extra midtraining. The midtraining data "involves exclusively the Tice et al. dataset (i.e. we do not mix in any other datasets)".
- All three then receive the same reasoning post-training: SFT and RL with verifiable rewards on math, coding and reasoning QA; no safety training.
- GPQA Diamond and CodeForces 2408 scores do not differ significantly between runs.

## Results

- QA evals from Tice et al. (close to the midtraining distribution): alignment midtraining > baseline > misalignment midtraining, with smaller differences than Tice et al. report; smaller again on the Textbook-Sourced split ("QA evals").
- Chat evals (Model Spec compliance, production deception, alignment generalization benchmark, emergent misalignment, blackmail, oversight sabotage, alignment questions): on all but two evals the aligned and misaligned midtraining runs score similarly; both score at or above the baseline; alignment improves over RL steps on all chat evals except Blackmail ("Chat evals").
- Agentic evals (Apollo scheming subset, impossible coding tasks): averaged across evals, the three runs are "not significantly different" ("Agentic evals", Figure 1).
- Summary sentence: misalignment-midtraining effects near the training distribution "tend to disappear once the model undergoes reasoning post training", and alignment midtraining "does not generalize to a different, more realistic chat and agent alignment evaluations".

## Stated limits and hypotheses ("Discussion and limitations")

- Only one data distribution (fictional scenarios) was tested; 340M tokens may be too few; earlier insertion points may persist better through RL; effects may reflect eval awareness or salience rather than alignment.

## Used in

ch-32 §8, Negative samples section.

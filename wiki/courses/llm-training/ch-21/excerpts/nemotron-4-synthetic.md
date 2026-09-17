---
chapter: ch-21
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/nemotron-4-synthetic.md
source_url: https://arxiv.org/abs/2406.11704
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: Nemotron-4 340B Technical Report — synthetic alignment data

This excerpt was rewritten on 2026-09-15 to match the verified library cards `nemotron-4-synthetic` and
`nemotron-4-synthetic-recipe` (checked 2026-09-14 against arXiv:2406.11704v2). The April 2026 version contained a
reconstructed loop (best/worst of K samples per prompt), a six-family task list, a "frozen anchor" explanation, and
staging rationales that the report does not state; they are removed.

## Human versus synthetic data (§3.2)
"we relied on only approximately 20K human-annotated data (10K for supervised fine-tuning, 10K Helpsteer2 data for
reward model training and preference fine-tuning), while our data generation pipeline synthesized over 98% of the
data used for supervised fine-tuning and preference fine-tuning."

## Prompt synthesis (§3.2.1)
- Generator: Mixtral-8x7B-Instruct-v0.1, chosen for its permissive license.
- Tasks with separate pipelines: open Q&A, writing, closed Q&A (from C4 documents), math and coding.
- Topics: the generator lists macro-topics, then subtopics per macro-topic; with manually collected topics, 3K topics
  in total. Math and coding prompts are seeded with 12K Python-related and 17K math-related keywords.
- Instruction-following prompts append verifiable format constraints; two-turn prompts take the first user turn
  from ShareGPT. LMSYS-Chat-1M prompts are also used; Mixtral responses to synthetic prompts have mean reward-model
  helpfulness 3.24 vs 3.04 for LMSYS prompts, which the report reads as LMSYS prompts being harder (Figure 3).

## Judges and filters
- **Reward model (§3.1):** Nemotron-4-340B-Reward is Nemotron-4-340B-Base with the final softmax replaced by a linear
  projection to five HelpSteer attributes, trained on 10K HelpSteer2 examples; RewardBench overall 92.0 (Table 4).
- **Preference ranking (§3.2.3):** ground truth or a Python verifier where available; otherwise LLM-as-judge in early
  iterations and Reward-Model-as-judge later (Chat-Hard accuracy 0.87 vs 0.54).
- **Dialogues (§3.2.2):** three turns, greedy decoding; dialogues below a reward-model score threshold are dropped.
- **Weak-to-strong (§3.2.4):** Mixtral-8x7B-Instruct-v0.1 data trains 340B-Interm-1; that model generates the next round.

## Staged SFT (§3.3.1)
"learning multiple behaviors concurrently can sometimes lead to conflicts between them ... We observe this phenomenon
particularly strongly in coding tasks, where adjusting the sampling weights for the data blend fails to align the
model to all coding tasks." The two-stage strategy "yields superior results across all downstream tasks" (no
single-stage numbers are given).
- Code SFT: Genetic Instruct (self-instruction and WizardCoder mutations from a limited number of seeds, with an LLM
  fitness check); about 800K samples after de-duplication and filtering; 1 epoch; constant LR 3e-7; global batch 128.
- General SFT: 200K-sample blend including 2% of Code SFT samples "to mitigate the risk of forgetting"; 3 epochs;
  batch 128; LR searched in [1e-7, 5e-7]; loss on assistant turns only.

## Stage effects (Table 6)
| After stage | MT-Bench (GPT-4-Turbo) | MMLU 0-shot | GSM8K 0-shot | HumanEval 0-shot | IFEval prompt-strict |
|---|---|---|---|---|---|
| Code SFT | 6.79 | 72.2 | 77.6 | 70.7 | 46.4 |
| + General SFT | 7.99 | 78.3 | 87.9 | 66.5 | 61.4 |
| + DPO | 7.90 | 78.4 | 88.5 | 67.1 | 61.7 |
| + RPO (iteration 1, 2, 3) | 8.21, 8.31, 8.22 | 78.5, 78.6, 78.7 | 91.1, 91.8, 92.3 | 70.7, 68.3, 73.2 | 78.2, 79.9, 79.9 |

The base model's HumanEval is 57.3 (text introducing Table 6).

## Negative signals (§3.3.2, §3.2.5)
- Under DPO the report observed both chosen and rejected likelihoods falling, added an SFT loss on chosen responses,
  and moved to RPO, whose target is the reward-model score gap.
- Refusal responses for tasks the model cannot do are ordinary SFT targets.

## Verification
- Values match the verified cards; the §3.2 and §3.3.1 quotations were re-read in the cached arXiv text on 2026-09-15.

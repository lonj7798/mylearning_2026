---
chapter: ch-45b
course: llm-training
phase: read
excerpt_of: arXiv:2601.15625 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2601.15625
created_at: "2026-09-15"
---

# Excerpt: Robust Tool Use via Fission-GRPO — Learning to Recover from Execution Errors

- **Authors:** Zhiwei Zhang, Fei Zhao, Rui Wang, Zezhong Wang, Bin Liang, Jiakang Wang, Yao Hu, Shaosheng Cao,
  Kam-Fai Wong
- **Year:** 2026 (arXiv v1 2026-01-22)
- **Source type:** paper
- **Used in:** [[read]] §5, Negative samples

## Abstract (verbatim)
"Large language models (LLMs) can call tools effectively, yet they remain brittle in multi-turn execution: after
a tool-call error, smaller models often fall into repetitive invalid re-invocations instead of interpreting the
feedback and recovering. This failure mode persists because current training paradigms do not explicitly teach
models how to recover from execution errors. In particular, standard reinforcement learning (RL) collapses rich
failure experience into sparse negative rewards, while pre-collected error-correction datasets become mismatched
to the policy's evolving failure modes. To bridge this gap, we propose Fission-GRPO, a framework that converts
execution errors into on-policy corrective supervision within the RL training loop. Our core mechanism fissions
each failed trajectory into a new training instance by augmenting it with diagnostic feedback from a fine-tuned
Error Simulator, then resampling multiple recovery rollouts on-policy. This enables the model to learn from the
precise errors it makes during exploration, rather than from static, pre-collected error cases. On BFCL v4
Multi-Turn, Fission-GRPO improves the error recovery rate of Qwen3-8B by 5.7% absolute and overall accuracy by
4.0% (from 42.75% to 46.75%), outperforming both RL baselines and specialized tool-use agents. The method further
generalizes to TAU-Bench and TAU2-Bench, achieving leading results across most settings with gains up to +17.4%."

## Position in the chapter's taxonomy
Under the course's four senses of "negative" ([[read]] §6 standard), Fission-GRPO converts a failure into
**negative as content**: the failed prefix plus diagnostic feedback becomes the input of a new on-policy training
instance, and the resampled recovery continuations are trained as ordinary positives or negatives by their own
reward. This is the opposite treatment from the masking used by [[deepswe]] and [[simpletir]], which remove the
failed trajectory from the gradient entirely.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2601.15625 (abstract page).
- Not extracted here: the Error Simulator's training data and size, the number of recovery rollouts per fission,
  the RL hyperparameters, and the per-setting TAU-Bench numbers behind "up to +17.4%". Only the abstract's own
  numbers are cited in the chapter.

---
chapter: ch-31
course: llm-training
phase: read
excerpt_of: arXiv:2602.15763v2 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2602.15763
created_at: "2026-09-15"
---

# Excerpt: GLM-5: from Vibe Coding to Agentic Engineering

- **Authors:** GLM-5 Team, Zhipu AI & Tsinghua University (byline)
- **Year:** 2026 (arXiv v2 2026-02-24)
- **Source type:** official technical report (code and models: https://github.com/zai-org/GLM-5)
- **Used in:** ch-31 §5.3, Negative samples and negative feedback, Recipe, Generalization lens

## Post-training pipeline (§1, §3)

- **Order.** Multi-task SFT, then Reasoning RL, Agentic RL, and General RL, then on-policy cross-stage distillation (§3 introduction, Fig. 5).
- **Stated placement of distillation.** §1 says on-policy cross-stage distillation was used "throughout this process to prevent catastrophic forgetting"; §3 calls it "the final refinement" and §3.5 "the final stage". The two descriptions are not reconciled in the text.
- **Relation to GLM-4.5.** GLM-4.5 used iterative self-distillation and outcome supervision for agents; GLM-5 adds asynchronous agent RL algorithms (§1).

## SFT details relevant to self-generated data (§3.1)

- SFT covers General Chat, Reasoning, and Coding & Agent data; maximum context 202,752 tokens.
- Logical reasoning data is synthesized with verifiable problems and rejection sampling; math and science problems are kept only if they are challenging for GLM-4.7.
- Coding and agent SFT data is improved "using expert reinforcement learning and rejection sampling". "Erroneous segments within trajectories are retained but masked out in the loss function, allowing the model to learn error correction behaviors without reinforcing incorrect actions."

## Reasoning RL backbone (§3.2)

GRPO with the IcePop technique for training–inference mismatch; the KL regularization term is removed; π_train is used for gradient updates and π_infer for sampling (Eq. 1).

## On-policy cross-stage distillation (§3.5)

- **Motivation.** "Sequentially optimizing for distinct objectives can lead to the cumulative degradation of previously acquired capabilities."
- **Teachers and prompts.** Final checkpoints of preceding stages are teachers; training prompts are sampled from the corresponding teachers' RL training sets and "mixed in appropriate proportions".
- **Objective.** The advantage in Eq. 1 is replaced by Â_{i,t} = sg[ log( π_teacher^infer(y_{i,t} | x, y_{i,<t}) / π_θ^train(y_{i,t} | x, y_{i,<t}) ) ] (Eq. 2), where sg is stop-gradient. Teacher logits are currently fetched from the inference engine.
- **Settings.** GRPO group size 1 and batch size 1,024, because the advantage comes from the teacher gap rather than a group of samples.
- **Evidence.** No ablation or number isolates the effect of this stage.

## Slide-generation case (§4.2.5; rejection sampling and masking)

For a slide-generation expert, RL reward functions are reused as a rejection-sampling filter with best-of-N selection; defective pages inside otherwise good trajectories are masked rather than discarded. The proportion of pages with a strict 16:9 aspect ratio rose from 40% to 92%, and human evaluation against GLM-4.5 gave an overall win rate of 67.5%. The report does not separate the contribution of rejection sampling and masking from RL.

## Verification

- Checked on 2026-09-15 against: https://arxiv.org/abs/2602.15763 (v2, 2026-02-24), §1, §3.1–3.5, and §4.2.5.
- Not reported by the source: mixture proportions for distillation prompts; number of distillation steps; per-stage capability numbers before and after distillation; SFT and RL learning rates.

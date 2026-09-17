---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: arXiv:2602.15763v2 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2602.15763
created_at: "2026-09-15"
---

# Excerpt: GLM-5: from Vibe Coding to Agentic Engineering

- **Authors:** GLM-5 Team, Zhipu AI & Tsinghua University (byline)
- **Year:** 2026 (arXiv v2 2026-02-24)
- **Source type:** official technical report (models and code: https://github.com/zai-org/GLM-5)
- **Used in:** ch-35 §4.3, Negative samples and negative feedback, Recipe

## Model
744B total parameters, 40B active; GLM-4.5 had 355B total and 32B active (§2.1).

## Where teacher data enters (§1, §3)
- Order: multi-task SFT → Reasoning RL → Agentic RL → General RL → on-policy cross-stage distillation (§3 introduction, Figure 5).
- §1 says on-policy cross-stage distillation was used "throughout this process to prevent catastrophic forgetting". §3 calls it "the final refinement", and §3.5 performs it "as the final stage". The report does not reconcile the two descriptions.

## SFT (§3.1)
- Categories: General Chat; Reasoning; Coding & Agent. Maximum SFT context 202,752 tokens.
- Logical reasoning: verifiable problems, data synthesized with rejection sampling.
- Math and science: "a difficulty-based filtering process is applied, retaining only problems that are challenging for the GLM-4.7 model."
- Coding and agent: SFT data improved "using expert reinforcement learning and rejection sampling. Erroneous segments within trajectories are retained but masked out in the loss function, allowing the model to learn error correction behaviors without reinforcing incorrect actions."

## Reasoning RL data (§3.2)
Difficulty filtering keeps problems "that GLM-4.7 solves correctly only rarely or fails consistently, while remaining solvable by stronger teacher models (e.g., GPT-5.2 xhigh and Gemini 3 Pro Preview)". The four domains (math, science, code, tool-integrated reasoning) are kept roughly balanced.

## On-policy cross-stage distillation (§3.5)
- Motivation: "sequentially optimizing for distinct objectives can lead to the cumulative degradation of previously acquired capabilities."
- Teachers: "the final checkpoints from the preceding training stages"; prompts "sampled from the corresponding teachers' RL training sets and mixed in appropriate proportions".
- Objective: the GRPO advantage in Eq. 1 is replaced by Â_{i,t} = sg[ log( π_teacher^infer(y_{i,t} | x, y_{i,<t}) / π_θ^train(y_{i,t} | x, y_{i,<t}) ) ] (Eq. 2). sg is stop-gradient. Teacher logits are fetched from the inference engine.
- Settings: group size 1 and batch size 1,024, "because it is no longer necessary to maintain a large group of samples per prompt to estimate advantages".

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2602.15763 (v2, 2026-02-24): §1, §2.1, §3.1, §3.2, §3.5.
- Not reported by the source: mixture proportions of distillation prompts; number of distillation steps; learning rates; any benchmark number before versus after the distillation stage.

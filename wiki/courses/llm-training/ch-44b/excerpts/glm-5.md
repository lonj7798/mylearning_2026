---
chapter: ch-44b
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
- **Used in:** [[read]] §1, §5, Distillation, Recipe

## Stage order (§1, §3)
- §1: "We implemented a sequential Reinforcement Learning pipeline—starting with Reasoning RL, followed by Agentic RL, and finishing with General RL. Crucially, we utilized On-Policy Cross-Stage Distillation throughout this process to prevent catastrophic forgetting".
- §3.5 performs on-policy cross-stage distillation "as the final stage ... to swiftly recover the skills acquired in earlier SFT and RL stages (Reasoning RL and General RL)". The two descriptions of placement are not reconciled in the text.

## Reasoning RL (§3.2)
- GRPO with IcePop for training-inference mismatch; the KL term is removed. Loss (Eq. 1) multiplies the clipped PPO term by pop(ρ_{i,t}, 1/β, β), where ρ_{i,t} = π_old^train / π_old^infer and pop(·) returns ρ inside [1/β, β] and 0 otherwise.
- Settings: β = 2, ε_low = 0.2, ε_high = 0.28, fully on-policy, group size 32, batch size 32.
- Mixed-domain reasoning RL over mathematics, science, code, and tool-integrated reasoning (TIR). Difficulty filter keeps problems "that GLM-4.7 solves correctly only rarely or fails consistently, while remaining solvable by stronger teacher models (e.g., GPT-5.2 xhigh and Gemini 3 Pro Preview)". Domain- and source-specific judge models or evaluation systems give binary outcome rewards. "We keep the overall mixture roughly balanced across the four domains, and consistently observe stable and significant gains in each domain under the mixed RL setting." No per-domain numbers are printed.

## General RL (§3.4)
- Three objective dimensions: foundational correctness (instruction-following failures, logical inconsistencies, factual inaccuracies, hallucinations, disfluencies), emotional intelligence, and task-specific quality (writing, text processing, subjective and objective QA, role-playing, translation).
- Hybrid reward system: rule-based rewards ("precise and interpretable ... limited to aspects expressible as deterministic rules"), outcome reward models ("low-variance signals and high training efficiency, but are more susceptible to reward hacking"), and generative reward models ("more robust to such exploitation, but tend to exhibit higher variance").
- Expert human-authored responses are introduced as "stylistic and qualitative anchors", motivated by the observation that "purely model-generated optimization tends to converge toward recognizably 'model-like' patterns—often verbose, formulaic".
- Mixing weights among reward types are not reported.

## On-policy cross-stage distillation (§3.5)
- Motivation: "sequentially optimizing for distinct objectives can lead to the cumulative degradation of previously acquired capabilities."
- Teachers are "the final checkpoints from the preceding training stages"; prompts are "sampled from the corresponding teachers' RL training sets and mixed in appropriate proportions".
- Advantage in Eq. 1 is replaced by Â_{i,t} = sg[ log( π_teacher^infer(y_{i,t} | x, y_{i,<t}) / π_θ^train(y_{i,t} | x, y_{i,<t}) ) ] (Eq. 2); sg is stop-gradient. Teacher logits are fetched from the inference engine.
- Group size 1 and batch size 1,024, because "the advantage is computed directly from the gap with the teacher models instead".

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2602.15763 (v2, 2026-02-24): §1, §3.2, §3.4, §3.5.
- Not reported by the source: mixing proportions of distillation prompts; distillation steps; reward-type weights in General RL; any benchmark number before versus after distillation or General RL.

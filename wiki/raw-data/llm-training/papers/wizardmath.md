<!-- scope: Evol-Instruct on math problems + RLEIF (Reinforcement Learning from Evol-Instruct Feedback)
     deps: [[evol-instruct]]
     see-also: [[wizardcoder]], [[math-shepherd]], [[prm800k]]
-->

# WizardMath: Empowering Mathematical Reasoning via Reinforced Evol-Instruct (RLEIF)
- **Core Insight:** Evolving math problems in *two* directions — downward to grade-school and upward to competition-hard — combined with reinforcement on both answer correctness *and* process-reward signals, closes most of the open-vs-frontier gap on math reasoning.
- **Guideline:** For math-specialist SFT+RL, take GSM8K/MATH seeds, apply bidirectional Evol-Instruct (make-easier + make-harder), generate candidate solutions, train an Instruction Reward Model (IRM) on instruction quality and a Process Reward Model (PRM) on step correctness, then PPO against IRM·PRM.
- **Authors:** Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, Yansong Tang (Microsoft + Tsinghua)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2308.09583
- **Relevant topics:** math reasoning, Evol-Instruct, process reward models, RLEIF

## Abstract
WizardMath applies the Evol-Instruct paradigm (from [[evol-instruct]]) to math problems in both directions: "downward evolution" produces easier grade-school-style variants; "upward evolution" produces harder, multi-step competition-style variants. The evolved dataset is used for SFT, then RLEIF — Reinforcement Learning from Evol-Instruct Feedback — trains an Instruction Reward Model plus a Process Reward Model and optimizes the policy with PPO against their product. WizardMath-70B surpasses GPT-3.5-Turbo, Claude 2, Gemini Pro, and GPT-4-early on GSM8K / MATH at release; the 7B-Mistral version beats prior open SOTA with higher data efficiency.

## Key Contributions
- **Math Evol-Instruct**: bidirectional (downward + upward) evolution operators specialized for math.
- **RLEIF**: joint optimization against an instruction reward model (IRM) and a process reward model (PRM).
- Public release of WizardMath 7B/13B/70B models.
- Established the "Evol + RLEIF" template for specialist vertical RL (later imitated in code and reasoning domains).

## Key Figures/Tables to Study
- **Figure illustrating bidirectional evolution** — a GSM8K seed, its simplified variants, and its MATH-hard variants.
- **Table: GSM8K / MATH scores across WizardMath sizes vs closed models.**
- **Ablation: Evol only vs RLEIF full** — RLEIF contributes several percentage points.

## Synthesis pipeline (REQUIRED — be concrete)
- **Seed input:** GSM8K + MATH training splits (and related open math datasets).

- **Generation step(s):**
  - **Downward evolution operators** (per Math Evol-Instruct): *reduce constraints*, *replace concepts with simpler ones*, *shorten the chain*, *make arithmetic easier*.
  - **Upward evolution operators**: *add constraints*, *compose with another concept*, *increase reasoning depth*, *require multiple solution steps*.
  - Each seed is evolved multiple times per direction to yield a broader difficulty spectrum.
  - Solutions generated via teacher LLM sampling with step-by-step format.

- **Filtering/rescoring:** answer-verifier (exact match for GSM8K-style; symbolic equivalence for MATH) rejects incorrect solutions; duplicate instruction filter.

- **RLEIF step:**
  - **IRM** — trained on pairs of evolved instructions to score instruction quality/evolution success.
  - **PRM** — trained on step-level labels (similar to [[prm800k]] lineage) to score partial-solution correctness.
  - PPO objective: maximize `IRM(instruction, response) × PRM(response_steps)` with KL penalty to SFT reference.

- **Output shape:** tens of thousands of evolved math problems with step-by-step solutions (not all publicly released). WizardMath model checkpoints released.

- **Teacher model(s):** GPT-3.5/GPT-4-class for evolution + solution generation; later iterations use WizardMath itself.

- **Cost estimate:** not disclosed.

## Quality / diversity evaluation
- WizardMath-70B: GSM8K ~81.6 / MATH ~22.7 at release — above GPT-3.5-Turbo, Claude 2.
- WizardMath-Mistral-7B: strong pareto point — beats LLemma-34B with fewer params at release.
- RLEIF ablation: +~3 points on GSM8K, +~2 on MATH over SFT-only.

## Risks + gotchas
- **Reward hacking**: PRM signal is noisy; overtraining on PRM can produce surface-correct chains with wrong answers.
- **Benchmark saturation** since release — newer 7B reasoning models now exceed WizardMath-70B.
- **License:** WizardLM-family data has had access restrictions; check before redistribution.
- **Process-reward labeling scale**: PRM data is expensive; WizardMath uses model-labeled step correctness, inheriting the teacher's error modes.

## Connections
- Direct math-specialized descendant of [[evol-instruct]].
- Companion to [[wizardcoder]] (Evol-Instruct on code).
- PRM component in the [[prm800k]] / [[math-shepherd]] / [[lets-verify]] lineage.
- Precursor to later math-RL recipes: [[grpo]] / [[rlvr-tulu3]] / [[deepseek-r1]] use verifiable-reward but drop the IRM and the evol-based data construction.

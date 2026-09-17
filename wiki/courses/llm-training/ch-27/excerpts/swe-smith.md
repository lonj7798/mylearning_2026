---
chapter: ch-27
course: llm-training
phase: read
excerpt_of: "SWE-smith: Scaling Data for Software Engineering Agents (arXiv:2504.21798v2)"
source_url: https://arxiv.org/abs/2504.21798
created_at: "2026-09-15"
note: "No library card exists for this source as of 2026-09-15; this excerpt is taken from the primary text."
---

# Excerpt: SWE-smith — synthesized SWE tasks across 128 repositories

**Authors:** John Yang, Kilian Lieret, Carlos E. Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, et al. (Stanford University; Princeton University; Alibaba Qwen). arXiv v1 2025-04, v2 2025-05-21.

## Pipeline (§2.1)
1. Build one execution environment per repository: SWE-agent runs on the latest commit for at most 100 steps to install the codebase and run its tests; a person verifies the instructions (about 7 minutes per repository), checks that more than 80% of existing tests pass, and creates a Docker image. Candidates are the 5,000 most downloaded PyPI packages (as of 2024-11-18) with at least 1,000 GitHub stars; the 12 SWE-bench test repositories are excluded.
2. Create task candidates with four strategies: LM Generation (modify or rewrite a function), Procedural Modification (13 AST transformations), Invert PRs ("PR Mirror"), and Combine Bugs.
3. Keep only candidates that break one or more existing tests.
4. Generate issue text with an LM.

## Dataset (§2.2, Table 2)
- 50k task instances (50,137) from 128 repositories; 381 instances per repository on average.
- Construction cost $1,360 ($1,000 bug generation, $160 repository installation, $200 issues for 10K bugs).
- Storage 295 GB, because tasks from one repository share an environment; the authors estimate 50–150 TB for the same number of SWE-bench-style images.

## Training and results (§3–4, App. F.1, Table 3)
- 5,016 expert trajectories from SWE-agent with claude-3-7-sonnet-20250219 (at most 75 steps, $2.00 cost limit), filtered to resolved instances; student inference uses the same 75-step limit at temperature 0.0.
- Qwen2.5-Coder-Instruct 7B and 32B, torchtune full fine-tuning, LR 5e-5, maximum 3 epochs, max context 32,768, 2–8 H100 GPUs.
- SWE-bench Verified pass@1: SWE-agent-LM-32B 40.2 (Lite 30.7); SWE-agent-LM-7B 15.2; SWE-Gym-32B 20.6; R2E-Gym-32B 34.4 (Table 3, no verifiers or multiple attempts).

## Ablations relevant to generality (§4.1)
- Repository count (Fig. 5): 700 trajectories from Procedural Modification tasks sampled from 4, 25, 50, 100 repositories; resolve rate 10.3, 11.5, 12.9, 15.1, described as approximately logarithmic.
- Repository specialization (Fig. 4): SymPy subset of Verified created after 2022-01-01 (22 instances) and 700 SymPy trajectories. 7B: 100-repository model 15.3 (Verified without SymPy) / 13.6 (SymPy) vs SymPy-only 14.0 / 21.2. 32B: SWE-agent-LM-32B 40.2 / 33.3 vs further SymPy fine-tuning 38.3 / 42.4.
- Bug strategy (Table 4, 1,000 instances each): PR Mirror 9.2, LM Rewrite 8.8, Procedural 8.6, LM Modify 5.7.
- Issue text (Table 5, 600 PR Mirror instances capped at 259 trajectories): LM-generated 7.7, original 7.8, fail-to-pass test 7.3, fixed template 6.4. Fixed templates gave 31% fewer unique actions (379 vs 550). With test-based issues the student attempted to reproduce the bug in 127 of 500 runs vs 379 with LM-generated issues.
- Difficulty (App.): SFT sets at difficulty scores 2/4/6/8 gave 12.4 / 10.8 / 13.6 / 12.2, with no strong correlation.

## Failure analysis (§4.2)
- More than 25% of SWE-agent-LM-32B trajectories contain a repeated action sequence of length ≥ 10, vs under 4% for Claude 3.7 Sonnet; a length-10 repetition corresponds to an 89% failure probability.
- SWE-bench Multilingual (300 instances, 9 additional languages) is introduced to test transfer beyond Python (§3).

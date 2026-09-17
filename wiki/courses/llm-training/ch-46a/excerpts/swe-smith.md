---
chapter: ch-46a
course: llm-training
phase: read
excerpt_of: "SWE-smith: Scaling Data for Software Engineering Agents (arXiv:2504.21798v2, 2025-05-21)"
source_url: https://arxiv.org/abs/2504.21798
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug swe-smith). Values below were read from the v2 PDF on 2026-09-15."
---

# Excerpt: SWE-smith — environment first, then synthesized tasks

**Authors:** John Yang, Kilian Lieret, Carlos E. Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, et al. (Stanford University; Princeton University; Independent; Alibaba Qwen).

## Design principle (§2)
> "The core principle of SWE-smith's collection strategy is to define an execution environment first, and then synthesize task instances within the environment. Conceptually, this is a simple inversion of SWE-bench's approach, which instead prioritizes identifying task instances, and then attempts to build an environment for each."

## Collection (§2.1)
- Environment: SWE-agent runs on the latest commit for at most 100 steps to install the repository and run its tests; the authors manually verify the instructions, check that more than 80% of existing tests pass, and build one Docker image per repository.
- Repository pool: the 5,000 most downloaded PyPI packages (2024-11-18), sorted by GitHub stars; packages with fewer than 1,000 stars and all 12 SWE-bench test repositories removed.
- Task candidates as `.diff` files from four strategies: LM Modify / LM Rewrite, Procedural Modification (AST transformations, Table 8), Combine Bugs, and PR Mirror (an LM reverts a PR's changes in the current version of the repository).
- Validation: > "only keep patches that break one or more existing, passing tests (referred to as Fail-to-Pass or F2P test(s))"; candidates whose test run exceeds two minutes are discarded.
- Issue text: an LM receives the patch, the source of a random F2P test, and the test output, and writes GitHub-issue-style text with reproduction code.
- Human labor: about 7 minutes per repository for installation parsing and about 1 minute for a test-output parser; "Creating SWE-smith took one author ∼20h of human labor."

## Statistics (§2.2, Tables 1-2)
| Strategy | Yield % | Instances | Cost per candidate |
|---|---|---|---|
| Combine | 96.9 | 10,092 | 0.00¢ |
| LM Modify | 56.0 | 17,887 | 0.38¢ |
| LM Rewrite | 35.0 | 4,173 | 3.93¢ |
| PR Mirror | 33.8 | 2,344 | 5.53¢ |
| Procedural | 40.2 | 15,641 | 0.00¢ |
| Total | 50.1 | 50,137 | 2.32¢ |
- 128 repositories; 381 instances per repository on average; total cost $1,360 ($1,000 bugs, $160 installation, $200 issues for 10K bugs).
- Environment storage (Table 2): SWE-smith 295 GB for 50k tasks; SWE-gym 6 TBs for 2.4k; R2E-gym (Subset) 4 TBs for 4.6k. The authors estimate 50-150 TB for 50k SWE-bench-style images.
- Difficulty classifier (Qwen 2.5 32B, 75.3% test accuracy): mean difficulty 5.27-5.72 across strategies vs SWE-bench 5.01 and SWE-gym 5.62.

## Training and results (§3-4, App. F.1, Table 3)
- Rejection sampling fine-tuning: claude-3-7-sonnet-20250219 in SWE-agent, at most 75 steps and $2.00; 17,906 attempts on 8,686 instances resolved 36% (6,457 trajectories); at most 3 trajectories per instance, giving 5,016.
- torchtune full fine-tuning, LR 5e-5, maximum 3 epochs, max context 32,768, 2-8 H100 GPUs.
- SWE-bench Verified pass@1: SWE-agent-LM-32B 40.2 (Lite 30.7); SWE-agent-LM-7B 15.2.

## Ablations (§4.1)
- Repositories (Fig. 5): 700 Procedural Modification trajectories drawn from pools of 4, 25, 50, 100 repositories (the text says 4; the figure axis prints 5): 10.3, 11.5, 12.9, 15.1% resolved, "approximately logarithmic".
- Specialization (Fig. 4): 7B trained on 700 SymPy trajectories: SymPy subset 21.2 vs 13.6 for the 100-repository model; Verified without SymPy 14.0 vs 15.3. 32B further trained on SymPy: 42.4 vs 33.3 on SymPy, 38.3 vs 40.2 without SymPy.
- Strategy (Table 4, 507 trajectories each): PR Mirror 9.2, LM Rewrite 8.8, Procedural 8.6, LM Modify 5.7.
- Issue text (Table 5, 259 trajectories each): original 7.8, LM 7.7, F2P test 7.3, fixed template 6.4; test-based issues cut bug-reproduction attempts from 379 to 127 of 500 runs.
- Difficulty (§4.1): SFT sets at difficulty 2/4/6/8 scored 12.4 / 10.8 / 13.6 / 12.2.

## Transfer beyond Python (App. F.4)
- SWE-bench Multilingual (300 tasks, 9 languages): Claude 3.7 Sonnet 43%, SWE-agent-LM-32B 8.4%, Qwen 2.5 Coder Instruct 6.5%. The authors observed edits "reflected syntax closer to Python" in non-Python repositories.

## Limitations (§6)
- The collection pipeline is Python-centric (Python `ast` library); only fine-tuning is explored, not RL.

## Used in

ch-46a §2 (environment-family specialization and repository-pool diversity) and §3 (task supply for the RL stage).

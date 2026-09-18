<!-- scope: executable SWE training environment (2,438 Docker-packaged Python tasks) plus rejection-sampling SFT and verifier training results
     deps: [[agentinstruct]]
     see-also: [[swe-rl]], [[openhands-data]], [[r2e-gym]], [[swe-rebench]]
-->

# Training Software Engineering Agents and Verifiers with SWE-Gym
- **Core Insight:** Fine-tuning Qwen-2.5-Coder-Instruct on 491 successful agent trajectories sampled inside SWE-Gym raises the SWE-Bench Verified resolve rate of the 32B model from 7.0% to 20.6%, and adding a trajectory verifier for best-of-16 selection raises it to 32.0% (§4.2 Table 3; §5.1.2).
- **Guideline:** When training an open-weight SWE agent, build executable per-instance environments with runnable unit tests first, because the same environments supply the success labels for rejection-sampling SFT and the success/failure labels for an outcome verifier (§3.1, §5.1.1).
- **Authors:** Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, Yizhe Zhang (UC Berkeley, UIUC, CMU, Apple)
- **Year:** 2024 (arXiv v1 2024-12; v2 2025-06); ICML 2025, PMLR 267
- **URL:** https://arxiv.org/abs/2412.21139
- **Source type:** paper
- **Relevant topics:** SWE agents, executable environments, rejection-sampling SFT, outcome reward models, inference-time scaling, SWE-Bench

## Abstract
SWE-Gym is a training environment for software-engineering agents. It contains 2,438 real-world task instances, each a Python codebase with an executable runtime environment, unit tests, and a natural-language task. The authors train agents on trajectories sampled in SWE-Gym and report up to 19% absolute gains in resolution rate on SWE-Bench Verified and Lite. They also train verifiers on sampled trajectories for inference-time scaling; combined with the fine-tuned agents this reaches 32.0% on SWE-Bench Verified and 26.0% on SWE-Bench Lite, which the paper states as state of the art for open-weight SWE agents at the time. SWE-Gym, the models, and the trajectories are released.

## Key Contributions
- An executable task environment: 2,438 unit-test-validated Python instances from 11 repositories, with pre-built Docker images released (§3.1).
- SWE-Gym Lite, a 230-instance subset built with the SWE-Bench Lite filtering pipeline (§3.2).
- Rejection-sampling fine-tuning results for Qwen-2.5-Coder-Instruct at 7B, 14B, and 32B under two scaffolds, OpenHands CodeActAgent 2.1 and MoatlessTools (§4.2, §4.3).
- An outcome-supervised verifier trained on execution-labeled trajectories, used for Best@k selection at inference (§5.1).
- Two negative results: self-improvement on OpenHands trajectories lowered the score, and the 32B MoatlessTools loop stopped improving after one iteration (§4.2, §4.3).

## Key Figures/Tables to Study
- Figure 1: training-data scaling (top) and inference-time Best@k scaling (bottom).
- Table 2: SWE-Gym vs SWE-Bench test-split statistics.
- Table 3: zero-shot vs fine-tuned resolve rate, empty-patch rate, stuck-in-loop rate, average turns.
- Table 4: MoatlessTools self-improvement iterations.
- Figure 3 / §5.1.2: Pass@k vs Best@k curves for the 32B agent and verifier.

## Technical Details
- SWE-Gym Raw contains 64,689 instances mined from 358 repositories; after semi-manual environment construction and execution validation, 2,438 instances from 11 repositories remain (§3.1).
- Validation required about 200 human annotation hours and 10,000 CPU core hours; the released Docker images total 6 TB (§3.1).
- Repository distribution is long-tailed: pandas is close to one third of instances, bokeh about one percent (§3.3, Fig. 2).
- Average gold patch edits 69.8 lines, 2.5 files, 4.1 functions; 10.0 fail-to-pass tests and 760.8 total tests per instance; issue text averages 239.8 words (Table 2).
- SWE-Gym Lite has 230 instances; it excludes multi-file edits, poorly described problem statements, very complex gold diffs, and error-message tests (§3.2).
- The SFT set is 491 successful trajectories obtained by rejection sampling from `gpt-4o-2024-08-06` and `claude-3-5-sonnet-20241022` at several temperatures; each successful trajectory averages about 19 turns and about 19,000 tokens (§4.2).
- SWE-Bench Verified resolve rate, zero-shot → fine-tuned (Table 3): 7B 1.8% → 10.6%; 14B 4.0% → 16.4%; 32B 7.0% → 20.6%.
- SWE-Bench Lite resolve rate, zero-shot → fine-tuned (Table 3): 7B 1.0% → 10.0%; 14B 2.7% → 12.7%; 32B 3.0% → 15.3%.
- Fine-tuning reduces the stuck-in-loop rate by 4.6–18.6 points across both test sets, except the 32B model on Lite, which rises by 1.5 points from an already low rate (§4.2).
- Verifier: an outcome-supervised reward model built by fine-tuning Qwen2.5-Coder-Instruct-32B to emit `<YES>` or `<NO>`; the score is r = exp(l_y) / (exp(l_y) + exp(l_n)), where l_y and l_n are the log probabilities of the two tokens (§5.1.1).
- Verifier data: 443 off-policy successful trajectories plus 875 on-policy successful trajectories from the fine-tuned 32B model, plus an equal count of unsuccessful trajectories sampled from each subset (1,318 each), for 2,636 trajectories total (§5.1.1).
- Best@k for the 32B agent on SWE-Bench Verified: 20.6 at k=1, 29.8 at k=8, 32.0 at k=16; the Lite figure is 26.0 (§5.1.2). Pass@k is higher than Best@k, which the authors attribute to verifier error.
- A LoRA-fine-tuned verifier scored 29.8@8 against 27.2@8 for full-parameter fine-tuning (§5.1.2).
- Data scaling on the 491 trajectories shows no sign of saturation at 491 (Fig. 1 top, §5.2); the authors conclude size and repository diversity are not yet the bottleneck.
- MoatlessTools self-improvement (Table 4), 30 rollouts per task at temperature 1.0 on SWE-Gym Lite: 7B 7.0% → 9.0% → 10.0%; 32B 19.0% → 19.7% → 19.7%. An instance cap of 2 trajectories per task performed slightly better than the uncapped set (§4.3, Table 6).
- Self-improvement under OpenHands failed: fine-tuning the base 32B model on 868 on-policy trajectories mixed with the 491 off-policy trajectories dropped the Lite resolve rate from 15.3% to 8.7% (§4.2).
- Evaluation uses temperature 0 unless stated; the verifier scaling runs use t = 0.5 (§4.1 footnote, Fig. 1 caption).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen-2.5-Coder-Instruct (SWE-Gym agent) | 7B / 14B / 32B | SFT | training examples | 491 successful trajectories | arXiv:2412.21139v2 §4.2 | verified 2026-09-18 | Fig. 1 top: resolve rate still rising at 491 trajectories |
| Qwen-2.5-Coder-Instruct (SWE-Gym agent) | 7B / 14B / 32B | SFT | fine-tuning method | full-parameter fine-tuning with torchtune | arXiv:2412.21139v2 §B.2 | verified 2026-09-18 | no ablation reported |
| Qwen-2.5-Coder-Instruct (SWE-Gym agent) | 7B / 14B / 32B | SFT | learning rate | 1e-4 | arXiv:2412.21139v2 §B.2 | verified 2026-09-18 | no ablation reported |
| Qwen-2.5-Coder-Instruct (SWE-Gym agent) | 7B / 14B / 32B | SFT | epochs | maximum 5 | arXiv:2412.21139v2 §B.2 | verified 2026-09-18 | no ablation reported |
| Qwen-2.5-Coder-Instruct (SWE-Gym agent) | 7B / 14B / 32B | SFT | global batch size (sequences) | 8 | arXiv:2412.21139v2 §B.2 | verified 2026-09-18 | no ablation reported |
| Qwen-2.5-Coder-Instruct (SWE-Gym agent) | 7B / 14B / 32B | SFT | max context length | 32,768 tokens | arXiv:2412.21139v2 §B.2 | verified 2026-09-18 | no ablation reported |
| Qwen-2.5-Coder-Instruct (SWE-Gym agent) | 7B / 14B / 32B | SFT | hardware | 2–8× NVIDIA H100 80GB on Modal | arXiv:2412.21139v2 §B.2 | verified 2026-09-18 | no ablation reported |
| Qwen-2.5-Coder-Instruct (SWE-Gym agent) | 32B | SFT | total GPU-hours | not reported (checked body, §B.2, acknowledgments) | — | not reported | — |
| Qwen2.5-Coder-Instruct verifier | 32B | reward-model | training trajectories | 2,636 (1,318 successful, 1,318 unsuccessful) | arXiv:2412.21139v2 §5.1.1 | verified 2026-09-18 | Fig. 8: mixed on-/off-policy data best, 27@8 |
| Qwen2.5-Coder-Instruct verifier | 32B | reward-model | fine-tuning method | LoRA (Unsloth) for the main result; full-parameter compared | arXiv:2412.21139v2 §5.1.2, §B.3 | verified 2026-09-18 | §5.1.2: LoRA 29.8@8 vs full fine-tuning 27.2@8 |
| SWE-Gym 32B agent + verifier | 32B | eval-gate | selection rule | Best@16 over rollouts at t = 0.5 | arXiv:2412.21139v2 §5.1.2, Fig. 1 | verified 2026-09-18 | Fig. 1 bottom: log-linear Best@k scaling |
| MoatlessTools self-improvement | 7B / 32B | SFT | rollouts per task | 30 at temperature 1.0 on SWE-Gym Lite | arXiv:2412.21139v2 §4.3 | verified 2026-09-18 | Table 6: instance cap of 2 slightly beats the full set |

## Findings relevant to agentic training and distillation
- The SFT trajectories are distilled from proprietary teachers (GPT-4o and Claude 3.5 Sonnet); the paper does not report a teacher-free variant that matches them (§4.2).
- On-policy self-improvement under the general-purpose OpenHands scaffold reduced performance (15.3% → 8.7% on Lite), while the constrained MoatlessTools scaffold allowed small gains for the 7B model (§4.2, §4.3). The authors attribute the difference to task horizon and action-space size.
- Gains transfer across two scaffolds but the paper evaluates only Python repository issues; no non-Python or non-SWE evaluation is reported.

## Connections
- [[swe-rl]] — trains without execution environments using a similarity reward; the SWE-RL paper lists SWE-Gym among baselines that distill from GPT-4o or Claude.
- [[openhands-data]] — the OpenHands CodeActAgent scaffold used for the main results.
- [[r2e-gym]], [[swe-rebench]] — later executable SWE task collections.
- [[rejection-sampling-finetuning]] — the policy improvement algorithm used here (filtered behavior cloning).

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2412.21139 (arXiv v2, 2025-06-06; ICML 2025 camera-ready text)
- Corrections to the previous card version:
  - Title "SWE-Gym: An Open Environment for Training Software Engineering Agents and Verifiers" → "Training Software Engineering Agents and Verifiers with SWE-Gym" (paper title page).
  - "7B: 3.0% → 15.3%, verifier 20.3%" → those are 32B SWE-Bench Lite numbers (3.0% → 15.3%); the 7B Verified numbers are 1.8% → 10.6% and no 20.3% figure appears (Table 3).
  - "32B version: 32.0% SWE-Bench Verified" → 32.0% is Best@16 with the trained verifier; plain fine-tuning gives 20.6% (§4.2 Table 3, §5.1.2).
  - "491 tasks immediately compatible with SWE-Bench Lite" → SWE-Gym Lite has 230 instances; 491 is the count of SFT trajectories (§3.2, §4.2).
  - "capture up to K=10 trajectories per task" → not stated; the SFT set is 491 trajectories total, and the MoatlessTools loop uses 30 rollouts per task (§4.2, §4.3).
  - "~20K–60K successful trajectories" → 491 successful trajectories (§4.2).
  - "teacher model Qwen-2.5-Coder-32B (primary), Claude-3.5-Sonnet (secondary)" → teachers are `gpt-4o-2024-08-06` and `claude-3-5-sonnet-20241022`; Qwen-2.5-Coder-Instruct is the student (§4.2).
  - "median ~15K tokens, tail to >100K" → about 19 turns and about 19,000 tokens on average (§4.2).
  - "SFT with verifier best-of-N gives +5 points over SFT alone" → +11.4 points, 20.6 to 32.0 on Verified at k=16 (§5.1.2).
- Removed as unsupported by the source: "~10K H100-hours for full 32B RS-SFT loop"; "trajectory count and verifier-N both show log-linear returns" for trajectory count (only Best@k is described as log-linear); the action-space list naming `str_replace_editor` / `execute_bash` / `browse` / `finish` (the paper says OpenHands gives a bash terminal and a file editor and that the browser feature was disabled, §4.2); "training filter requires full-pass not partial" as a stated design rule; "test leakage risk guarded by repo exclusion"; "Docker images rot" maintenance claim.
- Not reported by the source: total GPU-hours or dollar cost of the rollout phase; per-repository fine-tuning breakdowns; results on non-Python repositories.

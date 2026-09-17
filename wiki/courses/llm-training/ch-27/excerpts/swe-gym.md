---
chapter: ch-27
course: llm-training
phase: read
excerpt_of: "Training Software Engineering Agents and Verifiers with SWE-Gym (arXiv:2412.21139v2)"
source_url: https://arxiv.org/abs/2412.21139
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten from the primary text; the library card papers/swe-gym.md predates verification and contains values not in the paper)"
---

# Excerpt: SWE-Gym — executable repositories, teacher trajectories, and verifiers

**Authors:** Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, Yizhe Zhang. arXiv v1 2024-12, v2 2025-06-06; ICML 2025 (PMLR 267).

## Environment (§3, Table 2)
- 2,438 Python task instances from 11 repositories, each with pre-installed dependencies and executable unit tests.
- SWE-Gym Lite: 230 instances selected with the SWE-bench Lite filtering pipeline.
- Gold patches edit more lines than SWE-bench's (69.8 vs 32.8 lines on average), and models score lower on SWE-Gym than on SWE-bench (§3).

## Trajectory collection and SFT (§4.1–4.2, App. B.2)
- Scaffold: OpenHands CodeActAgent 2.1, general-purpose ReAct-style prompting with a bash terminal and file editor; browser feature disabled.
- Rejection sampling: 491 successful trajectories sampled from gpt-4o-2024-08-06 and claude-3-5-sonnet-20241022 at different temperatures; on average about 19 turns and about 19,000 tokens; the count is limited by compute budget.
- Example yield (App. B.4): gpt-4o-2024-08-06, temperature 0, on SWE-Gym Lite: 19 of 230 resolved (8.26%).
- Base models: Qwen-2.5-Coder-Instruct 7B, 14B, 32B. torchtune full fine-tuning, LR 1e-4, maximum 5 epochs, global batch 8, max context 32,768; 2–8 H100 80G GPUs.

## Results (Table 3, OpenHands; resolve rate %, zero-shot → fine-tuned)
| Size | Lite | Verified | Stuck in loop, Verified |
|---|---|---|---|
| 7B | 1.0 → 10.0 | 1.8 → 10.6 | 39.6 → 21.0 |
| 14B | 2.7 → 12.7 | 4.0 → 16.4 | 32.1 → 21.3 |
| 32B | 3.0 → 15.3 | 7.0 → 20.6 | 29.4 → 23.8 |

- Self-improvement (§4.2): the fine-tuned 32B model sampled 6 trajectories per instance at temperature 0.5 (868 successes); fine-tuning the base 32B on 868 on-policy + 491 teacher trajectories lowered Lite from 15.3 to 8.7. Authors: self-improvement "is not yet working".
- MoatlessTools scaffold (§4.3, Table 4): zero-shot Lite 7.0 (7B) and 19.0 (32B); two iterations of online rejection-sampling fine-tuning reach 10.0 and 19.7.

## Verifier and inference-time scaling (§5.1.1, Fig. 3)
- Outcome verifier: Qwen2.5-Coder-Instruct-32B trained to output <YES>/<NO> on 2,636 trajectories (1,318 successful: 443 off-policy + 875 on-policy; 1,318 unsuccessful); reward r = exp(l_y) / (exp(l_y) + exp(l_n)).
- 32B agent on Verified: pass@k 20.6 (k=1) → 37.8 (k=8) → 42.8 (k=16); Best@k 29.8 (k=8), 32.0 (k=16). LoRA verifier 29.8@8 vs full fine-tuning 27.2@8.
- MoatlessTools with verifier: 26.0 on Lite (§5.1.2).
- Training-data scaling (§5.2, Fig. 5): resolve rate increases with trajectory count up to 491; the authors state compute for sampling, not the number of tasks, is the limit.

## Not in the paper
"3.0% → 15.3% for 7B on Verified", "20.3% with verifier", "K=10 rollouts per task", "Qwen-2.5-Coder-32B as teacher", "491 tasks compatible with SWE-bench Lite", `browse` action, "~10K H100-hours".

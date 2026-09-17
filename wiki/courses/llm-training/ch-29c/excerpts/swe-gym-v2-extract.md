---
chapter: ch-29c
course: llm-training
phase: read
excerpt_of: "Training Software Engineering Agents and Verifiers with SWE-Gym (arXiv:2412.21139v2, 2025-06-06; ICML 2025)"
source_url: https://arxiv.org/abs/2412.21139
created_at: "2026-09-15"
note: "The library card [[swe-gym]] has no Verification section and contains numbers not found in the paper (K = 10 trajectories per task, ~10K H100-hours, 20K-60K trajectories, 7B 3.0% to 15.3% on Verified). ch-29c uses the values below, read from the v2 PDF on 2026-09-15."
---

# Excerpt: SWE-Gym environment construction and training results

## Construction (§3.1)
- SWE-Gym Raw: 64,689 task instances extracted from 358 repositories.
- Executable environments were created semi-manually for 11 repositories: dependencies configured per task instance from `requirements.txt`, CI scripts, or documentation at the time of the issue, then validated so that the gold patch passes more unit tests than the original code.
- > "This process required approximately 200 human annotation hours and 10,000 CPU core hours. After validation and filtering out failed instances, we obtained 2,438 unit-test-validated instances from 11 repositories."
- Pre-built Docker images for each instance total 6 TB.
- The 11 repositories are separate from those used in SWE-Bench "to avoid contamination" (§3).
- SWE-Gym Lite: 230 instances selected with the SWE-Bench Lite filters (§3.2).

## Training (§4.1-4.2, Table 3, App. B.2)
- Rejection sampling fine-tuning on 491 successful trajectories sampled from gpt-4o-2024-08-06 and claude-3-5-sonnet-20241022; about 19 turns and 19,000 tokens per trajectory.
- torchtune full fine-tuning, LR 1e-4, maximum 5 epochs, global batch 8, max context 32,768, Qwen-2.5-Coder-Instruct 7B/14B/32B.
- Resolve rate, zero-shot → fine-tuned (Table 3): Lite 7B 1.0 → 10.0, 14B 2.7 → 12.7, 32B 3.0 → 15.3; Verified 7B 1.8 → 10.6, 14B 4.0 → 16.4, 32B 7.0 → 20.6.
- Self-improvement: 32B fine-tuned on 868 on-policy plus 491 off-policy trajectories dropped on Lite from 15.3 to 8.7 (§4.2).

## Verifier (§5.1)
- Verifier trained on 2,636 trajectories (1,318 successful, 1,318 unsuccessful); Best@k with the 32B agent rose from 20.6@1 to 29.8@8 and 32.0@16, while Pass@k reached 42.8@16.

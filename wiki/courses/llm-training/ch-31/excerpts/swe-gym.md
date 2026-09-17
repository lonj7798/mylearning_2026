---
chapter: ch-31
course: llm-training
phase: read
excerpt_of: arXiv:2412.21139v2 (library card papers/swe-gym.md has no Verification section and several numbers that differ from the paper; this chapter-local extract was checked against the primary source)
source_url: https://arxiv.org/abs/2412.21139
created_at: "2026-09-15"
---

# Excerpt: Training Software Engineering Agents and Verifiers with SWE-Gym

- **Authors:** Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, et al. (UC Berkeley; UIUC; CMU; Apple)
- **Year:** 2024 (arXiv v1 2024-12; v2 2025-06-06; ICML 2025, PMLR 267)
- **Source type:** paper
- **Used in:** ch-31 §1.2, §4, §6.1, Negative samples and negative feedback, Recipe, Common mistakes

## Environment (§3)

2,438 Python tasks from pull requests in 11 repositories, each with an executable environment, unit tests, and a natural-language issue; repositories are disjoint from SWE-bench (§3, Table 1). SWE-Gym Lite is a subset of 230 tasks selected with the SWE-bench Lite filtering pipeline (§3.2).

## Training with teacher trajectories (§4.1–4.2)

- **Scaffolds.** OpenHands CodeAct (general-purpose, ReAct-style prompting, bash and file editor) and MoatlessTools (fixed workflow) (§4.1).
- **Policy improvement.** Rejection-sampling fine-tuning ("filtered behavior cloning") on successful trajectories (§4.1).
- **Data.** 491 successful trajectories from gpt-4o-2024-08-06 and claude-3-5-sonnet-20241022 at various temperatures; about 19 turns and 19,000 tokens on average (§4.2).
- **Results (Table 3, Qwen2.5-Coder-Instruct, OpenHands, temperature 0).**

| Size | SWE-bench Lite zero-shot → fine-tuned | SWE-bench Verified zero-shot → fine-tuned | Verified stuck-in-loop % |
|---|---|---|---|
| 7B | 1.0 → 10.0 | 1.8 → 10.6 | 39.6 → 21.0 |
| 14B | 2.7 → 12.7 | 4.0 → 16.4 | 32.1 → 21.3 |
| 32B | 3.0 → 15.3 | 7.0 → 20.6 | 29.4 → 23.8 |

Training data scaling shows no saturation up to 491 trajectories (Fig. 1 top).

## Self-improvement results

- **OpenHands, on-policy (§4.2).** The fine-tuned 32B model sampled 6 trajectories per SWE-Gym instance at temperature 0.5, giving 868 successful trajectories. Fine-tuning the base 32B model on these plus the 491 teacher trajectories changed SWE-bench Lite resolution from 15.3% to 8.7%, "suggesting that self-improvement is not yet working". The authors suggest PPO-style optimization or a stronger base model as future directions.
- **MoatlessTools, iterative rejection-sampling fine-tuning (§4.3, Table 4).** Each iteration: 30 rollouts per task at temperature 1.0 on SWE-Gym Lite, keep successful trajectories, fine-tune (LoRA for 32B). SWE-bench Lite resolution: 7B 7.0% → 9.0% → 10.0%; 32B 19.0% → 19.7% → 19.7%. Further iterations after two gave negligible gains. The authors attribute the limited 32B gain to the restricted action space and the rejection-sampling method.
- **Per-instance capping (§4.3, Fig. 6, Table 6).** Success probability per task is long-tailed, so training on all successes biases data toward easy tasks; a cap of 2 samples per task slightly outperformed the full dataset and trained faster. Among capped samples, trajectories with fewer model rounds are preferred.

## Verifier for inference-time scaling (§5.1.1)

- Qwen2.5-Coder-Instruct-32B fine-tuned to emit `<YES>` or `<NO>` for a trajectory; reward r = exp(l_y) / (exp(l_y) + exp(l_n)).
- Training data: 443 off-policy and 875 on-policy successful trajectories (kept under 32k tokens), plus an equal number of unsuccessful trajectories from each subset: 1,318 and 1,318, total 2,636.
- Results with the fine-tuned 32B agent: Pass@k rises from 20.6% (k = 1) to 37.8% (k = 8) and 42.8% (k = 16); Best@k from 20.6% to 29.8% (k = 8) and 32.0% (k = 16). LoRA verifier 29.8@8 vs full fine-tuning 27.2@8.
- Verifier data ablation (Fig. 8): mixing off-policy and on-policy data was best; off-policy-only data plateaued near 22%.

## Negative-sample classification (course standard §6.1)

In agent SFT, failed trajectories are discarded (negative marginal value). In the verifier, failed trajectories are training inputs with the target token `<NO>` under ordinary cross-entropy (negative as content for the verifier).

## Verification

- Checked on 2026-09-15 against: https://arxiv.org/abs/2412.21139 (v2, 2025-06-06), §3–§5 and Tables 1, 3, 4, 6.
- Differences from the library card `papers/swe-gym.md`: "Qwen-2.5-Coder-7B-Instruct: SWE-Bench Verified 3.0% → 15.3%" → 3.0% → 15.3% is the 32B model on SWE-bench Lite; 32B Verified is 7.0% → 20.6%, and 7B Verified is 1.8% → 10.6% (Table 3). "With verifier best-of-N: 20.3%" → Best@16 32.0% on Verified for 32B (§5.1.1). "K=10 trajectories per task", "~20K–60K successful trajectories", and "~10K H100-hours" do not appear in the paper; the teacher set is 491 trajectories (§4.2). The card omits the on-policy self-improvement negative result (§4.2).

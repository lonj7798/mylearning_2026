---
chapter: ch-51a
course: llm-training
phase: read
excerpt_of: "Measuring AI Ability to Complete Long Software Tasks (arXiv:2503.14499v4, 2026-07-10; v1 2025-03-18)"
source_url: https://arxiv.org/abs/2503.14499
created_at: "2026-09-15"
note: "No library card exists at wiki/raw-data/llm-training/**/metr-time-horizon.md on 2026-09-15. Values read from the v4 PDF on 2026-09-15."
---

# Excerpt: METR time horizon — the logistic fit and its assumptions

**Authors:** Thomas Kwa, Ben West (equal contribution), Joel Becker and colleagues; Elizabeth Barnes and Lawrence
Chan (Model Evaluation & Threat Research, METR). Used by [[read]] §8.

## Definition (Abstract, §1)

The **X%-task-completion time horizon** is "the length of tasks that models can complete approximately X% of the
time", where task length is the time a skilled human takes. The headline metric is the 50% time horizon.

## Task suites (§2.1)

170 tasks across three suites: HCAST, RE-Bench, and Software Atomic Actions (SWAA), a new set of 66 short tasks
that can separate pre-2023 models. Human baseliners with domain knowledge but no task-specific context are timed
on the same tasks.

## Scoring and the fit (§2.3, §3.1)

- Every task is reduced to a binary outcome. Continuously scored tasks are binarized at a task-specific threshold
  representing human performance; RE-Bench uses the average score of 7–9-hour human runs.
- The time horizon is fit by logistic regression:

```
p_success(agent, task) = σ( (log h_agent − log t_task) · β_agent )
```

  - `t_task`: the geometric mean time of successful human baselines for that task.
  - `h_agent`: the fitted 50% time horizon of the agent (the task length at which p_success = 0.5).
  - `β_agent`: the fitted slope, how sharply success falls as tasks get longer.
  - `σ`: the logistic function.
- Footnote 5 compares the fit to Item Response Theory: "unlike IRT, we use difficulty ratings directly based on
  human baseline time rather than ratings learned from agent performance."
- "The logistic fit is fairly good, though there is a jump in success rate between <1 minute SWAA tasks and
  >1 minute HCAST tasks" (Fig. 4 caption).

## Results (§3.2, §3.2.1)

- 12 frontier models from 2019 to 2025. GPT-2 has a 50% time horizon of 2 seconds; o3 has 110 minutes and succeeds
  at several tasks over 4 hours.
- Time horizon doubled every **207 days**, 95% bootstrapped CI 166–240 days (about ±19%), from an OLS regression
  of log(horizon) on release date. Error bars come from 10,000 samples of a three-level hierarchical bootstrap
  over task families, then tasks, then runs.
- "While there are wide error bars on each individual models' horizon lengths, these errors are highly correlated
  between models ... we are more confident in the slope of the time horizon trend than in the time horizon of any
  particular model" (§3.2).
- o3 lies above the long-run trend (p = 0.006), which may imply a faster trend in 2024 and early 2025 (§3.2).
- **80% horizon:** doubling time 204 days, close to the 207 days of the 50% horizon, "however, models' 80% time
  horizons are 4-6x shorter, suggesting that even models that sometimes succeed on difficult and diverse tasks
  cannot reliably perform tasks of moderate length." Time horizons at very high success rates (95%) cannot be
  confidently measured with this dataset (§3.2.1).
- Failure categories on sampled unsuccessful runs (Table 2, 31 GPT-4 1106 runs and 32 o1 runs): premature task
  abandonment 8 vs 16; repeating failed actions 12 vs 2; incorrect mental math/reasoning 6 vs 7; poor
  planning/tool choice 4 vs 6.

## External-validity checks (§4)

1. Replicating the method on SWE-bench Verified gives an exponential trend with a shorter doubling time (about 70
   days against 143 days for HCAST + SWAA + RE-Bench on 2024 models). The authors attribute part of this to
   SWE-bench Verified time annotations, which assume "an engineer who has had a few hours to familiarize
   themselves with the codebase" and differentially underestimate contractor time on the easiest tasks (§4.1).
2. A 16-factor "messiness" score (resource limits, novelty, dynamic environment, and others). Controlling for task
   length, models do worse on messier tasks, but the trend over time is similar for the low- and high-messiness
   subsets, with "no evidence of plateaus in performance trends specific to the higher messiness subset" (§4,
   §F.2). The mean messiness score of HCAST and RE-Bench tasks is 3.2 out of 16.
3. On an uncontaminated set of METR's internal pull requests, contractors take 5–18× longer than repository
   maintainers, and agent performance matches contractor times rather than maintainer times (§4, §C.2).

## Limits stated (§5)

"Time horizon is always measured relative to a domain, task distribution, and human baseliners' level of skill and
context, and can be difficult to measure in practice, especially for longer horizons or success rates near 100%."
The evaluation covers software and research tasks only.

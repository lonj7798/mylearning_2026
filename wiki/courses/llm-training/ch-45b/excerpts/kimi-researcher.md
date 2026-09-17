---
chapter: ch-45b
course: llm-training
phase: read
excerpt_of: "Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities (Moonshot AI blog, 2025-06-20)"
source_url: https://moonshotai.github.io/Kimi-Researcher/
created_at: "2026-09-15"
---

# Excerpt: Kimi-Researcher — End-to-End RL Training for Emerging Agentic Capabilities

- **Organization:** Moonshot AI (Kimi team)
- **Year:** 2025 (post dated 2025-06-20)
- **Source type:** official blog (no paper; no released weights)
- **Used in:** [[read]] §3, §4, §6, Negative samples, Generalization lens

## Results as reported
- Humanity's Last Exam (HLE): pass@1 rises "from 8.6% to 26.9%" over end-to-end RL training; pass@4 is 40.17%.
- xbench-DeepSearch: 69% pass@1, averaged across 4 runs.
- Behaviour: the agent performs "an average of 23 reasoning steps and explores over 200 URLs per task".
- The post reports no non-agentic general benchmark, and no ablation isolating any single component.

## Algorithm
- The policy is trained with "the REINFORCE algorithm" on strictly on-policy trajectories.
- **Negative-sample control (verbatim):** "Negative samples lead to a decrease in token probabilities, which
  increases the risk of entropy collapse during RL training. To address this, we discard some negative samples
  strategically, allowing the model to continue improving over a longer training period." The post does not state
  which negatives are discarded or what fraction.
- **Gamma decay (verbatim):** "the reward of step i becomes r×γ^(T − i), where r is the outcome reward, T is the
  number of steps, and 0<γ<1 represents the gamma-decay coeficient." The stated effect is a preference for
  shorter successful trajectories. The value of γ is not given.

## Context management
A context-management mechanism "extends a single rollout trajectory to over 50 iterations". In the reported
ablation, "a model trained with context management uses 30% more iterations" than a model trained without it.

## Infrastructure
Fully asynchronous rollouts behind gym-like interfaces; turn-level partial rollout resumes long-running tasks
with updated weights and delivers "substantial rollout acceleration (at least 1.5x)"; a unified sandbox runs on a
hybrid Kubernetes cloud.

## Verification
- Checked on 2026-09-15 against https://moonshotai.github.io/Kimi-Researcher/ (post dated 2025-06-20).
- Not reported by the post: model size and base checkpoint, γ, the discard rule for negative samples, training
  compute, data volumes, and any evaluation outside HLE and xbench-DeepSearch.

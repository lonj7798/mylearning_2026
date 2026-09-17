---
chapter: ch-32d
course: llm-training
phase: read
excerpt_of: Lei, Zhu, Li, Liu, Li, Yan, et al. (Tsinghua University; Alibaba Group), "State2State: Environment-Derived Mid-Training for LLM Agents", arXiv:2608.04934v1 (2026-08-05)
source_url: https://arxiv.org/abs/2608.04934
created_at: "2026-09-15"
source_type: paper
---

# Excerpt: State2State — state-reaching RL as an agent mid-training stage

No library card for this source exists at the time of writing (planned slug `state2state`). Only passages used by [[read]] §7, §9, the negatives section, and the Recipe table are extracted.

## Method (§3)

1. Exploration: "we deliberately adopt random exploration"; the explorer samples from valid actions (§3.2).
2. Task construction: remove invalid or uninformative observations, sample diverse targets, "replay the corresponding exploration trajectory from the same initial configuration for three times and retain only observations that can be consistently reached" (§3.3).
3. Reward (Eq. 1): r_t = 1 if match(o(s_t), o⋆), 0 otherwise, with normalized exact match in the textual environments (§3.3, §4.3).
4. Optimization: GRPO with dynamic sampling, "discarding rollout groups with identical rewards and continuing to sample new groups until the required batch size is reached" (§3.4).
5. The resulting policy initializes downstream RL on human-specified tasks (§3.4).

## Settings (§4.3, App. E.1)

verl and vLLM; actor LR 1e-6; KL loss coefficient 0.01, not added to the reward; train batch size 16; GRPO group size 8; training temperature 1.0, validation temperature 0.1; ALFWorld max prompt 4,096 and response 1,024; ScienceWorld response 1,024; State2State trained for 80 steps, with downstream training continued from the best checkpoint on the state-reaching validation set (§4.3). Models are labeled Qwen3-4B and Qwen3-8B.

## Results (Tables 1-5; task success rate, average of ID and OOD)

| Model | Setting | ALFWorld | ScienceWorld |
|---|---|---|---|
| Qwen3-4B | Base | 33.53 | 23.63 |
| Qwen3-4B | State2State only | 44.92 | 21.00 |
| Qwen3-4B | RL | 86.13 | 49.63 |
| Qwen3-4B | State2State + RL | 92.00 | 55.50 |
| Qwen3-8B | Base | 75.21 | 31.50 |
| Qwen3-8B | State2State only | 77.17 | 33.13 |
| Qwen3-8B | RL | 92.36 | 52.13 |
| Qwen3-8B | State2State + RL | 97.45 | 56.00 |

- Ordering on ScienceWorld, Qwen3-4B (Table 2): SFT 46.38; S2S + SFT 48.00; SFT + S2S 54.13; SFT + RL 58.38; S2S + SFT + RL 60.50; SFT + S2S + RL 65.88. The authors state that "applying State2State before SFT brings limited gains, likely because subsequent imitation learning can overwrite part of the exploratory environment priors."
- Exploration (Table 4): LLM explorer + RL 52.63; random explorer + RL 55.50.
- Cross-environment (Table 5, ALFWorld RL after ScienceWorld mid-training): none 86.13; ScienceWorld task RL 86.87; ScienceWorld State2State 89.44.

## Verification

- Checked on 2026-09-15 against https://arxiv.org/abs/2608.04934 (v1), §3-§5, Limitations, App. E.1.
- Not reported by the source: number of seeds; whether the Qwen3 checkpoints are base or post-trained releases beyond the label "Qwen3-4B/8B"; any evaluation outside ALFWorld, ScienceWorld, and MobileWorld; false-negative rate of the exact-match verifier.

---
chapter: ch-44a
course: llm-training
phase: read
excerpt_of: arXiv:2502.03373v1 (Demystifying Long Chain-of-Thought Reasoning in LLMs), §4.1-§4.5, App. C.1, App. E.5 (chapter-local verified extract; no library card exists for this slug on 2026-09-15)
source_url: https://arxiv.org/abs/2502.03373
created_at: "2026-09-15"
---

# Excerpt: Demystifying Long Chain-of-Thought Reasoning in LLMs

- **Authors:** Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, Xiang Yue (Carnegie Mellon University; Tsinghua University; IN.AI)
- **Year:** 2025 (arXiv v1 2025-02-05)
- **Source type:** paper
- **Setting:** PPO in OpenRLHF from Llama-3.1-8B and Qwen2.5-Math-7B, after SFT on long CoT distilled from QwQ-32B-Preview
  with the MATH training split; GAE λ = 1, γ = 1, prompt length 2048, generation length 14336, batch 512, actor LR 5e-7,
  KL 0.01 (App. E.5.2, E.5.3).
- **Used in:** ch-44a §2, §3, Common mistakes.

## Length instability under a correctness-only reward (§4.1, Figure 2)
With the "Classic Reward" (+1 for a correct answer), both models "increased their CoT length during training, eventually
reaching the context window limit. This led to a decline in training accuracy due to CoTs exceeding the allowable window
size." The exceed rate "leveled off at a certain threshold below 1 ... This suggests that exceeding the limit started to
apply significant downward pressure on the CoT length distribution, and highlights the context window size's role in
implicit length penalization. Notably, a trajectory might be penalized even without an explicit exceed-length penalty due
to reward or advantage normalization, both of which are standard in RL frameworks." The weaker Llama-3.1-8B "showed greater
fluctuations in CoT length compared to Qwen-2.5-Math-7B".

## Cosine reward (§4.2, App. C.1 Eq. 1)
Ordering constraints: correct CoTs score above wrong CoTs; shorter correct CoTs score above longer correct ones; shorter
wrong CoTs are penalized more than longer wrong ones.

```
R(C, L_gen) = CosFn(L_gen, L_max, r0^c, rL^c)   if C = 1
              CosFn(L_gen, L_max, r0^w, rL^w)   if C = 0
              r_e                               if L_gen = L_max

CosFn(t, T, η_min, η_max) = η_min + 0.5 (η_max − η_min) (1 + cos(tπ / T))
```
`C` correctness (0/1); `L_gen` generation length; `L_max` maximum length; `r0` and `rL` the rewards at length 0 and at
`L_max` for correct (c) and wrong (w) responses; `r_e` the exceed-length penalty. Values used in §4.2 and §3.2 runs:
`r0^c = +2`, `rL^c = +1`, `r0^w = −10`, `rL^w = 0`, `r_e = −10`; repetition penalty `P = −0.05` with N-gram size `N = 40`
(App. E.5.1, E.5.3). The reward is sparse, awarded once at the end of the CoT.

The Cosine Reward produced "more stable (a) training accuracy and (b) response length" than the Classic Reward (Figure 4).

## Hyperparameter behaviour (§4.3, App. Figure 9)
"if the reward for a correct answer increases with CoT length (r0^c < rL^c), the CoT length increases explosively. We also
see that the lower the correct reward relative to the wrong reward, the longer the CoT length. We interpret this as a kind
of trained risk aversion, where the ratio of the correct and wrong rewards determines how confident the model has to be
about an answer for it to derive a positive expected value from terminating its CoT with an answer." Reward A
(`r0^c = 0, rL^c = 10, r0^w = rL^w = 0`) "results in some performance degradation on downstream tasks due to the model's
reduced ability to stop within the context window".

## Context window size (§4.4, Figure 6)
Llama-3.1-8B trained at 4K, 8K and 16K context with the same number of training samples: "the model with a context window
size of 8K performed better than the model with 4K, as expected. However, we observed performance was better under 8K than
16K ... We see this as an indication that models need more training compute to learn to fully utilize longer context
window sizes."

## Length reward hacking (§4.5, Figures 10, 12)
"Length rewards will be hacked with enough compute ... but this can be mitigated using a repetition penalty." The paper
attributes the hack to upward pressure from the Cosine Reward when training accuracy is low: the model repeats content on
hard questions instead of producing new reasoning, and the branching frequency (count of the pivot word "alternatively")
falls with more training compute.

## Not reported
Truncation-rate values in tables, per-run seeds, and results of the cosine reward with group-baseline algorithms (the runs
use PPO with a value network).

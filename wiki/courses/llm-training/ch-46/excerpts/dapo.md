<!-- scope: DAPO (ByteDance Seed et al., 2025): four modifications to GRPO — clip-higher, dynamic sampling, token-level policy-gradient loss, overlong reward shaping — with a per-technique ablation on AIME 2024
     deps: [[grpo]]
     see-also: [[dr-grpo]], [[trl-grpo]], [[entropy-mechanism-llm-rl]], [[rlvr-beyond-base-model]]
-->

# DAPO: An Open-Source LLM Reinforcement Learning System at Scale
- **Core Insight:** Starting from Qwen2.5-32B base, naive GRPO reaches 30 points avg@32 on AIME 2024, and adding overlong filtering (36), clip-higher (38), soft overlong punishment (41), token-level loss (42) and dynamic sampling (50) gives DAPO's 50 points, above DeepSeek-R1-Zero-Qwen-32B at 47 and using half the training steps (Table 1, §4.2).
- **Guideline:** When a GRPO run shows falling entropy and a growing share of prompts whose rollouts are all correct, decouple the clip range (ε_low = 0.2, ε_high = 0.28) and resample each batch until every prompt has 0 < number of correct rollouts < G, because these were the two largest single contributions in the paper's ablation (+2 and +8 AIME points respectively, Table 1); the resampling costs extra generation.
- **Authors:** ByteDance Seed, Institute for AI Industry Research (AIR) Tsinghua University, The University of Hong Kong, SIA-Lab of Tsinghua AIR and ByteDance Seed (full author list in the report's Contributions section)
- **Year:** 2025 (arXiv v1 2025-03; v2 2025-05-20)
- **URL:** https://arxiv.org/abs/2503.14476
- **Source type:** official technical report
- **Relevant topics:** GRPO variants, entropy collapse, zero-variance groups, token-level loss, length truncation, reward noise

## Summary
The report describes an RL system built on verl that trains Qwen2.5-32B base to 50 points avg@32 on AIME 2024, and releases the algorithm, code and dataset. It identifies four problems with naive GRPO at scale — entropy collapse, vanishing gradients from prompts whose rollouts all share a reward, sample-level loss normalization that under-penalizes long low-quality outputs, and reward noise from truncated responses — and gives one modification for each.

## Key Contributions
- **Clip-Higher** (§3.1, Eq. 10): the clip range is decoupled into ε_low and ε_high; ε_high is raised so that low-probability tokens can gain probability, while ε_low is kept because raising it would push those tokens' probability toward zero.
- **Dynamic Sampling** (§3.2, Eq. 11): prompts whose rollouts are all correct or all incorrect are filtered out and the batch is refilled by over-sampling, keeping the number of prompts with non-zero advantage constant.
- **Token-level Policy Gradient Loss** (§3.3, Eq. 12): the loss is summed over all tokens in the group and divided by the total token count, rather than averaged per response first.
- **Overlong Reward Shaping** (§3.4): a soft length penalty over a cache window, and an Overlong Filtering variant that masks the loss of truncated samples.

## Key Figures/Tables to Study
- **Table 1:** per-technique AIME 2024 avg@32 ablation.
- **Figure 3b:** growth over training of the number of prompts whose rollouts are all correct.
- **Figure 5:** AIME accuracy and generation entropy with and without overlong reward shaping.

## Technical Details
- **Dynamic sampling (§3.2).** "If all outputs {o_i} of a particular prompt are correct and receive the same reward, the resulting advantage for this group is zero. A zero advantage results in zero policy gradients, shrinking the magnitude and increasing the noise sensitivity of the batch gradient." The constraint added to the objective is `0 < |{o_i | is_equivalent(a, o_i)}| < G` (Eq. 11). The number of all-correct prompts keeps rising during training (Figure 3b), so the effective batch shrinks without this filter.
- **Token-level loss (§3.3).** Under per-response averaging, tokens in longer responses contribute less to the loss; the report states that this both weakens learning from high-quality long samples and fails to penalize repetitive or degenerate patterns inside long samples, producing an unhealthy increase in entropy and response length (Figures 4a, 4b).
- **Configuration (§4.1).** verl framework; AdamW with constant learning rate 1e-6 and a linear warm-up over 20 rollout steps; rollout prompt batch 512 with 16 responses per prompt; training mini-batch 512, i.e. 16 gradient updates per rollout step; expected maximum length 16,384 tokens with a 4,096-token soft-punish cache, so maximum generation is 20,480 tokens; ε_low = 0.2, ε_high = 0.28. Evaluation on AIME repeats the set 32 times and reports avg@32 at temperature 1.0 and top-p 0.7.
- **Ablation (Table 1, AIME24 avg@32).** DeepSeek-R1-Zero-Qwen-32B 47; naive GRPO 30; + overlong filtering 36; + clip-higher 38; + soft overlong punishment 41; + token-level loss 42; + dynamic sampling (DAPO) 50.
- **Cost note (§3.2).** Dynamic sampling needs more generated samples per batch, but the report states the same performance is reached in a comparable amount of time because fewer steps are needed.
- **Token-level loss effect (§4.2).** The report states token-level loss brings less accuracy improvement than the other techniques but improves training stability and makes the length increase more healthy.

## Findings relevant to negative feedback and generality
- **Negative as gradient.** Clip-higher's asymmetry is explicitly sign-aware: the upper bound is loosened so that unlikely tokens can be promoted, while the lower bound is left tight because loosening it would drive suppressed tokens' probability to zero (§3.1).
- **Discarded signal.** Dynamic sampling discards both all-correct and all-incorrect groups; these carry no gradient under a group baseline, so the discard is about batch efficiency, not about the value of negatives (§3.2).
- **Reward noise from truncation.** A punitive reward for truncated samples penalizes sound reasoning for being long; masking those samples' loss stabilized training and raised AIME accuracy (§3.4, Figure 5).
- **Measurement limit.** All reported results are AIME 2024 avg@32 on one base model (Qwen2.5-32B). No held-out general-capability benchmarks are reported, and pass@k at large k is not measured; [[rlvr-beyond-base-model]] reports DAPO with the highest pass@1 and a drop at k = 256 in its own comparison (its §C.5).

## Connections
- [[grpo]] — the baseline objective all four modifications are applied to.
- [[dr-grpo]] — a different fix for the same length-normalization bias.
- [[trl-grpo]] — implements `loss_type="dapo"` as the default and `epsilon_high`; its docstring cites ε_high = 0.28 from this report.
- [[entropy-mechanism-llm-rl]] — measures clip-higher against covariance-based interventions on the same entropy-collapse problem.

## Verification
- Checked on 2026-09-15 against the cached primary text of https://arxiv.org/abs/2503.14476 (arXiv v2, 2025-05-20): Abstract, §3.1–§3.4, §4.1, §4.2, Table 1.
- Corrections to the previous card version: no previous card existed in the library for this slug. A claim carried by the earlier ch-46 read.md, that `top_entropy_quantile` implements "the DAPO / Muon trick", is not in this report; that parameter follows "Beyond the 80/20 Rule" (arXiv:2506.01939) per [[trl-grpo]] (grpo_config.py L276–281).
- Not reported by the source: held-out general-capability evaluations, pass@k at large k, the soft-punishment reward formula's coefficients beyond the cache size, number of RL steps for each ablation row.

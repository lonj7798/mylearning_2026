---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: github.com/verl-project/verl@753aed3 verl/trainer/ppo/core_algos.py and docs/algo/{grpo,dapo}.md; library card [[verl-grpo]] (verified 2026-09-14)
source_url: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/trainer/ppo/core_algos.py#L266-L530
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; aligned with the verified card, commit pinned)"
---

# Excerpt: verl's group advantage — what happens to all-equal and size-1 groups

Used by [[read]] §5, §6, §7, and the Common-mistakes table.

## `compute_grpo_outcome_advantage` (core_algos.py L311–329, verbatim)
```python
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
```
`scores[i]` is the summed token reward of response `i` (L304); groups are formed by prompt `uid`; `torch.std` is the sample standard deviation (divisor `G−1`); `ε = 1e-6` (L272). The same scalar is broadcast to every token of the response.

## Behaviour the chapter uses
- **All-equal group:** `r_i − mean = 0` exactly, so the advantage is 0 with or without the std division; the tokens stay in `response_mask` and still count in the denominator under the default `token-mean` aggregation (L329; L1170–1175).
- **Size-1 group:** mean 0 and std 1, so the advantage is the raw reward divided by `1 + 1e-6`. The default `actor_rollout_ref.rollout.n` is 1 (rollout.yaml L127), which is not a group baseline at all — the docs tell the user to raise it for GRPO.
- **Worked example at G = 4 with a binary reward** (derived from L319–326): one correct of four gives `μ = 0.25`, `σ = 0.5`, so +1.5 for the correct response and −0.5 for each wrong one; three correct gives +0.5 and −1.5; zero or four correct gives 0 for every response. Without std scaling the same cases give +0.75 / −0.25 and +0.25 / −0.75.
- **Pass@k estimator** (`grpo_passk`, L471–530, implementing arXiv:2503.19595): only the highest-reward response gets `r_max − r_second_max`; all advantages are ≥ 0, so no response receives a negative advantage.

## Documented configurations
- GRPO: `algorithm.adv_estimator=grpo`, `rollout.n > 1`, `actor.use_kl_loss=True` with `kl_loss_coef=0.001`, `kl_loss_type` one of `kl(k1) | abs | mse(k2) | low_var_kl(k3) | full`, `loss_agg_mode=token-mean` (docs/algo/grpo.md L29–47). The docs state that the original paper's sample-level aggregation (`seq-mean-token-mean`) "may be unstable in long-CoT scenarios".
- Dr. GRPO: `loss_agg_mode=seq-mean-token-sum-norm`, optional `loss_scale_factor` set to a constant such as the max response length, `use_kl_loss=False`, `algorithm.norm_adv_by_std_in_grpo=False` (grpo.md L57–62).
- DAPO: `clip_ratio_low: 0.2`, `clip_ratio_high: 0.28`; `algorithm.filter_groups` (default off) drops groups whose metric values are all equal and resamples; overlong buffer `max_response_length: 20480`, `overlong_buffer.len: 4096`, `penalty_factor: 1.0`, applied as `overlong_reward = min(-exceed_len / overlong_buffer_len * penalty_factor, 0)` (docs/algo/dapo.md).
- DAPO reproduction on Qwen2.5-32B (dapo.md L36–38): 52% AIME 2024 with dynamic sampling, 50% without, 44% without token-level loss and dynamic sampling. One run per row; the 44% run used different hardware and a different image, and no other benchmark is reported, so the table does not measure breadth.
- FAQ: "Most experiments in the paper, including the best-performant one, are run without Overlong Filtering because it's somehow overlapping with Overlong Reward Shaping" — a conflict with Table 1 of [[dapo]].

## Claims removed from the earlier version of this excerpt
"Singleton groups get zero advantage", "per-token broadcast is the source of the length bias" (the length term is set by `loss_agg_mode`, not by this function), and "verl is what most R1 reproductions run".

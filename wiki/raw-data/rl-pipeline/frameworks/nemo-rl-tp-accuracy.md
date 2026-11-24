<!-- scope: NVIDIA NeMo RL guide "Automodel Tensor Parallel Accuracy Issue" (docs/guides/automodel-tp-accuracy.md): token_mult_prob_error metric, reward-model loss vs TP size, batch-variant kernels (train_micro_batch_size vs logprob_batch_size), row-wise TP cross-GPU reductions, numerically stable colwise plan
     deps: []
     see-also: [[mai-rl-infrastructure]], [[thinking-machines-nondeterminism]], [[fp16-train-infer-mismatch]], [[truncated-importance-sampling]], [[verl-rollout]]
-->

# NeMo RL: Automodel Tensor Parallel Accuracy Issue
- **Core Insight:** In NeMo RL's Automodel (formerly DTensor) backend, TP > 1 produced persistent training–inference log-prob drift and degraded accuracy. The two dominant causes were (1) batch-variant kernels, triggered when `train_micro_batch_size` ≠ `logprob_batch_size`, and (2) row-wise TP, whose cross-GPU sums of partial outputs change the floating-point addition order.
- **Guideline:** Set `train_micro_batch_size` equal to `logprob_batch_size`; in strictly on-policy runs `loss_fn.force_on_policy_ratio` may set the ratio to 1; replace row-wise `o_proj` and `down_proj` sharding with a colwise plan that gathers inputs and produces replicated local outputs, so no inter-GPU add sits on the critical path. Alert when `token_mult_prob_error` exceeds 1.05.
- **Authors:** NVIDIA NeMo RL team (no named author in the file; added in PR #1767).
- **Year:** 2026 (first committed 2026-01-15 as `dtensor-tp-accuracy.md`, commit 67bfd4d24d; renamed to `automodel-tp-accuracy.md` 2026-10-19, commit c1d50c6e92).
- **URL:** https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/automodel-tp-accuracy.md
- **Source type:** framework documentation
- **Relevant topics:** tensor parallelism, row-wise vs column-wise sharding, batch-variant kernels, train–inference log-prob mismatch, importance ratio, reward-model numerics

## Summary
The guide reports three TP-related symptoms when using the Automodel policy worker: severe `token_mult_prob_error` spikes with an upward trend while post-training Qwen3-4B-Instruct-2507 (Fig. 1); reward-model validation loss that differs by TP size (Table 1); and degraded downstream performance with TP > 1 (Figs. 2–3). The problem was model-dependent (seen on Qwen3-4B, "not on Llama-3.1-8B-Instruct") and poorly reproducible. The analysis attributes it to a small set of interacting factors, with batch-variant kernels and the row-wise TP plan as the most significant.

## Key Contributions
- A training-vs-generation consistency metric, `token_mult_prob_error` = (1/Σm_i) Σ m_i · exp(|g_i − p_i|), with g = generation log-probs, p = policy log-probs, m = token mask; ideal value 1.0, investigate above 1.05.
- Root cause 1: unequal train and log-prob micro-batch sizes make `current_logprobs ≠ prev_logprobs`, so the on-policy ratio deviates from 1 (Figs. 4–5, DeepSeek-R1-Distill-Qwen-7B at TP=4).
- Root cause 2: row-wise TP sums partial outputs y = Σ y_i across GPUs; the result depends on collective pattern (ring/tree AllReduce), scheduling and library internals.
- A drop-in `numerical_stable_parallel_plan` (PR #1235).

## Key Figures/Tables to Study
**Table 1 — reward-model validation loss by TP size:**

| | TP=1 | TP=2 | TP=4 | TP=8 |
|---|---|---|---|---|
| With autocast | 0.6035 | 0.6010 | 0.5864 | 0.6021 |
| W/O autocast | 0.6035 | 0.6010 | 0.5864 | 0.6021 |

Identical rows with and without autocast show that mixed precision is not the cause; the guide attributes the difference to "how different TP plans partition and aggregate computations across devices". Fig. 6 shows KL between Hugging Face and NeMo RL logits on the same input and weights.

## Technical Details
**Why bad `token_mult_prob_error` does not always show in reward (Sec. "Overall Performance Degradation").** (1) Spikes, not continuous growth; (2) spikes are stochastic, not present in every run on the same batch; (3) with large datasets, sporadic spikes are diluted. The guide still treats the instability as unacceptable for an RL framework.

**Batch-variant kernels.** The ratio exp(current_logprobs − prev_logprobs) should equal 1 on-policy. A mismatch between `train_micro_batch_size` and `logprob_batch_size` sends the same sequence through different kernel paths (reduction order, tiling, fused-kernel heuristics, batch-size-conditioned algorithm selection). After aligning the two sizes, "the importance-sampling ratio (`probs_ratio`) becomes 1 as expected, and the observed accuracy issues disappear". `force_on_policy_ratio` is stated as valid only when data are guaranteed to come from the current policy.

**Row-wise vs column-wise.** For y = xWᵀ with TP = 2: row-wise splits Wᵀ by input dimension and needs y = y_1 + y_2 across GPUs; column-wise splits by output dimension and concatenates. On one device, the matmul and accumulation schedule is fixed for a given problem size, so rounding is "at least consistent"; across devices it is not guaranteed.

**Stable plan (changed entries only).**
```python
"model.layers.*.self_attn.o_proj": ColwiseParallel(
    input_layouts=Shard(-1), output_layouts=Replicate(), use_local_output=True),
"model.layers.*.mlp.down_proj": ColwiseParallel(
    input_layouts=Shard(-1), output_layouts=Replicate(), use_local_output=True),
```
The default plan has `RowwiseParallel()` for both; q/k/v, up/gate and `lm_head` stay colwise; `embed_tokens` stays `RowwiseParallel(input_layouts=Replicate())`.

**Other factors (treated as background noise).** Attention backend (SDPA vs flash attention 2), GPU architecture (Ampere vs Hopper), precision (BF16/FP16/FP8/FP32). References listed: [[fp16-train-infer-mismatch]], PyTorch reduced-precision reduction notes, "Systematic Outliers in LLMs", and the "When Speed Kills Stability" training-inference mismatch note. Future work: unify attention backends, TP-invariant kernels, compensated summation.

**Relevance.** [[mai-rl-infrastructure]] (p. 10) cites this guide for row-parallel reductions and solves the same problem differently: it replicates the attention output-projection weights after an all-gather in the sampler and runs the learner without TP.

## Verification
Fetched raw Markdown from `main` on 2026-11-24 (main HEAD a03ffa1c16; file last changed in c1d50c6e92). Commit dates from the GitHub API. Table 1 and code are copied from the file; figure images were not viewed.

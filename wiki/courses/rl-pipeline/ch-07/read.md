<!-- chapter: ch-07
     track: rl-pipeline
     kind: content
     title: Bitwise Sampler–Learner Parity
     deps: [ch-06]
     sources: [[mai-rl-infrastructure]], [[mai-thinking-1]], [[thinking-machines-nondeterminism]], [[flash-attention-stability]], [[nemo-rl-tp-accuracy]], [[deepseek-train-infer-consistency]], [[truncated-importance-sampling]], [[fp16-train-infer-mismatch]], [[verl-rollout-correction-and-lowprec]], [[slime-agentic-and-mismatch-examples]], [[slime]], [[open-instruct-rlvr-pipeline]], [[olmo-3-rl-pipeline]], [[scalerl]], [[areal-async-rl]]
     figures: figures/reduction-order.html
-->

# Chapter 7 — Bitwise Sampler–Learner Parity

> **Core insight.** Matching storage and compute precision (ch-06) does not make the sampler and the learner produce the same token probabilities. With identical checkpoints and inputs, the two still differ because they use different kernels, different parallelism, a different order of floating-point additions, and different attention tiling ([[mai-rl-infrastructure]] p. 9). Floating-point addition is not associative, so any change in the order or grouping of a reduction changes the rounded result. Rocket measured a larger sampler–learner log-probability gap at a later RL checkpoint than at initialization on identical inputs, which it reads as a sign that "optimization may amplify or exploit numerical differences" (p. 9). It removes the gap by forcing every reduction to have the same structure on both sides: (1) the sampler keeps attention tensor parallelism (TP) but replicates the output-projection weights and all-gathers head outputs, so no cross-GPU sum follows the projection, and the learner runs without TP; (2) MoE experts use expert parallelism without intra-expert TP, outputs return by all-to-all, and the routing-weighted sum is computed locally in FP32; (3) one FA4 attention implementation serves learner forward, sampler prefill and sampler decode, with sliding-window K-blocks aligned to the learner's tiles and one fixed KV-split size (p. 10). Open frameworks that do not reach parity correct the remaining gap at the loss: a truncated importance weight (TIS), token or sequence masking (MIS, IcePop, rejection), or a precision change (FP16 in both engines).
>
> **Guideline.** When the sampler and learner use different TP layouts, remove row-parallel cross-GPU sums after `o_proj` and `down_proj`, because NeMo RL measured reward-model validation loss of 0.6035 / 0.6010 / 0.5864 / 0.6021 at TP = 1 / 2 / 4 / 8 with identical values with and without autocast, which rules out mixed precision as the cause ([[nemo-rl-tp-accuracy]] Table 1). When the learner computes old log-probs with a different micro-batch size than the training forward, make them equal, because NeMo RL reports that after aligning `train_micro_batch_size` and `logprob_batch_size` the on-policy ratio became 1 and the accuracy issues disappeared. When parity is not achieved, record sampler log-probs and multiply the loss by min(π_learner/π_sampler, C) computed at θ_old and applied outside the PPO clip, because with INT8 rollouts on Qwen2.5-0.5B GSM8K this truncated weight reached BF16-level accuracy while PPO-IS and untruncated IS fell to near 0 ([[truncated-importance-sampling]] Fig. 2, Fig. 4). When you can change precision for RL fine-tuning of a pre-trained model, consider FP16 in both engines, because the sequence-level mismatch was about 24× smaller than BF16 and BF16 token-level TIS still collapsed on the paper's sanity test ([[fp16-train-infer-mismatch]] Sec. 3.5, 4.2). In every case, log a mismatch metric from step 0, because the gap is not constant during RL ([[mai-rl-infrastructure]] p. 9) and runs that collapsed showed growing mismatch beforehand ([[fp16-train-infer-mismatch]] Sec. 4.2).

---

## Why this chapter matters

An RL step computes a gradient on tokens that the sampler generated, using probabilities that the learner computes. The policy-gradient estimate is E_{a∼π_sampler}[R · ∇ log π_learner(a)] ([[truncated-importance-sampling]] Summary). If π_sampler ≠ π_learner for the same weights, the run is off-policy even when it is configured as on-policy, and the size of that off-policyness is set by implementation details nobody chose as hyperparameters.

Two separate terms make up the sampler–learner log-probability difference. slime's systems documentation writes it as

```
log p_{v_t}(a|h) − log q_{v_s}(a|h)
   = [log p_{v_t} − log p_{v_s}]   (parameter staleness)
   + [log p_{v_s} − log q_{v_s}]   (same-version engine mismatch)
```

where `p` is the learner, `q` the sampler, `v_s` the weight version that sampled the token and `v_t` the version that trains on it (`slime@79aa41a docs/en/advanced/rl-systems.md` L33–41). Chapters ch-03 to ch-05 manage the first term. This chapter is about the second term. The same document states that deterministic kernels "can reduce the second term, not the first" (L41).

Rocket's Fig. 3 shows why removing the second term matters for interpretation: the mean |log-prob difference| per rollout rises with staleness with a linear fit R² = 0.76, and at staleness 0 the label reads "Same weights: exact match" ([[mai-rl-infrastructure]] Fig. 3, p. 5). The exact match at staleness 0 is the result of the work in this chapter. Without it, the intercept of that plot would be a nonzero engine-mismatch floor, and the measured staleness effect would be mixed with numerical noise.

For a weeks-long climb from a base model, the post argues that mismatches "have little time to compound" in a short distilled-start run, while climbing from base removes that escape (p. 1). That is an argument, not a measurement of compounding; the measured evidence is the later-checkpoint gap (p. 9) and the Fig. 6 comparison (§6 below).

---

## §1 Sources of mismatch beyond precision

The mechanisms below each change the bits of a token's log-probability while the weights and input stay fixed.

**Floating-point non-associativity.** `(0.1 + 1e20) − 1e20` evaluates to 0, and `0.1 + (1e20 − 1e20)` evaluates to 0.1. Summing one 8-element array in random orders 10,000 times produced 102 distinct results ([[thinking-machines-nondeterminism]] Technical Details). Every mechanism below is a way of changing the order or grouping of a sum.

**Batch-variant kernels.** A kernel that changes its reduction strategy with batch size gives a token different numerics depending on how many other tokens share its batch. The Thinking Machines post identifies split reductions in RMSNorm, split-K and stream-K in matmul, smaller tensor-core instructions at small shapes, and split-KV attention with a fixed *number* of splits. It also shows that atomic adds are not the cause: the LLM forward pass "usually" contains none ([[thinking-machines-nondeterminism]] Technical Details). Measured effect: Qwen3-235B-A22B-Instruct-2507, 1,000 completions at temperature 0 of 1,000 tokens each gave 80 unique completions, identical for 102 tokens, then 992 continued with "Queens, New York" and 8 with "New York City" (Experiments).

In an RL learner, the same effect appears when the old-log-prob forward and the training forward use different micro-batch sizes. NeMo RL found that unequal `train_micro_batch_size` and `logprob_batch_size` made `current_logprobs ≠ prev_logprobs`, so the on-policy ratio was not 1 ([[nemo-rl-tp-accuracy]] Technical Details).

**Parallelism.** Under row-parallel TP, y = xWᵀ is split by the input dimension and each GPU produces a partial output; the final value is y = Σ y_i summed across GPUs. The order of that sum depends on the collective (ring or tree all-reduce), scheduling and library internals. On one device the matmul and accumulation schedule is fixed for a given problem size, so rounding is "at least consistent" ([[nemo-rl-tp-accuracy]] Technical Details). The TIS authors counted high-mismatch responses (max |p_sampler − p_learner| > 0.5) on 512 DAPO-Math prompts: one response with vLLM TP1 / FSDP SP1, two with sampler TP2, "double digits" when the learner also used SP8; equal TP on both sides (TP2/TP2, TP4/TP4) gave few. Mean mismatch and KL showed no significant difference across these configurations, so the effect is in the tail ([[truncated-importance-sampling]] Mismatch factors).

**Attention tiling.** FlashAttention computes softmax online over K/V tiles and rescales partial results per tile. At BF16, its output deviates from an FP64 reference about 10× more than standard attention does; deviation grows with sequence length at fixed tile size and shrinks with larger tiles ([[flash-attention-stability]] Sec. IV, Figs. 4–6). Two attention implementations with different tile shapes therefore give different outputs on identical inputs.

**Chunking and cache layout.** Attention must give the same numerics whether a sequence is processed at once or in chunks (chunked prefill, prefix cache). With block size 32, handling 80 cached tokens and 48 new tokens separately produces 5 blocks instead of 4, which changes the reduction grouping ([[thinking-machines-nondeterminism]] Technical Details). Decode uses split-KV; FlashInfer's balanced scheduling chooses the split size from occupancy, which is not invariant.

**Discrete choices that amplify small differences.** A tiny logit difference can flip an MoE top-k expert selection or move a token across a top-p boundary. MAI-Thinking-1 replays the sampler's MoE routing and top-p mask in the learner; without top-p mask replay it reports "catastrophic off-policy mismatch, causing divergence within a few training steps" ([[mai-thinking-1]] Sec. 3.1.3, 3.6.4). The TIS authors expect MoE to be worse than dense because of "precision-sensitive routing" and special kernels (Mismatch factors).

**Precision.** BF16 has 7 mantissa bits and FP16 has 10; the next representable value above 1 is 1.007812 in BF16 and 1.000977 in FP16 ([[fp16-train-infer-mismatch]] Table 1). Coarser rounding makes every order difference above visible in more cases.

slime's documentation gives a first-order view of how logit perturbations reach log-probs. If sampler logits are z + δ and learner logits are z, then log q_a − log p_a ≈ δ_a − Σ_j p_j δ_j: a shift common to all logits cancels; only token-dependent perturbations change probabilities, and the approximation fails near discontinuities such as expert selection (`rl-systems.md` L47–57).

| Source | Mechanism | Measured symptom | Source |
|---|---|---|---|
| Non-associativity | sum order | 102 results from 10,000 orders of 8 numbers | [[thinking-machines-nondeterminism]] |
| Batch variance | split-K, split reductions, split-KV count | 80 unique completions of 1,000 at T = 0 | [[thinking-machines-nondeterminism]] |
| Micro-batch mismatch | different kernel path for logprob vs train forward | ratio ≠ 1 on-policy; fixed by equal sizes | [[nemo-rl-tp-accuracy]] |
| Row-parallel TP | cross-GPU partial sums | RM loss 0.5864–0.6035 across TP 1–8 | [[nemo-rl-tp-accuracy]] Table 1 |
| Attention tiling | per-tile rescaling | ~10× deviation of Flash vs standard at BF16 | [[flash-attention-stability]] |
| Top-p / routing | discrete boundary flips | divergence within a few steps without top-p mask replay | [[mai-thinking-1]] Sec. 3.1.3 |
| Precision | 7 vs 10 mantissa bits | ~24× larger sequence-level mismatch in BF16 than FP16 | [[fp16-train-infer-mismatch]] |

---

## §2 Why the gap matters, and why it grows during RL

**Effect on the gradient.** With identical parameters, some tokens had probability near 1 under vLLM and near 0 under FSDP ([[truncated-importance-sampling]] Core Insight). In the DAPO run on Qwen2.5-32B the maximum token-probability difference was ≈ 1.0 (Experiments). Tokens with such gaps receive gradients computed as if the learner had sampled them.

**Observed growth.**
- Rocket: on identical inputs, a later RL checkpoint showed a larger sampler–learner gap than the initialization checkpoint ([[mai-rl-infrastructure]] p. 9). This is one controlled comparison. The mechanism ("may amplify or exploit") is stated as a suggestion, not tested.
- [[fp16-train-infer-mismatch]] Sec. 4.2: runs that collapsed in BF16 showed growing mismatch before the collapse, with π and µ moving toward extreme values (one near 1, the other near 0). This is a temporal association across runs, not an intervention.
- Response length: 20K-token responses had higher max mismatch than 4K responses, and the first 4K tokens of a 20K response already exceeded an independent 4K response ([[truncated-importance-sampling]] Mismatch factors). Long agentic and long-CoT rollouts are therefore the worst case.
- MAI-Thinking-1 reports that mixed-language chains of thought "correlate with spikes in log-probability divergence between the training and inference policies" and adds a language-consistency reward partly for that reason ([[mai-thinking-1]] Sec. 3.1.2). This is a correlation; the report does not show that removing foreign tokens removes the divergence spikes.

**Proposed mechanism for collapse.** For tokens with π_learner/π_sampler < 1 and negative advantage, the learner keeps pushing probability down because the sampler, which generated the token, does not reflect the decrease; the TIS authors propose this can collapse entropy ([[truncated-importance-sampling]] Failure signatures). INT8-rollout runs showed entropy "dropping below 0.2 and continuing to decrease", abnormally long responses, and a k1 KL estimate that was frequently negative. The mechanism is a hypothesis consistent with these signatures; the post does not isolate it.

**End-to-end comparison on one setup.** Thinking Machines ran RLVR on Bigmath from Qwen 2.5-VL instruct 8B with max rollout 4,096: without importance weighting, reward collapsed with a loss spike around step 318 and a matching KL spike; with importance weighting KL stayed "around 0.001 with occasional spikes"; with a bitwise-identical sampler and trainer, KL was a flat line at 0 and training was smooth ([[thinking-machines-nondeterminism]] Experiments). One run per condition.

---

## §3 Rocket fix 1: no cross-GPU sum after the attention output projection

**Problem.** In a standard Megatron-style TP layout, q/k/v projections are column-parallel (each GPU owns some heads) and `o_proj` is row-parallel: each GPU multiplies its local head outputs by its slice of `W_o` and the partial results are summed by an all-reduce. The all-reduce order is a property of the collective and the rank count. A learner with a different TP degree (or none) sums in a different order. The two outputs differ in the last bits, and that difference propagates through the remaining layers.

**Rocket's change** ([[mai-rl-infrastructure]] p. 10):
1. The sampler keeps TP for attention "for performance" (q/k/v and the attention computation stay sharded by head).
2. Before `o_proj`, the sampler all-gathers the attention-head outputs, so every GPU holds each token's full concatenated head vector.
3. The `o_proj` weights are replicated on every TP rank, so each GPU computes a token's full projection with one local matmul. No reduction across GPUs follows the projection.
4. The learner "operates without TP, allowing its summation order to match the sampler's".

The open-source equivalent is NeMo RL's "numerically stable" plan, which replaces the row-wise entries with column-wise ones whose output is replicated (quoted from `docs/guides/automodel-tp-accuracy.md` at NeMo RL main a03ffa1c16, via [[nemo-rl-tp-accuracy]]):

```python
"model.layers.*.self_attn.o_proj": ColwiseParallel(
    input_layouts=Shard(-1), output_layouts=Replicate(), use_local_output=True),
"model.layers.*.mlp.down_proj": ColwiseParallel(
    input_layouts=Shard(-1), output_layouts=Replicate(), use_local_output=True),
```

The default plan has `RowwiseParallel()` for both entries. NeMo RL changes `down_proj` as well because its target models have dense MLPs; Rocket's MoE layers do not use intra-expert TP at all (fix 2), so only the attention projection needs the change. Rocket does not report the memory cost of replicating `W_o` or the bandwidth of the extra all-gather; it states only that parity was achieved "with no noticeable performance degradation" (p. 10).

**Worked example.** The interactive figure [reduction-order.html](figures/reduction-order.html) computes a toy `o_proj` dot product in emulated BF16: (a) on one device with a fixed order, as the learner does; (b) row-parallel over 2 or 4 ranks with partial sums rounded to BF16 and all-reduced in ring or tree order; (c) Rocket's all-gather plus replicated weights. It counts how often (b) differs bitwise from (a) over random inputs and shows that (c) matches (a) in every trial. The inputs are synthetic; the figure demonstrates the mechanism, not Rocket's measured rate.

---

## §4 Rocket fix 2: local FP32 combination of MoE expert outputs

**Problem.** A token routed to k experts needs the weighted sum Σ_i g_i · E_i(x), where g_i are normalized routing weights. If experts are split by intra-expert TP, each expert output itself requires a cross-GPU reduction; if the weighted sum is accumulated in BF16 as expert outputs arrive, the arrival order from the all-to-all changes the result.

**Rocket's change** ([[mai-rl-infrastructure]] p. 10): experts are distributed by expert parallelism (EP) with no TP inside an expert; an all-to-all returns the individual expert outputs to the token's source device; the weighted sum is computed there, locally, in FP32. "Matching the accumulation precision and reduction order between the sampler and learner eliminates discrepancies introduced by expert combination."

**Open counterpart: slime's GLM-5 alignment path.** slime documents a train/rollout log-prob alignment mode for the GLM-5 structure that uses the same rule: "the token owner performs the weighted reduction in slot order and FP32", and "Ordinary Megatron all-to-all is not an alignment backend for this path" (`slime@79aa41a docs/en/advanced/reproducibility.md` L63–66). The learner-side MoE wrapper applies the router probability after the second expert GEMM in FP32, matching SGLang:

```python
# slime@79aa41a slime/backends/megatron_utils/alignment/deepgemm_moe_forward.py L86-114 (docstring trimmed)
def _apply_router_probability_fp32_inplace(
    down_output: torch.Tensor,
    permuted_probs: torch.Tensor,
) -> torch.Tensor:
    """Apply the post-fc2 router probability without a full-size FP32 temporary.

    SGLang performs this multiply in FP32 and casts the result back to the
    activation dtype. ...  Processing independent row chunks is numerically identical
    while keeping the FP32 workspace bounded.
    """
    if down_output.ndim != 2:
        raise RuntimeError(f"Expected a 2D MoE output, got shape {tuple(down_output.shape)}")
    probabilities_fp32 = permuted_probs.detach().reshape(-1, 1).to(torch.float32)
    ...
    for start in range(0, down_output.shape[0], _ROUTER_PROBABILITY_CHUNK_ROWS):
        end = min(start + _ROUTER_PROBABILITY_CHUNK_ROWS, down_output.shape[0])
        scaled = (down_output[start:end].to(torch.float32) * probabilities_fp32[start:end]).to(down_output.dtype)
        down_output[start:end].copy_(scaled)
    return down_output
```

The same document lists the other conditions slime needed: DeepGEMM batch-invariant block-FP8 forward for dense and grouped-MoE layers, an FP32 MoE router, a deterministic DSA sparse-attention path, and matched KV-cache dtype (L58–72). It states that "the LM head stays bf16 on both train and rollout — matching precision, not fp32, is what aligns" (L62). The regression gate trains a 6-layer GLM-5.2 through a real Megatron → SGLang weight update and asserts `train_rollout_logprob_abs_diff < 1e-6`; the H100 reference value is about 2e-7 (`tests/test_glm52_6layer_deterministic_e2e.py` L6–7, L50; doc L74). A second gate requires an absolute hidden-state difference of exactly zero for decoder layers 0–5 (doc L76–78). The environment variables shared by both sides are centralized in one function (`slime/backends/megatron_utils/alignment/env.py` L19–56), which includes `SGLANG_DEEPGEMM_BATCH_INVARIANT=1` and Megatron flags that make the training side call SGLang's fused kernels (`MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS`, `..._ROUTER_GEMM`, `..._ROPE`).

Two points of contrast with the FP32-head literature. [[scalerl]] reports that computing LM-head logits in FP32 raised its fitted asymptote from 0.52 to 0.61 (Fig. 5b), and the TIS authors tried an FP32 `lm_head` patch in vLLM and found "the mismatch problem still exists" ([[truncated-importance-sampling]] Summary). slime's statement and the TIS result agree: higher precision on one side does not create equality; equal precision and equal order do. ScaleRL's gain is measured; why it occurs (smaller mismatch or something else) is not isolated in the card.

**MAI-Thinking-1 used a different approach.** Rocket v1 kept BF16 in both learner and inference and added MoE routing replay and top-p mask replay ([[mai-thinking-1]] Sec. 3.6.4). Routing replay forces the same discrete expert choice; it does not make the continuous combination bitwise equal. The Base 2 work in the infrastructure post goes further and removes the numeric differences themselves.

---

## §5 Rocket fix 3: one attention kernel, aligned tiles, one fixed KV split

**Problem.** The learner's forward pass processes whole sequences in tiles chosen for training throughput. Sampler prefill processes prompts (with prefix cache), and sampler decode processes one query token at a time against a long KV cache, usually with split-KV to keep SMs busy. Three code paths, three tilings, three reduction groupings.

**Rocket's change** ([[mai-rl-infrastructure]] p. 10):
1. One shared FA4 implementation is used for learner forward, sampler prefill and sampler decode.
2. For sliding-window attention, the sampler's K-block boundaries are aligned with the learner's tiles, and positions outside the valid window are masked. The compact KV-cache layout used for sliding windows would otherwise "change token grouping in the online softmax and attention-value reductions".
3. The number of KV tokens per split is tuned once "to maximize overall throughput in our training setup", then that fixed value and matching split boundaries are used in all three paths.

Point 3 is the Thinking Machines rule: use a fixed split *size*, not a fixed *number* of splits. With KV length 1,000 and split size 256, the splits are 256, 256, 256, 232, independent of batch or occupancy, where a fixed count of four would give 250 each ([[thinking-machines-nondeterminism]] Key Contributions). Thinking Machines also updates the KV cache and page table before the attention kernel so cached and new tokens are processed in one block structure.

**DeepSeek-V4's alternative.** DeepSeek-V4 does not use split-KV for decode at all. Kernel 1 computes a whole sequence on one SM for fully occupied waves; kernel 2 uses several SMs per sequence for the last partially filled wave, with a calculation path designed so that "its accumulation order is the same as that of the first kernel", exchanging data through distributed shared memory. Matmuls move from cuBLAS to DeepGEMM end to end, and split-K is abandoned "in most scenarios" ([[deepseek-train-infer-consistency]] Sec. 3.3). DeepSeek's target is batch invariance inside each kernel; Rocket's target is equal reduction structure across three execution paths. Both remove the same term.

**Learner-side determinism already existed in MAI's stack.** The MAI-Thinking-1 report describes the YOLO learner as bitwise reproducible across runs, with deterministic accumulations "in prescribed orders": RMSNorm backward uses a two-stage tiled reduction with a fixed-order finalization kernel, and MoE top-k uses a stable sort to avoid nondeterministic tie-breaking ([[mai-thinking-1]] Sec. 2.8, read from the PDF). Run-to-run determinism of the learner is a prerequisite for parity but is not parity: it says two learner runs agree, not that learner and sampler agree.

**Cost evidence.**

| Implementation | Reported cost | Evidence type | Source |
|---|---|---|---|
| Thinking Machines batch-invariant vLLM, Qwen3-8B, 1,000 seqs | 26 s default → 55 s unoptimized deterministic → 42 s with improved attention | measured, one GPU | [[thinking-machines-nondeterminism]] |
| Batch-invariant Triton matmul | "about 20%" slower than cuBLAS (unoptimized) | measured | [[thinking-machines-nondeterminism]] |
| DeepSeek-V4 dual-kernel decode | "negligible" | stated, no number | [[deepseek-train-infer-consistency]] |
| DeepSeek-V4 DeepGEMM without split-K | "match or even surpass" split-K in most scenarios | stated | [[deepseek-train-infer-consistency]] |
| Rocket, all three fixes | "no noticeable performance degradation" | stated, no number | [[mai-rl-infrastructure]] p. 10 |
| FP32 inference + BF16 training (alternative) | "nearly three times slower" inference | measured | [[fp16-train-infer-mismatch]] Sec. 4.4 |

---

## §6 Evidence that parity improves RL

**Rocket Fig. 6** ([[mai-rl-infrastructure]] p. 9). Model L42 from the MAI Base 2 scaling ladder: 42 layers, 7B active and 180B total parameters. Two RL runs start from the same checkpoint and were launched roughly one month apart. The baseline run "predates the numerical consistency improvements"; the bitwise-matched run "incorporates them". The y-axis is mean pass@1 on an in-house math reasoning benchmark (axis ≈ 20–60%), the x-axis checkpoint iteration 20–2,000 on a log scale. The matched run "achieves higher evaluation scores at comparable training steps". Curve values were not read from the figure.

How strong is this evidence? It is one pair of runs. Because the runs were launched a month apart and the baseline predates the improvements, any other change made to the code or configuration in that month is not excluded by the text. The post presents it as a comparison, not as a controlled ablation; I read it as consistent with parity helping, with confounds not ruled out.

**Thinking Machines** (§2 above): smooth training and KL = 0 with bitwise parity vs collapse without correction. One run per arm, one model.

**MAI-Thinking-1:** BF16 in both engines "yields substantially smaller numerics gaps and more stable training than the lower-precision alternatives we evaluated" (Sec. 3.6.4). No numbers are given.

**What is not reported anywhere in the sources read:** a parity ablation at the 3T-parameter scale, the size of the residual gap in Rocket before the fixes, or which of the three fixes contributes most.

---

## §7 When parity is not achieved: the open-source answers

Most open stacks pair a vLLM or SGLang sampler with an FSDP or Megatron learner and do not reach bitwise parity. They use one of four answers.

### §7.1 Truncated importance sampling (TIS)

The correction multiplies the gradient by a truncated ratio of learner to sampler probability ([[truncated-importance-sampling]] Key Contributions):

- REINFORCE: E_{a∼π_sampler}[min(π_learner(a, θ)/π_sampler(a, θ), C) · R(a) · ∇ log π_learner(a, θ)].
- PPO: the PPO ratio stays π_learner(θ)/π_learner(θ_old); a separate factor min(π_learner(a, θ_old)/π_sampler(a, θ_old), C) multiplies the clipped PPO term.

Why the two variants it replaces fail (TIS Analysis):
- *Vanilla IS* (no truncation) multiplies gradient variance by the squared ratio; a ratio of 16 amplifies noise 256×, against 4× under TIS with C = 2 and 64× with C = 8.
- *PPO-IS* puts π_learner(θ)/π_sampler(θ_old) inside the clip. That ratio is not 1 even at θ = θ_old, so tokens are clipped "with high possibility" and training is less informative.

Measured results: with INT8 rollouts PPO-IS and vanilla IS reach near 0 accuracy, and TIS "outperforms both variants consistently, especially in cases where the gap is large (e.g., FP8/INT8)" (Fig. 4). INT8 + TIS reaches performance similar to BF16 on Qwen2.5-0.5B GSM8K (Fig. 2). Limits: with a small gap (DeepSeek-R1-Distill-Qwen-1.5B, BF16 rollouts) TIS "cannot bring performance gain" (Fig. 3); the main-experiment value of C is not stated in the text; the 32B DAPO run covered only the first 250 steps.

Production use: Olmo 3 32B Think used TIS with cap 2.0 ([[olmo-3-rl-pipeline]] Table 49), and open-instruct implements it as the ρ correction (below). The slime coding-agent example trains with `--use-tis --tis-clip 2 --tis-clip-low 0` ([[slime-agentic-and-mismatch-examples]]).

### §7.2 Masking and rejection: MIS, IcePop, veto

Truncation keeps every token with a capped weight. Masking removes tokens or whole sequences whose ratio falls outside a band. verl implements both in one function; the IS weight is clamped (TIS) for a single upper bound or zeroed outside [lower, upper] (IcePop) for a two-sided bound:

```python
# verl@8357f1e verl/trainer/ppo/rollout_corr_helper.py L603-616
# Zero out weights for padding tokens using response mask
raw_rollout_is_weights = raw_rollout_is_weights * response_mask

# Apply TIS for a single upper bound and IcePop for a lower_upper string.
if not use_icepop:
    rollout_is_weights = raw_rollout_is_weights.clamp(max=rollout_is_threshold_upper)
else:
    assert rollout_is_threshold_lower is not None
    token_kept_mask = (raw_rollout_is_weights >= rollout_is_threshold_lower) & (
        raw_rollout_is_weights <= rollout_is_threshold_upper
    )
    rollout_is_weights = torch.where(
        token_kept_mask, raw_rollout_is_weights, torch.zeros_like(raw_rollout_is_weights)
    )
```

The verl docstring labels token-level IS "biased, low variance" and sequence-level IS "unbiased, high variance" (L556–557). Separately from the weights, verl can apply a rejection mask from divergence estimators (`token_k1`, `seq_max_k2`, etc.) that edits `response_mask` (L196–260) ([[verl-rollout-correction-and-lowprec]]).

slime's mismatch helper adds a per-sequence veto: if any token in a sequence has log-ratio below log(1e-4), the whole sequence gets weight 0:

```python
# slime@79aa41a examples/train_infer_mismatch_helper/mis.py L92-96, L113
log_veto_threshold = torch.log(torch.tensor(veto_threshold, device=log_ratio.device))
# For each sequence, if it has any catastrophic tokens, return 0 for the sequence
catastrophic_tokens = (log_ratio < log_veto_threshold) & loss_mask.bool()
has_catastrophic = catastrophic_tokens.any()
veto_mask = (~has_catastrophic).float().expand_as(log_ratio)
...
return weights.clamp(0, upper_bound) * loss_mask   # truncate (TIS)
```

Defaults in `mis.yaml`: token-level TIS and rejection, truncate mode, bounds 0.5 / 2.0, veto threshold 1.0e-4, batch-normalized weights ([[slime-agentic-and-mismatch-examples]]). The card notes that `.any()` runs over whatever tensor the caller passes, so per-sequence behavior depends on the caller passing one sequence at a time; the caller was not traced.

open-instruct unifies clamping and masking in one ρ = π^train_old / π^infer_old correction:

```python
# open-instruct@f020e9a open_instruct/grpo_utils.py L453-456, L478-490
logprob_diff = torch.where(
    response_mask, (old_logprob - vllm_logprobs).clamp(-10.0, 10.0), torch.zeros_like(old_logprob)
)
rho = torch.exp(logprob_diff)
...
    dropped_low, dropped_high = _rho_drop_masks(
        rho_effective, response_mask, config.rho_mask_lower_bound, config.rho_mask_upper_bound
    )
in_range = response_mask & ~dropped_low & ~dropped_high
rho_clamped = rho_effective
if config.rho_clamp_lower_bound > 0.0:
    rho_clamped = torch.clamp(rho_clamped, min=config.rho_clamp_lower_bound)
if config.rho_clamp_upper_bound > 0.0:
    rho_clamped = torch.clamp(rho_clamped, max=config.rho_clamp_upper_bound)
weights = torch.where(in_range, rho_clamped, torch.zeros_like(rho_clamped))
```

Defaults: `use_rho_correction = True`, clamp upper 2.0, masks off (L129–141). A `rho_mask_sequence_level` option applies the mask on the sequence-mean log ratio, which the docstring labels "DeepSeek-V3.2 style" (L143–144).

AReaL's decoupled PPO also masks rather than truncates: samples whose ratio exceeds a threshold are dropped ([[truncated-importance-sampling]] Relation to other corrections; [[areal-async-rl]]). Its search-agent example uses rejection on the ratio with `upper: 5.0` ([[areal-agent-pipelines]] config).

**Which works better under BF16?** [[fp16-train-infer-mismatch]] tested on a "perfectible" dataset where a reliable method should approach 100% training accuracy (1,460 MATH problems at 20–80% initial accuracy for DeepSeek-R1-Distill-Qwen-1.5B; C = 3 for IS methods). Under BF16, vanilla GRPO peaked at 73% (VeRL) / 84% (Oat) and collapsed; GRPO-Token-TIS collapsed after 82% / 88%; GSPO lasted longer "despite not using the inference policy µ at all"; only GRPO-Seq-MIS did not collapse, but it converged slowly and peaked at 95% training accuracy and 34% AIME 2024 (Sec. 4.2). The authors describe a bias–variance trade-off under BF16: low-variance biased estimators converge fast then collapse; less biased ones are stable but slow (Sec. 6). One model and one sanity task; the ordering may not hold for other setups.

### §7.3 FP16 in both engines

[[fp16-train-infer-mismatch]] argues the mismatch is mainly a rounding-precision problem. Offline, FP16 sequence-level log-ratio mismatch was about 24× smaller than BF16 (Fig. 2; panel annotations KL = 7.64 for BF16 vs 0.32 for FP16, argument order not legible). With FP16 for sampler and learner, the algorithms in §7.2 became "almost indistinguishable" and the problem "nearly on-policy" (Sec. 4.3). The precision ablation shows that raising only inference precision prolongs stability, FP32 inference with BF16 training is fully stable but nearly three times slower, and FP16 for both gives the lowest mismatch and no speed loss (Sec. 4.4). Results extended to Qwen3-30B-A3B-Base (MoE), LoRA, Qwen3-14B-Base and OctoThinker-3B (Sec. 5).

Higher precision alone did not raise inference scores: DeepSeek-R1-Distill-Qwen-1.5B scored 50.38 / 50.60 / 51.54 on AMC23 (8K) in BF16 / FP16 / FP32 and 22.60 / 20.10 / 22.30 on AIME24 (8K) (Table 2). The benefit is consistency between engines, not accuracy of either engine.

Limits stated by the authors: FP16 is not claimed universally optimal; very large models may overflow; efficiency pushes toward FP8 (Sec. 6). The dynamic-range argument assumes RL fine-tuning of an already pre-trained model. Rocket chose a different path: lower precision for storage (MXFP8, ch-06) plus parity of every reduction, not more mantissa bits.

### §7.4 Deterministic modes that do not give parity

Deterministic inference makes the sampler reproducible; it does not make it equal to the learner. slime's reproducibility mode combines SGLang deterministic inference (`--sglang-enable-deterministic-inference`, `--sglang-attention-backend flashinfer`) with Megatron `--deterministic-mode` and environment settings `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` for "bitwise experiment reproduction" (`docs/en/advanced/reproducibility.md` L3–25). The TIS authors measured that SGLang's deterministic mode alone "does not noticeably reduce the gap" between sampler and learner ([[truncated-importance-sampling]] Mismatch factors). Parity requires the cross-engine work of §3–§5 (or slime's GLM-5 aligned path in §4).

### §7.5 Summary of options

| Approach | Removes | Cost | Strongest evidence | Source |
|---|---|---|---|---|
| Bitwise parity (Rocket) | engine-mismatch term entirely | kernel and layout engineering; replicated `o_proj` | Fig. 6 pair of runs; TM KL = 0 | [[mai-rl-infrastructure]], [[thinking-machines-nondeterminism]] |
| Batch-invariant kernels (DeepSeek-V4) | batch dependence within each kernel | DeepGEMM, dual-kernel decode | stated, no RL ablation in card | [[deepseek-train-infer-consistency]] |
| Equal micro-batch sizes, colwise `o_proj` (NeMo RL) | two dominant TP-related causes | none stated | ratio becomes 1; accuracy issues gone | [[nemo-rl-tp-accuracy]] |
| TIS | variance of large ratios; bias remains | extra old-log-prob forward unless bypass mode | INT8 rollouts restored to BF16 level | [[truncated-importance-sampling]] |
| MIS / IcePop / veto | outlier tokens or sequences | discarded samples | Seq-MIS only non-collapsing BF16 method (slow) | [[fp16-train-infer-mismatch]] |
| FP16 both sides | most rounding mismatch | loss scaling; overflow risk at large scale | 24× smaller mismatch; algorithms converge | [[fp16-train-infer-mismatch]] |
| Routing + top-p mask replay | discrete flips | mask storage and transfer | divergence without top-p replay | [[mai-thinking-1]] |

The extra forward pass is a real cost of decoupled corrections: [[fp16-train-infer-mismatch]] estimates "approximately 25%" more training cost for TIS/MIS in GRPO, assuming backward = 2× forward (Sec. 2.1.1). verl's bypass mode sets π_old = π_rollout and skips the old-log-prob forward, trading that cost for using the sampler's (mismatched) log-probs as the PPO anchor ([[verl-rollout-correction-and-lowprec]] doc L780–833).

---

## §8 Parity and staleness are separate budgets

Parity removes the second term of slime's decomposition. The first term, parameter staleness, remains in any asynchronous run, and it still needs a correction or a bound. MAI-Thinking-1 used 5 gradient steps per inference update, dropped rollouts more than 8 updates (40 steps) stale, and clipped all ratio branches at r_max = 50 ([[mai-thinking-1]] Sec. 3.1.5). The Base 2 post does not state which loss-side correction Rocket uses for staleness; it states only the levers of ch-04 and the parity work of this chapter.

Parity also changes what a ratio means. Without parity, a token's ratio mixes "the policy changed" with "the engines disagree". With parity, the ratio at staleness 0 is exactly 1, so any departure from 1 is attributable to policy change. That is why Rocket can plot Fig. 3 against staleness and fit a line through an exact-match origin.

A practical consequence: a TIS cap or mask tuned to absorb engine mismatch may clip tokens whose ratio reflects real policy change once staleness grows. slime's documentation warns that "TIS/SC can be useful, but do not justify an unlimited stale queue" (`rl-systems.md` L41).

---

## §9 Measuring the mismatch

Every framework read for this chapter logs a mismatch statistic. Pick one and log it from the first step.

| Metric | Definition | Threshold or reference | Source |
|---|---|---|---|
| `token_mult_prob_error` | (1/Σm_i) Σ m_i · exp(\|g_i − p_i\|), g = generation log-probs, p = policy log-probs | ideal 1.0; investigate above 1.05 | [[nemo-rl-tp-accuracy]] |
| `rollout_corr/kl` | KL between rollout and training log-probs | BF16 0.0006–0.0008 vs MXFP8 0.0063–0.0071 on Qwen3-8B GSM8K | [[verl-rollout-correction-and-lowprec]] |
| `val/rho_*` | ρ histogram, clip fraction, drop fractions | — | open-instruct `grpo_utils.py` L456–497 |
| `catastrophic_token_fraction` | share of tokens with log-ratio below log(veto) | veto 1e-4 | slime `mis.py` L98–99 |
| max / mean \|p_sampler − p_learner\| | per response | max > 0.5 counted as high | [[truncated-importance-sampling]] |
| `train_rollout_logprob_abs_diff` | absolute log-prob difference | < 1e-6 gate (≈ 2e-7 observed) | slime GLM-5 gate |

NeMo RL explains why a bad mismatch metric does not always show in reward: the spikes are sporadic, not present in every run on the same batch, and diluted over large datasets ([[nemo-rl-tp-accuracy]] Technical Details). verl recommends logging mismatch before choosing a correction; setting both `rollout_is` and `rollout_rs` to null while recording rollout log-probs gives metrics only ([[verl-rollout-correction-and-lowprec]] doc L834–845).

**A parity test.** Take one checkpoint and a fixed set of prompts. Generate with the sampler, recording token ids and log-probs. Run the learner forward on the same token ids. Compare per-token log-probs bitwise and report the fraction of tokens that differ and the max absolute difference. Repeat at a later RL checkpoint, because Rocket found the gap larger there (p. 9). Repeat with long responses, because mismatch grows with length ([[truncated-importance-sampling]]). Repeat with the sampler under load (different batch compositions), because batch variance only appears when other requests share the batch ([[thinking-machines-nondeterminism]]).

---

## Recipe

1. **Instrument first.** Record sampler token ids and log-probs for every rollout. Log one mismatch metric (§9) per step, plus its maximum over tokens.
2. **Run the parity test** (§9) at initialization, with sampler load varied, long and short responses, and repeat it every few hundred steps.
3. **Remove cheap causes.** Make the learner's log-prob micro-batch size equal to its training micro-batch size. Use the same TP degree on both sides or remove row-parallel sums after `o_proj` (and `down_proj` for dense MLPs).
4. **Match discrete choices.** Replay top-p masks if top-p < 1; replay MoE routing if exact MoE parity is not reached.
5. **Decide whether to pursue bitwise parity.** Pursue it if rollouts are long, the run is long, and you own the kernels on both sides (Rocket, slime GLM-5 path). Requirements: one attention implementation for learner forward, prefill and decode; aligned tiles; fixed KV split size; batch-invariant matmuls; FP32 local MoE combine in a fixed slot order; the same precision for LM head and router on both sides.
6. **If not, correct at the loss.** Default to token-level TIS with C ≈ 2 applied outside the PPO clip (Olmo 3 32B, slime coding agent), plus a sequence veto for catastrophic tokens. If training is in BF16 and collapses with token-level TIS, test sequence-level MIS or FP16 in both engines.
7. **Keep staleness separate.** Bound staleness by admission or drop rules (ch-03, ch-04); do not widen the IS cap to absorb it.
8. **Alert.** Alert on `token_mult_prob_error` > 1.05 or a rising trend in KL, entropy falling below its usual band, and response length rising abnormally (the INT8 failure signature).

---

## Common mistakes and how to detect them

| Mistake | Symptom | Detection |
|---|---|---|
| Assuming BF16 on both sides means on-policy | ratio ≠ 1 at staleness 0 | parity test at staleness 0; Rocket Fig. 3 intercept |
| Different micro-batch sizes for old-log-prob and training forward | sporadic ratio spikes; model-dependent | NeMo `token_mult_prob_error`; compare sizes in config |
| Sampler TP ≠ learner TP with row-parallel `o_proj` | high-mismatch tail responses, mean unchanged | count responses with max diff > 0.5 by TP configuration |
| Upcasting only one side (e.g. FP32 LM head in sampler only) | mismatch persists | TIS authors' FP32 `lm_head` patch result; slime L62 |
| Folding the sampler ratio into the PPO clip (PPO-IS) | most tokens clipped; slow or no learning | clip fraction near 1 at θ = θ_old |
| Untruncated IS | gradient-norm spikes | ratio histogram tail; variance ∝ ratio² |
| Treating deterministic inference as parity | gap unchanged | TIS authors' SGLang deterministic-mode result |
| Checking parity only at initialization | gap grows later | repeat test at later checkpoints (Rocket p. 9) |
| Top-p < 1 without mask replay | divergence within a few steps | MAI-Thinking-1 Sec. 3.1.3 |
| Reading flat reward as "mismatch is harmless" | spikes diluted over a large dataset | log mismatch directly; NeMo dilution note |

---

## Check your understanding

1. Rocket keeps tensor parallelism for attention in the sampler but makes the learner run without TP. Why does replicating `o_proj` and all-gathering head outputs make the two summation orders equal, and what would break if the learner also used TP with a different degree?
2. Thinking Machines shows that the LLM forward pass contains no atomic adds. Why, then, is inference output nondeterministic for a user of a busy server, and why does a fixed split *size* (not count) for split-KV fix it?
3. slime decomposes the sampler–learner log-ratio into parameter staleness and same-version engine mismatch. Explain why bitwise parity makes Rocket's Fig. 3 interpretable, and why parity does not remove the need for a staleness bound.
4. The TIS authors found that PPO-IS performs poorly even when the gap is moderate. Explain, using the value of the ratio at θ = θ_old, why putting the sampler ratio inside the clip harms learning while applying it outside does not.
5. slime states that matching precision, not FP32, is what aligns the LM head, while ScaleRL measured a higher asymptote with an FP32 LM head. Are these claims in conflict? What experiment would separate "less mismatch" from other effects of FP32 logits?
6. Rocket's Fig. 6 compares two runs launched about a month apart. What can and cannot be concluded from it about the causal effect of parity, and what control would make it an ablation?
7. Under BF16, [[fp16-train-infer-mismatch]] found sequence-level MIS stable but slow and token-level TIS fast but collapsing. Explain this using bias and variance of the two estimators, and say why FP16 makes the choice matter less.

---

## Connections

- ch-03 (staleness definition and Fig. 3), ch-04 (staleness levers), ch-05 (weight transfer: after a transfer the sampler holds the same weights, which this chapter makes produce the same numbers), ch-06 (MXFP8 storage and fake quantization: precision parity, the prerequisite for this chapter), ch-10 (recording sampler log-probs and top-p masks as part of rollout design), ch-22 (capstone parity plan).
- Algorithm mechanics: [llm-training ch-40](../../llm-training/ch-40/read.md) for the PPO/GRPO ratio and clip that TIS sits beside; [llm-training ch-43](../../llm-training/ch-43/read.md) for entropy collapse, the failure signature of large mismatch; [llm-training ch-45b](../../llm-training/ch-45b/read.md) for long multi-turn rollouts, where mismatch grows with length.
- Sources: [[mai-rl-infrastructure]], [[mai-thinking-1]], [[thinking-machines-nondeterminism]], [[flash-attention-stability]], [[nemo-rl-tp-accuracy]], [[deepseek-train-infer-consistency]], [[truncated-importance-sampling]], [[fp16-train-infer-mismatch]], [[verl-rollout-correction-and-lowprec]], [[slime-agentic-and-mismatch-examples]], [[slime]], [[olmo-3-rl-pipeline]], [[scalerl]], [[areal-async-rl]].

---

## Sources

- [[mai-rl-infrastructure]] — Rocket's three parity fixes (p. 10), the later-checkpoint gap (p. 9), Fig. 3 exact match at staleness 0, Fig. 6 L42 comparison.
- [[mai-thinking-1]] — Rocket v1 numerics (BF16 both sides, routing and top-p mask replay), top-p replay failure without it, language-reward correlation, YOLO deterministic accumulation (Sec. 2.8, Sec. 3.1.2–3.1.3, 3.6.4).
- [[thinking-machines-nondeterminism]] — non-associativity, batch invariance, fixed split-size split-KV, Qwen3-235B completion counts, runtime costs, RL KL = 0 experiment.
- [[flash-attention-stability]] — tiling-dependent FlashAttention deviation (~10× at BF16), sequence-length and tile-size trends.
- [[nemo-rl-tp-accuracy]] — micro-batch mismatch, row-wise TP sums, Table 1 by TP size, colwise plan, `token_mult_prob_error`.
- [[deepseek-train-infer-consistency]] — batch-invariant decode attention without split-KV, DeepGEMM without split-K, deterministic backward.
- [[truncated-importance-sampling]] — TIS for REINFORCE and PPO, variance argument, INT8/FP8 results, mismatch by parallelism and length, deterministic-mode result.
- [[fp16-train-infer-mismatch]] — BF16 vs FP16 mismatch (~24×), sanity test, TIS/MIS/GSPO under BF16, precision ablation, cost of extra forward.
- [[verl-rollout-correction-and-lowprec]] — verl TIS/IcePop/rejection implementation, decoupled vs bypass, metrics-only mode, MXFP8 KL.
- [[slime-agentic-and-mismatch-examples]] — slime mismatch helper (TIS, MIS, veto) and coding-agent TIS flags.
- [[slime]] — slime trainer and docs; the GLM-5 alignment path and reproducibility mode were read directly in the clone at 79aa41a.
- [[olmo-3-rl-pipeline]] — open-instruct ρ correction defaults and Olmo 3 TIS cap 2.0.
- [[scalerl]] — FP32 LM head asymptote 0.52 → 0.61.
- [[areal-async-rl]] — decoupled PPO with ratio-threshold dropping.

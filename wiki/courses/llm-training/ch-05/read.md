<!-- chapter: ch-05
     track: foundations
     kind: content
     title: Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen
     deps: [ch-03, ch-04]
     sources: [[fsdp-sft]], [[llama-3]], [[llama-3-recipe]], [[deepseek-v3]], [[deepseek-v3-recipe]], [[olmo-3]], [[olmo-core-olmo3-configs]], [[ring-attention]], [[deepspeed-ulysses]], [[large-batch-training-noise-scale]]
     figures: figures/parallelism-layout.html, figures/fsdp-memory.html
     revised: 2026-09 (generality revision)
-->

# Chapter 5 — Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen

> **Core insight.** A parallelism layout is a placement decision, and most of it does not change what
> the model is trained on. Four quantities do change it: the number of tokens in a global batch, the
> sequence length, the number of tokens seen in total, and the composition of each batch. Llama 3.1
> 405B used three different layouts during pre-training — TP 8 / CP 1 / PP 16 / DP 64 on 8,192 GPUs,
> TP 8 / CP 1 / PP 16 / DP 128 on 16,384 GPUs, and TP 8 / CP 16 / PP 16 / DP 8 at sequence length
> 131,072 — and held tokens per batch at 16M across all three ([[llama-3]] Table 4). Holding that
> quantity fixed is what keeps the three layouts comparable as one training run.
>
> **Guideline.** When you change a parallel degree, restate the global batch in tokens and the
> sequences per data-parallel group before changing anything else, because Llama 3 attributes the MFU
> drop from 43% to 41% between its 8,192-GPU and 16,384-GPU layouts to the smaller batch per DP group
> needed to hold global tokens per batch constant ([[llama-3]] §3.3.2). When the target sequence
> length no longer fits after tensor and data parallelism are set, add context parallelism rather than
> shrinking the batch, because Llama 3 reached sequence length 131,072 with CP 16 while keeping 16M
> tokens per batch (Table 4). When the interconnect between context-parallel ranks is inter-node
> rather than intra-node, prefer an all-gather-K/V or all-to-all method over ring attention, because
> ring attention needs a local sequence of about 149,500 tokens to hide communication on A100
> InfiniBand versus about 6,200 on A100 NVLink ([[ring-attention]] Table 2).

---

## Why this chapter matters for a general-purpose model

The pipeline position is pre-training and mid-training, with the same arithmetic reappearing at SFT and RL. [[ch-03]] fixed the schedule, the batch size, and the initialization as learning decisions. [[ch-04]] fixed what a training sequence contains. This chapter is about the machinery that executes those decisions on many devices, restricted to the part of that machinery that can change them.

Two of this chapter's quantities set the breadth of the resulting model directly.

- **Tokens seen** is the budget that pays for coverage of domains, languages, and formats. Llama 3's own scaling-law fit put the compute-optimal point for 3.8 × 10^25 FLOPs at 402B parameters over 16.55T tokens, and the released 405B model was trained on 15.6T tokens ([[llama-3]] §3.2.1, Abstract). Throughput determines how many tokens a fixed cluster-time budget yields, so a layout that halves MFU halves the breadth a fixed budget can pay for. The allocation question itself belongs to [[ch-08a]].
- **Sequence length** bounds which capabilities can be trained at all. A long-context stage is not possible if the layout cannot hold one 131,072-token sequence, and that constraint is what context parallelism removes.

The remaining quantity, **batch composition**, is the one most easily damaged without an error being raised. The intended mixture is a property of the corpus; the realized mixture is a property of how shards were assigned to data-parallel ranks and of what the data loader restored after a resume.

Memory mechanics — the per-component byte ledger, attention kernels, activation checkpointing policies, and the parallelism taxonomy itself — are taught in the learner's separate training-memory course, ch-06 ("The Attention Kernel Zoo: SDPA, xFormers, SageAttention, Ring, Paged") and ch-07 ("Parallelism Taxonomy: DP / ZeRO / FSDP / TP / SP / PP / EP / CP"). This chapter does not re-derive them.

---

## §1 The five axes, as a reference

Each axis partitions one dimension of the training computation. The table states what is split, what is communicated, and whether the axis can change a learning quantity.

| Axis | What is partitioned | Collective per layer or step | Changes a learning quantity? |
|---|---|---|---|
| Data parallel (DP) | the batch, and under sharding also parameters, gradients, optimizer state | AllGather + ReduceScatter (sharded) or AllReduce (replicated) | Yes — DP degree × per-rank batch is the global batch |
| Tensor parallel (TP) | individual weight matrices within a layer | AllReduce inside each block | No, if the global batch is held fixed |
| Pipeline parallel (PP) | layers into stages | point-to-point activations between stages | Indirectly — micro-batch count interacts with the batch |
| Context parallel (CP) | positions within one sequence | K/V all-gather, all-to-all, or ring rotation | Yes — CP sets the maximum trainable sequence length |
| Expert parallel (EP) | experts of an MoE layer across ranks | all-to-all dispatch and combine | Yes — routing balance changes which experts see which data |

**Definitions.** *Global batch* is the number of tokens whose gradients are averaged into one optimizer step. *Tokens per step* is the same number. *Tokens seen* is the global batch times the number of optimizer steps. *World size* is the number of participating devices.

The identity that relates them:

```
world_size = TP × PP × CP × DP
global_batch_tokens = DP × sequences_per_DP_group × sequence_length
tokens_seen = global_batch_tokens × optimizer_steps
```

`TP`, `PP`, `CP`, `DP` are the degrees of the four axes; `sequences_per_DP_group` is the number of training sequences one data-parallel group consumes per optimizer step, accumulated over one or more micro-batches.

**Sharded data parallelism in one formula.** FSDP's single control is the sharding factor `F`, the number of ranks a parameter is sharded over: `F = 1` is replication, `F = W` is full sharding, and `1 < F < W` is hybrid sharding ([[fsdp-sft]] §3.2). For a model of `Ψ` elements split into `N` units with element counts `ψ_1 … ψ_N`, the paper gives

```
peak parameter memory ∈ O( Σ_i ψ_i / F  +  max_i ψ_i )
collectives per iteration ∈ O(N)
```

`Ψ` is the total parameter count, `ψ_i` the element count of FSDP unit `i`, `F` the sharding factor, and `N` the number of units. The second term is **the largest single unit**, not the model ([[fsdp-sft]] §3.2.1). Under mixed precision the first term is counted at full-precision bytes and the second at low-precision bytes (§4.4).

Cross-host traffic per GPU for a model of `M` bytes on `W` accelerators in hosts of `G` ([[fsdp-sft]] §3.2.2): `2M(W−1)/W` for full replication, `3M(W−1)/W` for full sharding, and `2M(W−1)/(GW)` for hybrid sharding at `F = W/G`. The ratio of the first two is the paper's stated "1.5x communication overhead and volume over DDP" (§3.2.1).

[figures/fsdp-memory.html](figures/fsdp-memory.html) evaluates both formulas for a chosen `(Ψ, N, F, W, G)` and shows the three traffic regimes side by side.

**Placement.** Llama 3 orders the four groups as `[TP, CP, PP, DP]` because "the innermost parallelism requires the highest network bandwidth and lowest latency, and hence is usually constrained to within the same server", while the outermost — FSDP — "can tolerate longer network latency by asynchronously prefetching sharded model weights and reducing gradients" ([[llama-3]] §3.3.2). The 405B runs place TP 8 inside one 8-GPU server.

---

## §2 Global batch, tokens per step, and tokens seen

### 2.1 The problem, stated measurably

Two layouts that produce the same `global_batch_tokens`, the same sequence length, and the same data order produce the same sequence of optimizer states up to numerical reordering. Two layouts that do not are running different experiments. The measurable failure is a run whose realized tokens per step differ from the configured value, which changes the learning-rate-per-token schedule of [[ch-03]] without changing any line of the schedule configuration.

### 2.2 Worked example: Llama 3.1 405B, Table 4

Table 4 of [[llama-3]] as printed:

| GPUs | TP | CP | PP | DP | Seq. Len. | Batch size/DP | Tokens/Batch | TFLOPs/GPU | BF16 MFU |
|---|---|---|---|---|---|---|---|---|---|
| 8,192 | 8 | 1 | 16 | 64 | 8,192 | 32 | 16M | 430 | 43% |
| 16,384 | 8 | 1 | 16 | 128 | 8,192 | 16 | 16M | 400 | 41% |
| 16,384 | 8 | 16 | 16 | 8 | 131,072 | 16 | 16M | 380 | 38% |

Check the identity on each row.

- Row 1: `8 × 1 × 16 × 64 = 8,192` devices. `64 × 32 × 8,192 = 16,777,216` tokens ≈ 16M.
- Row 2: `8 × 1 × 16 × 128 = 16,384`. `128 × 16 × 8,192 = 16,777,216` tokens.
- Row 3: `8 × 16 × 16 × 8 = 16,384`. `8 × 16 × 131,072 = 16,777,216` tokens.

Row 1 to row 2 doubles the devices. The global batch is held at 16M, so the sequences per DP group fall from 32 to 16. Llama 3 attributes the MFU drop from 43% to 41% to exactly this: "the lower batch size per DP group needed to keep the global tokens per batch constant during training" ([[llama-3]] §3.3.2). Row 2 to row 3 raises the sequence length 16× and takes CP from 1 to 16, so DP falls from 128 to 8 at the same world size, and the batch stays at 16M with 16 sequences per DP group of 131,072 tokens each.

**Implication.** Doubling the cluster at fixed global batch raises the number of optimizer steps completed per wall-clock hour without changing the learning problem, and costs some efficiency. Doubling the cluster and doubling the batch instead is a different decision, evaluated in §2.4.

[figures/parallelism-layout.html](figures/parallelism-layout.html) lets you set the four degrees and the batch definition and reports the world size, sequences per DP group, accumulated micro-steps, and the divisibility flags; the three Llama 3 rows and five OLMo 3 stages are loadable as presets.

### 2.3 Worked example: OLMo 3, where the DP degree is derived

OLMo 3 trains in three stages — pre-training, mid-training, and a long-context stage ([[olmo-3]]) — and the per-stage scripts are released. The OLMo-core scripts print the batch and the CP degree but not the DP degree; the GPU count comes from the stage table in the repository README ([[olmo-core-olmo3-configs]]). Every numeric OLMo 3 value in this chapter is taken from those scripts and that README.

OLMo 3 32B long-context: 1,024 H100s, `DEFAULT_SEQUENCE_LENGTH = 65536`, `GLOBAL_BATCH_SIZE = 8 * 1024 * 1024`, `cp_config=TransformerContextParallelConfig.llama3(degree=8, head_stride=4)`, `rank_microbatch_size=sequence_length`.

1. DP degree: `1024 / 8 = 128`, since TP and PP are absent from the script.
2. Sequences per step: `8,388,608 / 65,536 = 128`.
3. Sequences per DP group: `128 / 128 = 1`.
4. Micro-steps: `rank_microbatch_size` equals one sequence, so one micro-step; no gradient accumulation.
5. Tokens per device inside attention: `65,536 / 8 = 8,192`, which is the value the script's own comment gives: `# 64k tokens per instance -> 8k tokens per device`.

OLMo 3 7B long-context: 256 H100s, CP 8, `GLOBAL_BATCH_SIZE = 65536 * 64`. DP is `256 / 8 = 32`, sequences per step `4,194,304 / 65,536 = 64`, sequences per DP group `64 / 32 = 2`, and since `rank_microbatch_size=sequence_length` the run accumulates 2 micro-steps per optimizer step. The script annotates the same line with `# for CP we want only 1 instance per rank`.

**Conditions and limits.** These derivations assume the README GPU count and the script belong to the same run, which the README asserts by linking each stage row to its script. The scripts print no MFU and no world-size product; the DP degrees above are derived, not reported.

### 2.4 The batch size above which added width stops saving optimizer steps

[[large-batch-training-noise-scale]] defines the critical batch size `B_crit` from the hyperbolic trade-off between optimizer steps `S` and examples processed `E` to reach a fixed target:

```
S/S_min − 1 = (E/E_min − 1)^(−1),   B_crit = E_min / S_min
```

`S_min` is the minimum steps over all batch sizes, `E_min` the minimum examples, and `B_crit` the batch at which a run uses 2× the minimum steps and 2× the minimum examples (Eq. 2.11–2.12, §2.3).

Worked example from the paper's own relations (App. D.1, `δS = 1 + B_noise/B`, `δE = B + B_noise`, relative to the minima), with `B_noise = 1,000`:

| Batch B | Steps vs minimum | Examples vs minimum |
|---|---|---|
| 250 | 5× | 1.25× |
| 1,000 | 2× | 2× |
| 4,000 | 1.25× | 5× |

Each pair satisfies Eq. 2.11. Above `B_crit`, added data-parallel width converts into examples consumed rather than steps saved. The paper's estimator of `B_crit` is the simple gradient noise scale `B_simple = tr(Σ)/|G|²`, which matched `B_crit` at the order-of-magnitude level across 8 tasks whose `B_crit` ranged from 20 to over 10 million (§3, Table 1, Fig. 4). `B_crit` rises as the loss falls: Billion Word moved from 700 at the start to 100,000 averaged over training (Table 1).

**Conditions and limits.** The tasks in Table 1 are MNIST-to-Dota scale, not frontier LLM pre-training; the ratio `B_simple / B_crit` itself varies by about an order of magnitude across them (§5). Production runs instead ramp the batch: Llama 3.1 405B went from 4M tokens at sequence 4,096 to 16M after 2.87T tokens ([[llama-3-recipe]], v3 §3.4.1), and DeepSeek-V3 raised its batch from 3,072 to 15,360 over the first 469B tokens ([[deepseek-v3-recipe]] §4.2; the report prints no unit for that number).

**Implication for a general-purpose model.** Batch size is a data-efficiency decision, not a systems decision. A cluster expansion that raises the global batch past `B_crit` without an error being raised spends the token budget — the same budget that pays for domain breadth — on redundant gradient averaging.

---

## §3 Data sharding across ranks, and whether the mixture survives

### 3.1 Definition and the measurable problem

The **intended mixture** is the token share each source contributes to the corpus. The **realized mixture** is the token share each source contributes to the sequence of global batches the optimizer actually saw. They differ when shard assignment, sampling, or resume logic changes which files a rank reads. The measurable symptom is per-domain held-out loss that moves without any change to the mixture configuration.

### 3.2 Mechanism

1. The corpus is tokenized into numbered shards, one file per source subset.
2. A data-loader seed fixes a permutation over instances; each DP rank takes a strided or contiguous slice of that permutation.
3. Each rank's slice is read in order; the global batch for a step is the union of the slices of all DP ranks at that step.
4. On resume, the loader position must be restored, or the permutation restarts and already-seen instances are re-consumed.

The OLMo 3 7B long-context script makes all four steps explicit ([[olmo-core-olmo3-configs]]; the 32B script is identical in structure and differs in the mixture name and `num_workers`):

```python
dataset_config = NumpyPackedFSLDatasetConfig.from_data_mix(
    DataMix.OLMo_longmino_mix_0625,
    ...
    generate_doc_lengths=True,  # enables intra-document masking
    source_group_size=8,
    source_permutation_seed=123,
)

data_loader_config = NumpyDataLoaderConfig(
    global_batch_size=GLOBAL_BATCH_SIZE,
    seed=34521,
    num_workers=4,
)
```

`source_group_size=8` groups sources before permuting them, and `source_permutation_seed=123` fixes that permutation; `seed=34521` fixes the instance order. Two runs that differ in any of these three integers see the same corpus in a different per-step composition.

### 3.3 Resume is a mixture decision, not only a correctness decision

The same script loads a stage-2 checkpoint with

```python
load_path=".../Olmo-3-1025-7B/stage2/step47684/",
load_strategy=LoadStrategy.always,
load_trainer_state=False,
load_optim_state=True,
```

`load_trainer_state=False` with `load_optim_state=True` keeps the optimizer moments and discards the trainer's own state, including its data-loader position. That is the intended behaviour here, because stage 3 trains on a different mixture (`OLMo_longmino_mix_0625`) from stage 2. The same two flags on a *restart of the same stage* would restart the data order from instance zero while keeping the optimizer state, so the run would re-consume the beginning of the permutation and never reach its tail. No error is raised and the loss curve continues smoothly, because the re-consumed data is in-distribution (Interpretation; the flags and their semantics are from the released script, the failure mode is this course's reading of them).

### 3.4 What changing the DP degree does to composition

At fixed global batch, raising DP lowers sequences per DP group (§2.2). When that count reaches 1, each optimizer step draws exactly `DP` sequences, so the per-step mixture is a sample of size `DP` from the permutation and cannot be made more representative without raising the global batch. OLMo 3 32B pre-training sits at that boundary: global batch 8,388,608 tokens at sequence 8,192 gives 1,024 sequences, and `DP = 1,024`, so each rank contributes exactly one sequence per step ([[olmo-core-olmo3-configs]]). Two consequences follow. A layout in which `sequences_per_step` is not an integer multiple of `DP` requires the loader to pad or drop, and either choice changes the realized tokens per step. And at one sequence per rank there is no per-rank averaging left to smooth a rank whose shard happens to be single-source, so shard assignment, not only the mixture specification, sets what a step contains.

**Implication for a general-purpose model.** Domain breadth is specified in the mixture ([[ch-13]]) and delivered by the loader. The delivery step has its own seeds, its own sharding, and its own resume semantics, and none of them are visible in the mixture specification.

---

## §4 Long context: context and sequence parallelism

This section covers how a sequence longer than one device can hold is trained without changing the global batch.

### 4.1 Three methods

**Ring attention** ([[ring-attention]]). Each host holds one query block; key-value blocks rotate around a ring. While a host computes blockwise attention against the K/V block it holds, it sends that block onward and receives the next, so communication is overlapped with computation (§3, Figure 2a). The abstract states the method "enables training and inference of sequences that are up to device count times longer than those achievable by prior memory-efficient Transformers, without resorting to approximations or incurring additional communication and computation overheads".

**DeepSpeed Ulysses** ([[deepspeed-ulysses]]). The sequence is split `N/P` across `P` devices. Before attention, an all-to-all over Q, K, V gives each device the full sequence for a non-overlapping subset of attention heads; attention is computed per head; a second all-to-all returns the output to sequence-parallel layout (§3.1, Figure 2).

**All-gather-K/V context parallelism** ([[llama-3]] §3.3.2). Each rank all-gathers K and V, then computes attention for its local Q chunk. Llama 3 states two reasons for choosing this over the ring structure: "it is easier and more flexible to support different types of attention masks in all-gather based CP attention, such as the document mask", and the gathered K and V "are much smaller than Q tensor due to the use of GQA", so with attention at `O(S²)` against an all-gather at `O(S)` "the all-gather overhead [is] negligible".

### 4.2 Communication, with the numbers each source prints

Ulysses gives a closed form (§3.2). With hidden size `h`, sequence length `N`, and parallel degree `P`, the per-link volume of an all-to-all for an aggregate message `M` over `P` GPUs is `M/P`. Ulysses does one all-to-all of `3Nh` (QKV) and one of `Nh` (output) per layer, so:

```
Ulysses per-link volume     = 4Nh / P      →  O(N/P)
Megatron-LM SP per-link vol = 4Nh          →  O(N), P times larger
```

The Megatron comparison is the paper's own: two all-gathers and two reduce-scatters of `Nh` per layer, each costing `M` rather than `M/P` when `P ≫ 1` (§3.2). The Ulysses volume "is constant when both N and P are increased proportionally".

Ring attention gives a hardware condition instead of a volume (§3). Blockwise attention costs `4dc²` FLOPs for a block of size `c` at hidden size `d`, and the K/V blocks cost `4cd` bytes, so with `F` FLOPS per host and bandwidth `B` the minimal block size for overlap is `c = F/B` and the minimal local sequence is `s = 6c` (Table 2 caption). The ring-attention paper writes `F` for FLOPS; §1 of this chapter uses `F` for the FSDP sharding factor, and the two are unrelated.

Three of the five rows of Table 2 are reproduced here; the TPU v4 and TPU v5e rows are omitted, and so is the HBM column.

| Spec per host | FLOPS (TF) | Interconnect bandwidth (GB/s) | Minimal block size (×1e3) | Minimal local sequence (×1e3) |
|---|---|---|---|---|
| A100 NVLink | 312 | 300 | 1.0 | 6.2 |
| A100 InfiniBand | 312 | 12.5 | 24.5 | 149.5 |
| TPU v3 | 123 | 112 | 1.1 | 6.6 |

Worked reading: the FLOPS-to-bandwidth ratio is `312e12 / 300e9 ≈ 1.04e3` for NVLink and `312e12 / 12.5e9 ≈ 25.0e3` for InfiniBand, so the bandwidth ratio `300 / 12.5 = 24` carries straight into the block size. The minimal local sequence follows at `6c`. The recomputed InfiniBand entries are `25.0e3` and `149.8e3` against the `24.5` and `149.5` the table prints, a rounding difference in the source, not a different claim. Below that local sequence length, the K/V transfer is not hidden and the method's "no added overhead" property does not hold ([[ring-attention]] Table 2, §3).

### 4.3 What production runs chose

Llama 3.1 405B long-context stage: 16,384 GPUs, TP 8, **CP 16**, PP 16, DP 8, sequence 131,072, 16M tokens per batch, 380 TFLOPs/GPU, BF16 MFU 38% ([[llama-3]] Table 4). The sequence is partitioned into `2 × CP` chunks, and "The i-th CP rank received both the i-th and the (2 × CP − 1 − i)-th chunks", which balances the work that a causal mask makes uneven (§3.3.2).

OLMo 3 long-context stages: `cp_config=TransformerContextParallelConfig.llama3(degree=8, head_stride=4)` for both 7B (256 H100s) and 32B (1,024 H100s) at sequence 65,536 ([[olmo-core-olmo3-configs]]). The constructor name identifies the all-gather-K/V method, not ring attention.

**Replicated practice, not a controlled result**: two independent groups — Meta for Llama 3.1 405B and Ai2 for OLMo 3 — selected all-gather-K/V context parallelism for their long-context stages, and both state or configure a document mask alongside it ([[llama-3]] §3.3.2; the OLMo 3 scripts set `generate_doc_lengths=True  # enables intra-document masking`).

**Conditions and limits.** Neither source ablates CP method against throughput. Ulysses' reported gains — "2.5x faster with 4x longer sequence length than the existing method SOTA baseline", over 175 TFLOPs/GPU at over 54% of peak, up to 256 A100s, and 1M-token sequences on a 1.2B GPT — are measured against Megatron-LM sequence parallelism on GPT models, not against all-gather CP ([[deepspeed-ulysses]] Abstract, §1, §4.1).

**Interpretation.** Ulysses computes attention head-parallel across `P` devices, so `P` is bounded by the number of attention heads available to split. The paper states the head-parallel structure (§3.1, §3.4) but prints no divisibility rule; treat the bound as implied rather than reported. Under GQA with 8 KV heads — OLMo 3 32B's configuration ([[olmo-core-olmo3-configs]]) — that bound is tighter than it appears from the query-head count.

**Implication for a general-purpose model.** CP is what makes long-context capability trainable at all. [[ch-32b]] covers the data mixture and short-context regression of a context-extension stage, and [[ch-32c]] covers the gap between claimed and effective context length; this chapter covers only whether the layout can hold the sequence.

---

## §5 Expert parallelism and load balancing

### 5.1 Definition and the problem

In a Mixture-of-Experts layer, each token is routed to a subset of expert FFNs. **Expert parallelism** places different experts on different ranks, so routing becomes an all-to-all dispatch and an all-to-all combine. The measurable problem is load imbalance: if routing concentrates tokens on a few experts, those ranks become the step's latency, and a common remedy — an auxiliary balancing loss — changes the routing the model would otherwise learn.

### 5.2 Mechanism: auxiliary-loss-free balancing

DeepSeek-V3 adds a per-expert bias `b_i` to the affinity score **only for top-K selection**; the gate value still uses the unbiased affinity `s_{i,t}`. After each step, `b_i` decreases by `γ` for overloaded experts and increases by `γ` for underloaded experts ([[deepseek-v3]] §2.1.2, Eq. 16). A sequence-wise balance loss with a small `α` is retained to prevent extreme imbalance inside a single sequence (Eq. 17–20). Configured values: `γ = 0.001` for the first 14.3T tokens and `0.0` for the remaining 500B; `α = 0.0001` ([[deepseek-v3-recipe]] §4.2).

Because the bias enters selection but not the gate, the balancing pressure does not appear in the gradient of the language-modelling loss.

### 5.3 Evidence

[[deepseek-v3]] Table 5, auxiliary-loss-based versus auxiliary-loss-free, at two scales — a 15.7B-total (2.4B active) model on 1.33T tokens and a 228.7B-total (20.9B active) model on 578B tokens. Seven of the table's ten benchmark rows are reproduced here; DROP, TriviaQA, and NaturalQuestions are omitted:

| Benchmark (metric) | Small, aux-loss | Small, aux-free | Large, aux-loss | Large, aux-free |
|---|---|---|---|---|
| Pile-test (BPB, lower better) | 0.727 | 0.724 | 0.656 | 0.652 |
| BBH (EM, 3-shot) | 37.3 | 39.3 | 66.7 | 67.9 |
| MMLU (EM, 5-shot) | 51.0 | 51.8 | 68.3 | 67.2 |
| HumanEval (pass@1) | 22.0 | 22.6 | 40.2 | 46.3 |
| MBPP (pass@1, 3-shot) | 36.6 | 35.8 | 59.2 | 61.2 |
| GSM8K (EM, 8-shot) | 27.1 | 29.6 | 70.7 | 74.5 |
| MATH (EM, 4-shot) | 10.9 | 11.1 | 37.2 | 39.6 |

**Result (single study).** The auxiliary-loss-free strategy is ahead on most rows at both scales, and behind on MMLU at the large scale and MBPP at the small scale. One seed per cell; the report gives no error bars.

The mechanism the report offers is scope. Batch-wise balancing "does not enforce in-domain balance on each sequence", which "allows experts to better specialize in different domains"; expert load recorded per Pile domain at 16B shows greater specialization for the auxiliary-loss-free model (§4.5.3, Figure 9). A batch-wise auxiliary loss reaches the same validation loss as the auxiliary-loss-free method: 1B MoE, 2.258 (sequence-wise) versus 2.253 (auxiliary-loss-free) versus 2.253 (batch-wise); 3B MoE, 2.085 versus 2.080 versus 2.080 (§4.5.3).

**Conditions and limits.** The report names two efficiency challenges for batch-wise balancing, beginning with load imbalance within a batch (§4.5.3). The 1B and 3B validation-loss comparison is validation loss only, not downstream evaluation.

### 5.4 Layout, and why routing choices are communication choices

DeepSeek-V3's training layout is "16-way Pipeline Parallelism (PP), 64-way Expert Parallelism (EP) spanning 8 nodes, and ZeRO-1 Data Parallelism (DP)" on a cluster of 2,048 H800 GPUs, with **no tensor parallelism**: the memory optimizations are what "enable us to train DeepSeek-V3 without using costly Tensor Parallelism (TP)" ([[deepseek-v3]] §3.1, §3.2; [[deepseek-v3-recipe]] §3.2, §4.2).

Node-limited routing caps each token at `M = 4` nodes (§2.1.2). The kernel design uses the bandwidth difference directly: "NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)", so each token is sent over IB to the matching in-node index on each target node and forwarded over NVLink from there. The consequence the report draws: a token selects "an average of 3.2 experts per node without incurring additional overhead", so although 8 routed experts are selected in practice, the report states it "can scale up this number to a maximum of 13 experts (4 nodes × 3.2 experts/node) while preserving the same communication cost", and "only 20 SMs are sufficient to fully utilize the bandwidths of IB and NVLink" (§3.2.2). DualPipe is the pipeline schedule that overlaps computation with communication across a forward and backward chunk pair (§3.2.1); the cross-node all-to-all kernels are a separate contribution (§3.2.2).

Because balancing succeeds, "DeepSeek-V3 does not drop any tokens during training" (§2.1.2) — no training token is discarded by the routing layer.

**Implication for a general-purpose model.** Load balancing is where an efficiency mechanism reaches into what the model learns. A balancing pressure applied per sequence forces every sequence to use every expert, which suppresses the domain specialization that per-batch balancing permits, and DeepSeek-V3 measures a small but consistent downstream cost for the per-sequence version (§4.5.3, Table 5).

---

## Recipe

Rows quote published runs. `Status` records reading at the locus on the stated date.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3.1 405B | 405B | pretrain-stable | Parallelism, 8K context | 8,192 GPUs: TP 8, CP 1, PP 16, DP 64; seq 8,192; 32 seq per DP; 16M tokens/batch; 430 TFLOPs/GPU; BF16 MFU 43% | arXiv:2407.21783 v3 Table 4 | verified 2026-09-17 | no ablation reported |
| Llama 3.1 405B | 405B | pretrain-stable | Parallelism, 8K context, larger cluster | 16,384 GPUs: TP 8, CP 1, PP 16, DP 128; 16 seq per DP; 16M tokens/batch; 400 TFLOPs/GPU; MFU 41% | v3 Table 4, §3.3.2 | verified 2026-09-17 | §3.3.2 attributes the 43%→41% MFU drop to the smaller batch per DP group at constant global tokens |
| Llama 3.1 405B | 405B | long-context | Parallelism | 16,384 GPUs: TP 8, CP 16, PP 16, DP 8; seq 131,072; 16 seq per DP; 16M tokens/batch; 380 TFLOPs/GPU; MFU 38% | v3 Table 4 | verified 2026-09-17 | no ablation reported |
| Llama 3.1 405B | 405B | pretrain-stable | CP method | all-gather K and V, then attention on local Q chunk; sequence split into 2×CP chunks, rank i holds chunks i and 2·CP−1−i | v3 §3.3.2 | verified 2026-09-17 | chosen for mask flexibility and for small K/V under GQA; no ablation printed |
| Llama 3.1 405B | 405B | pretrain-stable | PP bubble ratio | interleaved schedule with V stages per rank; bubble ratio (PP−1)/(V·M), M = micro-batches | v3 §3.3.2 | verified 2026-09-17 | no measured bubble number printed |
| Llama 3.1 405B | 405B | pretrain-stable | Batch schedule | 4M tokens at seq 4,096 → "8M sequences of 8,192 tokens" after 252M tokens → 16M after 2.87T tokens | v3 §3.4.1 ([[llama-3-recipe]]) | verified 2026-09-14 | smaller early batch "to improve training stability" |
| DeepSeek-V3 | 671B total / 37B active | pretrain-stable | Parallelism | 16-way PP; 64-way EP across 8 nodes; ZeRO-1 DP; no TP; cluster of 2,048 H800 | arXiv:2412.19437 v2 §3.1, §3.2 | verified 2026-09-17 | no ablation reported |
| DeepSeek-V3 | 671B / 37B active | pretrain-stable | Node-limited routing | at most M = 4 nodes per token; average 3.2 experts per node; 20 SMs for communication; NVLink 160 GB/s vs IB 50 GB/s | v2 §2.1.2, §3.2.2 | verified 2026-09-17 | no ablation reported |
| DeepSeek-V3 | 671B / 37B active | pretrain-stable | Balancing | bias update γ = 0.001 for first 14.3T tokens, 0.0 for last 500B; sequence-wise balance loss α = 0.0001 | v2 §4.2 ([[deepseek-v3-recipe]]) | verified 2026-09-14 | Table 5 at 15.7B/1.33T and 228.7B/578B; §4.5.3 validation losses at 1B and 3B |
| DeepSeek-V3 | 671B / 37B active | pretrain-stable | Batch schedule | 3,072 → 15,360 over the first 469B tokens, then 15,360 (unit not printed) | v2 §4.2 | verified 2026-09-14 | no ablation reported |
| OLMo 3 7B | 7B | long-context | Layout and batch | 256 H100; CP degree 8 (`...llama3(degree=8, head_stride=4)`); `hsdp` with `shard_degree=1`; seq 65,536; global batch 65536×64 ≈ 4.19M tokens; `rank_microbatch_size=sequence_length` | OLMo-core `src/scripts/official/OLMo3/OLMo-3-1025-7B-long-context.py` + `README.md` ([[olmo-core-olmo3-configs]]) | verified 2026-09-17 | no ablation reported |
| OLMo 3 32B | 32B | long-context | Layout and batch | 1,024 H100; CP degree 8; `hsdp` with `shard_degree=8`; seq 65,536; global batch 8,388,608 tokens; AC budget 0.3 | `OLMo-3-1025-32B-long-context.py` + `README.md` | verified 2026-09-17 | no ablation reported |
| OLMo 3 32B | 32B | pretrain-stable | Layout and batch | 1,024 H100; `hsdp` with `shard_degree=64`; seq 8,192; global batch 8,388,608 tokens; AC budget 0.5 | `OLMo-3-1025-32B-pretrain.py` + `README.md` | verified 2026-09-17 | no ablation reported |
| OLMo 3 7B | 7B | long-context | Resume flags | `load_trainer_state=False`, `load_optim_state=True`, `load_strategy=LoadStrategy.always` | `OLMo-3-1025-7B-long-context.py` | verified 2026-09-17 | stage 3 uses a different mixture from stage 2 |
| OLMo 3 7B / 32B | 7B, 32B | long-context | Data-loader seeds | `source_group_size=8`, `source_permutation_seed=123`, loader `seed=34521` | same scripts | verified 2026-09-17 | no ablation reported |
| minGPT 175B (FSDP paper) | 175B | benchmark | Throughput | >173 and >186 TFLOPS/GPU at batch 1 and 2 on 128–512 A100 80GB; ~55% and ~60% of the 312 TFLOPS BF16 peak | arXiv:2304.11277 §5.4, Figure 7b | verified 2026-09-17 | full sharding + prefetch + rate limiter + activation checkpointing + BF16 + Adam (§5.4) |
| GPT 1.2B (Ulysses paper) | 1.2B | benchmark | Sequence scaling | up to 1M tokens; evaluation up to 256 A100; >175 TFLOPs/GPU (>54% of peak); 2.5× throughput and 4× sequence length vs Megatron-LM SP | arXiv:2309.14509 v2 Abstract, §1, §4.1 | verified 2026-09-17 | measured against Megatron-LM sequence parallelism only |

**Starting point for a small general-purpose run.** For a single 8-GPU NVLink node, no published row above applies, since every row is a multi-node frontier run. The transferable parts are the identities of §1 and two verified conditions. First, keep the global batch in tokens fixed when the device count changes, as Llama 3.1 405B did across three layouts at 16M tokens per batch (v3 Table 4). Second, if context parallelism is used on a node, the NVLink row of [[ring-attention]] Table 2 gives a minimal local sequence of about 6,200 tokens for ring attention on A100-class hardware (312 TF, 300 GB/s); below that, choose an all-gather-K/V or all-to-all method instead. Every other number in the table above is scoped to its run and should not be transferred.

---

## Generalization lens

**(a) What increases breadth.**

- Holding global tokens per step fixed while the layout changes keeps the learning problem identical across cluster sizes, so throughput improvements convert into tokens seen rather than into a different optimization trajectory ([[llama-3]] Table 4: 16M tokens per batch across three layouts).
- Context parallelism makes long-sequence training possible without shrinking the batch: Llama 3.1 405B trained at sequence 131,072 with CP 16 while keeping 16M tokens per batch (Table 4). Long-context capability is a breadth axis that no amount of short-context data supplies.
- Per-batch rather than per-sequence expert balancing permits domain specialization and is ahead on most of the ten rows of Table 5 at both ablation scales ([[deepseek-v3]] Table 5, §4.5.3).

**(b) What causes narrowing or forgetting.**

- A global batch above the critical batch size converts token budget into redundant averaging: at `B = 4·B_noise` a run needs 5× the minimum examples for 1.25× the minimum steps ([[large-batch-training-noise-scale]] Eq. 2.11–2.12, App. D.1).
- A layout that cannot hold the target sequence forces truncation or a shorter stage, which removes the long-context portion of the corpus from what the model ever sees.
- Sharding and resume errors change the realized mixture without changing its specification, so domain coverage silently narrows toward whichever shards the loader repeats (§3.3; Interpretation).
- A per-sequence balancing loss forces every sequence to spread across experts, which [[deepseek-v3]] §4.5.3 associates with reduced expert specialization and a small validation-loss cost at 1B and 3B.

**(c) How to measure it at this stage.**

- Log realized tokens per optimizer step and compare against the configured global batch every step, not at start-up only.
- Log per-source or per-shard token counts consumed per step; a mixture that drifts from the specification shows here before it shows in evaluation.
- Log MFU or TFLOPs/GPU per layout change, as Llama 3 does per Table 4 row, so a placement regression is separable from a data regression.
- Track per-domain held-out loss rather than one aggregate loss; the aggregate hides a domain that stopped arriving. In-loop evaluation design is [[ch-06]].

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Changing DP degree without restating the global batch | Loss curve shifts after a cluster resize with no configuration change to the schedule | Recompute `DP × sequences_per_DP × sequence_length` before and after; compare with the logged tokens per step |
| Global batch not an integer multiple of `DP × sequence_length` | Realized tokens per step differ from the configured value; some ranks idle | Assert divisibility at start-up; the flags in [figures/parallelism-layout.html](figures/parallelism-layout.html) show which of the two divisions fails |
| Assuming FSDP holds a full copy of the model during all-gather | Memory estimates far above what the run actually uses | Use `O(Σψ_i/F + max_i ψ_i)`: the second term is the largest unit, roughly one block under block-wise wrapping ([[fsdp-sft]] §3.2.1) |
| Placing a TP group across hosts | MFU falls with no change to batch or data | Verify the group order `[TP, CP, PP, DP]` and that the TP group size divides the accelerators per host ([[llama-3]] §3.3.2) |
| Using ring attention below its overlap threshold | Throughput falls as CP degree rises rather than staying flat | Compute `c = FLOPS/Bandwidth` and `s = 6c` for the actual interconnect; compare with the local sequence length ([[ring-attention]] Table 2) |
| Setting CP degree above the number of splittable heads for an all-to-all method | Initialization failure or an unexpected fallback path | Read the head count from the model config; under GQA use the KV-head count ([[deepspeed-ulysses]] §3.1, §3.4; Interpretation) |
| Resuming with the optimizer state but not the data-loader position | Smooth loss curve, but the corpus tail is never reached | Log cumulative tokens consumed per source; compare against the mixture specification (§3.3) |
| Copying a parallelism layout across model families | A layout with TP applied to a model whose report states it trained without TP | Read the layout at its locus: DeepSeek-V3 used PP 16, EP 64, ZeRO-1 DP, and no TP ([[deepseek-v3]] §3.2) |
| Applying a per-sequence balancing loss at frontier MoE scale | Expert load uniform within every sequence; per-domain specialization absent | Record expert load per domain, as in [[deepseek-v3]] §4.5.3, Figure 9 |

---

## Check your understanding

1. Llama 3.1 405B moved from 8,192 to 16,384 GPUs and kept tokens per batch at 16M. Explain why MFU fell from 43% to 41%, and say which quantity had to change to hold the batch constant.
2. Row 3 of Table 4 raises the sequence length from 8,192 to 131,072 at the same world size and the same tokens per batch. Derive the CP and DP degrees that make those three constraints consistent, and explain why DP had to fall.
3. The OLMo 3 7B long-context script sets `rank_microbatch_size=sequence_length` with the comment "for CP we want only 1 instance per rank". Given 256 GPUs, CP 8, sequence 65,536, and a global batch of 65536×64 tokens, how many micro-steps does each optimizer step accumulate, and why is that number not 1?
4. A run resumes with `load_optim_state=True` and `load_trainer_state=False` in the middle of a single stage. The loss curve is continuous and no error is raised. Explain what the model stops seeing, and name one metric that would reveal it.
5. Ring attention claims no added communication overhead, and Table 2 lists a minimal local sequence of about 6,200 tokens on A100 NVLink and about 149,500 on A100 InfiniBand. Explain what produces the 24× difference and what breaks when the local sequence is below the threshold.
6. DeepSeek-V3's auxiliary-loss-free bias enters top-K selection but not the gate value. Explain why that placement matters for the gradient, and what the report measures as the downstream consequence of balancing per sequence instead of per batch.
7. Two engineers propose the same throughput improvement. One doubles the cluster at a fixed global batch; the other doubles the global batch at a fixed cluster. Using the steps-versus-examples relation of [[large-batch-training-noise-scale]], state what each one changes about tokens seen at a fixed target loss.
8. A mixture specification says 17% code. Describe two distinct mechanisms in this chapter by which the realized code share of the batches the optimizer saw could differ from 17%, with no error raised.

---

## Connections

- Previous chapter: [[ch-04]] — *Sequence Packing, Loss Masking, and Chat Templates*. Packing fixes what one sequence contains; this chapter fixes how many such sequences enter one optimizer step.
- Dependency: [[ch-03]] — *Learning-Rate Schedules, Batch Size, Initialization, and Normalization*. The batch size chosen there is realized, or not, by the layout here.
- Next chapter: [[ch-06]] — *Checkpointing, In-Loop Evaluation, and Checkpoint Selection*. The per-domain and per-shard instrumentation this chapter calls for is designed there, with what a resumable checkpoint must contain.
- Forward: [[ch-08a]] — *Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability*. Tokens seen is that allocation's input; throughput converts a compute budget into it.
- Forward: [[ch-13]] — *Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale*. The intended mixture is specified there and delivered by the sharding mechanics of §3.
- Forward: [[ch-32b]] — *Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression* — and [[ch-32c]] — *Claimed versus Effective Context Length and Long-Context Evaluation*. §4 covers only whether the layout can hold the sequence.
- Sibling course: training-memory ch-06 (*The Attention Kernel Zoo*) and ch-07 (*Parallelism Taxonomy: DP / ZeRO / FSDP / TP / SP / PP / EP / CP*) hold the memory ledger and the primitive-by-primitive derivations this chapter references instead of repeating.

---

## Sources

- [[fsdp-sft]] — §1, Recipe: sharding factor `F`, peak parameter memory `O(Σψ_i/F + max_i ψ_i)`, the three cross-host traffic expressions, the 1.5× ratio against DDP, GPT-175B throughput (chapter copy: `excerpts/fsdp-sft.md`, rewritten 2026-09).
- [[llama-3]] — §1, §2, §4, Recipe: Table 4 layouts and MFU, the `[TP, CP, PP, DP]` placement rule, all-gather-K/V CP with `2 × CP` chunk assignment, interleaved pipeline bubble ratio.
- [[llama-3-recipe]] — §2.4, Recipe: the 405B batch-size ramp with its locus.
- [[deepseek-v3]] — §5, Recipe: PP 16 / EP 64 / ZeRO-1 DP without TP, node-limited routing and IB-to-NVLink kernels, auxiliary-loss-free balancing, Table 5, §4.5.3.
- [[deepseek-v3-recipe]] — §2.4, §5: balancing coefficients and batch-size schedule with loci.
- [[olmo-core-olmo3-configs]] — §2.3, §3, §4.3, Recipe: chapter excerpt of the released OLMo 3 scripts and README stage table (GPU counts, batch and sequence literals, `cp_config`, `dp_config`, loader seeds, resume flags).
- [[olmo-3]] — §2.3: the three-stage training procedure only. Every numeric OLMo 3 value in this chapter comes from the released scripts instead.
- [[ring-attention]] — §4: chapter excerpt with the ring mechanism, the `c = F/B` and `s = 6c` overlap condition, and Table 2.
- [[deepspeed-ulysses]] — §4: chapter excerpt with the two all-to-all steps, `4Nh/P` versus `4Nh` per-link volume, ZeRO-3 over the combined DP and SP group, reported scaling.
- [[large-batch-training-noise-scale]] — §2.4: the steps-versus-examples relation, `B_crit`, the `B_simple = tr(Σ)/|G|²` estimator, and the Table 1 range across 8 tasks.

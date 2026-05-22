---
chapter: ch-13
course: llm-inference
phase: read
excerpt_of: "GPipe (Huang et al. 2018) + Megatron-LM Interleaved Pipeline (Narayanan et al. 2021)"
source_url: https://arxiv.org/abs/1811.06965
created_at: "2026-05-21"
---

# Excerpt: Pipeline Parallelism for Inference

**Authors:** Yanping Huang et al. (GPipe, 2018); Deepak Narayanan et al. (Megatron interleaved, 2021)
**Year:** 2018 (GPipe); 2021 (Megatron interleaved)
**Venue:** NeurIPS 2019 (GPipe); SC '21 (Megatron)
**URLs:** https://arxiv.org/abs/1811.06965 (GPipe), https://arxiv.org/abs/2104.04473 (Megatron interleaved)
**Raw-data source:** [[raw-data/pipeline-parallel-inference]]

---

## The basic 1F1B schedule

For `S` stages and `M` microbatches, one-forward-one-backward at training gives:

```
Time →
Stage 0:  F0 F1 F2 F3 . . . .         . . . . B3 B2 B1 B0
Stage 1:  .  F0 F1 F2 F3 . . .       . . . B3 B2 B1 B0 .
Stage 2:  .  .  F0 F1 F2 F3 . .     . . B3 B2 B1 B0 . .
Stage 3:  .  .  .  F0 F1 F2 F3 .   . B3 B2 B1 B0 . . .
                                ^^^
                              bubble
```

For inference there is no backward pass; only the forward fill/drain remains:

```
Time →
Stage 0:  F0 F1 F2 F3 . . .
Stage 1:  .  F0 F1 F2 F3 . .
Stage 2:  .  .  F0 F1 F2 F3 .
Stage 3:  .  .  .  F0 F1 F2 F3
```

---

## The bubble cost

For `S` stages and `M` microbatches, useful work is `S · M` stage-steps; total elapsed is `S + M - 1` stage-steps. Bubble fraction:

```math
\text{bubble} = \frac{S - 1}{S + M - 1}
```

| S | M | bubble |
|---|---|--------|
| 4 | 4 | 50% |
| 4 | 8 | 27% |
| 4 | 32 | 9% |
| 8 | 32 | 18% |
| 16 | 32 | 32% |

**Inference implication**: PP needs `M ≫ S` to hide the bubble. A single decode step has *one* batch per microbatch — so PP at decode only works when continuous batching ([[continuous-batching]]) feeds enough concurrent requests to fill the pipeline.

---

## Cross-stage activation transfer

Per stage transition, send `[batch_tokens, d_model]` bytes. For `d_model=8192`, batch=32, bf16:

```math
\text{per transfer} = 32 \cdot 8192 \cdot 2 = 512 \text{ KB}
```

With `S` stages, `S-1` transfers per microbatch. On InfiniBand HDR (~25 GB/s), each transfer is ~20 µs. For PP=4: 60 µs of cross-stage traffic per microbatch — small.

**Contrast with TP across IB**: TP=8 would cost ~6 ms of all-reduce per decode step. PP=4 costs ~60 µs of activation transfers per microbatch. PP wins by ~100× on cross-node traffic.

This is the entire reason for "TP intra-node, PP inter-node".

---

## Megatron interleaved 1F1B

Each physical stage owns `v` non-contiguous *virtual stages*. E.g. for `S=4` physical and `v=2` virtual:

```
Physical stage 0 owns virtual stages {0, 4}    (layers 0..9 + layers 40..49)
Physical stage 1 owns virtual stages {1, 5}    (layers 10..19 + layers 50..59)
Physical stage 2 owns virtual stages {2, 6}    (layers 20..29 + layers 60..69)
Physical stage 3 owns virtual stages {3, 7}    (layers 30..39 + layers 70..79)
```

Bubble shrinks by factor `~v`: `(S-1)/(v(S+M-1))` instead of `(S-1)/(S+M-1)`.

**Cost**: `v` times as many activation transfers per microbatch. For training this is worth it; for inference, usually not — the extra IB traffic eats the bubble win.

---

## PP at decode in practice (vLLM)

vLLM's V1 engine ([[vllm-scheduler]]) treats each in-flight decode token as a microbatch from PP's perspective. At each scheduler tick:

1. Stage 0 emits a partial activation for each scheduled request.
2. The activation streams down the pipeline; each stage runs its layer group on receipt.
3. Stage `S-1` produces logits → sampler → next token.
4. Meanwhile, stage 0 is already starting the *next* request's activation.

At steady state with `M ≥ S` concurrent requests, the pipeline is full. Below that threshold, you idle stages.

```bash
# vLLM
vllm serve meta-llama/Llama-3-405B \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --gpu-memory-utilization 0.92
```

---

## Pitfalls

- **Low-QPS chat**: PP=4 with 1-2 in-flight requests → 50-75% pipeline idle. TTFT improves vs single-GPU only if model didn't fit; TPOT suffers.
- **Stage imbalance**: 20 layers on stage 0 vs 21 on stage 1 → slowest stage bottlenecks. Layer counts must be exactly equal.
- **Activation dtype**: bf16 transfers are 2× faster than fp32. Always send bf16 across stages.
- **First stage cold-start**: the first request entering an idle pipeline pays the full fill cost. SLOs must account for the cold-start tail.

---

## Connections

- [[excerpts/tensor-parallel-inference]] — the intra-node companion.
- [[excerpts/expert-parallel-inference]] — orthogonal axis for MoE.
- [[ch-13]] — parent chapter.

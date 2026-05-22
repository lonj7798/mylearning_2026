---
chapter: ch-13
course: llm-inference
phase: read
excerpt_of: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (Shoeybi et al. 2019) + vLLM / TRT-LLM serving knobs"
source_url: https://arxiv.org/abs/1909.08053
created_at: "2026-05-21"
---

# Excerpt: Tensor Parallelism for Inference

**Authors:** Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (Megatron-LM, 2019); vLLM team; TensorRT-LLM team
**Year:** 2019 (Megatron); 2023+ for inference adaptation
**Venue:** Megatron-LM technical report
**URLs:** https://arxiv.org/abs/1909.08053 ; https://docs.vllm.ai/en/v0.7.2/serving/distributed_serving.html
**Raw-data source:** [[raw-data/tensor-parallel-inference]]

---

## The MLP partition (Megatron, Section 3)

A transformer MLP is `Y = GeLU(X · A) · B`. Megatron splits A column-wise and B row-wise:

```math
A = [A_1, A_2, \ldots, A_N]      \quad \text{(split along output dim)}
```
```math
B = \begin{bmatrix} B_1 \\ B_2 \\ \vdots \\ B_N \end{bmatrix}      \quad \text{(split along input dim)}
```

Per rank `i`:

```math
Y_i = \text{GeLU}(X \cdot A_i) \cdot B_i
```

```math
Y = \text{AllReduce}(\{Y_i\})       \quad \text{(SUM)}
```

The GeLU is local because it's elementwise — so we get away with **one all-reduce per MLP**, not two. This is the entire reason TP scales: synchronization happens *only at the row-parallel output*, not at every nonlinearity.

---

## The attention partition (Megatron, Section 3, continued)

For `H` heads, partition Q/K/V projection column-wise so each rank owns `H/N` heads. The per-head softmax-attention is local. The output projection is partitioned row-wise, with one final all-reduce:

```math
Q_i, K_i, V_i = X \cdot W_{qkv,i}      \quad \text{(local, H/N heads)}
```
```math
\text{Attn}_i = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right) V_i
```
```math
Y_i = \text{Attn}_i \cdot W_{O,i}
```
```math
Y = \text{AllReduce}(\{Y_i\})
```

**Two all-reduces per transformer block (one MLP, one attention)**. For 80 blocks (Llama-3-70B), that's 160 collectives per forward pass.

---

## All-reduce cost — the per-step overhead

Each all-reduce carries `batch_tokens × d_model × dtype_bytes`. For bf16:

```math
\text{bytes per AR} = 2 \cdot \text{batch\_tokens} \cdot d_{\text{model}}
```

Ring all-reduce moves `2(N-1)/N` of this volume per rank. For `N=8` GPUs:

```math
\text{bytes per AR per rank} = \frac{14}{8} \cdot \text{batch\_tokens} \cdot d_{\text{model}}
```

Llama-3-70B (`d_model=8192`), decode at batch=32:

```math
\text{per AR} = \frac{14}{8} \cdot 32 \cdot 8192 \cdot 2 \approx 0.92 \text{ MB}
```

Per layer (2 ARs): 1.8 MB. Per decode step (80 layers): ~150 MB moved per rank.

On NVLink 4.0 (~450 GB/s effective ring bandwidth): ~330 µs of AR cost — negligible vs the 30-50 ms forward pass.

On InfiniBand HDR (200 Gb/s ≈ 25 GB/s): ~6 ms of AR cost — now ~15-20% of the decode latency. This is why TP across IB is a bad idea.

---

## KV-cache sharding

With TP=`N`, each rank holds `kv_heads / N` KV heads' worth of cache. The PagedAttention block table is per-rank but indexed identically across ranks — same block IDs everywhere, different physical content.

**Divisibility rule**: `kv_heads % tp_size == 0` must hold. Llama-3-70B has 8 KV heads → TP={1,2,4,8} legal, TP=16 forces head replication and doubles per-rank KV bytes.

---

## Vocabulary / embedding parallelism

The LM head's `[d_model, vocab_size]` matrix is split column-wise — each rank owns `vocab_size / N` rows. Sampling can either:

1. **All-reduce logits**: gather full `[batch, vocab]` to every rank → sample → broadcast. Costly at large vocab.
2. **Sharded sampler**: pick top-k locally → tree-reduce across ranks → sample. ~10× cheaper for top-p / top-k sampling.

vLLM and TRT-LLM both implement (2); option (1) only happens when you ask for the full logits (e.g. for log-prob computation).

---

## vLLM / TRT-LLM / SGLang configuration

```bash
# vLLM
vllm serve meta-llama/Llama-3-70B-Instruct --tensor-parallel-size 8

# TensorRT-LLM (engine build time — TP is baked into the engine)
trtllm-build --checkpoint_dir ... --output_dir ... --tp_size 8

# SGLang
python -m sglang.launch_server --model ... --tp 8
```

vLLM auto-launches one worker process per TP rank; TRT-LLM bakes TP into the precompiled engine plan; SGLang uses the same per-rank pattern as vLLM.

---

## Pitfalls

- **TP across IB.** Don't do TP=4 on 4 GPUs spread across nodes; the AR cost will eat the compute savings.
- **`kv_heads` non-divisible.** TP=16 on Llama-3-70B replicates KV heads, doubling per-rank KV-cache footprint.
- **Hidden host-side dtype conversion.** Some NCCL paths up-cast bf16→fp32 for reduction (correct) but then down-cast on send (extra latency). Pin `NCCL_PROTO=Simple` if you see anomalies.
- **Captured CUDA graphs must include the all-reduce.** vLLM's piecewise CUDA graphs break at collective boundaries — one graph per block, not one graph per layer.

---

## Connections

- [[excerpts/pipeline-parallel-inference]] — across nodes where TP becomes uneconomic.
- [[excerpts/expert-parallel-inference]] — orthogonal axis for MoE.
- [[ch-13]] — parent synthesis chapter.

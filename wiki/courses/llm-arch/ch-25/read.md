# Chapter 25: KV-Cache and Serving

<!-- scope: KV-cache memory management, PagedAttention, continuous batching, prefix caching, architecture-serving co-design
     deps: [[ch-07]], [[ch-18]]
     see-also: [[ch-26]], [[ch-27]]
-->

## Overview

[[ch-07]] established the core tradeoff: autoregressive decoding is memory-bandwidth-bound, not compute-bound. Every decoding step loads the entire KV cache from HBM, and the attention variant — MHA, GQA, MLA, sliding window — determines how many bytes that entails per token, per layer. But knowing the *size* of the cache is only half the serving problem. The other half is *managing* that memory across hundreds or thousands of concurrent requests, each growing its cache token-by-token, each finishing at unpredictable times.

This chapter crosses from model architecture into systems design. The KV cache is where architecture meets infrastructure: the attention variant you chose during training determines the memory footprint per request, and the memory management system you deploy determines how many requests you can serve simultaneously. Get the systems wrong, and even a perfectly GQA-optimized model wastes 60-80% of its GPU memory on fragmentation. Get them right, and you can serve 2-4x more requests on the same hardware.

The progression follows the serving stack bottom-up: first the KV cache itself (what it stores, how it grows), then PagedAttention (how to manage that memory efficiently), then continuous batching (how to schedule requests), then prefix caching (how to share computation), and finally the budget arithmetic that connects architecture to serving capacity.

The KV cache has a conceptual ancestor worth naming: Neural Turing Machines ([[neural-turing-machines|paper]], Graves et al. 2014) introduced *differentiable external memory* accessed via attention — a controller reading and writing an explicit memory matrix through content-based addressing. A modern Transformer's KV cache is exactly this pattern specialized to sequence history: each decode step queries an external memory via scaled dot-product attention, and the cache is the memory matrix written to during prefill. PagedAttention's block tables then add an OS-style virtual memory layer on top of that same substrate.

---

## 1. The KV Cache: What It Stores and Why

During autoregressive generation, each new token needs to attend to all previous tokens. Without caching, generating the $n$-th token requires recomputing keys and values for all $n-1$ previous tokens — $O(n^2)$ total work across a full generation. The KV cache eliminates this redundancy by storing previously computed key and value vectors and reusing them.

As Raschka ([[raschka-kv-cache|blog]]) demonstrates with a concrete implementation:

```python
def forward(self, x, use_cache=False):
    keys_new = self.W_key(x)
    values_new = self.W_value(x)
    queries = self.W_query(x)

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
```

With caching, each decoding step computes K and V only for the *new* token, then concatenates with the cached tensors. Per-step cost drops from $O(n)$ to $O(1)$ in key/value computation (though the attention computation itself remains $O(n)$ because the new query still attends to all cached positions). On Raschka's benchmarks, this yields a **5.3x speedup** for a 124M parameter model generating 200 tokens on CPU.

### Memory Growth Per Token

The KV cache grows linearly with sequence length. Per token, per layer, the cache stores:

$$\text{KV bytes/token/layer} = 2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{precision}$$

The factor of 2 accounts for both keys and values. The critical variable is $n_{\text{kv\_heads}}$ — this is where the attention variant from [[ch-07]] directly determines serving cost:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">KV Cache Memory Per Request: Architecture Determines Serving Cost</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Model</th>
<th style="text-align:right; padding:8px;">Attention</th>
<th style="text-align:right; padding:8px;">KV Heads</th>
<th style="text-align:right; padding:8px;">Cache/Token/Layer</th>
<th style="text-align:right; padding:8px;">Cache @ 4K seq</th>
<th style="text-align:right; padding:8px;">Cache @ 128K seq</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">Hypothetical 70B MHA</td>
<td style="text-align:right; padding:8px;">MHA</td>
<td style="text-align:right; padding:8px;">64</td>
<td style="text-align:right; padding:8px;">32 KB</td>
<td style="text-align:right; padding:8px;">10.0 GB</td>
<td style="text-align:right; padding:8px;">320 GB</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">Llama 3 405B</td>
<td style="text-align:right; padding:8px;">GQA</td>
<td style="text-align:right; padding:8px;">8</td>
<td style="text-align:right; padding:8px;">4 KB</td>
<td style="text-align:right; padding:8px;">2.0 GB</td>
<td style="text-align:right; padding:8px;">64.5 GB</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ffd93d; font-weight:bold;">DeepSeek-V2</td>
<td style="text-align:right; padding:8px;">MLA</td>
<td style="text-align:right; padding:8px;">-</td>
<td style="text-align:right; padding:8px;">~2 KB*</td>
<td style="text-align:right; padding:8px;">0.5 GB</td>
<td style="text-align:right; padding:8px;">14.4 GB</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#a29bfe; font-weight:bold;">Mistral 7B</td>
<td style="text-align:right; padding:8px;">GQA+SWA</td>
<td style="text-align:right; padding:8px;">8</td>
<td style="text-align:right; padding:8px;">4 KB</td>
<td style="text-align:right; padding:8px;">0.5 GB</td>
<td style="text-align:right; padding:8px;">0.5 GB (capped)</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
*MLA caches a 576-dim latent (512 + 64 RoPE) per token per layer instead of full KV heads.<br>
Llama 3 405B: 2 x 8 x 128 x 126 layers x seq_len x 2 bytes. Mistral 7B: capped at window size 4096 x 32 layers.
</div>
</div>

Mistral 7B ([[mistral-7b|report]]) is the most striking entry: its sliding window attention caps the cache at the window size regardless of sequence length. At 128K context, its cache is identical to its cache at 4K — both are bounded by $W = 4096$ positions. The rolling buffer implementation overwrites position $t - W$ when storing position $t$:

```python
cache_index = t % W  # Circular buffer overwrites old positions
```

### The Naive Implementation Problem

Raschka's code above uses `torch.cat` to grow the cache. This is correct but inefficient — every concatenation allocates a new tensor and copies all previous data. For a 70B model generating 4096 tokens, this means thousands of allocation-copy-free cycles, each larger than the last. The alternative is **pre-allocation**: reserve a buffer for the maximum sequence length at the start:

```python
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim))
# Write into slices rather than concatenating:
cache_k[:, :, current_pos, :] = new_key
```

Pre-allocation eliminates the copy overhead but introduces a new problem: you must reserve memory for the *maximum possible* sequence length, even if most requests are much shorter. For Llama 3 with 128K context, pre-allocating the full cache per request consumes ~64 GB — you could serve exactly one request per A100. This is the memory management problem that PagedAttention solves.

---

## 2. PagedAttention: Virtual Memory for the KV Cache

The key insight from the PagedAttention paper ([[paged-attention|paper]]): existing LLM serving systems waste **60-80% of KV cache memory** due to fragmentation and over-reservation. The analogy to operating systems is precise — early OS designs allocated contiguous memory blocks for each process, wasting space on internal fragmentation (allocated but unused space within a block) and external fragmentation (free space broken into unusable chunks between allocated blocks). Virtual memory and paging solved this by decoupling the logical address space from physical memory layout.

PagedAttention applies exactly the same idea to KV cache.

### The Block Table Abstraction

Instead of allocating a contiguous memory region for each request's KV cache, PagedAttention divides GPU memory into fixed-size **blocks** (typically 16 tokens per block). Each request maintains a **block table** — a mapping from logical block indices to physical block locations in GPU memory:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">PagedAttention: Logical vs Physical Memory</div>
<div style="display:flex; gap:40px; align-items:flex-start; justify-content:center; flex-wrap:wrap;">
<div>
<div style="color:#4ecdc4; font-weight:bold; font-size:12px; margin-bottom:8px;">Request A (logical view: contiguous)</div>
<div style="display:flex; gap:2px; margin-bottom:12px;">
<div style="width:50px; height:35px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold; border:1px solid #4ecdc4;">Blk 0</div>
<div style="width:50px; height:35px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold; border:1px solid #4ecdc4;">Blk 1</div>
<div style="width:50px; height:35px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold; border:1px solid #4ecdc4;">Blk 2</div>
</div>
<div style="color:#e94560; font-weight:bold; font-size:12px; margin-bottom:8px;">Request B (logical view: contiguous)</div>
<div style="display:flex; gap:2px;">
<div style="width:50px; height:35px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#e94560; font-size:10px; font-weight:bold; border:1px solid #e94560;">Blk 0</div>
<div style="width:50px; height:35px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#e94560; font-size:10px; font-weight:bold; border:1px solid #e94560;">Blk 1</div>
</div>
</div>
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center;">
<div style="color:#888; font-size:20px;">&#8594;</div>
<div style="color:#888; font-size:10px;">block<br>table</div>
</div>
<div>
<div style="color:#ffd93d; font-weight:bold; font-size:12px; margin-bottom:8px;">GPU Memory (physical: non-contiguous)</div>
<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:2px;">
<div style="width:50px; height:35px; background:#4ecdc4; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#1a1a2e; font-size:9px; font-weight:bold;">A:0</div>
<div style="width:50px; height:35px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:9px; font-weight:bold;">B:0</div>
<div style="width:50px; height:35px; background:#4ecdc4; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#1a1a2e; font-size:9px; font-weight:bold;">A:1</div>
<div style="width:50px; height:35px; background:#333; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#666; font-size:9px;">free</div>
<div style="width:50px; height:35px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:9px; font-weight:bold;">B:1</div>
<div style="width:50px; height:35px; background:#333; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#666; font-size:9px;">free</div>
<div style="width:50px; height:35px; background:#4ecdc4; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#1a1a2e; font-size:9px; font-weight:bold;">A:2</div>
<div style="width:50px; height:35px; background:#333; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#666; font-size:9px;">free</div>
</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:12px; text-align:center;">
Requests see contiguous logical blocks. Physical memory is allocated on demand from a free-block pool.<br>
Waste is limited to the last block of each request (internal fragmentation < 1 block per request).
</div>
</div>

For a detailed interactive visualization of this virtual memory system, see [figures/paged-attention.html](figures/paged-attention.html).

### Why This Matters: The Waste Arithmetic

Consider serving Llama 3 70B ([[llama-3|report]]) with a 4K max context. Per request, the KV cache can grow up to 1.25 GB (GQA with 8 KV heads, 80 layers). Without PagedAttention:

- **Pre-allocation approach:** Reserve 1.25 GB per request slot. If average generation is only 512 tokens (not 4096), you waste 87.5% of reserved memory.
- **Dynamic allocation approach:** Allocate contiguously as tokens are generated. As requests arrive and complete at different times, freed memory fragments into unusable gaps.

The PagedAttention paper measured that existing systems (FasterTransformer, Orca) wasted 60-80% of KV cache memory through these mechanisms. PagedAttention reduces waste to *less than one block per request* — internal fragmentation of the last partially-filled block.

### The Custom Kernel Cost

There is a real tradeoff. Standard attention kernels assume contiguous KV storage — a single pointer plus stride to read all keys. PagedAttention requires a custom kernel that looks up each block's physical location through the block table, introducing pointer indirection at every block boundary. This is slightly less efficient per operation than contiguous access. But the throughput gain from serving 2-4x more concurrent requests vastly outweighs the per-operation overhead.

### Copy-on-Write: Sharing Within Requests

For beam search or parallel sampling, multiple candidate sequences share a common prefix. Without sharing, each beam copies the prefix KV cache — duplicating potentially gigabytes of data. PagedAttention uses **copy-on-write** (exactly like the OS mechanism): beams share physical blocks for the common prefix, and a block is copied only when one beam diverges and needs to write different values. Reference counting tracks how many sequences point to each block.

---

## 3. Continuous Batching: Dynamic Request Scheduling

Traditional (static) batching waits until a fixed batch of requests is assembled, processes them together, and returns all results when the *last* request completes. The problem: LLM generation lengths vary dramatically. A request that generates 10 tokens sits idle while its batch-mate generates 2000 tokens — the GPU processes padding for the short request while doing real work for the long one.

**Continuous batching** (also called iteration-level scheduling, pioneered by Orca) makes scheduling decisions at *every decoding iteration*:

1. After each token generation step, check which requests have finished (hit EOS or max length)
2. Remove completed requests immediately
3. Insert new waiting requests into the batch immediately
4. The batch composition changes at every step

For an animated walkthrough of how continuous batching improves GPU utilization compared to static batching, see [figures/continuous-batching.html](figures/continuous-batching.html).

### The Two Phases: Prefill and Decode

Serving has two distinct computational phases with very different hardware characteristics:

**Prefill** processes the entire input prompt in parallel. This is compute-bound (large matrix multiplications) and benefits from high GPU arithmetic throughput. Prefill processes all $n$ prompt tokens simultaneously, computing all keys and values for the prompt in one forward pass.

**Decode** generates tokens one at a time (per request). This is memory-bandwidth-bound — the arithmetic per step is tiny (one row of Q times the full KV cache), but the model must load the entire KV cache from HBM. This is the regime where [[ch-07]]'s attention variants matter: GQA reduces bytes loaded by $H/G$ times, which translates nearly directly to proportional speedup because the GPU is waiting on memory reads.

The prefill/decode distinction creates a scheduling challenge: prefill wants large batch parallelism for compute efficiency, decode wants small batch sizes for low latency. Modern serving systems handle this by **chunked prefill** — breaking long prompts into chunks that interleave with decode iterations of active requests. Mistral 7B ([[mistral-7b|report]]) explicitly optimizes for this with its pre-fill chunking strategy, segmenting input prompts into window-sized chunks.

### Throughput Impact

Continuous batching's advantage is directly proportional to variance in generation lengths. If all requests generate exactly the same number of tokens, static batching wastes nothing. In practice, generation lengths span 10 tokens to 4000+ tokens, and continuous batching recovers 50-90% of the GPU utilization lost to static batching's padding.

Combined with PagedAttention, continuous batching enables **2-4x throughput improvement** over static batching systems. The two are synergistic: PagedAttention makes it possible to have many requests in flight simultaneously (efficient memory), and continuous batching ensures the GPU is always doing useful work (efficient scheduling).

---

## 4. Prefix Caching: Sharing Across Requests

Many serving scenarios involve requests that share a common prefix — the most obvious being a system prompt. If 1000 requests per minute all begin with the same 2000-token system prompt, naively computing and storing 2000 tokens of KV cache per request wastes enormous compute and memory.

**Prefix caching** (also called prompt caching) stores the KV cache for common prefixes and shares it across requests. PagedAttention's block table makes this natural: the shared prefix maps to the same physical blocks for all requests. Each request's block table starts with pointers to the shared prefix blocks, then diverges into request-specific blocks for the unique suffix.

### What Can Be Shared

- **System prompts:** The most common case. A model serving an API typically has a fixed system prompt prepended to every request. Caching its KV saves both the computation of prefill and the memory for per-request storage.
- **Few-shot examples:** When the same examples are prepended to multiple requests.
- **Conversation history:** In multi-turn conversations, the shared conversation prefix up to the latest turn can be cached and reused.

### The Compute Savings

For a system prompt of $P$ tokens with $R$ requests per second:

- **Without prefix caching:** $P \times R$ tokens of prefill compute per second
- **With prefix caching:** $P$ tokens computed once, reused $R$ times. Savings: $(R - 1) \times P$ tokens of prefill compute per second

For a 2000-token system prompt at 100 req/s, this saves 199,800 tokens/s of prefill compute — the equivalent of freeing up multiple GPUs.

### Implementation via PagedAttention

PagedAttention's block structure makes prefix caching efficient. The shared prefix occupies a fixed set of physical blocks. Each new request's block table begins with pointers to these blocks (read-only), then allocates new blocks for the divergent suffix. The copy-on-write mechanism from Section 2 handles the boundary correctly: if a request somehow needed to modify a shared block, it would copy-on-write, but in practice, prefix blocks are read-only during generation.

---

## 5. Memory Budget Calculation: From Architecture to Throughput

The central question for LLM serving capacity is: given a GPU with $M$ bytes of memory, after loading model weights, how many concurrent requests can be served? The answer is a direct function of the attention variant.

### The Budget Equation

$$\text{Max concurrent requests} = \frac{M_{\text{GPU}} - M_{\text{weights}} - M_{\text{activations}}}{M_{\text{KV per request}}}$$

where:

$$M_{\text{KV per request}} = 2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times L \times S_{\text{avg}} \times \text{bytes\_per\_param}$$

$L$ = number of layers, $S_{\text{avg}}$ = average sequence length (prompt + generation).

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Serving Capacity: Concurrent Requests on a Single A100 80GB</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Model</th>
<th style="text-align:right; padding:8px;">Weights (FP16)</th>
<th style="text-align:right; padding:8px;">KV/req @ 2K avg</th>
<th style="text-align:right; padding:8px;">Max Concurrent</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560;">Llama 2 7B (GQA-8, 32L)</td>
<td style="text-align:right; padding:8px;">14 GB</td>
<td style="text-align:right; padding:8px;">0.25 GB</td>
<td style="text-align:right; padding:8px; color:#4ecdc4; font-weight:bold;">~264</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560;">Mistral 7B (GQA-8+SWA, 32L)</td>
<td style="text-align:right; padding:8px;">14 GB</td>
<td style="text-align:right; padding:8px;">0.13 GB*</td>
<td style="text-align:right; padding:8px; color:#4ecdc4; font-weight:bold;">~508</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560;">Llama 3 70B (GQA-8, 80L)</td>
<td style="text-align:right; padding:8px;">140 GB**</td>
<td style="text-align:right; padding:8px;">0.63 GB</td>
<td style="text-align:right; padding:8px; color:#ffd93d;">needs multi-GPU</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560;">DeepSeek-V2 (MLA, 60L)</td>
<td style="text-align:right; padding:8px;">~42 GB***</td>
<td style="text-align:right; padding:8px;">0.13 GB</td>
<td style="text-align:right; padding:8px; color:#4ecdc4; font-weight:bold;">~292</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
*Mistral 7B: SWA caps effective seq len at 4096 regardless of actual sequence length, calculated with min(avg_seq, 4096).<br>
**70B model requires tensor-parallel across multiple GPUs; numbers here are per-GPU share of a multi-GPU setup.<br>
***DeepSeek-V2: 21B active params (MoE), but weights across experts are larger; shown as approximate FP16 active-param footprint.
</div>
</div>

### Worked Example: Llama 3 8B on A100 80GB

Llama 3 8B ([[llama-3|report]]): 32 layers, 8 KV heads (GQA), $d_{\text{head}} = 128$, FP16.

1. **Model weights:** 8B params x 2 bytes = 16 GB
2. **Available for KV cache:** 80 - 16 - 2 (activations/overhead) = 62 GB
3. **KV per token per layer:** $2 \times 8 \times 128 \times 2 = 4{,}096$ bytes = 4 KB
4. **KV per token (all layers):** $4 \times 32 = 128$ KB
5. **KV per request @ 2K avg seq:** $128 \times 2{,}048 = 256$ MB
6. **Max concurrent requests:** $62{,}000 / 256 \approx 242$

Now the same calculation for a hypothetical MHA version (32 KV heads instead of 8):

- KV per token per layer: $2 \times 32 \times 128 \times 2 = 16{,}384$ bytes = 16 KB
- KV per request @ 2K: $16 \times 32 \times 2{,}048 = 1{,}024$ MB = 1 GB
- Max concurrent: $62{,}000 / 1{,}024 \approx 60$

**GQA gives 4x the serving capacity.** This is not a theoretical argument — it is the direct reason that Llama 3 uses GQA with 8 KV heads at every model size ([[llama-3|report]]). The 405B model also uses only 8 KV heads despite having 128 query heads — a 16:1 ratio — because the serving capacity gain at 128K context is existential.

### MLA's Budget Advantage

DeepSeek-V2 ([[deepseek-v2|report]]) caches only the 576-dimensional latent ($d_c = 512$ + $d_h^R = 64$ for decoupled RoPE) per token per layer, regardless of the number of attention heads (128). Compare:

- **GQA-8 equivalent:** $2 \times 8 \times 128 = 2{,}048$ dims per token per layer
- **MLA:** $576$ dims per token per layer

That is a **3.6x advantage over GQA-8**, or equivalently, 3.6x more concurrent requests at the same average sequence length. The paper reports **5.76x generation throughput improvement** over the MHA-based DeepSeek 67B predecessor.

---

## 6. Architecture-Serving Co-Design

The serving stack is not independent of the model architecture. Architectural decisions made during training — often months before deployment — lock in the serving cost envelope. This section connects the architectural choices from [[ch-07]] and [[ch-18]] to their serving consequences.

### GQA: The Industry Default for a Reason

Llama 3 ([[llama-3|report]]) uses 8 KV heads at every scale (8B, 70B, 405B). This is a serving-first decision: 8 KV heads provide enough representational diversity (the quality evidence from [[ch-07]]) while keeping per-request memory 4-16x lower than MHA. Critically, GQA is supported natively by every major serving framework — vLLM, TensorRT-LLM, llama.cpp, SGLang — with heavily optimized kernels. As the Raschka survey ([[raschka-kv-cache|blog]]) notes, the ecosystem maturity of GQA often matters more than the architectural superiority of alternatives.

### MLA: Maximum Compression, Kernel Complexity

DeepSeek-V2's MLA ([[deepseek-v2|report]]) achieves 93.3% KV cache reduction while *improving* quality — the low-rank latent bottleneck acts as a regularizer. But the up-projection from latent to full keys/values during attention requires custom CUDA kernels, and inference tooling support is still maturing. For 100B+ models where the memory savings justify the engineering cost, MLA is the superior choice. For smaller models, GQA's tooling maturity wins.

### Sliding Window: Bounded Memory, Bounded Context

Mistral 7B ([[mistral-7b|report]]) combines GQA (8 KV heads, 4x reduction in cache per token) with sliding window attention ($W = 4096$, bounded total cache). The multiplicative effect is powerful: GQA reduces the *width* of each cached token, SWA bounds the *number* of cached tokens. The result is a model that can process arbitrary-length sequences with a *fixed* memory budget for KV cache.

The hybrid approach used by later models (Gemma 3: 5:1 local/global ratio) preserves bounded memory for most layers while maintaining exact retrieval capability through occasional global attention layers.

### The Co-Design Principle

The relationship flows in both directions:

**Architecture constrains serving:** If you choose MHA with 64 KV heads, no amount of systems optimization can make your model serve 100+ concurrent requests at 128K context on reasonable hardware. The memory budget is set.

**Serving constraints should inform architecture:** When Llama 3 chose 8 KV heads for the 405B model (16:1 query-to-KV ratio), they were not just optimizing training quality — they were designing for a world where the 405B model would need to serve thousands of concurrent requests across data centers. The attention variant is as much an infrastructure decision as a modeling one.

---

## Core Insights from the Literature

### Insight 1: KV cache fragmentation, not cache size, is the serving bottleneck
**Paper:** Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" ([[paged-attention|paper]])

Before PagedAttention, the community focused on reducing KV cache *size* (GQA, MQA, MLA). The PagedAttention paper revealed that even with size-optimized caches, **60-80% of allocated KV memory was wasted** due to fragmentation and pre-allocation overhead. The OS virtual memory analogy — fixed-size blocks, page tables, copy-on-write — eliminated nearly all waste, enabling 2-4x throughput improvement without changing the model architecture at all. **Guideline:** Always serve through a PagedAttention-based engine (vLLM, SGLang, TensorRT-LLM). The memory management system is as important as the attention variant for serving throughput.

### Insight 2: Static batching fundamentally underutilizes the GPU
**Paper:** Kwon et al., PagedAttention ([[paged-attention|paper]]); Yu et al., Orca (2022)

Static batching pads all requests to the longest sequence in the batch, wasting compute on padding tokens. The waste is proportional to generation length variance — which in practice is enormous (10 tokens to 4000+). Continuous batching (iteration-level scheduling) makes admission and eviction decisions at every decoding step, keeping GPU utilization consistently high. **Guideline:** Never deploy static batching for production serving. The throughput loss is 2-4x compared to continuous batching, and every modern serving framework supports it by default.

### Insight 3: The attention variant chosen during training sets the serving cost ceiling
**Sources:** Llama 3 ([[llama-3|report]]), DeepSeek-V2 ([[deepseek-v2|report]]), Mistral 7B ([[mistral-7b|report]])

This is the co-design insight. Llama 3 standardized on 8 KV heads across all model sizes — not because 8 is optimal for training quality, but because it provides 4-16x serving capacity improvement over MHA with minimal quality loss. DeepSeek-V2 went further with MLA's 93.3% reduction but required custom inference kernels. Mistral 7B added sliding window for bounded memory. Each choice trades architectural simplicity for serving efficiency. **Guideline:** When designing a model architecture, compute the KV cache memory budget at your target serving scale *before* finalizing the attention configuration. A model that cannot be served efficiently is a model that cannot be deployed.

### Insight 4: Prefix caching turns shared structure into serving leverage
**Paper:** Kwon et al., PagedAttention ([[paged-attention|paper]])

PagedAttention's block-based memory enables natural prefix sharing: multiple requests with the same system prompt point to the same physical KV blocks. For API serving with a fixed system prompt, this eliminates both the compute (prefill) and memory (per-request KV storage) cost of the shared prefix. The savings scale linearly with request volume. **Guideline:** Design your serving architecture to exploit prefix sharing. Use consistent system prompts, and configure your serving engine to enable prefix caching (vLLM enables this by default for detected common prefixes).

---

## Key Takeaways

1. **The KV cache is a space-time tradeoff:** It eliminates $O(n^2)$ redundant computation during generation at the cost of $O(n)$ memory that grows with sequence length. The attention variant (MHA/GQA/MLA/SWA) determines the constant factor.

2. **Memory *management* matters as much as memory *size*.** PagedAttention's OS-inspired virtual memory approach eliminates 60-80% of KV cache waste through block-based allocation, enabling 2-4x throughput improvements without model changes.

3. **Continuous batching keeps the GPU busy.** By scheduling at every decoding iteration rather than waiting for batch completion, continuous batching recovers the GPU utilization lost to generation length variance.

4. **Prefix caching converts shared structure into performance.** System prompts, few-shot examples, and conversation history can be computed once and shared across requests, saving both compute and memory proportional to request volume.

5. **The memory budget equation connects architecture to serving capacity.** It is: (GPU memory - weights - overhead) / (KV per request). The attention variant directly sets the denominator. GQA-8 provides 4x more capacity than MHA; MLA provides ~3.6x more than GQA-8; SWA bounds the denominator regardless of sequence length.

6. **Architecture and serving co-design is not optional.** The attention variant chosen during training — months before deployment — sets the serving cost ceiling. Llama 3's universal GQA-8, DeepSeek-V2's MLA, and Mistral's SWA are all serving-informed architectural decisions.

7. **Tooling maturity is a constraint.** GQA dominates partly because PagedAttention, continuous batching, and prefix caching are all optimized for it in every major serving framework. MLA's superior compression requires custom kernels that are still maturing.

---

## References

- [[paged-attention|Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) (paper)]] — PagedAttention, vLLM
- [[raschka-kv-cache|Raschka, "Understanding and Coding the KV Cache in LLMs from Scratch" (2025) (blog)]] — KV cache implementation and benchmarks
- [[llama-3|Meta AI, "The Llama 3 Herd of Models" (2024) (report)]] — GQA standardization across scales
- [[deepseek-v2|DeepSeek AI, "DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model" (2024) (report)]] — MLA and serving throughput
- [[mistral-7b|Jiang et al., "Mistral 7B" (2023) (report)]] — Sliding window attention and rolling buffer cache
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (2022) — Continuous batching / iteration-level scheduling
- [[neural-turing-machines|Graves, A., Wayne, G., & Danihelka, I. "Neural Turing Machines." arXiv:1410.5401, 2014. — paper]] — Differentiable external memory via attention; conceptual ancestor of KV cache as content-addressed memory

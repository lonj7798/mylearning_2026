---
chapter: ch-17
course: llm-inference
phase: read
excerpt_of: "SGLang: Efficient Execution of Structured Language Model Programs (Zheng et al. 2023) + SGLang serving docs"
source_url: https://arxiv.org/abs/2312.07104
created_at: "2026-05-21"
---

# Excerpt: SGLang Architecture — frontend DSL + RadixAttention runtime

**Authors:** Lianmin Zheng et al. (SGLang team, UC Berkeley + LMSYS)
**Year:** 2023–present
**URLs:** https://arxiv.org/abs/2312.07104 (paper) / https://docs.sglang.ai/ (docs) / https://github.com/sgl-project/sglang (source)
**Raw-data source:** [[raw-data/sglang]]

---

## The four-layer stack

```
┌────────────────────────────────────────────────────────────┐
│ 1. Frontend Language (Python DSL)                          │
│    @sgl.function decorator wraps a multi-step LLM program. │
│    Primitives: gen, select, fork, system, user, assistant. │
│    Compiled to a request graph the backend can pre-plan.   │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ 2. Server / API layer                                      │
│    OpenAI /v1/chat/completions, /v1/completions            │
│    Native /generate (text + sampling_params + constraints) │
│    Offline Engine (in-process Python, no HTTP)             │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ 3. Scheduler (managers/scheduler.py)                       │
│    Waiting / running queues + cache-aware admission        │
│    Chunked prefill + continuous batching                   │
│    Coordinates grammar, LoRA, speculative decoding         │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ 4. Backend / Model Worker                                  │
│    RadixAttention KV cache (mem_cache/radix_cache.py)      │
│    HiCache tiers (L1 GPU → L2 host → L3 distributed)       │
│    Attention kernels (FlashInfer / Triton / CUTLASS)       │
└────────────────────────────────────────────────────────────┘
```

The structural contrast with vLLM: SGLang's frontend can *tell* the backend that a `fork()` is coming, so the backend keeps the shared prefix KV pinned across branches. vLLM has no equivalent signal — its prefix cache must rediscover the sharing post-hoc via block hashing.

---

## Why the frontend exists

A naive multi-step agent in plain OpenAI client code:

```python
ctx = system + user + "Step 1?"
step1 = client.chat.completions.create(messages=ctx).choices[0].message
ctx = ctx + step1.content + "Step 2?"
step2 = client.chat.completions.create(messages=ctx).choices[0].message
# ... etc
```

Each call is opaque to the server. Even with vLLM's automatic prefix cache, the server has to *guess* via hash matching whether the new prompt shares prefix with the previous one. The hash matches work but they're after-the-fact.

The same program in SGLang DSL:

```python
@sgl.function
def stepwise(s, problem):
    s += sgl.system("Think step by step.")
    s += sgl.user(problem)
    for i in range(5):
        s += sgl.assistant(sgl.gen(f"step_{i}", max_tokens=128))
        s += sgl.user("Next?")
```

The runtime sees the whole 5-step program graph at function entry. Each `gen` is a suspension point — the runtime knows step `i+1` will extend the same prefix that step `i` finished with, so it never even considers evicting those blocks while the program is live.

---

## The three primitive operations

```python
sgl.gen(name, max_tokens=..., stop=..., temperature=...)
    → autoregressive generation bound to a variable `name`
    → backend: one prefill on the suffix + decode loop

sgl.select(name, choices=["yes", "no", "maybe"])
    → constrained generation: pick exactly one of choices
    → backend: short generation under grammar mask + score comparison

s.fork(N)  →  [s_0, s_1, ..., s_{N-1}]
    → N parallel branches from the same point
    → backend: ref-count the shared prefix node N times; admit branches in parallel
```

The `fork` primitive is what no plain OpenAI API gives you. It's the explicit "I am about to issue N requests sharing this exact prefix; please pin it" hint that lets RadixAttention drive 90 %+ hit rate on tree-search and self-consistency workloads.

---

## The OpenAI-compatible bypass

For 80 % of production traffic, SGLang serves OpenAI-shaped requests:

```bash
python -m sglang.launch_server --model meta-llama/Llama-3-8B-Instruct --port 30000

curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3", "messages": [...], "stream": true}'
```

The frontend DSL is opt-in. The backend optimizations (RadixAttention, HiCache, chunked prefill) work regardless of which surface clients use — but the frontend gives the backend strictly more information to work with.

---

## Connections

- [[excerpts/sglang-radixattention]] — the core data structure this architecture is built around.
- [[excerpts/sglang-scheduler]] — the cache-aware admission that turns RadixAttention into throughput.
- [[ch-17]] — parent synthesis of the SGLang stack.
- [[ch-16]] — vLLM architecture, for direct comparison.

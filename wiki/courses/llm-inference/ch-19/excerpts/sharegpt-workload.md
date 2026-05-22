---
chapter: ch-19
course: llm-inference
phase: read
excerpt_of: "ShareGPT — the de-facto LLM serving benchmark workload"
source_url: https://docs.vllm.ai/en/latest/contributing/benchmarks.html
created_at: "2026-05-21"
---

# Excerpt: ShareGPT Workload — heavy-tailed prompt + response distributions

**Authors:** ShareGPT dataset community + vLLM/SGLang benchmark maintainers
**Year:** 2023–2026
**URLs:** https://docs.vllm.ai/en/latest/contributing/benchmarks.html + dataset commonly available as `ShareGPT_V3_unfiltered_cleaned_split.json`
**Raw-data source:** [[raw-data/sharegpt-workload]]

---

## What ShareGPT is

A scrape of real user-assistant conversations from the ChatGPT era. Used in serving benchmarks to extract realistic prompt and response length distributions, not for the textual content itself.

Typical (post-filter) distributions:

```
Prompt length (tokens):    median ~250, mean ~600, p95 ~2k, p99 ~8k
Response length (tokens):  median ~400, mean ~550, p95 ~1.5k, p99 ~4k
```

Heavy-tailed in both dimensions — short conversations dominate by count, long ones dominate by tokens.

---

## How it's replayed

```python
# vllm/benchmarks/benchmark_serving.py (simplified)
def sample_sharegpt(num_samples, tokenizer, min_input=4, max_input=4096,
                    min_output=4, max_output=2048):
    raw = json.load(open("ShareGPT_V3_unfiltered_cleaned_split.json"))
    sampled = []
    for conv in raw:
        if len(conv["conversations"]) < 2:
            continue
        prompt = conv["conversations"][0]["value"]
        response = conv["conversations"][1]["value"]
        p_tok = len(tokenizer.encode(prompt))
        r_tok = len(tokenizer.encode(response))
        if (min_input <= p_tok <= max_input
            and min_output <= r_tok <= max_output):
            sampled.append((prompt, r_tok))
        if len(sampled) >= num_samples:
            break
    return sampled
```

Two replay modes:

### Open-loop (request rate) — right for serving benchmarks

```python
import asyncio, random
async def open_loop(prompts, rate):
    # Poisson arrival at `rate` requests/sec.
    for prompt, out_len in prompts:
        await issue_request(prompt, out_len)        # fire and forget
        delay = random.expovariate(rate)            # exponential inter-arrival
        await asyncio.sleep(delay)
```

Captures queuing effects — if the server can't keep up, requests pile up and tail latency grows.

### Closed-loop (concurrency) — right for engine-capacity benchmarks

```python
async def closed_loop(prompts, concurrency):
    sem = asyncio.Semaphore(concurrency)
    async def worker(prompt, out_len):
        async with sem:
            await issue_request(prompt, out_len)
    await asyncio.gather(*[worker(p, l) for p, l in prompts])
```

Keeps N in flight; never goes above the server's capacity. Measures engine throughput in isolation.

---

## Disclosure requirements

When publishing a ShareGPT-based benchmark, disclose all of:

| Setting | Why |
|---------|-----|
| Source JSON file URL + SHA | Different snapshots differ |
| Tokenizer (model + revision) | Affects token-count filters |
| Filters: min/max prompt + output | Excluding p99 changes distribution |
| Output-length replay policy | Replay exact length vs generate to EOS |
| Random seed | Sample order matters under burst |
| Chat template applied | Or not — changes input tokens |
| Number of prompts | Smaller N has more variance |

Without these, results are not reproducible.

---

## Alternatives

| Workload | When to use |
|----------|-------------|
| Synthetic uniform length | Microbenchmarks; controlled isolation |
| **ShareGPT** | OSS default for chat-like serving |
| LongBench / RULER | Long-context (≥32k tokens) |
| Production trace replay | If you have it; anonymized |
| CodeAlpaca | Code-completion serving |
| GSM-8k / MATH | Reasoning-trace serving (long output) |
| Generated-shared-prefix | Test prefix-cache behavior (SGLang `--dataset-name generated-shared-prefix`) |

---

## The shared-prefix variant — SGLang's bench addition

For prefix-cache-aware testing:

```bash
python -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --random-input-len 4096 \
    --gen-shared-prefix-num-groups 8 \
    --gen-shared-prefix-prompts-per-group 32 \
    --gen-shared-prefix-system-prompt-len 2048
```

Generates 8 distinct system-prompt prefixes, each used by 32 requests. Directly exercises RadixAttention / APC behavior. Use this to expose the *prefix-cache value*; ShareGPT alone won't.

---

## Connections

- [[excerpts/ttft-tpot-itl]] — the metrics to compute from each replayed request.
- [[excerpts/vllm-benchmarks]] — the script that sampled ShareGPT first.
- [[excerpts/sglang-benchmarks]] — adds the shared-prefix variant.
- [[ch-19]] — parent synthesis.

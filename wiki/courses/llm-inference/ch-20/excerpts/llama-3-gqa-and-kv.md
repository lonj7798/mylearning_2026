---
chapter: ch-20
course: llm-inference
phase: read
excerpt_of: "Llama 3 / 3.1 / 3.2 inference-relevant facts"
source_url: https://ai.meta.com/blog/meta-llama-3/
created_at: "2026-05-21"
---

# Excerpt: Llama 3 GQA-8 geometry + the EOT serving gotcha

**Authors:** Meta AI
**Year:** 2024–2025
**Releases:** Llama 3 (Apr 2024, 8B/70B), Llama 3.1 (Jul 2024, +405B + 128k), Llama 3.2 (Sep 2024, 1B/3B + vision)
**URL:** https://ai.meta.com/blog/meta-llama-3/
**Raw-data source:** [[raw-data/llama-3-inference]]

---

## The GQA-8 family rule

Meta fixed `kv_heads = 8` for every Llama 3 size from 8B through 405B. The replication ratio (query heads per KV head) grows with model size:

| Size | Layers | Q heads | KV heads | Repl. | head_dim | KV bytes/token (BF16) |
|------|-------:|--------:|---------:|------:|---------:|----------------------:|
| 8B   | 32     | 32      | 8        | 4:1   | 128      | 131 072  (= 2·32·8·128·2) |
| 70B  | 80     | 64      | 8        | 8:1   | 128      | 327 680  (= 2·80·8·128·2) |
| 405B | 126    | 128     | 8        | 16:1  | 128      | 516 096  (= 2·126·8·128·2) |

The serving consequence: KV cache scales with **layer count**, not query-head count. Going 8B → 70B is 2.5× the KV cost. Going 8B → 405B is 3.9× the KV cost — even though param count grew 50×.

This is exactly why GQA is the "default for inference" architecture choice: a Llama-3-70B with naive MHA-64 would cache `64 / 8 = 8×` more KV — 2.6 MB/token instead of 320 KB/token — making 70B unservable at 128 k context on any current single-node deployment.

---

## The 128 k context KV-budget math

Llama 3.1 supports 128 k. For 70B at BF16:

```
KV per request at 128k = 327 680 bytes/token × 131 072 tokens
                       = ~40 GB
```

On an 8×H100 node with TP=8, the 70B weights take ~140 GB (BF16) split across 8 GPUs = ~17.5 GB/GPU. Free KV budget per GPU ≈ 50 GB. At 40 GB/request, **one** max-context request takes more than that — meaning you cannot run a single 128 k request without spilling. Practical deployments either (a) cap `--max-model-len` to 32 k, (b) shard KV across GPUs via sequence parallelism, or (c) quantize KV to FP8 (halving cost).

This is the kind of math every Llama 3 production deployment must do once before launch. The model card advertises 128 k; the serving system enforces a tighter limit.

---

## The EOT-vs-EOS serving footgun

Llama 3 introduced a 128k-token vocabulary with multiple "end" tokens:

| Token | ID | Meaning |
|-------|---:|---------|
| `<\|begin_of_text\|>` | 128000 | BOS for completions |
| `<\|end_of_text\|>` | 128001 | end of *document* (pretraining) |
| `<\|start_header_id\|>` | 128006 | role marker open |
| `<\|end_header_id\|>` | 128007 | role marker close |
| `<\|eot_id\|>` | **128009** | **end of turn (chat)** |

After instruction tuning, the chat model emits `<|eot_id|>` (id 128009), **not** `<|end_of_text|>` (id 128001), at the end of each assistant turn. If your serving stack only adds 128001 to `stop_token_ids` (the tokenizer's `eos_token_id`), the model will:

1. emit the answer + `<|eot_id|>`
2. continue past it (because 128009 wasn't a stop)
3. roleplay both sides of a conversation until `max_tokens`

The fix in vLLM / SGLang / HF:

```python
# vLLM
SamplingParams(
    max_tokens=2048,
    stop_token_ids=[128001, 128009],  # both EOS and EOT
)
```

For Llama-3 *Instruct* checkpoints, the tokenizer's `eos_token_id` is already set to 128009 in recent HF releases — but if you load an older snapshot, double-check. This bug is silent: the response looks "fine" until you read the second paragraph.

---

## Tokenizer + chat template

The chat template (HF format):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

Two non-obvious requirements:
- The blank line after `<|end_header_id|>` is *load-bearing* — it's part of the canonical template and dropping it changes tokenisation slightly enough to hurt downstream eval scores.
- The assistant turn opens with `<|start_header_id|>assistant<|end_header_id|>\n\n` *with no closing `<|eot_id|>`* — the model must emit it.

`apply_chat_template(messages, tokenize=True, add_generation_prompt=True)` handles all of this. Never construct the prompt string by hand.

---

## Quantization in production

For Llama 3 70B specifically, the dominant production quantization paths are:

| Path | Bits/wt | Weight memory | Latency vs BF16 | Where |
|------|--------:|--------------:|----------------:|-------|
| BF16 | 16 | 140 GB | 1.0× | Reference, multi-GPU |
| FP8 (W8A8) | 8 | 70 GB | 1.0–1.2× | TensorRT-LLM, vLLM |
| W4 GPTQ + Marlin | 4 | 35 GB | 1.5–2× faster | vLLM, SGLang |
| AWQ-W4 | 4 | 35 GB | 1.5–2× faster | vLLM |

Marlin-class W4 kernels (see ch-19 of model-quantization course for the kernel detail) hit 80–90 % of FP16 quality on standard evals with ~1.7× throughput speedup at batch 16. This is the deployment-default for Llama 3 70B in mid-2025.

---

## Connections

- [[excerpts/moe-param-accounting]] — Llama 3 is dense so accounting is trivial; the contrast with Mixtral / Qwen-3-MoE clarifies the "active vs total" distinction.
- [[ch-03]] — the per-token KV formula `2 · layers · kv_heads · head_dim · dtype` is the only equation needed to derive every Llama 3 KV number.
- [[ch-21]] — the lab uses `meta-llama/Meta-Llama-3-8B-Instruct` as the head-to-head workhorse.

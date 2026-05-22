---
chapter: ch-20
course: llm-inference
phase: read
excerpt_of: "Qwen3 hybrid thinking mode + serving cost implications"
source_url: https://qwenlm.github.io/blog/qwen3/
created_at: "2026-05-21"
---

# Excerpt: Qwen 3 `enable_thinking` — a serving-level knob

**Authors:** Qwen Team, Alibaba
**Year:** April 2025
**URL:** https://qwenlm.github.io/blog/qwen3/
**Raw-data source:** [[raw-data/qwen-3-inference]]

---

## What the knob actually does

Qwen 3's chat template inspects `enable_thinking` (default **`True`**). When true, the model emits a `<think>...</think>` block of internal reasoning before the visible answer. Both the thinking tokens and the visible tokens are real autoregressive output — they cost decode time, hold KV cache, and count against `max_tokens`.

Unlike DeepSeek R1, this is the *same* checkpoint with two modes selected at inference; there is no separate "reasoning model" weight set. The training mixed thinking and non-thinking traces so the model learns to gate on the template flag.

---

## Serving cost arithmetic

For a simple multi-step problem, here is the cost difference. Visible answer: ~100 tokens. Thinking block: ~1500 tokens. TPOT: 25 ms/token (Qwen-3-32B on H100, BF16).

| Mode | Output tokens | Wall time |
|------|---------------:|----------:|
| `enable_thinking=False` | 100 | 2.5 s |
| `enable_thinking=True`  | 1600 | 40 s |

A 16× latency multiplier from one config flag. For an SLO-bound deployment, ignoring this turns "p95 TTFT 200 ms / TPOT 25 ms" into "p95 end-to-end 40 s" — and the operator only finds out when monitoring catches it.

KV cache cost scales the same way: 1600 tokens of cache held for the full 40 s vs 100 tokens held for 2.5 s. Concurrent capacity drops by roughly the same multiple.

---

## How to disable / control

Three equivalent ways:

```python
# 1. Template kwarg (vLLM, SGLang, HF)
client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[...],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

# 2. Soft-switch in system prompt
messages = [
    {"role": "system", "content": "You are a concise assistant. /no_think"},
    ...
]

# 3. Construct prompt manually with the template flag
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False,
)
```

`/no_think` is parsed by the chat template's prepass and is the lowest-friction option for end-user-controlled apps.

---

## Length-controlled thinking

Qwen 3 also exposes the convention `/think (max=N)` in the system prompt, which the model interprets as a soft budget for the thinking block. This is not enforced by the runtime (no token-counting stop logic) — the model is *trained* to emit roughly the requested budget. Compliance is high but not perfect; pair it with a hard `max_tokens` if the SLO is strict.

---

## The serving-stack implications

Five things to do for any production Qwen 3 deployment:

1. **Decide the default explicitly.** Don't ship the upstream default; pick `enable_thinking=False` for low-latency chat, `True` for analytical workloads.
2. **Expose it per-request.** Make it a request-level option so users / your routing layer can pick.
3. **Size capacity for the thinking case.** Concurrent-request budget should assume 1k–2k output tokens, not 100.
4. **Strip the `<think>` block before returning to end users** in most consumer apps. The block is internal reasoning — keep it server-side for audit, return only what's after `</think>`.
5. **Charge correctly.** If you bill by output tokens, the thinking tokens are real tokens you pay for. Decide whether to bill them through or absorb them.

---

## Comparison with DeepSeek R1 thinking

| | Qwen 3 thinking | DeepSeek R1 |
|---|---|---|
| Mechanism | One model, template flag | Dedicated reasoning model |
| Thinking length | 200–2000 tokens (typical) | 500–20000 tokens (typical) |
| Off-switch | `enable_thinking=False` | Cannot disable (model is trained to think) |
| Architecture | Dense or MoE GQA | MoE + MLA (671B) |
| Distilled variants | n/a (just use non-thinking mode) | Yes, R1-Distill-{Qwen,Llama}-{7,14,32,70}B |

For a serving stack, the practical difference: Qwen 3 thinking is *opt-in per request*; R1 thinking is *always on, length-dependent on the prompt*. Capacity planning differs accordingly.

---

## Connections

- [[ch-20]] §2 (Qwen 3) and §4 (DeepSeek R1) — both reasoning-style patterns covered in the main chapter.
- [[ch-19]] — the TPOT × output_length lens; thinking modes are the canonical case where the headline TPOT looks fine but user latency is 10×.
- [[ch-21]] — the lab uses Qwen-1.8B for the resource-constrained path; that model predates the thinking-mode feature so the knob is moot there.

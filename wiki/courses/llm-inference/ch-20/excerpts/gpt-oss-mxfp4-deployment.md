---
chapter: ch-20
course: llm-inference
phase: read
excerpt_of: "OpenAI gpt-oss model card + MXFP4 MoE weight format"
source_url: https://openai.com/index/introducing-gpt-oss/
created_at: "2026-05-21"
---

# Excerpt: GPT-OSS MXFP4 — why 120B fits a single 80 GB GPU

**Authors:** OpenAI
**Year:** August 2025
**URL:** https://openai.com/index/introducing-gpt-oss/
**License:** Apache 2.0
**Raw-data source:** [[raw-data/gpt-oss-inference]]

---

## The headline target

gpt-oss-120b is designed to fit on a single 80 GB GPU (H100, H200, B100). gpt-oss-20b is designed to fit on a 16 GB consumer GPU (RTX 4080 / 4090 with KV-cache caveats). This is the only frontier open release in 2025 whose memory budget is the *headline* — quality is competitive with GPT-4-mini-class models, but the design goal was deployability.

Two architectural decisions make the budget achievable:

1. **MoE sparsity** — only 5.1B of 117B params activate per token (≈4.4 %).
2. **MXFP4 MoE weights** — expert weights are pre-quantized to 4-bit floating point.

---

## MXFP4 in one paragraph

MXFP4 = **Microscaling FP4**, an OCP-standardised microscaling format. Each block of 32 elements shares one 8-bit (E8M0) scale exponent; each element is FP4 (E2M1: 1 sign, 2 exponent, 1 mantissa). Per-element storage = `4 + 8/32 = 4.25` bits effective. The block-scale layout matches Blackwell's hardware FP4 GEMM units; on Hopper / Ada it's emulated via FP16 dequant.

For comparison:

| Format | Bits/element | Effective with scales | Comment |
|--------|-------------:|---------------------:|---------|
| BF16 | 16 | 16 | Reference |
| FP8 (E4M3) | 8 | 8 (per-tensor) or ~8.06 (per-row) | DeepSeek V3 native |
| MXFP4 | 4 | **4.25** | GPT-OSS MoE weights |
| INT4 (GPTQ g=128) | 4 | ~4.13 | Common open-quant target |

MXFP4's per-element format is *floating-point* not integer, so the dynamic range is wider — meaningful for MoE expert outputs which have heavy-tailed distributions. The per-32-element scale block prevents one outlier from polluting a long row.

---

## Memory budget breakdown — 120B on H100

```
gpt-oss-120b @ MXFP4 (MoE) + BF16 (attention) on a single H100 80 GB:

  MoE expert weights (the bulk, 110B params × 0.5 bytes/MXFP4)   ≈ 55 GB
  MXFP4 scale blocks (~1 % overhead)                              ≈  1 GB
  Attention + embedding + norm (~7B params × 2 bytes BF16)        ≈ 14 GB
  Activations + workspace during forward (estimate)               ≈  3 GB
  ────────────────────────────────────────────────────────────────────────
  Subtotal                                                         ≈ 73 GB
  KV cache budget left                                             ≈  7 GB
```

7 GB / 74 KB-per-token ≈ 95k token-equivalents of KV. At a context cap of 16 k tokens that's ~5 concurrent requests; at 8 k it's ~12; at 128 k it's <1 (single max-context request blocks everyone).

The serving recipe: set `--max-model-len 16384` and `--max-num-seqs 8` as defaults; raise either only if your workload demands and you've measured the budget.

In **BF16** the same model would need 117B × 2 = 234 GB → 3×H100 minimum. MXFP4 is not optional for 120B; it is the deployment format.

---

## Memory budget — 20B on a 16 GB consumer GPU

```
gpt-oss-20b @ MXFP4 (MoE) + BF16 (attention):

  MoE expert weights (~15B × 0.5 bytes)                ≈ 7.5 GB
  Scales                                               ≈ 0.2 GB
  Attention / embedding / norm (~6B × 2 bytes)         ≈ 12  GB  ← problem
```

The 20B model's MoE weights fit easily; the attention + embeddings push the BF16 budget over 16 GB. Realistic single-GPU 16 GB deployments quantize the attention path to FP8 (vLLM `--quantization fp8` for non-MoE layers), bringing the non-MoE budget to ~6 GB and freeing room for KV.

---

## The Harmony chat template

gpt-oss uses a bespoke template — *not* ChatML, *not* Llama 3's format. The minimal Harmony skeleton:

```
<|start|>system
You are GPT-OSS. Reasoning: medium.
<|end|>
<|start|>user
What is 17 × 23?
<|end|>
<|start|>assistant
<|channel|>analysis
17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391.
<|end|>
<|start|>assistant
<|channel|>final
391
<|end|>
```

Three non-obvious things:

1. **`<|channel|>analysis` vs `<|channel|>final`.** Internal scratchpad vs visible answer, similar to R1's `<think>` vs answer split — but each is its own message turn, not nested. The serving stack should hide the `analysis` channel from end users.
2. **`Reasoning: low / medium / high` in the system prompt.** This is the `reasoning_effort` knob from the model card; it actually changes the model's behaviour because it was trained on the labels.
3. **Tool calls** get their own `<|channel|>commentary to=tool.foo<|message|>` form. If you don't speak Harmony, the model can still talk to you but tool-use breaks.

vLLM 0.6+ and SGLang ship native Harmony renderers; HF Transformers added the template in 4.46. Confirm the version before deploying.

---

## reasoning_effort = serving knob, not UX knob

| Setting | Typical extra analysis tokens | Latency multiplier (T=0) |
|---------|-------------------------------:|--------------------------:|
| `low` | 50–200 | 1.0× |
| `medium` (default) | 200–800 | 2.5× |
| `high` | 500–5000 | 6× |

Treat `reasoning_effort` like `enable_thinking` in [[excerpts/qwen-3-thinking-mode]] — a capacity-affecting parameter that the routing layer should set, not the model's UX defaults.

---

## Connections

- [[excerpts/deepseek-mla-compression]] — V3 attacks the *attention KV* memory; MXFP4 attacks the *weight* memory. Different surfaces, same goal: frontier capability under tight memory budgets.
- [[excerpts/moe-param-accounting]] — `117B total / 5.1B active` is the standard MoE accounting; treat the rest the same way.
- [[ch-13]] — TP for the dense attention layer + EP for the experts is a typical deployment topology, even for single-node 120B.

<!-- chapter: ch-20
     phase: benchmarks-production
     title: Production Model Inference Reports — Llama 3 / Qwen 3 / DeepSeek V3+R1 / GPT-OSS / Mixtral / Gemma / Phi
     deps: [ch-13, ch-19]
     sources: [[llama-3-inference]], [[qwen-3-inference]], [[deepseek-v3-inference]], [[deepseek-r1-inference]], [[gpt-oss-inference]], [[mixtral-inference]], [[gemma-inference]], [[phi-inference]]
     figures: figures/model-inference-spec-table.html
-->

# Chapter 20 — Production Model Inference Reports

> **Core insight.** Every open frontier model in 2025 has chosen its inference profile by trading three knobs against each other: **KV-cache geometry** (MHA → GQA → MLA → MQA-banded), **active-vs-total parameters** (dense → MoE-sparse), and **serving dtype** (BF16 → FP8 → MXFP4). The model card tells you which trade-offs were made; the serving stack must match. Mis-reading any of these knobs is the difference between sizing for 8 GB of KV cache per request and 800 MB.
>
> **Guideline.** Before benchmarking a new open model, walk down a fixed checklist: `(active_params, total_params, layers, kv_heads, head_dim, context, dtype, attention_pattern, chat_template, reasoning_mode_default)`. The KV-cache memory formula from [[ch-03]] is parameterised by exactly the fields in that checklist — every other planning number falls out of it.

---

## Why this chapter exists

Earlier chapters built the mechanism vocabulary: KV-cache geometry (ch-03), MoE expert parallelism (ch-13), benchmark methodology (ch-19). This chapter is the *audit pass*: we read the actual deployment specs of the eight open frontier families that dominate 2025–2026 inference workloads and produce a single comparison table you can sanity-check any new release against.

The audit matters because every model card hides at least one footgun:

- Llama 3.1 says "128k context." The KV-cache cost at that context is 30 GB *per request* — almost 40 % of an H100.
- Qwen 3's `enable_thinking=True` is the default; ignoring it doubles your TPOT budget vs the model card's headline.
- DeepSeek V3 looks like a 671B model but only 37B activate per token — MLA cuts the cache to ~5 % of what a dense 671B would need.
- GPT-OSS-120B fits on a single 80 GB GPU only because the MoE weights ship in MXFP4 — load it in BF16 and you OOM by 3×.
- Mixtral's 8×7B name suggests 56B params; the actual checkpoint is 46.7B because attention is shared across experts.

The point of the chapter is to surface every such gotcha *in one place* and end with a comparison table the learner can read like a serving sizing spec sheet.

---

## 1. Llama 3 / 3.1 / 3.2 — the dense GQA baseline

[[llama-3-inference]]. Meta released Llama 3 (8B, 70B) in mid-2024 and Llama 3.1 (8B, 70B, 405B, 128k context) shortly after. Llama 3.2 added 1B/3B edge models and 11B/90B vision variants. For serving, **Llama 3 is the dense-GQA reference target** that every other open model is implicitly compared against.

**Attention geometry.**
- 8B: 32 layers, **32 query heads / 8 KV heads** (GQA-8), head_dim 128 → per-token KV = `2 · 32 · 8 · 128 · 2` bytes (BF16) = **128 KB/token**.
- 70B: 80 layers, 64 query heads / **8 KV heads** (GQA-8 again), head_dim 128 → per-token KV = `2 · 80 · 8 · 128 · 2` = **320 KB/token**.
- 405B: 126 layers, 128 query heads / **8 KV heads** (GQA-16, K replication ratio = 16:1), head_dim 128 → per-token KV = `2 · 126 · 8 · 128 · 2` = **504 KB/token**.

Two key observations:
1. **KV heads = 8 for all sizes.** Meta deliberately fixed the KV-head count across model scales. The serving-side consequence: KV-cache memory scales with *layer count*, not query-head count, as you go from 8B → 70B → 405B. This is exactly the design that GQA was meant to enable for inference economics (see [[ch-02]]).
2. **128 k context implications.** Llama 3.1 supports 128 k. At 70B + BF16, that's `320 KB · 128k = 40 GB` of KV cache per single request. An H100 with 80 GB has ~30 GB free for KV after BF16 weights (140 GB but split across multiple GPUs in TP=4-8). The practical implication: *do not* default to 128 k context window in your `--max-model-len`; size it to your workload's actual length distribution.

**Tokenizer + chat template.** Llama 3 introduced a 128 k vocabulary tokenizer (vs Llama 2's 32 k). The chat template uses `<|begin_of_text|>`, `<|start_header_id|>system<|end_header_id|>`, `<|eot_id|>` — and the EOT vs EOS distinction is the single most common silent bug: the model emits `<|eot_id|>` (id 128009), not `<|end_of_text|>` (128001), so serving stacks must add `128009` to `stop_token_ids` or they will run on past the assistant turn.

**Quantization in production.** All major runtimes (vLLM, SGLang, TensorRT-LLM) ship Marlin-class W4 INT4 kernels for Llama 3. A practical 70B serving deployment runs at W4A16 on 2×H100 with TP=2, giving ~80 ms TTFT and ~30 ms/token at batch 16.

---

## 2. Qwen 3 — dense + MoE under one tokenizer with hybrid thinking

[[qwen-3-inference]]. Alibaba's Qwen 3 family (released April 2025) is unusual in shipping *both* dense and MoE checkpoints under one architecture lineage, plus a *user-controllable* thinking mode that changes inference cost by 2–10×.

**Lineup.**
- Dense: 0.6B, 1.7B, 4B, 8B, 14B, 32B (all GQA, RoPE).
- MoE: **30B-A3B** (3B active of 30B total, 128 experts, 8 active) and **235B-A22B** (22B active of 235B total).
- All variants nominally 32k context; YaRN extension to 128 k available via config.

**Hybrid thinking.** The Qwen 3 chat template has an `enable_thinking` flag (default `True`). When enabled, the model emits a `<think>...</think>` reasoning block before the visible answer. This is *not* a reasoning model in the DeepSeek-R1 sense; it's the same weights generating extra output tokens to think out loud. For serving:

- **Latency.** Thinking output is real generation, so each `<think>` token costs decode time. Expected total output for a multi-step math problem: 200 visible tokens + 800-2000 thinking tokens → 5-10× the headline latency.
- **Memory.** The KV cache holds the thinking tokens for the duration of the request. Plan for it.
- **Disabling.** Pass `chat_template_kwargs={"enable_thinking": False}` (vLLM) or add `/no_think` to the system prompt to suppress.

**MoE serving notes.** Qwen 3 30B-A3B is the most interesting deployment target: only 3B activate per token, but all 30B weights must reside on GPUs. With BF16 that's 60 GB — fits on 1×H100. With expert parallelism (EP=4) across an H100 node, you split the 30B weights and pay all-to-all communication on every layer. EP only beats TP for this size when batch ≥ 16 and the all-to-all is overlapped with attention.

**YaRN context extension.** Qwen 3 ships with a `rope_scaling: {type: "yarn", factor: 4.0}` config block enabling 128k. **Do not just pass `--max-model-len 131072`** — also confirm `rope_scaling` is applied at load time, or the model will silently emit garbage past 32 k tokens.

---

## 3. DeepSeek V3 — 671B MoE + MLA + FP8 in one package

[[deepseek-v3-inference]]. The most architecturally ambitious open release of 2025. DeepSeek V3 is the model that broke the "MoE-doesn't-fit-on-X-GPUs" assumption for frontier-scale serving.

**Headline numbers.**
- 671B total parameters, **37B active per token**.
- 256 routed experts + 1 shared expert; 8 experts active per token.
- 61 layers, MLA (Multi-head Latent Attention) instead of GQA.
- 128 k context, FP8 native training and inference.

**MLA — what it actually compresses.** Standard GQA caches `[K, V]` of shape `(kv_heads · head_dim)` per token per layer. MLA caches a *compressed latent* `c_KV` of shape `(d_c,)` with `d_c ≪ kv_heads · head_dim`, plus a small RoPE-coupled `k_rope` slice. At inference time the K and V tensors are reconstructed from `c_KV` on the fly via two small projections that are absorbed into the WQ and WO matrices.

The serving consequence is dramatic: DeepSeek V3's MLA stores **~70 KB/token** at FP8 — comparable to *Llama-3-8B's* KV cost, despite being an 18× larger model. (For reference, a hypothetical naive-MHA version of V3 would store ~1.5 MB/token, 20× more.) MLA is the single architectural feature that makes a 671B model serveable on 8×H100.

**Native FP8.** V3 was trained directly in FP8 with per-block scaling (1×128 activation tiles, 128×128 weight blocks) and FP32 accumulation. The deployment payload is FP8 weights + per-block scales (~14 bytes per scale, ~1 % overhead). Loading at BF16 *can* be done for accuracy comparison but doubles VRAM and costs accuracy because the model was never trained outside FP8's distribution. **Always serve at FP8 unless you have a measured quality reason not to.**

**Expert parallelism.** Production V3 deployments use TP=8 (intra-node NVLink) × EP=8 or EP=16 (inter-node InfiniBand) for the 256 experts. The all-to-all traffic on every layer is large enough that DeepSeek released DeepEP, a custom NCCL-fused expert-routing kernel, to make this practical. See [[ch-13]] for the parallelism background.

---

## 4. DeepSeek R1 — reasoning model with long CoT generation

[[deepseek-r1-inference]]. R1 is V3 trained for chain-of-thought reasoning. The *architecture* is V3's (MoE + MLA + 671B/37B); the *inference behaviour* is qualitatively different and changes serving sizing.

**The long-CoT pattern.** R1 emits a `<think>...</think>` reasoning trace before the answer. Typical generation lengths:

| Task | Visible answer | Reasoning trace | Total output |
|------|---------------:|----------------:|-------------:|
| Simple Q&A | ~50 tokens | ~200 tokens | ~250 |
| Math word problem | ~100 tokens | 1k–4k tokens | ~1k–4k |
| Code with debug | ~200 tokens | 2k–8k tokens | ~2k–8k |
| Olympiad math | ~150 tokens | 5k–20k tokens | ~5k–20k |

The headline implication for serving: **TPOT × output length, not TTFT, dominates user latency for R1.** An 8k-token CoT at 30 ms/token = 4 minutes of decode. The user experiences this as "the model is thinking"; the operator experiences it as 4 minutes of KV-cache residency per request.

**Distilled variants.** R1 ships with distilled dense versions: R1-Distill-Llama-8B, R1-Distill-Qwen-7B, R1-Distill-Llama-70B, etc. These transplant the long-CoT *behaviour* onto dense backbones, so you get the latency profile without the MoE serving complexity. R1-Distill-Llama-70B at 2×H100 + W4 is a reasonable cost-conscious deployment of R1-style reasoning.

**Serving knobs for reasoning.**
- `max_tokens` must be set to at least 8k for hard problems; 4k clips most chains.
- `temperature` is recommended at 0.6 (not 1.0) per the model card to reduce CoT divergence.
- The `<think>` block should not be returned to the end user verbatim in most chat applications — strip it before streaming, but keep it server-side for audit.

---

## 5. GPT-OSS — MXFP4 MoE designed for memory budgets

[[gpt-oss-inference]]. OpenAI's August 2025 open release. Two sizes designed explicitly around GPU memory tiers.

| | gpt-oss-20b | gpt-oss-120b |
|---|---|---|
| Total params | 21 B | 117 B |
| Active per token | 3.6 B | 5.1 B |
| Experts | 32 (4 active) | 128 (4 active) |
| Context | 128 k | 128 k |
| Attention | grouped MQA + alternating dense/banded local | same |
| Headline target | 16 GB GPU | single 80 GB GPU |

**MXFP4 weights for MoE.** The 120B model ships with MoE expert weights pre-quantized in **MXFP4** (4-bit floating-point with shared 8-bit exponent per 32-element block). This is what lets a 117B-parameter model fit in 80 GB:
- 117B params × 0.5 bytes (MXFP4) ≈ 58 GB weights.
- ~5 GB MXFP4 scales overhead.
- ~10–15 GB for KV cache + activations + workspace.
- Total ≈ 75 GB, fits a single H100.

In BF16 the same weights would be 234 GB — requires 3×H100. MXFP4 is not optional for 120B; it is the deployment format.

**Harmony format.** GPT-OSS uses a non-standard chat template called the *Harmony response format*: messages get explicit `<|start|>role<|message|>...<|end|>` markers plus structured tool-call regions. The serving stack must apply the Harmony template, not the generic ChatML, or the model emits garbled output. vLLM and SGLang both ship native Harmony support; HF Transformers needs `apply_chat_template(messages, ..., template="harmony")`.

**Reasoning effort.** GPT-OSS exposes a `reasoning_effort ∈ {low, medium, high}` knob in the system prompt. It controls the length of the model's internal scratchpad. Default is `medium`; `low` cuts output length ~3×, `high` extends it ~2×. Treat this as a *serving* parameter (it gates capacity), not just a UX preference.

---

## 6. Mixtral 8×7B / 8×22B — the MoE that made sparse routing mainstream

[[mixtral-inference]]. Mistral's December 2023 release was the first open MoE that was actually deployable. Two sizes:

| | Mixtral 8×7B | Mixtral 8×22B |
|---|---|---|
| Total params | 46.7 B | 141 B |
| Active per token | 12.9 B | 39 B |
| Experts | 8 (2 active) | 8 (2 active) |
| Layers | 32 | 56 |
| KV heads | 8 (GQA) | 8 (GQA) |
| Context | 32 k (sliding 4k) | 64 k |

**Why "8×7B" is not 56B.** The `8×` refers only to the FFN expert weights. Attention (QKV, O) is shared across experts. Mistral's accounting: shared attention + embeddings + LayerNorm = ~13 B params; per-expert FFN = ~4.2 B. So the total is `13 + 8 × 4.2 ≈ 46.7 B`, not 56 B. Tokens still route to 2 of 8 experts per layer, so active params = `13 + 2 × 4.2 ≈ 12.9 B`. Read every MoE param count this way.

**Sliding window attention** (Mixtral 8×7B specifically). The first 32 layers use sliding 4096-token attention; only the last layers are full. For inference: the KV cache for sliding layers can be capped at 4096 entries per layer, saving memory at long context. Most serving stacks apply this automatically — but if you fork Mixtral support, the cache-trimming logic is the easy bug.

**Serving pattern.** Mixtral 8×22B at BF16 needs ~280 GB weights → TP=4 minimum on 4×H100 (or 8×A100). Expert parallelism is not strictly required (only 8 experts) but cuts inter-GPU all-reduce vs pure TP.

---

## 7. Gemma 2/3 — dense edge models with sliding+full attention layering

[[gemma-inference]]. Google's open-weight family.

**Gemma 2** (2024): 2B, 9B, 27B dense. GQA, alternating local (4k window) + global attention every other layer. 8 k base context.

**Gemma 3** (2025): 1B, 4B, 12B, 27B with native multimodality (image+text), 128 k context for the 27B variant via combined sliding+global. The interesting deployment fact: **5:1 local-to-global layer ratio.** Of every 6 layers, 5 use 1024-token sliding window attention and 1 uses full attention. KV-cache cost is therefore dominated by the global layers — for 27B that's ~16 % of layers caching long-range state, the rest cap at 1024 entries.

**Edge focus.** Gemma 3-1B/4B target single-GPU and on-device inference. llama.cpp + GGUF Q4_K_M is the canonical local-deployment path; vLLM/SGLang also support all sizes server-side.

---

## 8. Phi 3 / 3.5 / 4 — small-model serving as a first-class target

[[phi-inference]]. Microsoft's family. Phi-3-mini (3.8B), Phi-3-small (7B), Phi-3-medium (14B), Phi-3.5-mini, Phi-4 (14B).

**Architecturally boring on purpose.** Dense, GQA (Phi-3-medium uses 10/40 KV/Q heads), RoPE, 4 k base context with 128 k variants via LongRoPE. No MoE, no MLA, no exotic attention. The point is *high quality at small size*, so the inference profile is "dense GQA model that fits on a phone." On an RTX 4090, Phi-4-14B at W4 runs at ~80 tokens/sec single-batch — typical edge/desktop deployment.

**ONNX / DirectML route.** Microsoft ships native ONNX checkpoints for Windows AI Toolkit. For pure server deployments, vLLM and SGLang both have native support; nothing exotic.

---

## 9. The comparison table

This table is the deliverable a serving engineer references when sizing a new deployment. Numbers are at BF16 unless noted, single-request KV cost.

| Model | Active / Total | Layers | KV heads | head_dim | KV bytes/token | Context | Attention | Native dtype | Notable serving knob |
|-------|---------------:|-------:|---------:|---------:|---------------:|--------:|-----------|-------------|----------------------|
| **Llama-3-8B** | 8B / 8B | 32 | 8 | 128 | 131 KB | 8k / 128k (3.1) | GQA-8 | BF16 | EOT id 128009 in stop list |
| **Llama-3-70B** | 70B / 70B | 80 | 8 | 128 | 327 KB | 8k / 128k (3.1) | GQA-8 | BF16 | TP=2-4 typical; W4 Marlin |
| **Llama-3.1-405B** | 405B / 405B | 126 | 8 | 128 | 516 KB | 128k | GQA-16 | BF16 / FP8 | TP=8 + W8 or FP8 for 1-node |
| **Qwen-3-8B** | 8B / 8B | 36 | 8 | 128 | 147 KB | 32k / 128k YaRN | GQA-8 | BF16 | `enable_thinking` default ON |
| **Qwen-3-30B-A3B** | 3B / 30B | 48 | 4 | 128 | 98 KB | 32k / 128k YaRN | GQA + MoE | BF16 | 128 experts, 8 active |
| **Qwen-3-235B-A22B** | 22B / 235B | 94 | 8 | 128 | 385 KB | 32k / 128k | GQA + MoE | BF16 | EP across nodes |
| **DeepSeek-V3** | 37B / 671B | 61 | n/a (MLA) | latent 512 | ~70 KB (FP8) | 128k | MLA + MoE | **FP8 native** | 256 experts; DeepEP all-to-all |
| **DeepSeek-R1** | 37B / 671B | 61 | n/a (MLA) | latent 512 | ~70 KB (FP8) | 128k | MLA + MoE | FP8 | `max_tokens ≥ 8k`, T=0.6 |
| **GPT-OSS-20B** | 3.6B / 21B | 24 | 8 | 64 | 49 KB | 128k | GQA + banded MoE | MXFP4 (MoE) | Harmony chat template |
| **GPT-OSS-120B** | 5.1B / 117B | 36 | 8 | 64 | 74 KB | 128k | GQA + banded MoE | **MXFP4 (MoE)** | Single-H100 deployment |
| **Mixtral-8×7B** | 12.9B / 46.7B | 32 | 8 | 128 | 131 KB | 32k (4k sliding) | GQA + MoE | BF16 | 2 of 8 experts per token |
| **Mixtral-8×22B** | 39B / 141B | 56 | 8 | 128 | 229 KB | 64k | GQA + MoE | BF16 | TP=4 minimum |
| **Gemma-2-27B** | 27B / 27B | 46 | 16 | 128 | 377 KB | 8k | GQA + 50% local 4k | BF16 | local layers cap KV |
| **Gemma-3-27B** | 27B / 27B | 62 | 16 | 128 | 508 KB | 128k | GQA + 5:1 local:global | BF16 | only ~16% of layers full-context |
| **Phi-4-14B** | 14B / 14B | 40 | 10 | 128 | 205 KB | 16k | GQA-4 | BF16 | ONNX / DirectML on Windows |

See `figures/model-inference-spec-table.html` for an interactive version with KV-memory-at-context-length sliders.

---

## 10. Three sizing exercises against the table

**Exercise A — "I want to serve Llama-3-70B on 1×H100 (80 GB)."**
Weights at BF16 = 140 GB → does not fit. At W4 Marlin = 35 GB. KV cost at 8k = 327 KB · 8192 = 2.7 GB/request. Budget 35 GB weights + 5 GB activations/workspace = 40 GB used; 40 GB free for KV. At 2.7 GB/request you get ~15 concurrent requests at 8k context. Tighten context to 4k and you double to 30.

**Exercise B — "I want to serve DeepSeek V3 on 8×H100."**
8 × 80 = 640 GB. FP8 weights = 671 GB → does NOT fit naively. With expert sharding (EP=8, each GPU holds 1/8 of experts) + the shared/dense parts, ~80 GB/GPU. KV at MLA = 70 KB · 128k = 9 GB per max-context request, replicated nowhere thanks to MLA. Concurrent 128 k requests: ~5–8. This is why frontier MoE serving uses 16-GPU or larger pools.

**Exercise C — "I want to serve GPT-OSS-120B on 1×H100."**
117B × 0.5 = 58.5 GB weights + 5 GB scales + 10 GB activations = ~75 GB. KV at 74 KB × 128k = 9.5 GB per max-context request — does not fit a second one. Realistic deployment: cap `--max-model-len 16384`, KV per request drops to 1.2 GB, get ~6–8 concurrent requests. This matches OpenAI's own published serving guidance.

---

## Connections and what's next

- **Back to [[ch-03]]** — every "KV bytes/token" number in the table comes from the formula `2 · layers · kv_heads · head_dim · dtype_bytes`. The exception is MLA (DeepSeek V3/R1), which replaces the formula's `kv_heads · head_dim` factor with the latent dimension `d_c`.
- **Back to [[ch-13]]** — TP/PP/EP choices for the MoE models. Mixtral, Qwen-3-MoE, DeepSeek-V3, and GPT-OSS each pick different combinations.
- **Back to [[ch-19]]** — the metrics framework (TTFT/TPOT/ITL/goodput) for benchmarking the models in this table. The R1 long-CoT pattern is the canonical case where TPOT × output-length dominates user latency.
- **Forward to [[ch-21]]** — the lab puts these models on a benchmark harness and runs vLLM vs SGLang head-to-head with Llama-3-8B as the workhorse.
- **Forward to [[ch-22]]** — the capstone asks you to reproduce a paper that *would* change one row of this table.

## Further reading

- [[llama-3-inference]], [[qwen-3-inference]], [[deepseek-v3-inference]], [[deepseek-r1-inference]], [[gpt-oss-inference]], [[mixtral-inference]], [[gemma-inference]], [[phi-inference]] — the per-model excerpts referenced throughout.
- HuggingFace model cards for each model — always the ultimate source for the chat template + tokenizer.

## Companion visualization

**[figures/model-inference-spec-table.html](figures/model-inference-spec-table.html)** — sortable / filterable spec sheet with a context-length slider that recomputes per-request KV-cache cost live for each model in the table.

## Excerpts

- [[excerpts/llama-3-gqa-and-kv]] — Llama 3 GQA-8 design + the EOT-token serving gotcha.
- [[excerpts/deepseek-mla-compression]] — what MLA actually compresses, with the latent-dim arithmetic.
- [[excerpts/qwen-3-thinking-mode]] — the `enable_thinking` knob and how it changes TPOT × output-length.
- [[excerpts/gpt-oss-mxfp4-deployment]] — MXFP4 MoE weight format + the single-H100 memory budget.
- [[excerpts/moe-param-accounting]] — `8×7B ≠ 56B`: how to read MoE parameter counts across Mixtral / Qwen-3 / DeepSeek / GPT-OSS.

<!-- scope: Groq LPU inference hardware and quantization story
     deps: [[int8]], [[fp8-e4m3]]
     see-also: [[nvidia-h100-fp8]]
-->

# Groq — LPU Inference and Quantization Disclosures
- **Core Insight:** Groq's LPU (Language Processing Unit) is a deterministic compile-time-scheduled SRAM-resident inference accelerator; quantization is treated as a deployment-time compile target rather than a runtime decision, with the compiler statically scheduling INT8 and FP8 paths to maximize the on-chip SRAM bandwidth.
- **Guideline:** Treat Groq's quant story as architecturally distinct from GPU stacks — there's no dynamic quantization at runtime, the compiler bakes in the bit-width per layer; benchmark only on the production model you intend to serve.
- **Authors:** Groq (Jonathan Ross + team)
- **Year:** 2023-2024 (public benchmarks and disclosures)
- **URL:** https://groq.com/blog/ ; https://wow.groq.com/
- **Relevant topics:** LPU, deterministic scheduling, SRAM-resident weights, FP8, INT8

## Summary
Groq's LPU is a fundamentally different inference architecture: instead of HBM + tensor cores like an H100, the chip is dominated by 230 MB of SRAM with a deterministic dataflow compiler. There is no caching, no speculation, no dynamic scheduling — the compiler statically assigns each tensor to a SRAM bank and schedules every instruction at compile time. Quantization fits this architecture as a compile-time choice: the user picks INT8 or FP8 per layer, the compiler generates the corresponding microcode. Groq has published benchmarks showing >500 tokens/sec on Llama-3-70B and >300 tokens/sec on Mixtral 8x22B at INT8 — numbers that GPU stacks reach only with multiple H100s and batch>32. The quantization story is less algorithm-rich than GPU frameworks (no SmoothQuant / GPTQ / AWQ tooling published) — Groq seems to use per-tensor symmetric INT8 with simple absmax calibration.

## Key Points
- LPU architecture: 230 MB SRAM, deterministic compile-time scheduling.
- No KV cache hierarchy — everything lives in SRAM during inference.
- Quantization is a compile-time per-layer choice.
- Published benchmarks: Llama-3-70B at 500+ tok/sec, Mixtral at 300+ tok/sec.
- Quantization algorithm: per-tensor symmetric INT8 (most likely) or FP8; not disclosed in detail.
- Batch=1 latency is the headline metric; throughput at batch>1 less competitive vs GPU.

## Technical Details

### Hardware (LPU v1)
- 230 MB on-die SRAM (vs ~50 MB L2 on H100).
- 750+ TOPS INT8 per chip.
- Deterministic dataflow: every cycle is predicted at compile time.
- Multiple chips connected via Groq's deterministic networking.

### Why deterministic matters for quantization
- No cache misses → no jitter → predictable latency.
- The compiler knows exactly when each tensor will be in which SRAM bank.
- Quantized data fits more compactly in SRAM; INT8 doubles the effective capacity vs FP16.

### Inference flow
1. User submits a compiled model package (Groq's compiler output).
2. The compiler has pre-decided: which layers are INT8, which are FP8, where each tensor lives.
3. Runtime is just data streaming through the pre-scheduled program — no decisions.

### Published benchmark numbers (Groq blog)
| Model | Format | Throughput (tok/sec/user) |
|-------|--------|---------------------------|
| Llama-3-70B | INT8 | 500-800 |
| Llama-3-8B | INT8 | 1200+ |
| Mixtral 8x22B | INT8 | 300+ |
| Llama-2-70B | INT8 | 250+ |

### What Groq does not publish
- The exact PTQ algorithm (whether they use SmoothQuant, RTN, or something internal).
- Accuracy drop from INT8 vs FP16 for each model.
- The compiler stack publicly (closed-source).

### Speculative inference details (community reverse-engineering)
- Weights stored INT8, on-chip SRAM.
- Activations likely INT8 with per-tensor scale.
- KV cache resident in SRAM; no spill to DRAM.
- Multi-chip deployments split layers across LPUs deterministically.

### Caveats
- Per-user batch=1 latency is the marketing metric; at high batch GPU stacks compete well.
- LPU economics differ: SRAM-heavy chip = lower memory-bandwidth-per-dollar but much higher hit rate.
- Not available as commodity hardware — Groq runs its own cloud and licenses the chips selectively.

### Why this matters for the quant course
- Demonstrates that quantization can be a hardware-architecture-level concern, not just an algorithm.
- The dataflow / compile-time scheduling story is unique among LLM-serving stacks.
- A useful counterpoint to GPU-centric quant narratives (GPTQ / AWQ / Marlin).

## Connections
- [[nvidia-h100-fp8]] — GPU-centric alternative architecture for comparison.
- [[int8]] / [[fp8-e4m3]] — formats Groq compiles into.
- [[character-ai-quant-deployment]] — adjacent production-cost write-up from a different angle.

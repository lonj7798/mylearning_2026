<!-- scope: Intel Advanced Matrix Extensions (AMX) INT8/BF16 tile instructions
     deps: [[int8]], [[bf16]]
     see-also: [[intel-quantization]]
-->

# Intel AMX — Advanced Matrix Extensions (Sapphire Rapids and later)
- **Core Insight:** Intel introduced a CPU-side tile-matrix coprocessor with the Sapphire Rapids Xeon (4th-gen Scalable, 2023) that gives x86 servers a native INT8 / BF16 matmul instruction (`TMUL`) capable of competing with mid-tier GPUs on small-batch LLM inference.
- **Guideline:** Use `oneDNN` / `IPEX` to dispatch to AMX automatically — never target the intrinsics directly; quantize weights to INT8 with per-channel scales for the best CPU INT8 throughput.
- **Authors:** Intel Architecture team (Sapphire Rapids whitepaper + Intel ISA Reference)
- **Year:** 2023 (Sapphire Rapids), extended 2024 (Emerald Rapids), 2025 (Granite Rapids)
- **URL:** https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html
- **Relevant topics:** AMX, TMUL, INT8 CPU inference, BF16, oneDNN

## Summary
AMX adds a new architectural state (eight 1KB tile registers, `TMM0`–`TMM7`) and a `TMUL` matrix-multiply instruction that performs a tile-level INT8 or BF16 matmul per cycle. The tile registers hold a `TILECFG`-defined 2D matrix up to 16 rows × 64 bytes; `TMUL` then computes `C += A × B` where A is M×K BF16/INT8, B is K×N BF16/INT8, and C is M×N FP32/INT32 accumulated. On Sapphire Rapids, one core can sustain ~2,048 INT8 ops/cycle (vs ~512 with AVX-512 VNNI), delivering ~410 GOPS/core at 2 GHz. The Granite Rapids generation (2025) extends AMX to FP16, complex tile shapes, and per-tile data-type selection. AMX is the reason `llama.cpp` and `vLLM-CPU` builds for Xeon SPR can hit double-digit tokens/sec on 70B-class quantized models.

## Key Points
- Tile registers: 8 × 1KB (`TMM0`–`TMM7`); configured by `LDTILECFG`.
- `TMUL` instruction: one tile matmul per cycle (BF16 or INT8 element type).
- INT8 throughput per core (SPR @ 2 GHz): ~410 GOPS dense.
- Granite Rapids (2025): adds FP16 tile type, complex shapes.
- Software entry points: oneDNN, Intel Extension for PyTorch (IPEX), Intel Neural Compressor.

## Technical Details

### Hardware feature names
- ISA extensions: `AMX-TILE` (config), `AMX-INT8` (INT8 TMUL), `AMX-BF16` (BF16 TMUL).
- Granite Rapids additions: `AMX-FP16`, `AMX-COMPLEX`.
- Detect via `CPUID.(EAX=7,ECX=0)`: bits 24/25 of EDX = AMX-TILE/AMX-INT8.

### Key instructions
| Mnemonic | Operation |
|----------|-----------|
| `LDTILECFG` | load tile-configuration descriptor (rows, cols per tile) |
| `TILELOADD` | load tile from memory (strided) |
| `TILESTORED` | store tile to memory |
| `TDPBSSD` / `TDPBUSD` | tile dot-product, INT8 signed/unsigned, accumulate INT32 |
| `TDPBF16PS` | tile dot-product, BF16, accumulate FP32 |
| `TILEZERO` | zero a tile |
| `TILERELEASE` | release AMX state |

### Tile shape constraints
- Up to 16 rows per tile.
- Up to 64 bytes per row (= 16 INT32 / 32 BF16 / 64 INT8 elements).
- A and B tiles must agree on K (inner) dimension: K = 32 for INT8, K = 16 for BF16.

### Throughput (Sapphire Rapids, per core)
| Op | Throughput (GOPS @ 2 GHz) |
|----|---------------------------|
| AVX-512 INT8 (VNNI) | ~128 |
| AVX-512 BF16 | ~64 |
| AMX-INT8 | ~410 |
| AMX-BF16 | ~205 |

### Software stack
- `oneDNN` primitives `inner_product` and `matmul` dispatch to AMX when the input dtype is `int8` or `bf16`.
- `intel-extension-for-pytorch` (IPEX) installs `torch.compile` backends that lower `nn.Linear` to AMX.
- `Intel Neural Compressor` provides INT8 PTQ flows specifically targeted at AMX deployment.

## Connections
- [[int8]] — operand format for `TDPBSSD`.
- [[bf16]] — operand format for `TDPBF16PS`.
- [[intel-quantization]] — Intel Neural Compressor team that productionizes AMX quant flows.
- [[nvidia-h100-fp8]] — GPU competitor whose FP8 tensor cores AMX competes with on small-batch inference.

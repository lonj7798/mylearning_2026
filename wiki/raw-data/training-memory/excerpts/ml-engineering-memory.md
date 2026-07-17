# ML Engineering — Memory Usage Anatomy
<!-- slug: ml-engineering-memory · type: doc · source: https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md -->

**Core Insight.** Training with AdamW in mixed precision costs a minimum of 18 bytes per parameter (6 for weights + 8 for optimizer + 4 for gradients) before activations — and activation memory is the component that explodes non-linearly with sequence length and can dwarf the static 18-byte floor.

**Guideline.** Use "18 bytes/param + activation memory + temp memory" as the pre-flight estimate for any full fine-tune run. If activation memory at your target seq-len exceeds the 18-byte floor, gradient checkpointing is not optional.

## Technical Details

- **Six named memory residents during training:**
  1. Model weights
  2. Optimizer states
  3. Gradients
  4. Forward activations (for grad computation)
  5. Temporary buffers (freed per-op)
  6. Functionality-specific overhead (model-dependent)
- **Weight bytes by training mode:**
  - FP32 only: 4 B/param
  - Mixed precision: 6 B/param (2 B bf16 working + 4 B fp32 master)
- **Optimizer bytes (per param):**
  - Standard AdamW (fp32 states): 8 B
  - BF16 AdamW: 4 B
  - SGD w/ momentum / LION / Adafactor: 4 B
  - 8-bit quantized (bitsandbytes): 2 B
- **Gradient bytes:** 4 B/param (fp32 or mixed-half); 2 B if non-mixed fp16.
- **Static floor identity:** 6 + 8 + 4 = **18 B/param** for standard mixed-precision AdamW.
- **Concrete activation example:** Llama-3-8B, batch=1, seq=32,768 — activation memory with checkpointing: ~31 GB; without checkpointing: ~240 GB. The 8× gap is entirely from the `seq²` attention term.
- **PyTorch CUDA init tax:** "When PyTorch uses CUDA for the first time, it may use up 0.5–2 GB of GPU memory" before any model is loaded; subtract from usable capacity.
- **Training-memory angle:** The canonical practitioner checklist for GPU memory accounting. The 18 B/param floor is the figure most engineers cite when estimating whether a model fits; the Llama-3-8B activation example concretizes why checkpointing becomes mandatory at long contexts even with a single batch.

## Citation
Stas Bekman, "Machine Learning Engineering Open Book," stas00/ml-engineering, ongoing.
https://github.com/stas00/ml-engineering

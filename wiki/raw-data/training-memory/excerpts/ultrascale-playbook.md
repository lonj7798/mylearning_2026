# The Ultra-Scale Playbook
<!-- slug: ultrascale-playbook · type: doc · source: https://nanotron-ultrascale-playbook.static.hf.space/ -->

**Core Insight.** The 4-component training-memory ledger (weights + gradients + optimizer states + activations) has a hard static floor of 16 N bytes for mixed-precision AdamW, but activations become "by far the largest memory burden" once batch size or sequence length grows — and selective recomputation breaks that wall at only 2.7% extra compute for a 70% activation saving.

**Guideline.** Use the activation formula `m_act = L·seq·bs·h·(34 + 5·n_heads·seq/h)` to estimate activation memory before choosing a recomputation strategy. Default to selective recomputation (discard attention matrices, keep MLP activations); reserve full recomputation for extreme memory pressure (adds 30–40% compute).

## Technical Details

- **Static floor (mixed precision, N parameters):**
  - BF16 working weights: 2N bytes
  - FP32 master copy: 4N bytes
  - BF16 gradients: 2N bytes
  - Adam fp32 optimizer (momentum + variance): 8N bytes
  - **Total: 16N bytes** ("or 20N with FP32 gradient accumulation")
- **Scale reality check:** "As soon as we reach 7B (!), weights and optimizer requirements already starts to add up significantly and exceed the size of a typical GPU memory, e.g. 80GB for a H100 GPU."
- **Activation memory formula:** `m_act = L · seq · bs · h · (34 + 5·n_heads·seq/h)` bytes
  - Scales **linearly** with batch size; **quadratically** with seq length (the `seq²` attention term).
- **Full recomputation:** stores only layer-boundary activations; recomputes everything inside each layer. Cost: "up to 30–40%" extra training time.
- **Selective recomputation:** discards attention-score matrices (cheap to recompute because FlashAttention fuses them); keeps MLP and LayerNorm activations. Result for GPT-3 175B: "70% activation memory reduction at a 2.7% compute cost."
- **Peak memory warning:** "The first training step shows different memory patterns than subsequent steps" — optimizer states materialize only after step 1; OOM can appear on step 2 even if step 1 succeeds.
- **Training-memory angle:** Establishes the canonical four-bucket anatomy used by nanotron and is the source most modern playbooks reference for the 16 N floor and the quadratic seq-length activation formula. Directly drives chapter decisions on recomputation strategy and batch sizing.

## Citation
Guilherme Penedo et al. (HuggingFace / nanotron team), "The Ultra-Scale Playbook: Training LLMs on GPU Clusters," 2025.
https://nanotron-ultrascale-playbook.static.hf.space/ · https://huggingface.co/spaces/nanotron/ultrascale-playbook

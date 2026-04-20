<!-- scope: recurrent-depth transformer with MLA/GQA and sparse MoE
     deps: [[ch-05]]
     see-also: [[raschka-attention-variants]], [[hf-mixture-of-experts]]
-->

# OpenMythos -- Recurrent-Depth Transformer Implementation

- **Core Insight:** Recurrent-Depth Transformer combines switchable MLA/GQA attention with sparse MoE.
- **Guideline:** Study this as an example of how cutting-edge techniques are combined in new architectures.

- **URL:** https://github.com/kyegomez/OpenMythos
- **Type:** repo (open-source model architecture)
- **Relevant chapters:** transformer architecture, recurrent transformers, mixture of experts, attention variants (MLA, GQA), implicit chain-of-thought

## Content

### Project Overview

OpenMythos is an independent, community-driven theoretical implementation of a **Recurrent-Depth Transformer (RDT)** architecture. It reconstructs a looped transformer with three processing stages, implementing implicit chain-of-thought reasoning through continuous latent space updates rather than explicit token outputs.

- **Author:** Kye Gomez
- **License:** MIT (2026)
- **Install:** `pip install open-mythos`

### Core Architecture: Three-Stage Processing

The model divides processing into three sequential blocks:

**1. Prelude**
- Standard transformer layers executed once
- Initial feature extraction and embedding transformation

**2. Recurrent Block**
- Looped transformer layers (up to `max_loop_iters`)
- Same weights executed multiple times per forward pass
- Hidden state update rule: `h_{t+1} = A * h_t + B * e + Transformer(h_t, e)`
- Enables "deeper thinking" through iterative refinement

**3. Coda**
- Final transformer layers executed once
- Produces output logits

### Key Features

**Switchable Attention**
- Multi-head Latent Attention (MLA) -- compressed latent key/value projections
- Grouped Query Attention (GQA) -- shared key/value heads across query groups
- Configurable per model via `MythosConfig`

**Sparse Mixture of Experts (MoE)**
- Routed experts: tokens dispatched to top-k experts via gating
- Shared experts: all tokens pass through common expert layers
- Compute-adaptive reasoning -- different inputs use different capacity

**Recurrent Architecture**
- Weight sharing across loop iterations (parameter efficient)
- Variable depth at inference time via `n_loops` parameter
- Depth extrapolation: train with fewer loops, inference with more

### Pre-configured Model Scales

Models range from 1B to 1T parameters with pre-defined configurations.

### Usage Example

```python
from open_mythos.main import OpenMythos, MythosConfig

cfg = MythosConfig(
    vocab_size=1000,
    dim=256,
    n_heads=8
)
model = OpenMythos(cfg)

ids = torch.randint(0, cfg.vocab_size, (2, 16))
logits = model(ids, n_loops=4)
```

### Training

- Supports single and multi-GPU setups using PyTorch DDP
- 3B model trains on FineWeb-Edu dataset
- Optimizer: Muon for weight matrices, AdamW for embeddings
- Training scripts included in the repository

### Central Hypothesis

The architecture implements **implicit chain-of-thought reasoning** through continuous latent space updates rather than explicit token outputs. By looping through the recurrent block multiple times, the model performs iterative refinement in latent space, enabling:

- Systematic generalization
- Depth extrapolation (using more loops at inference than training)
- Adaptive compute per input (analogous to "thinking longer" on harder problems)

## Why This Is Useful

OpenMythos demonstrates several frontier ideas in LLM architecture that go beyond the standard decoder-only transformer:

1. **Recurrent depth** -- The three-stage (Prelude/Recurrent/Coda) design with weight-shared looping is a concrete implementation of the "universal transformer" concept. It shows how to get deeper effective computation without deeper (more parameters) models.

2. **MLA attention** -- Multi-head Latent Attention (from DeepSeek-V2) is one of the most important recent innovations in attention mechanism design. Seeing it implemented alongside GQA in a switchable configuration is instructive.

3. **Sparse MoE integration** -- The combination of routed + shared experts with recurrent depth shows how mixture-of-experts can be composed with other architectural innovations.

4. **Implicit reasoning** -- The latent-space iterative refinement approach is the architectural foundation behind "thinking" models (like o1-style reasoning), but implemented through architecture rather than prompting. This is directly relevant to understanding how reasoning capabilities emerge from model design.

5. **Practical code** -- The repository provides clean, runnable PyTorch code with configurable scales from 1B to 1T, making it suitable for studying architecture at different scales.

For an LLM architecture course, OpenMythos is a valuable case study in how multiple architectural innovations (recurrence, MLA, MoE, implicit reasoning) can be composed into a single coherent design.

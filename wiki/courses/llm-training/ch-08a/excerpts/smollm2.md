---
chapter: ch-08a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2502.02737v1 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2502.02737
created_at: "2026-09-15"
---

# Excerpt: SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model

**Report:** Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, et al. (Hugging Face). arXiv v1 2025-02-04 read. Source type: official technical report. This excerpt covers token budgets and pre-training schedule only.

## Token budgets (Abstract, §4, §6)
- SmolLM2 1.7B is trained on ~11 trillion tokens with a multi-stage mixture of web text, math, code, and instruction data; the abstract uses the word "overtrain" (Abstract).
- 11T tokens is "approximately two epochs on our collected datasets" (§4).
- SmolLM2-360M is trained on 4T tokens and SmolLM2-135M on 2T tokens; data ablations were re-run at the target training length for these sizes (§6).
- Stated rationale: longer training "deviates from the Chinchilla-optimal guidelines", but "the resulting performance gains and reduced inference costs make extended training a worthwhile trade-off", citing de Vries (2023) (§4).
- Examples cited by the report (secondary claims about other models): Qwen2-1.5B on 7T tokens, Qwen2.5-1.5B on 18T, Llama3.2-1B distilled on 9T (§4).

## Schedule and setup for 1.7B (§4.1, App. A Table 6)
- AdamW (β1, β2) = (0.9, 0.95); Warmup-Stable-Decay schedule chosen "to avoid setting a fixed training duration"; 2,000 warmup steps; peak LR 5.0 × 10^−4; decay to zero over 10% of total training steps (§4.1).
- 256 H100 GPUs, nanotron framework; vocabulary 49,152 (§4.1).
- 24 layers, model dimension 2,048, FFN dimension 8,192, 32 attention heads, sequence length 2,048 before context extension, 2M tokens per batch, tied embeddings, RoPE θ = 10,000 (App. A Table 6).
- Stages by tokens: 0–6T, 6–8T, 8–10T (stable phase), 10–11T (decay) (Table 8). StarCoderData limited to 10% of the stage-1 mixture to give about 4 epochs of that code source over 11T tokens (§4.2).

## Verification
- Read on 2026-09-15 against arXiv:2502.02737v1 PDF text (Abstract, §4–§4.2, §6, App. A, App. E.1).
- Not reported by the source: a scaling-law analysis of the token budget; post-training plasticity measurements as a function of pre-training tokens.

---
chapter: ch-12
course: model-quantization
phase: read
excerpt_of: "LLM-QAT: Data-Free Quantization Aware Training for Large Language Models"
source_url: https://arxiv.org/abs/2305.17888
created_at: "2026-05-21"
---

# Excerpt: LLM-QAT — data-free QAT via teacher self-generation

**Authors:** Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, Vikas Chandra
**Year:** 2023
**URL:** https://arxiv.org/abs/2305.17888
**Raw-data source:** [[raw-data/llm-qat]]

---

## The premise

Below 4 bits PTQ saturates and you need QAT — but LLM pretraining data (Common Crawl / Pile / curated mixes) is mostly unavailable or licensed-restricted. The workaround: let the FP teacher **generate its own calibration corpus** by sampling from itself, then distill into the quantized student with full-distribution KL.

This is the recipe that brings **KV-cache quantization** (W4A8KV4, W4A4KV4) into the same QAT loop — configurations that pure PTQ cannot deliver.

---

## Data-free calibration corpus

Sample ~100k sequences from the FP teacher:
- Sample temperature T = 1.0.
- First 3–5 tokens from a vocab-random or BPE-balanced prompt; rest free-generated.
- Length ≈ 1024 tokens per sequence.
- **No external dataset required** — no Pile, no C4, no instruction sets.

---

## Distillation loss

Per token position `t`, the student matches the teacher's **full** output distribution:

```math
\mathcal{L}_{\text{distill}} = -\sum_t \sum_v p_T(v\,|\,x_{<t}) \cdot \log p_S^{\text{quant}}(v\,|\,x_{<t})
                            = \sum_t \mathrm{KL}\big(p_T(\cdot\,|\,x_{<t})\ \|\ p_S^{\text{quant}}(\cdot\,|\,x_{<t})\big) + H(p_T)
```

- `p_T` from the FP teacher; `p_S^{quant}` from the simulated-quantized student.
- **Full vocab** → richer signal than one-hot CE.
- Crucial at very low bits: matching just the argmax loses the teacher's uncertainty information.

---

## Quantization (simulated during QAT)

- **Weights:** per-channel symmetric (per-row), b ∈ {4, 3, 2} bits, STE through `round`.
- **Activations:** per-token dynamic, b ∈ {8, 6, 4} bits, STE.
- **KV cache:** per-token (K), per-channel (V) — separate scales per layer. `b_KV ∈ {8, 4}`.

All quantization is **simulated**: forward uses `dequant(quant(·))`; backward uses STE ([[straight-through-estimator]]).

---

## Training schedule

- One epoch over the 100k self-generated sequences.
- AdamW, lr ≈ 1e-5 (small — the student's job is repair, not learn).
- 32–256 GPUs depending on scale; LLaMA-30B fits with FSDP.

---

## Tested configurations (Table 4 results)

| Config | Weight bits | Activation bits | KV bits |
|---|---|---|---|
| W4A8 | 4 | 8 | 16 |
| W4A6 | 4 | 6 | 16 |
| W4A4 | 4 | 4 | 16 |
| **W4A8KV4** | 4 | 8 | **4** |

At W4A6 and W4A4, LLM-QAT beats GPTQ + SmoothQuant by large margins (PTQ cannot reach these activation precisions at acceptable quality).

W4A8KV4 is the configuration that production inference stacks (e.g. **QServe**, see [[ch-14]]) eventually adopt for serving — LLM-QAT proved the quality target is reachable.

---

## Hyperparameters

| Knob | Value |
|---|---|
| Self-gen corpus | 100k sequences × 1024 tokens |
| Sampling temperature | 1.0 |
| Loss | full-distribution KL |
| Epochs | 1 |
| Optimizer | AdamW, lr 1e-5 |
| Quant sim | STE through round |

---

## When to pick LLM-QAT over QLoRA / LoftQ

- **Activations below A8:** mandatory (PEFT methods don't quantize activations).
- **KV-cache quantization in the training loop:** mandatory.
- **2-bit weights with no quality compromise:** consider (LoftQ handles 2-bit weight-only; LLM-QAT handles 2-bit with activation quant too).

The cost: full QAT compute (days × multi-GPU) vs QLoRA's single-GPU PEFT. Use LLM-QAT only when you genuinely need sub-A8 or KV-quant.

---

## Connections

- PTQ rivals it succeeds at low bits: [[gptq]], [[awq]], [[smoothquant]].
- PEFT-style alternatives that avoid full QAT cost: [[qa-lora]], [[loftq]], [[peqa]].
- Self-distillation extension to sub-4-bit: [[bitdistiller]].
- KV-cache quant lineage carried forward: [[kivi]], [[kvquant]], [[gear]] ([[ch-15]]).
- Block-wise QAT successor: [[efficientqat]].

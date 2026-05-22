<!-- scope: LLM-QAT — data-free QAT via teacher self-generation, with KV-cache quant
     deps: [[straight-through-estimator]], [[zeroquant]]
     see-also: [[qa-lora]], [[loftq]], [[bitdistiller]], [[kivi]]
-->

# LLM-QAT: Data-Free Quantization Aware Training for Large Language Models
- **Core Insight:** Below 4 bits, PTQ saturates and you need QAT — but LLM pretraining data is unavailable; the workaround is to let the FP teacher *generate its own calibration corpus* by sampling from itself, then distill into the quantized student with cross-entropy on full output distributions instead of one-hot labels.
- **Guideline:** When pushing LLaMA-class models to ≤4-bit weights + ≤8-bit activations (and especially when quantizing the KV cache), use LLM-QAT: generate ~100k sequences from the FP teacher, train the quantized student with logit-distillation `KL(p_T || p_S)` for ~1 epoch, STE through round.
- **Authors:** Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, Vikas Chandra
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2305.17888
- **Relevant topics:** data-free QAT, teacher-generated calibration, KV cache quantization, sub-4-bit LLM

## Abstract
LLM-QAT brings true QAT to LLMs without access to original training data by having the pretrained FP teacher generate its own calibration corpus via next-token sampling (~100k sequences). The quantized student is then trained with full-distribution distillation `−Σ p_T(y) log p_S(y)` rather than next-token cross-entropy, which forces matching not just the argmax but the full output distribution — critical at very low bits. LLM-QAT also brings activation and KV-cache quantization into the same QAT loop, enabling W4A8KV4 and W4A4 configurations that pure PTQ cannot. Tested on LLaMA 7B/13B/30B with consistent gains over GPTQ at low-bit settings.

## Key Contributions
- **Data-free QAT via teacher self-generation**: sample from the FP teacher to build a calibration corpus, sidestepping the need for the original pretraining data.
- **Logit-distillation loss** (full distribution KL) instead of one-hot cross-entropy — preserves the rich uncertainty signal of the teacher.
- Brings KV cache (K and V activations during inference) into the QAT loop, enabling W4A8KV4 and W4A4KV4.
- First QAT to push below 4 bits at LLaMA scale and hold accuracy.

## Key Figures/Tables to Study
- **Figure 3:** sampled vs random calibration data — sampled wins by 1–3 ppl at 4-bit.
- **Table 4:** W4A8 / W4A6 / W4A4 / W4A8KV4 on LLaMA — LLM-QAT beats GPTQ + SmoothQuant by large margins below A8.

## Technical Details

### Data-free calibration corpus
Sample ~100k sequences from the FP teacher:
- Sample temperature T = 1.0.
- First 3–5 tokens from a vocab-random or BPE-balanced prompt; rest free-generated.
- Length ≈ 1024 tokens per sequence.
- No external dataset required (no Pile, no C4, no instruction sets).

### Distillation loss
Per token position t, the student matches the teacher's *full* distribution:
```
L_distill = − Σ_t Σ_v  p_T(v | x_{<t}) · log p_S^{quant}(v | x_{<t})
          = Σ_t  KL( p_T(·|x_{<t}) || p_S^{quant}(·|x_{<t}) )  + H(p_T)
```
- `p_T` from FP teacher; `p_S^{quant}` from the simulated-quantized student.
- Full vocab → richer signal than one-hot CE.

### Quantization (simulated during QAT)
- Weights: per-channel symmetric (per-row), b ∈ {4, 3, 2} bits, STE through `round`.
- Activations: per-token dynamic, b ∈ {8, 6, 4} bits, STE.
- KV cache: per-token (K), per-channel (V) — separate scales per layer. b_KV ∈ {8, 4}.

All quantization is **simulated**: forward uses `dequant(quant(·))`; backward uses straight-through gradient.

### Training schedule
- One epoch over the 100k self-generated sequences.
- AdamW, lr ≈ 1e-5 (small; the student's main job is repair, not learn).
- 32–256 GPUs depending on scale; LLaMA-30B fits with FSDP.

### Tested configurations
| Config | Weight | Activation | KV |
|--------|--------|------------|----|
| W4A8 | 4 | 8 | 16 |
| W4A6 | 4 | 6 | 16 |
| W4A4 | 4 | 4 | 16 |
| W4A8KV4 | 4 | 8 | 4 |

### Hyperparameters
| Knob | Value |
|------|-------|
| Self-gen corpus | 100k sequences × 1024 tokens |
| Temperature | 1.0 |
| Loss | full-distribution KL |
| Epochs | 1 |
| Optimizer | AdamW, lr 1e-5 |
| Quant sim | STE through round |

## Connections
- PTQ rivals it succeeds at low bits: [[gptq]], [[awq]], [[smoothquant]].
- PEFT-style alternatives that avoid full QAT cost: [[qa-lora]], [[loftq]], [[peqa]].
- Self-distillation extension to sub-4-bit: [[bitdistiller]].
- KV-cache quant lineage carried forward: [[kivi]], [[kvquant]], [[gear]].
- Block-wise QAT successor: [[efficientqat]].

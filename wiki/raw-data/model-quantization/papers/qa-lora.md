<!-- scope: QA-LoRA — group-wise quantization-aware LoRA that merges natively into INT4 base
     deps: [[qlora]], [[lora]]
     see-also: [[loftq]], [[peqa]], [[llm-qat]]
-->

# QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models
- **Core Insight:** QLoRA-style fine-tuning leaves an *imbalanced degrees of freedom* problem — the FP LoRA adapter has far more representational freedom than the INT4 base, so merging the trained adapter back into the INT4 base degrades quality; fix it by making the adapter *group-wise* (same group structure as the INT4 quantization), so adapter and base have matching DOF and the merge is lossless.
- **Guideline:** When you need the deployment artifact to remain pure INT4 (no FP16 LoRA path at inference), use QA-LoRA with `group_size = 32–128` matching the base quant; one bias-style adapter per group lets the post-training merge fold cleanly into the INT4 weight.
- **Authors:** Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, Xiaopeng Zhang, Qi Tian
- **Year:** 2023 (ICLR 2024)
- **URL:** https://arxiv.org/abs/2309.14717
- **Relevant topics:** quantization-aware LoRA, group-wise adapter, merge-free quantized fine-tuning

## Abstract
QLoRA gives a 4-bit base + BF16 LoRA artifact; to deploy you must either keep both (~FP16 inference cost) or merge LoRA into the base (which forces re-quantizing the merged weight, regressing accuracy). QA-LoRA traces this to a mismatch in degrees of freedom: standard LoRA `BA` is fully dense and per-element, while the INT4 base has one scale per group. QA-LoRA introduces a **group-wise low-rank operator** that adds one scalar correction per (output, group) — the same granularity as the quantization scales — so the trained adaptation merges into the INT4 scales without information loss. At inference, the merged model is pure INT4 with no LoRA branch. Matches QLoRA fine-tune quality with a *quantized* deployment artifact.

## Key Contributions
- Identifies the **imbalanced DOF** problem in QLoRA: 16-bit dense LoRA + 4-bit grouped base cannot merge losslessly.
- Replaces LoRA's `B A` with a **group-wise adapter** matching the base's group granularity.
- After training, the adapter folds directly into the per-group quantization scales (or zero-points) — no merge-induced quality drop, no FP16 sidecar at inference.
- Demonstrates parity with QLoRA on LLaMA/LLaMA-2 across multiple instruction datasets while shipping a pure-INT4 model.

## Key Figures/Tables to Study
- **Figure 2:** the DOF diagram — FP LoRA degrees vs INT4 base degrees side-by-side; QA-LoRA's grouped adapter matches.
- **Table 3:** post-merge accuracy — QLoRA loses 0.5–2 ppl after re-quantization; QA-LoRA loses 0.0.

## Technical Details

### Standard QLoRA path (the problem)
For a quantized layer `W_q ∈ INT4^{d_out × d_in}` with per-group (G) scales `s ∈ R^{d_out × (d_in/G)}`:
```
y = dequant(W_q, s) · x + (α/r) B A · x        (BF16 LoRA branch)
```
Merging `ΔW = (α/r) BA` (BF16, dense per-element) back into `W_q` requires re-quantizing `dequant(W_q, s) + ΔW` → information loss, because ΔW has `d_out · d_in` DOF while the INT4 representation has only `d_out · d_in / G + d_out · d_in · log2(16)/G_scale` effective DOF.

### QA-LoRA's group-wise operator
Replace `B A` (full-rank-r dense) with a **group-wise additive scalar** `δ ∈ R^{d_out × (d_in/G)}` — one trainable scalar per (output, group) cell, exactly matching the quantization grid:
```
y = dequant(W_q, s + δ) · x
```
(or equivalently, δ added to a per-group zero-point.)

This is a `(d_out / G)`-sized correction per output channel. Same total trainable parameters as a low-rank decomposition with rank r ≈ d_in/G.

### Merge step
Post-training, set `s_new = s + δ`. The deployed weight is `W_q` (unchanged INT4) with the updated scales. No re-quantization, no FP16 branch.

### Hyperparameters (recipe)
| Knob | Value |
|------|-------|
| Base quant | INT4 group-wise (G = 32 or 128) |
| Adapter granularity | one scalar per (d_out, group) |
| Optimizer | AdamW, lr 2e-4 |
| Datasets tested | Alpaca, GSM8K, etc. |
| Deployment | pure INT4 after merge |

## Connections
- Direct comparison: [[qlora]] (BF16 adapter, requires keeping LoRA at inference or accepting merge loss).
- Joint quant+LoRA init alternative: [[loftq]].
- Scale-only fine-tuning sibling: [[peqa]] (updates only scales, no per-group delta).
- True QAT for LLM: [[llm-qat]].
- Outlier-aware PEFT alternative: [[owq]] (weak-column tuning).

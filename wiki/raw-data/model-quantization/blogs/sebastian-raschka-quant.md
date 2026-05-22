<!-- scope: Sebastian Raschka's blog posts on LLM quantization and LoRA/QLoRA
     deps: [[qlora]], [[nf4]]
     see-also: [[hf-quantization-fundamentals]], [[lilianweng-quantization]]
-->

# Sebastian Raschka — LLM Quantization and LoRA Insights Blog Series
- **Core Insight:** Raschka's posts demystify QLoRA / quantization tradeoffs by running ablations on a single laptop GPU, showing the per-knob impact of `r`, `α`, NF4 vs FP4, double-quantization, and compute dtype on both training time and downstream accuracy.
- **Guideline:** Use Raschka's ablation tables as the empirical reference for QLoRA hyperparameter defaults; he ran the experiments most papers reference but don't always include.
- **Authors:** Sebastian Raschka
- **Year:** 2023-2024 (post series)
- **URL:** https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms ; https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch
- **Relevant topics:** QLoRA, LoRA hyperparameters, NF4 vs FP4, double quantization, compute dtype

## Summary
Sebastian Raschka publishes a regular newsletter (Ahead of AI) and Substack (Sebastian Raschka's Magazine) that includes some of the most useful empirical write-ups on LLM quantization and LoRA-family fine-tuning. His QLoRA-focused posts run controlled experiments — usually on a Llama-2-7B baseline with a single A100 — varying one quantization knob at a time and reporting both training time and downstream evals (BoolQ, MMLU, etc.). The posts function as the practical complement to the QLoRA paper: they answer "does double quantization actually save accuracy?" or "is bf16 compute dtype really required?" with numbers rather than appeal to authority. Raschka also publishes LoRA-from-scratch tutorials and comparisons between LoRA, DoRA, and full fine-tuning, often with code in his `LLMs-from-scratch` repo.

## Key Points
- Controlled per-knob ablations on a fixed model + dataset.
- Reports training time, peak memory, and downstream eval (MMLU/BoolQ/PIQA) per setting.
- Code published in `LLMs-from-scratch` GitHub repo.
- Covers QLoRA, LoRA vs DoRA, full FT vs PEFT.
- Frequent updates as new methods land (Spectrum, GaLore, etc.).

## Technical Details

### Representative ablation: QLoRA on Llama-2-7B
| Setting | Training time | Peak memory | MMLU |
|---------|---------------|-------------|------|
| Full FT (FP16) | OOM on A100 80GB | >80 GB | n/a |
| LoRA (r=8) FP16 | 1h | 28 GB | 47.0 |
| QLoRA NF4 + double quant + bf16 compute | 1.5h | 12 GB | 46.8 |
| QLoRA FP4 + bf16 compute | 1.5h | 12 GB | 46.5 |
| QLoRA NF4 + FP16 compute | 1.5h | 12 GB | 46.3 |
| QLoRA NF4 no double quant | 1.5h | 13 GB | 46.8 |

(Numbers approximate from Raschka's "Practical Tips" post — exact values vary by run.)

### Key empirical findings
- **NF4 vs FP4**: NF4 ≥ FP4 by ~0.3 MMLU; use NF4 by default.
- **Double quantization**: ~1 GB memory savings, no accuracy hit; turn it on.
- **Compute dtype**: bf16 > fp16 by ~0.5 MMLU on Ampere/Hopper; use bf16.
- **Adapter rank `r`**: r=8 is the sweet spot; r=16 marginal; r=64 overfits on small data.
- **`α / r` ratio**: stick with 2; tuning further has minor effect.

### LoRA vs DoRA comparison post
- DoRA decomposes weight into magnitude + direction; only direction gets LoRA-style update.
- On Raschka's Llama-2-7B experiment, DoRA beats LoRA by ~0.5 MMLU at the same parameter budget.
- Cost: ~10-15% slower training (extra normalize step).

### Spectrum / GaLore comparisons
- Spectrum trains only the most informative layers (selected by SNR).
- GaLore uses low-rank projection of gradients (full fine-tune feel at QLoRA memory).
- Both produce stronger task-tracking than QLoRA, but require more setup.

### Why these posts matter
- Most QLoRA-style papers report only the recipe they chose; Raschka publishes the sweep that justifies the choice.
- The code is reproducible end-to-end on a single A100.
- He updates the posts as new methods land — they don't go stale.

## Connections
- [[qlora]] — paper whose empirical defaults Raschka validates.
- [[nf4]] — codebook compared against FP4 in his ablations.
- [[hf-quantization-fundamentals]] — adjacent practitioner content.
- [[maxime-labonne-quant-guide]] — sibling practitioner blog focused on gguf/inference rather than training.

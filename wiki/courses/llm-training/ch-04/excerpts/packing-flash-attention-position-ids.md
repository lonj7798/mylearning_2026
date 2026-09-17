---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Kundu, Lee, Wynter, Ganti, Mishra — "Enhancing Training Efficiency Using Packing with Flash Attention" (IBM Research)
source_url: https://arxiv.org/abs/2407.09105
created_at: "2026-09-15"
note: "No library card for this artifact on 2026-09-15; written from the primary PDF (arXiv v6, 2024-09-01). The library card papers/packed-vs-unpacked-ablation.md lists this work but has no primary source of its own and is not cited."
---

# Excerpt: Kundu et al. 2024 — Packing with position IDs and FlashAttention

**Artifact:** arXiv:2407.09105 (v1 2024-07; read as v6).

## Mechanism (§3)

- Online minibatch collating concatenates the examples of a minibatch into one row of length Σ L_i and provides position ids `[0..L_1−1, 0..L_2−1, …]`. This is available as `DataCollatorWithFlattening` in Transformers 4.44 and as `padding_free=True` in TRL's `DataCollatorForCompletionOnlyLM` (§3.1).
- The models' `_flash_attention_forward()` receives `position_ids`; "When attention_mask is None in the case of number of examples > batch size, we compute cu_seq_len from position_ids and use the flash_attn_varlen_func()" (§3.3).
- The PaddingFreeCollator converts "the first label_id of each example into -100, then concatenate[s]" (§3.4). Example with lengths 4, 8, 5, 11: labels start `[−100, 11, 12, 13, −100, 21, …]` and position ids `[0, 1, 2, 3, 0, 1, …]` (§3.5).
- Table 1 characteristics: FixedLengthPacking (without position ids) has no correct cross-attention and breaks examples; FixedLengthPacking+PosID has correct attention but still breaks examples; MiniBatchPacking+PosID has correct attention and no broken examples.

## Results (§4)

Setting: 20K-example subsets of FLAN, OrcaMath, and the Stack; one node of 8 A100-80GB GPUs with FSDP; one epoch, gradient accumulation 2, maximum sequence length 4,096, minibatch 4 per GPU.

Mistral-7B on FLAN_20k (Table 2):

| Batching | Tokens/s | Validation loss |
|---|---|---|
| padding | 742 | 1.129 |
| packing without position IDs | 2,986 | 1.306 |
| offline packing with position IDs (585 packed rows) | 3,010 | 1.284 |
| online minibatch packing with position IDs | 1,408 | 1.127 |

The "Rows" column is the number of training rows in one epoch: 19,961 with padding and minibatch packing, 2,294 for fixed-length packing without position ids, and 585 for offline packing with position ids. The authors attribute the higher loss of offline packing to far fewer optimisation steps in one epoch; minibatch packing keeps the step count and matches padding (§4.1, §4.3). The difference between the two offline packed variants is not consistent across models: Llama-2 1.579 (without position ids) vs 1.578 (with); Granite 1.740 vs 1.767 (Table 2). Throughput benefits hold across the tested architectures except Gemma-7B and Qwen1.5-MoE-A2.7B (§4.1).

## Use in ch-04

The position-id reset is how the Hugging Face padding-free path encodes document boundaries. Continuous position ids in that path remove the boundaries even though RoPE itself is invariant to a constant offset.

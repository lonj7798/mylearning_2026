---
chapter: ch-36
course: llm-training
phase: read
excerpt_of: primary source arXiv:2407.09105v6 (no library card; the synthesis card packed-vs-unpacked-ablation is marked "No verifiable primary source" and is not cited)
source_url: https://arxiv.org/abs/2407.09105
created_at: "2026-09-15"
---

# Excerpt: Enhancing Training Efficiency Using Packing with Flash Attention (Kundu et al.)

**Paper:** Achintya Kundu, Rhui Dih Lee, Laura Wynter, Raghu Kiran Ganti, Mayank Mishra (IBM Research). arXiv v1 2024-07; read at v6 (2024-09-01). Source type: paper.

## Mechanism (§3.3-3.5)

- "When attention_mask is None in the case of number of examples > batch size, we compute cu_seq_len from position_ids and use the flash_attn_varlen_func()" (§3.3).
- Padding-free collator (§3.4): concatenate all examples of the mini-batch into `input_ids`; "Convert the first label_id of each example into -100, then concatenate" for `labels`; "Generate position ids for each example and concatenate them all" for `position_ids`.
- Worked example (§3.5), four examples of lengths 4, 8, 5, 11:

```
input_ids:    [[10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27,
                30, 31, 32, 33, 34, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 410]]
labels:       [[−100, 11, 12, 13, −100, 21, 22, 23, 24, 25, 26, 27,
                −100, 31, 32, 33, 34, −100, 41, 42, 43, 44, 45, 46, 47, 48, 49, 410]]
position_ids: [[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7,
                0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
```

- Available as `DataCollatorWithFlattening` in Transformers 4.44 and through `padding_free=True` in TRL's `DataCollatorForCompletionOnlyLM` (§3.1).

## Table 2 (FLAN 20K subset; one epoch; gradient accumulation 2; max sequence length 4096; mini-batch 4 per GPU), Mistral-7B rows

| Batching | Rows | Tokens/s | Validation loss |
|---|---|---|---|
| padding | 19,961 | 742 | 1.129 |
| packing without position IDs | 2,294 | 2,986 | 1.306 |
| offline packing with position IDs | 585 | 3,010 | 1.284 |
| online mini-batch packing with position IDs | 19,961 | 1,408 | 1.127 |

§4.1: "Due to the fact that far fewer optimisation steps are taken with such maximal packing, the loss does not decrease as fast, and its effect is confirmed by the validation loss ('VLoss') after one epoch." Mini-batch packing keeps the number of optimizer steps. The paper reports no seeds or run-to-run variance.

## Used in

ch-36 §2.3 (packed-attention and packed-label tests) and §2.4 (why training curves are not an equivalence test).

---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: arXiv:2107.02027v2; arXiv:2407.09105v6; arXiv:2410.08081v3
source_url: https://arxiv.org/abs/2107.02027
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from primary sources)"
---

# Excerpt: Packing — what three studies measured

Used by [[read]] §4. The earlier version of this excerpt quoted a throughput formula, a 0.01-nat diagnostic, and four failure modes from the card `packed-vs-unpacked-ablation`, which has no primary source; none of those items appears below.

## 1. Krell, Kosec, Perez, Fitzgibbon (arXiv:2107.02027v2) — BERT, not decoder SFT

Abstract (verbatim): padding is such "that up to 50% of all tokens can be padding. In less common, but not extreme, cases (e.g. GLUE-cola with sequence length 128), the ratio is up to 89%." The 50% figure refers to Wikipedia BERT pre-training data. The packed model is kept equivalent to the unpacked one with a block-diagonal attention mask and per-sequence position indices (§3.2.1-3.2.2; card [[sequence-packing]]).

Ablation (§4.2.1): without the position adjustment, "the loss and accuracy almost match. However, the accuracy stalls at 71.8% and does not reach the target accuracy of 72.1%." BERT uses learned absolute position embeddings, so this result does not transfer directly to rotary embeddings.

## 2. Kundu, Lee, Wynter, Ganti, Mishra (IBM Research), "Enhancing Training Efficiency Using Packing with Flash Attention" (arXiv:2407.09105v6)

Mechanism (§3.3, verbatim): "we modify the models' _flash_attention_forward(), adding the argument position_ids and extracting the number of examples in the batch from the position_ids. When attention_mask is None in the case of number of examples > batch size, we compute cu_seq_len from position_ids and use the flash_attn_varlen_func()." Boundaries are therefore derived from `position_ids` in this implementation.

Setting (§4.1): FLAN 20K subset, one epoch, gradient accumulation 2, maximum sequence length 4096, mini-batch 4 per GPU. Table 2, Mistral-7B:

| Batching | pos_id | Rows | Tokens/s | Validation loss |
|---|---|---|---|---|
| padding | no | 19,961 | 742 | 1.129 |
| packing to sequence length, no position IDs | no | 2,294 | 2,986 | 1.306 |
| offline packing to bs × msl rows, with position IDs | yes | 585 | 3,010 | 1.284 |
| online mini-batch packing, with position IDs | yes | 19,961 | 1,408 | 1.127 |

Llama-2 rows in the same table: 1.266, 1.579, 1.578, 1.262. Across the 10 models of Table 2, validation loss of packing without position IDs minus offline packing with position IDs ranges from −0.089 (Phi-2: 3.375 vs 3.464) to +0.170 (Falcon: 2.585 vs 2.415); the two rows also differ in row count.

Table 4 (§4.3, Mistral-7B, FLAN 20K, one epoch; bs, msl, gas as printed) compares the two at identical step counts: FixedLengthPacking (2, 4096, 32), 35 steps, validation loss 1.294 against FixedLengthPacking+PosID 1.221; (2, 4096, 4), 281 steps, 1.252 against 1.170. Padding at 311 steps gave 1.117 (2, 4096, 32). The authors attribute the remaining gap of offline packing to "the fewer number of optimisation update steps performed". Explanation (§4.1, verbatim): "Due to the fact that far fewer optimisation steps are taken with such maximal packing, the loss does not decrease as fast, and its effect is confirmed by the validation loss ("VLoss") after one epoch." Mini-batch packing "achieves the same optimal loss pattern and hence validation loss, as the inefficient padding-based approach." Benefits are consistent across architectures "with the exception of Gemma-7B and Qwen1.5-MoE-A2.7B".

## 3. Wang, Wang, Wang, Li, Hovy, Guo, "Packing Analysis: Packing Is More Appropriate for Large Models or Datasets in Supervised Fine-tuning" (arXiv:2410.08081v3)

Setting (§4.1.2, Table 2): LLaMA-3-8B and LLaMA-3-70B; LR 1e-5; maximum length 4096; warmup ratio 0.2; 4 epochs (8B) or 3 epochs (70B); loss only on tokens after the assistant header. The paper does not state that attention is reset between packed conversations; §3.3.3 argues that the [EOS] token separates samples.

Table 3, average benchmark score, padding / random packing / greedy packing:

| Data | 8B | 70B |
|---|---|---|
| WildChat (GPT-4), 69K | 49.58 / 49.46 / 50.6 | 61.50 / 65.97 / 65.92 |
| Open-source 1M | 54.3 / 54.95 / 55.05 | 66.12 / 67.26 / 67.54 |

Table 5: 70B on WildChat (GPT-4), 9,533 s with padding against 3,749 s with random packing. §5.3: packing a single-turn set (filtered 200K OpenHermes 2.5) gave a MATH drop that returned to normal after adding 1/40 to 1/20 multi-turn conversations. No seeds or run-to-run variance are reported.

## Connections

- [[read]] §4; [[ch-04]] for packing mechanics.
- [[smol-training-playbook]] — SmolLM3 packing throughput and small-data effect.

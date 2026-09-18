---
chapter: ch-08
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2407.09105
source_url: https://arxiv.org/abs/2407.09105
created_at: "2026-09-17"
revised: 2026-09 (generality revision)
---

# Excerpt: Kundu and Lee 2024, "Enhancing Training Efficiency Using Packing with Flash Attention"

**Artifact:** Achintya Kundu, Rhui Dih Lee, Laura Wynter, Raghu Kiran Ganti, Mayank Mishra (IBM Research),
arXiv:2407.09105 (v1 2024-07; v6 2024-09-01).
**Read on:** 2026-09-17 from the PDF text cached at `scratchpad/sources/kundu-packing-fa.txt`.
**Source type:** paper.

The library card [[packed-vs-unpacked-ablation]] is marked "no verifiable primary source" and must not be
cited as evidence; this excerpt records the primary numbers ch-08 uses for the packed-versus-unpacked test.

---

## Mechanism

- Position IDs restart at 0 for each packed example; `cu_seq_len` is derived from `position_ids` and
  `flash_attn_varlen_func()` is called with it (§3.3).
- The padding-free collator returns input IDs, labels, and position IDs of each example after flattening (§3, L194).
- Exposed as `DataCollatorWithFlattening` in Transformers 4.44 and as `padding_free=True` in TRL's
  collator (§3.1).

## Setting

20K-example subsets of FLAN, OrcaMath, and The Stack; one epoch; gradient accumulation 2; maximum sequence
length 4096; per-GPU minibatch 4; one node of 8 A100-80GB GPUs with FSDP (§4, §4.1). Ten models were tested,
including `mistralai/Mistral-7B-v0.1` and `Llama-2-7B-fp16`.

## Table 2 (FLAN_20k, one epoch) — rows read at the locus

| Model | Batching | Rows | Tok/s | Peak mem (MB) | Validation loss |
|---|---|---|---|---|---|
| Mistral-7B | padding (no packing) | 19961 | 742 | 30625 | 1.129 |
| Mistral-7B | packing without position IDs | 2294 | 2986 | 30783 | 1.306 |
| Mistral-7B | offline flat packing with position IDs | 585 | 3010 | 30783 | 1.284 |
| Mistral-7B | online minibatch packing with position IDs | 19961 | 1408 | 24549 | 1.127 |
| Llama-2-7B | padding | 19958 | 771 | 29234 | 1.266 |
| Llama-2-7B | packing without position IDs | 2346 | 3091 | 29433 | 1.579 |
| Llama-2-7B | online minibatch packing with position IDs | 19958 | 1455 | 23854 | 1.262 |
| Granite-8B-code | padding | 19947 | 693 | 36277 | 1.169 |
| Granite-8B-code | online minibatch packing with position IDs | 19947 | 1358 | 27538 | 1.169 |
| Falcon-7B | padding | 19969 | 890 | 29894 | 1.842 |
| Falcon-7B | packing without position IDs | 2168 | 3401 | 30106 | 2.585 |

Two facts ch-08 uses from this table:

1. **Validation loss after one epoch of SFT on a 7B-8B base model is between 1.1 and 1.9 nats**, not
   `ln(V)`. A step-1 SFT loss near `ln(V)` indicates a broken input pipeline, not a healthy start.
2. **Offline packing with correct position IDs does not reproduce the padded loss in this setting**
   (1.284 vs 1.129 for Mistral-7B). The authors attribute the gap to the far smaller number of optimisation
   steps taken in one epoch when rows are maximally packed, and show that online minibatch packing, which keeps
   the row count, matches the padded loss (1.127 vs 1.129) (§4.1, §4.3). Packing without position IDs is worse
   again (1.306), and the gap is larger for Llama-2 (1.579 vs 1.266) and Falcon (2.585 vs 1.842).

## Conditions and limits stated by the source

One seed per configuration; no variance is reported. Benefits are described as consistent across
architectures except Gemma-7B and Qwen1.5-MoE-A2.7B (§4.1). Bin-packing sample-selection variants add limited
benefit over simpler packing (§4.3).

## Connections

- [[trl-sft-trainer]] — `padding_free`, `seq_lengths`, and the `position_ids == 0` label mask implement the
  mechanism described in §3.3-3.5 of this paper.
- [[sequence-packing]] — Krell et al. 2021, the earlier block-diagonal-attention result on an encoder model.

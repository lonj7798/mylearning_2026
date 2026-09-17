---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Krell, Kosec, Perez, Fitzgibbon — "Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance"
source_url: https://arxiv.org/abs/2107.02027
created_at: "2026-04-23"
revised_at: "2026-09-15 (generality revision; rewritten from the primary PDF, arXiv v2)"
---

# Excerpt: Krell et al. 2021 — Sequence packing without cross-contamination

**Artifact:** arXiv:2107.02027 (v1 2021-07; read as v2). Authors: Mario Michael Krell, Matej Kosec, Sergio P. Perez, Andrew Fitzgibbon (Graphcore).
**Setting:** BERT, a bidirectional encoder with learned absolute position embeddings. The paper does not study decoder SFT or chat data.

## What ch-04 uses

- **Padding share.** Up to 50% of tokens are padding in BERT pre-training on Wikipedia at sequence length 512; for GLUE CoLA at sequence length 128 the share is 89% (abstract; §1).
- **SPFHP.** Shortest-pack-first histogram packing works on the sequence-length histogram (bin size 1), traverses it from longest to shortest, and assigns each histogram bin to the pack with the most remaining space (worst-fit), subject to a maximum packing depth (§3.1.1). It is not "longest sequence that still fits".
- **NNLSHP.** Non-negative least squares histogram packing restates packing as a weighted non-negative least squares problem over the histogram (§3.1.2).
- **Efficiency (Table 1, IPU).**

  | Packing depth | Algorithm | Efficiency (%) | Packing factor | Estimated realized speed-up |
  |---|---|---|---|---|
  | 1 | NONE | 50.0 | 1.00 | 1.000 |
  | 3 | SPFHP | 89.4 | 1.79 | 1.716 |
  | 8 | SPFHP | 98.9 | 1.98 | 1.895 |
  | max | SPFHP | 99.6 | 1.99 | 1.905 |
  | 3 | NNLSHP | 99.7 | 2.00 | 1.913 |

  In MLPerf phase-2 training the measured total speed-up exceeded 2x (§4.2, Fig. 3).
- **Required adjustments.** (1) Position embeddings: position indices restart per sequence, e.g. [0, 1, 0, 1, 2] for lengths 2 and 3, via an embedding look-up (§3.2.1). (2) Attention: a block-diagonal mask built from sequence ids, `zero_one_mask = tf.equal(mask, mask.T)`, applied before the softmax (§3.2.2, Fig. 2). (3) Per-sequence losses: without unpacking, a per-sequence loss is weighted per pack, so "a single sequence would contribute to the loss with the same weight as a pack of three sequences" (§3.2.3).
- **Ablation (§4.2.1, Fig. 4).** Without the attention-mask adjustment, training loss and accuracy "worsen drastically" and longer training does not recover them. Without the position adjustment, loss and accuracy almost match, but MLM accuracy stalls at 71.8% against the 72.1% target.
- **Convergence.** Packed and unpacked learning curves nearly match when normalized by samples processed at the same effective batch size; changing batch size and hyperparameters changes early convergence (§4.2, Fig. 3).
- **Downstream check.** After packed pre-training, SQuAD 1.1 F1 is 88.32 (base) and 90.65 (large) against reference 88.5 and 90.9 (§4.3, Table 3).

## Limits for decoder LLMs

- The position-reset result applies to learned absolute position embeddings. With RoPE and a correct block-diagonal mask, attention depends only on relative offsets (see ch-04 §3).
- The 50% and 89% padding shares are BERT/GLUE figures. They are not measurements of chat or instruction mixtures.
- The library card `papers/sequence-packing.md` (2026-07 version) states "zero accuracy impact", "instruction datasets have 50-89% padding", and describes SPFHP as picking the longest sequence that fits. None of these is stated by the paper in that form.

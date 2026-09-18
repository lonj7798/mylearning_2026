---
chapter: ch-05
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2309.14509
source_type: paper
verified_on: "2026-09-17"
verified_against: "arXiv:2309.14509v2 (4 Oct 2023), cached primary text"
---

# Excerpt: DeepSpeed Ulysses — System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models

**Authors:** Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song,
Samyam Rajbhandari, Yuxiong He (Microsoft)
**Version read:** arXiv:2309.14509v2, 4 Oct 2023 (v1 2023-09)
**Used by:** [[read]] §4 (context and sequence parallelism)

This file holds only statements checked against the primary text. The library has no card for this
artifact at the time of writing; this excerpt is the chapter's citable locus.

## Mechanism

From §3.1 and Figure 2: the input sequence of length `N` is partitioned across `P` devices, so each
device holds `N/P` positions. Each local partition is projected into its own Q, K, V.

1. An all-to-all collective over the QKV embeddings gives every device the **full sequence** for a
   **non-overlapping subset of attention heads** (§1, §3.1).
2. Attention is then computed per head, `Softmax((QKᵀ)/√d)V`, with each device holding whole heads
   (§3.1, Eq. 1).
3. A second all-to-all transforms the output context tensor back to sequence-parallel `N/P` layout for
   the following operators (MLP matmul, layer norm) (§3.1).

Quoted: "attention computation is head parallelism with full attention per head but just with fewer
heads, thus attention computation can be replaced with any type of attention mechanisms, e.g., dense
attention and various forms of sparse attention" (§3.4).

## Communication analysis (§3.2)

Symbols as defined in §3.2: `h` = hidden size, `N` = sequence length, `P` = sequence-parallel degree,
`M` = aggregate message size.

- "the communication volume transmitted per link for an all-to-all for aggregate message of size M
  over P GPUs is M/P" (§3.2).
- Per transformer layer, Ulysses does one all-to-all with aggregate message `3Nh` (QKV) and one with
  `Nh` (output context), so the volume per link is `4Nh/P`, complexity `O(N/P)` (§3.2).
- "Note that this communication volume is constant when both N and P are increased proportionally"
  (§3.2).
- Megatron-LM sequence parallelism does two all-gathers of `Nh` and two reduce-scatters of `Nh` per
  layer; each costs `M` rather than `M/P` when `P ≫ 1`, so its volume per link is `4Nh`, "which is P
  times larger than that for DeepSpeed sequence parallelism" (§3.2).

## Memory and composition with ZeRO (§3.3)

Sequence parallelism reduces activation memory but "does not impact the memory consumed by the model
states". Ulysses is therefore integrated with ZeRO-3, and ZeRO-3 partitioning is extended to "the
combination of data parallel and sequence parallel ranks": model states are partitioned and gradients
are reduced across the combined data-parallel and sequence-parallel group (§3.3).

## Reported results

- "DeepSpeed-Ulysses trains 2.5x faster with 4x longer sequence length than the existing method SOTA
  baseline" (Abstract).
- "Communication reduction of over 10x compared to existing systems, resulting in throughput
  improvements of up to 2.5x, and sustained throughput of over 175 TFlops/GPU (over 54% of hardware
  peak)" (§1, contributions).
- Evaluation is on GPT models using up to 256 A100 GPUs (§4).
- Sequence-length strong scaling: up to 1 million tokens on a 1.2B-parameter GPT model, with sequence
  length increasing linearly with GPU count at similar computation throughput (§4.1, Figure 3).
- With sparse attention and ZeRO-3, "sequence parallelism leveraging ZeRO-3 scales to 4x longer
  sequence lengths than Megatron-LM" (§4.3).

## Not stated by the source (checked)

- No explicit divisibility rule bounding `P` by the number of attention heads. The bound is implied by
  the head-parallel attention step of §3.1 and §3.4, and the chapter labels it as an interpretation.
- No production-model context-parallel degree; no MoE interaction.

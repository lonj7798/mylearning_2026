---
chapter: ch-05
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2310.01889
source_type: paper
verified_on: "2026-09-17"
verified_against: "arXiv:2310.01889v4 (27 Nov 2023), cached primary text"
---

# Excerpt: Ring Attention with Blockwise Transformers for Near-Infinite Context

**Authors:** Hao Liu, Matei Zaharia, Pieter Abbeel (UC Berkeley)
**Version read:** arXiv:2310.01889v4, 27 Nov 2023 (v1 2023-10)
**Used by:** [[read]] §4 (context and sequence parallelism)

This file holds only statements checked against the primary text. The library has no card for
this artifact at the time of writing; this excerpt is the chapter's citable locus.

## What the method is

Quoted from the abstract:

> "We present a novel approach, Ring Attention with Blockwise Transformers (Ring Attention), which
> leverages blockwise computation of self-attention and feedforward to distribute long sequences
> across multiple devices while fully overlapping the communication of key-value blocks with the
> computation of blockwise attention. Our approach enables training and inference of sequences that
> are up to device count times longer than those achievable by prior memory-efficient Transformers,
> without resorting to approximations or incurring additional communication and computation
> overheads." (Abstract)

Mechanism, as described in §3: each host holds one query block. Key-value blocks rotate around a ring
of hosts. While a host computes blockwise attention for its local query block against the key-value
block it currently holds, it sends that key-value block to the next host and receives the next one from
the previous host (§3, Figure 2a). The analysis "involves interactions only with the immediately
previous and next hosts in a circular configuration", so it applies to both GPU all-to-all topology and
TPU torus topology (§3).

## The overlap condition (arithmetic intensity between hosts)

Symbols as defined in §3: `F` = FLOPS per host, `B` = bandwidth between hosts, `c` = block size,
`d` = hidden size, `s` = sequence length per device.

- Blockwise self-attention costs `2dc²` FLOPs for the attention scores and `2dc²` FLOPs for the
  score-by-value product, so `4dc²` FLOPs in total. The projections of Q, K, V and the blockwise
  feedforward are excluded because they add compute without adding inter-host communication (§3).
- Key and value blocks: the paper writes "both key and value blocks require a total of 2cd bytes.
  Thus, the combined communication demand is 4cd bytes" (§3). The two sentences are consistent only if
  the first counts elements and the second bytes at two bytes per element; the chapter uses the second
  figure, `4cd` bytes, which is the one the overlap condition is derived from.
- Minimal block size required for overlap: `c = FLOPS / Bandwidth`; minimal sequence length per
  device `s = 6c` (Table 2 caption).

Table 2, three of its five rows ("Minimal sequence length needed on each device"; interconnect
bandwidth is the bandwidth between hosts). The TPU v4 row (275 TF, 32 GB, 268 GB/s, 1.0, 6.2) and the
TPU v5e row (196 TF, 16 GB, 186 GB/s, 1.1, 6.3) are printed in the paper and omitted here:

| Spec per host | FLOPS (TF) | HBM (GB) | Interconnect bandwidth (GB/s) | Minimal block size (×1e3) | Minimal sequence len (×1e3) |
|---|---|---|---|---|---|
| A100 NVLink | 312 | 80 | 300 | 1.0 | 6.2 |
| A100 InfiniBand | 312 | 80 | 12.5 | 24.5 | 149.5 |
| TPU v3 | 123 | 16 | 112 | 1.1 | 6.6 |

Reading of the table used by the chapter: the per-device sequence length below which communication
stops being hidden is set by the ratio of per-host FLOPS to inter-host bandwidth, so the same method
needs about 24× longer local sequences over InfiniBand than over NVLink (Table 2).

## Activation size

Table 1 ("maximum activation sizes among different Transformer architectures", bytes per layer, bfloat16;
`b` batch, `h` hidden, `n` heads, `s` sequence, `c` block size, with `c` independent of `s`):

| Layer type | Self-attention | FeedForward | Total |
|---|---|---|---|
| Vanilla | 2bns² | 8bsh | 2bhs² |
| Memory-efficient attention | 2bsh + 4bch | 8bsh | 8bsh |
| Memory-efficient attention and feedforward | 2bsh | 2bsh | 2bsh |
| Ring Attention | 6bch | 2bch | 6bch |

## Not stated by the source (checked)

- No context-parallel degree for any named production model.
- No comparison against all-gather-based context parallelism (that method is described in
  [[llama-3]] §3.3.2, which cites this paper as the ring-structured alternative).

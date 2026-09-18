---
chapter: ch-56
course: llm-training
phase: read
excerpt_of: (no library card at the time of writing; extracted from the primary source)
source_url: https://arxiv.org/abs/2310.01889
created_at: "2026-09-17"
revised: "2026-09 (generality revision)"
---

# Excerpt: Ring Attention with Blockwise Transformers for Near-Infinite Context

**Primary source:** Hao Liu, Matei Zaharia, Pieter Abbeel (UC Berkeley). arXiv:2310.01889 v4, 2023-11-27
(v1 2023-10). Read on 2026-09-17.
**Source type:** paper.

---

## Core claim

Blockwise attention and feedforward are computed on per-device blocks of one sequence while key-value
blocks rotate around a ring of devices, so the transfer of the next block overlaps the computation on the
current one. The abstract states this enables sequences up to device-count times longer than prior
memory-efficient transformers, "without resorting to approximations or incurring additional communication
and computation overheads."

---

## The two numbers ch-56 uses

### 1. Activation memory (Table 1, bytes per layer, bfloat16)

| Layer type | Self-attention | Feedforward | Total |
|---|---|---|---|
| Vanilla | 2bns² | 8bsh | 2bhs² |
| Memory-efficient attention | 2bsh + 4bch | 8bsh | 8bsh |
| Memory-efficient attention and feedforward | 2bsh | 2bsh | 2bsh |
| **Ring attention** | 6bch | 2bch | **6bch** |

b is batch size, h is hidden dimension, n is number of heads, s is sequence length, c is block size. The
ring-attention total contains no s: memory scales with the per-device block, not the full sequence. Six
blocks are held per host — the query block, the current key and value blocks, two incoming key and value
blocks, and the output block (§3.2).

### 2. The overlap condition (§3.2)

Blockwise attention on a block of size c costs 4dc² FLOPs and moves 4cd bytes of key and value data.
Overlap requires 4dc²/F ≥ 4cd/B, that is **c ≥ F/B**, where F is per-host FLOPS and B is unidirectional
interconnect bandwidth. The minimal sequence length per device is **s = 6c**.

Table 2, minimal block size and minimal sequence length per host:

| Spec per host | FLOPS (TF) | HBM (GB) | Interconnect (GB/s) | Minimal block size | Minimal sequence length |
|---|---|---|---|---|---|
| A100 NVLink | 312 | 80 | 300 | 1.0 × 10³ | 6.2 × 10³ |
| A100 InfiniBand | 312 | 80 | 12.5 | 24.5 × 10³ | 149.5 × 10³ |
| TPU v3 | 123 | 16 | 112 | 1.1 × 10³ | 6.6 × 10³ |
| TPU v4 | 275 | 32 | 268 | 1.0 × 10³ | 6.2 × 10³ |
| TPU v5e | 196 | 16 | 186 | 1.1 × 10³ | 6.3 × 10³ |

The paper's reading of this table: the requirement is met easily on high-bandwidth interconnects and is
"more strict" for GPUs connected via InfiniBand (§3.2).

---

## Conditions and limits

The implementation described in the paper is in JAX and uses `jax.lax.ppermute` for the ring exchange; the
arithmetic above is an analysis of the overlap condition, not a measured speed-up for any particular
training stack. OpenRLHF's `--ds.ring_attn_size` uses the third-party `ring_flash_attn` package
(`openrlhf/models/ring_attn_utils.py`), and neither OpenRLHF nor this paper reports a measurement for that
combination.

---

## Links

[[openrlhf-ppo]] · [[async-rollout]]

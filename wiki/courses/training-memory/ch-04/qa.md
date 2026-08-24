<!-- qa: ch-04 — Attention Is a Memory Problem: O(N^2) and Why the Kernel Decides
     companion to [[read]] · append-only across cycles · kernels/index only -->

# ch-04 Q&A — Attention as a Memory Problem

Clarifying questions raised while reading [[read]]. Kernels only; full detail in `read.md`.
Prior chapters: `ch-01/`[[qa]] (ledger), `ch-02/`[[qa]] (precision, logit spike), `ch-03/`[[qa]] (activations).

---

### Q1 — observation: attention eats the most memory of anything

**Correct, and the crossover is startlingly low.** Setting [[ch-03]]'s two activation terms equal (with SP):

```
linear term 34  =  attention term 5as/h      ->    s = 34h/(5a)
h = 4096, a = 32   ->   s ~= 870 tokens
```

Past **~870 tokens** attention outweighs everything else combined — so at any modern context (4k/8k/32k) it always dominates.

**Boson-scale numbers** (`s=4096, h=4096 illustrative, a=32, t=8, L=80, b=1`, 64 GPUs, ZeRO-3):

| item | per GPU |
|---|---|
| **attention `s²` term** | **26.8 GB** |
| all other activations (with SP) | 5.7 GB |
| entire static ledger (432 GB ÷ 64) | 6.75 GB |

Attention alone is **~4× the whole static ledger per GPU** — bigger than weights + gradients + 12 B/param optimizer states put together. (Un-mitigated figure; [[ch-03]] §3's selective recomputation deletes it, which is why that is effectively mandatory rather than optional.)

**But it is not always first place:**

| regime | largest item |
|---|---|
| full fine-tune, short seq, few GPUs | optimizer states (12 B/param) |
| LoRA (gradients/optimizer ≈ 0) | attention |
| long sequence | attention, overwhelmingly |
| more GPUs | tilts toward attention |

That last row is the structural one: **ZeRO/FSDP divides the static ledger by `W`, but activations are not divided** — each GPU still processes its own batch, so per-GPU activation is unchanged by scaling out.

```
add GPUs ->  static:     432/W   falls
             activation: unchanged
```

So the larger the run, the more attention becomes the *only* remaining problem — which is why FlashAttention mattered as much as it did.

**And `s²` runs away:** 4k → 32k is 8× in both `sbh` and `5as/h`, hence **64×**: `26.8 GB → 1,718 GB` on one GPU. At that point long context is not expensive, it is *impossible* — the premise of this chapter, which §2's online softmax and [[ch-05]]'s FlashAttention convert to `O(s)`.

**One line:** attention overtakes all other activation memory past only ~870 tokens and at boson scale is ~4× the entire per-GPU static ledger; scaling out shrinks the static side but never the activation side, and `s²` grows 64× from 4k to 32k — so attention is the one term that must be attacked structurally. See read.md §1.2/§5, [[ch-03]] qa-deep-2 Q6, [[ch-05]].

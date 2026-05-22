---
chapter: ch-22
course: model-quantization
phase: read
excerpt_of: "Reproduction-memo template + worked example for the ch-22 capstone"
created_at: "2026-05-21"
---

# Excerpt: Reproduction Memo Template + Worked Example

**Use as:** the literal scaffolding for `capstone-memo.md` in your `quant-capstone/` repo.

---

## Memo template (copy into `capstone-memo.md`)

```markdown
# Reproduction Memo — <Method> on <Model>

**Method:** <KIVI | KVQuant | GEAR | TurboQuant | NVFP4 inference>
**Paper:** <citation + arXiv link>
**Reproduced on:** <e.g., 1 × H100 80 GB, CUDA 12.4, PyTorch 2.5>
**Base model:** <meta-llama/Llama-3-8B | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | ...>
**Code:** https://github.com/<you>/quant-capstone @ <commit-sha>
**Date:** <YYYY-MM-DD>

## 1. Method summary (one paragraph from the paper, in your own words)

<One paragraph. State the method's core mechanism without referring to the paper.
If you can't write this paragraph cold, you don't yet understand the paper well enough
to have reproduced it.>

## 2. Reproduction setup

| Axis | This reproduction | Paper |
|------|-------------------|-------|
| Model | <your model> | <paper's model> |
| Bit budget | <e.g., INT2 KV> | <paper> |
| Calibration | <corpus + size + seed; "none" for data-oblivious methods> | <paper> |
| Hardware | <yours> | <paper's> |
| Framework | <PyTorch + HF transformers v...> | <paper's framework> |
| Long-context lengths | <e.g., 4K / 8K / 16K> | <paper's> |
| Eval seeds | <e.g., 5 seeds> | <paper> |

## 3. Headline numbers

| Metric | Paper | This reproduction | Gap | Within 2%? |
|--------|-------|-------------------|-----|-----------|
| Wikitext-2 PPL gap vs FP16 | x | y ± CI | y - x | ✓ / ✗ |
| LongBench avg (delta vs FP16) | x | y ± CI | y - x | ✓ / ✗ |
| NIAH @ 32K @ 50% depth | x | y | y - x | ✓ / ✗ |
| RULER multi-hop @ 32K | x | y | y - x | ✓ / ✗ |
| Throughput (if applicable) | x | y | y - x | ✓ / ✗ |
| Peak memory reduction | x | y | y - x | ✓ / ✗ |

## 4. Verdict

<One sentence. Either:
 - "Reproduced within 2% on all reported metrics."
 - "Reproduced within 2% on quality metrics; throughput not reproduced because
   <hardware reason>."
 - "Did NOT reproduce: gap of X% on metric Y, attributed to Z."

The "did not reproduce" verdict is fully acceptable if the explanation in §5 is solid.>

## 5. Where the gap lives (if any)

<This section is the actual learning outcome of the capstone. If §4 says "reproduced,"
this section is "what surprised me along the way." If §4 says "did not reproduce,"
this section is the diagnostic walk-through.

Diagnostic walk-through pattern:
 1. Observed: <PPL gap 0.45 on Llama-3-8B; paper reports 0.18 on Llama-2-7B>
 2. Hypothesis 1: implementation bug. Test: <ran on Llama-2-7B, got 0.20 → within 2% of paper>.
    Rules out implementation bug; gap is model-dependent.
 3. Hypothesis 2: GQA differences. Llama-3 uses GQA (8 KV heads vs 32 attention heads);
    Llama-2 uses MHA (32 KV heads). Test: <measured per-channel K outlier distribution on
    both models; Llama-3-8B's GQA K cache shows higher per-channel concentration of outliers,
    consistent with the gap>.
 4. Recommended extension: <KIVI at 2-bit should add per-channel outlier isolation
    (KVQuant-style sparse path) for GQA architectures, or use 3-bit instead of 2-bit
    on GQA models>.

A diagnostic walk-through this clean is publishable as a workshop note.>

## 6. Implementation notes

<Bullet list of non-trivial implementation decisions, each with rationale:
 - "Used HF transformers Cache subclass instead of monkey-patching attention because
   <reason>."
 - "Group size 32 along token axis; aligned to PagedAttention block size 16 by maintaining
   a 16-element streaming buffer + 16-element packed past."
 - "Pre-RoPE quantization required modifying _attention_forward, not just KV storage."
 - "k-means codebook trained for 50 iterations (paper says ≥20); convergence verified at 30.">

## 7. What the paper did not say

<This is where the field actually advances. Bullet list of things you discovered during
reproduction that the paper didn't fully document:
 - "Calibration set sensitivity: 128 vs 64 sequences changed PPL by 0.05; paper says only
   '128 sequences from Pile'."
 - "Initialization of k-means codebooks: uniform initialization gives 0.3 PPL worse than
   quantile-initialization; paper doesn't specify."
 - "The streaming SVD orthogonalization step is missing from Algorithm 1 but present in
   the released code."

Each of these is a reproducibility-improvement contribution to the field.>

## 8. Optional stretch (if attempted)

<Document any stretch goals attempted and their outcomes.
 - W4A4 compounding: ...
 - vLLM custom backend: ...
 - Cross-method comparison: ...>

## 9. Limitations of this reproduction

<Honest enumeration of what this memo does NOT claim:
 - "Throughput not measured: emulated FP4 on Ampere, not native on Blackwell."
 - "Did not test on multi-turn chat distribution; eval is on Wikitext / C4 / academic benchmarks."
 - "Single-seed runs on long-context (compute budget)."
 - "Did not reproduce on 13B / 30B / 70B scales.">
```

---

## Worked example (excerpted)

The following is a sample memo for a KIVI reproduction on Llama-3-8B that *did not* match the paper exactly — and the §5 walk-through that turned it into a publishable workshop note.

```markdown
# Reproduction Memo — KIVI on Llama-3-8B

**Method:** KIVI
**Paper:** Liu et al., ICML 2024 (arXiv:2402.02750)
**Base model:** meta-llama/Llama-3-8B
**Hardware:** 1 × H100 80 GB

## 3. Headline numbers

| Metric | Paper | This reproduction | Gap | Within 2%? |
|--------|-------|-------------------|-----|-----------|
| Wikitext-2 PPL gap (INT2 KV) | 0.18 | 0.45 ± 0.03 | +150% | ✗ |
| LongBench avg delta | -0.9 pp | -2.4 pp | -1.5 pp | ✗ |
| NIAH @ 32K @ 50% depth | 95% | 78% | -17 pp | ✗ |
| Throughput vs FP16 | 2.6× | 2.4× | -7.7% | borderline ✗ |

## 4. Verdict

Did NOT reproduce on Llama-3-8B. KIVI at INT2 KV holds Llama-2-7B (control reproduction
confirms 0.20 PPL gap, matching paper's 0.18 within 2%) but breaks down on Llama-3-8B.
The root cause is GQA's smaller KV-head count amplifying per-channel outlier concentration.

## 5. Where the gap lives

1. Observed: PPL gap 0.45 on Llama-3-8B; paper reports 0.18 on Llama-2-7B.
2. Hypothesis 1: implementation bug. Test: ran the same code on Llama-2-7B; got 0.20 PPL gap.
   Within 2% of paper → not an implementation bug.
3. Hypothesis 2: Llama-3-8B's larger vocabulary (128K vs 32K) inflates per-token PPL.
   Test: re-computed PPL using Llama-2's tokenization on the same text; still 0.40 gap → not vocabulary.
4. Hypothesis 3: GQA differences. Llama-3-8B uses GQA (32 attention heads, 8 KV heads, group=4);
   Llama-2-7B is MHA (32 attention heads, 32 KV heads). Test: measured per-channel K outlier
   distribution by head:
   - Llama-2-7B: max per-channel scale / median per-channel scale ≈ 15× per head.
   - Llama-3-8B: max per-channel scale / median per-channel scale ≈ 80× per head (5× more concentrated).
   Each Llama-3 K head is shared across 4 attention heads, so the outlier channels carry
   the union of outliers from all 4 attention heads. Per-channel INT2 cannot represent
   80× dynamic range without crushing the bulk.
5. Diagnostic confirmation: re-ran KIVI with the top-0.5% K-channel elements held in FP16
   (à la SpQR/KVQuant dense-and-sparse). Result: PPL gap 0.21 ± 0.02, within 2% of paper's
   Llama-2-7B number.
6. Recommended extension: KIVI at INT2 should add a dense-and-sparse path for GQA architectures,
   bringing it close to KVQuant for these models. Alternative: target INT3 instead of INT2
   on GQA models (PPL gap was 0.22 at INT3 in our measurements).

## 7. What the paper did not say

- The KIVI paper's evaluation is on Llama-2 / Falcon / Mistral, all MHA. GQA's per-channel
  outlier concentration is qualitatively different. The paper's "tuning-free, drop-in"
  claim does not transfer to GQA models at INT2.
- The streaming chunk size (g=32) needed adjustment on Llama-3-8B's KV layout. With
  PagedAttention block size 16, the per-channel scales refresh on a 32-token boundary
  that does not align with the 16-token attention block. We maintained a separate
  FP16 residual buffer of 16 tokens to bridge this; the paper's implementation appears
  to assume a 32-token PagedAttention block.

## 9. Limitations

- Single hardware platform (H100); did not test on Ampere or MI300X.
- Did not test on 30B+ models.
- LongBench evaluated at 16K context only; paper used 32K. Our 32K runs OOM'd on
  H100 80 GB at our chosen batch size; reducing batch size to 1 was not done due to
  wall-clock budget.
```

This memo *did not reproduce the paper* — and is a more valuable contribution because it identifies a real, reproducible gap in the published method's transfer to GQA architectures. A workshop note titled "KIVI Breaks on GQA: A Reproduction Study" would be acceptable at venues like the MLSys Workshop on Quantization with this content.

---

## Connections

- [[ch-22]] §the-reproduction-workflow — chapter section.
- [[ch-22]] §what-reproduction-means — the framing this memo template enacts.
- [[ch-20]] §6 — the evaluation harness the memo summarises.

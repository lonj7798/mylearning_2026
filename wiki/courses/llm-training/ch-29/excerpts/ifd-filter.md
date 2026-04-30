---
chapter: ch-29
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ifd.md + wiki/raw-data/llm-training/papers/cherry-llm.md
source_url: https://arxiv.org/abs/2308.12032
created_at: "2026-04-23"
---

# Excerpt: IFD — the difficulty filter ch-29's Stage 2 implements

**Source library:**
- `wiki/raw-data/llm-training/papers/ifd.md` (metric definition)
- `wiki/raw-data/llm-training/papers/cherry-llm.md` (pipeline)

**Artifact:** IFD = PPL(a|q) / PPL(a), warm-up → score → top-K

---

## Why this source anchors ch-29

After format and before dedup, ch-29 needs a filter that quantifies *which instructions actually help their responses*. [[alpagasus]] does this with a teacher rating and a threshold; [[deita]] with a complexity × quality scorer; [[cherry-llm]] does it with one number computable from two forward passes of the target model. The key property: **no external judge, no teacher, no label cost**. For a lab on a $3–$20 API budget this is the right trade; the stronger [[deita]] pipeline is for production.

---

## The metric, stated exactly

From [[ifd]] (lines 15–18):

```
PPL_cond(a | q) = exp( -1/|a| · Σ_t log p_M(a_t | q, a_<t) )
PPL_uncond(a)   = exp( -1/|a| · Σ_t log p_M(a_t |    a_<t) )
IFD(q, a)       = PPL_cond(a | q) / PPL_uncond(a)
```

Interpretation bands (lines 22–26):
- `IFD < 1` — `q` reduces response uncertainty (good, informative).
- `IFD ≈ 1` — `q` and `a` effectively independent (drop).
- `IFD > 1` — `q` *hurts* predictability of `a` (likely mismatched or pathological, drop).
- The hardest-but-valid samples cluster at IFD just below 1 — the "cherry" samples.

Ch-29's `ifd_filter` implements exactly this: filter `< 1`, sort by score desc (highest first within the valid band), keep top `k_frac`.

---

## The warm-up step — why it is non-optional

From [[cherry-llm]] (line 38): *"Warm-up step: fine-tune target LM on a small random subset (~1K) for 1 epoch to calibrate perplexity estimates."*

A vanilla base model's IFD scores are uncalibrated because the model has not seen the chat template. Skipping warm-up is the #1 way to produce IFD histograms that center above 1 across the entire pool — not because the pool is bad, but because the scorer has not adapted to its own prompt format.

Ch-29's `ifd_filter_check.py` acceptance gate (gate 2) is specifically for this failure mode: if your histogram centers above 1, go back and warm up.

---

## Practical rules from the source

From [[ifd]] (lines 36–40):

- Use BF16/FP16 — IFD is precision-stable.
- Batch-size 1 works; prefer right-padding for conditional PPL.
- **Use the same tokenizer & system prompt in both passes** — otherwise the ratio is biased.
- For very long responses, consider length-normalizing separately per PPL.

The "same tokenizer" rule is why ch-29's `neg_log_prob` takes `tok` as a parameter and is called twice with the same object — if you accidentally tokenize the `PPL_uncond` pass without a BOS while the `PPL_cond` pass has one, every IFD score is shifted by a constant factor.

---

## Why top 10–15% specifically

From [[cherry-llm]] (lines 19, 46): keeping the top 10% beats the full set; below 5% loses coverage. Ch-29's default `keep_frac=0.15` is the high end of this band, chosen because the synthetic pool is noisier than the public Alpaca pool Cherry-LLM was calibrated against — a slightly larger keep fraction gives dedup more substrate to work with.

---

## A known limitation, relevant to ch-29

From [[cherry-llm]] (line 51): *"IFD does not measure factual correctness — only conditioning signal; incorrect but hard responses can score highly."*

This is precisely why ch-29's Stage 3 verifier gate exists on the reasoning and tool-call subsets. IFD will happily keep a confidently-wrong math answer if the question and answer are tightly coupled; the exact-match or execution check in Stage 3 catches it. The two stages are complementary, not redundant.

---

## What ch-29 keeps, changes, drops

| Cherry-LLM / IFD default | Ch-29 choice | Reason |
|--------------------------|--------------|--------|
| Warm-up: 1K random, 1 epoch | Same | attested; cheap |
| `keep_frac`: top 10% | `keep_frac=0.15` | slightly larger pool feeds dedup + verifier |
| Metric scored once | Same | single pass through the pool |
| Score on target model | Same | ch-29's ~1B SFT base is the scorer too |
| Alternative: Superfiltering (0.5B proxy) | Documented but not used | resource-constrained path could swap in for speed |

---

## Connections

- **ch-23** — the full-read chapter on [[cherry-llm]] / [[ifd]].
- **ch-24** — [[deita]] is the stronger three-axis alternative.
- **ch-22** — [[lima]] is the baseline ch-29 matches size against.
- **ch-29 §4** — the IFD-removed ablation is the clearest test of whether difficulty-selection is doing work on *your* synthetic pool.

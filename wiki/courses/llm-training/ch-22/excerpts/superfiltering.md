---
chapter: ch-22
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/superfiltering.md
source_url: https://arxiv.org/abs/2402.00530
created_at: "2026-04-23"
---

# Excerpt: Superfiltering — weak-to-strong IFD transfer

**Source library:** `wiki/raw-data/llm-training/papers/superfiltering.md`
**Paper:** Li et al. 2024, "Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning" (ACL 2024).

---

## Why this source anchors ch-22

Superfiltering is the empirical claim that makes IFD industrially practical at large-pool scale. Without it, IFD requires scoring the target model over the entire pool — a cost that grows linearly with pool size and model size. Superfiltering shows that a 125M-parameter proxy (GPT-2) produces IFD *rankings* that are highly correlated (Spearman ρ) with the target model's IFD rankings. Since selection needs only the ranking and not the absolute values, you can filter with the tiny proxy and train with the large target — ~20× cheaper filtering.

The ch-22 reading: **rank-stability across scale is a property of the IFD metric, not an accident**. IFD is picking up a data-intrinsic signal that is mostly invariant to the scoring model's absolute capability. This is what makes it deployable.

---

## The rank-correlation claim

From the source (lines 14-15):

> The authors measure Spearman's ρ between GPT-2 IFD rankings and Llama-2-7B IFD rankings on Alpaca/WizardLM — the correlation is high.

The paper does not publish a single universal ρ — it varies by dataset and by tokenization alignment between proxy and target. The headline: across the tested pairs, the top decile of proxy-IFD and top decile of target-IFD overlap enough that the downstream SFT result is indistinguishable.

This is a *rank* correlation, not a value correlation. GPT-2's absolute IFDs are not comparable to Llama-2-7B's absolute IFDs. The ordering is what transfers.

---

## The 20× speedup — where it comes from

From the source (lines 15, 40-43):

- GPT-2-125M: ~200× fewer parameters than Llama-2-7B.
- Inference speed scales approximately with parameter count at batch-size 1.
- Two forward passes per sample (IFD_cond + IFD_uncond) are the per-sample filtering cost.
- Net: roughly 20-50× filtering-time speedup in practice, after accounting for I/O and batch effects.

The subtle operational win: the proxy fits on one GPU and scores the pool in a single node-hours-to-wall-clock-hours regime, while the target model requires multi-GPU just to run at batch=1. Superfiltering moves filtering from the multi-node job queue to a laptop-plus-one-GPU job.

---

## The family-mismatch failure mode

From the source (lines 45-48):

> Weak-proxy family mismatch: transferability depends on the proxy having a plausibly similar tokenizer + capability profile; using a domain-specific proxy can skew selection.

Two concrete failures to guard against:

1. **Tokenizer mismatch** — if the proxy tokenizes numbers / code / Chinese differently from the target, the `|a|` token count differs, and the per-token normalization in PPL differs, and IFD rankings drift. Use a proxy with the same tokenizer family if possible.
2. **Capability mismatch** — a code-specialist proxy scoring an English-chat pool will over-weight code-adjacent responses. Match proxy domain to pool domain.

The practical advice: GPT-2 is a good default proxy for English chat; Qwen-0.5B is a better default for multilingual / code-including pools.

---

## Why rank-stability is a metric property

The deeper point ch-22 makes via Superfiltering: not every selection signal is rank-stable across scale. [[less]]-style gradient-similarity is *less* rank-stable (gradients diverge in geometry as capability scales). [[alpagasus]]-style LLM-rating is stable only if proxy and target raters agree (which is a separate alignment question).

IFD's stability traces to its construction: the *ratio* of two perplexities normalizes away the absolute-capability term, leaving behind a data-intrinsic quantity. Changing the proxy changes both numerator and denominator by the same scale factor (approximately), so the ratio survives.

---

## Pipeline

From the source (lines 34-38):

1. Choose weak proxy (GPT-2-125M or Qwen-0.5B).
2. Warm proxy on ~1K random subset for 1 epoch.
3. Compute IFD for all pool samples with proxy.
4. Keep top ~15% by IFD.
5. Train target (7B+) on selected subset.

Same as Cherry-LLM except steps 2-3 run on the small proxy.

---

## Connections

- **[[ch-22]]** §4 — the rank-stability slot.
- **[[ifd]] / [[cherry-llm]]** — the signal being scaled down.
- **[[deita]]** — alternative approach that needs a dedicated 13B scorer rather than a weak proxy.
- **[[less]]** — contrast: gradient-similarity is not as rank-stable across scale.
- **[[prismatic-synthesis]]** — later 2025 work also relies on proxy-model rank-stability for gradient-based filters.

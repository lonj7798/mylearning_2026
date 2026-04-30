---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ultrafeedback-construction.md
source_url: https://arxiv.org/abs/2310.01377
created_at: "2026-04-23"
---

# Excerpt: UltraFeedback — 4-aspect rubric and preference-sampling pipeline

**Source library:** `wiki/raw-data/llm-training/papers/ultrafeedback-construction.md`
**Year / authors:** 2023/2024 / Cui, Yuan, Ding et al. (Tsinghua + OpenBMB).

---

## Why this source anchors ch-15

UltraFeedback replaces the human labeler with GPT-4 and replaces the two-axis HH rubric with a *four*-aspect rubric. It is the canonical 2023–2024 synthetic-preference pipeline, and therefore the canonical counterexample to ch-15 §5's "when humans override judges" rule. For preference data where the rubric is sufficiently precisely written that GPT-4's ~80% agreement ceiling ([[judge-llm-bias]]) is tolerable, UltraFeedback-style pipelines dominate on cost and scale. Ch-15 §4's close-pair mining and §5's decision table are both direct generalizations of UltraFeedback's design decisions.

---

## The prompt sourcing — diversity by construction

```
# ultrafeedback-construction.md, §3.1
Prompt pool: 64K prompts from six heterogeneous sources:
  UltraChat       — synthetic multi-turn,
  ShareGPT        — real user logs,
  Evol-Instruct   — complexity-evolved,
  TruthfulQA      — truthfulness stress,
  FalseQA         — intentionally false premises,
  FLAN            — instruction diversity.

Post-dedup and length/quality filter.
```

This is the prompt-curation side of a preference-data pipeline, which ch-15 §4 assumes as given. The UF choice to *deliberately* mix six sources of different shape (synthetic vs real, easy vs adversarial) is the sampling-policy analogue of what [[tulu-3-sft-mix]] later does at 939K scale with skill-specific submixes. Both are responses to the observation that **a preference dataset's downstream usefulness is bounded by the prompt distribution's match to the target policy's deployment distribution**.

---

## The response fleet — 17 models, deliberate quality spread

The central design innovation. For each prompt, 4 out of 17 models generate responses, with the fleet spanning capability tiers:

| Tier | Models |
|---|---|
| Top | GPT-4, GPT-3.5-turbo |
| Open strong | Llama-2-70B-chat, WizardLM-70B, Vicuna-33B |
| Mid | Llama-2-13B/7B-chat, MPT-30B-chat, Falcon-40B-instruct |
| Weak | Alpaca-7B, Pythia-12B, StableLM, Dolly, Starchat |

The reason for the deliberate spread is ch-15 §4's observation in reverse: **you want the annotator (GPT-4 as judge, here) to see a preference gap that is *informative*, not trivially resolved**. A pool of four GPT-4 responses gives near-zero variance; a pool of four Dolly responses gives high variance but low quality. Mixing creates a distribution of gaps that spans the informative band.

For on-policy preference data — which [[tulu-3]] and Llama-3's RSFT loop prefer — the UltraFeedback fleet structure breaks: you can't generate from 17 models and call it on-policy to one. The tradeoff is explicit in ch-15 §4: off-policy pairs are cheaper at the generation stage but carry less signal for policy gradient; on-policy pairs are more expensive but generalize better. UltraFeedback chose the off-policy side for cost reasons, and the result is that UltraFeedback-trained RMs are reused across many target policies with only modest loss of fit.

---

## The 4-aspect rubric — annotation granularity

```
# ultrafeedback-construction.md, §3.3
For each (prompt, response), GPT-4 produces:
  instruction-following (0–10) + natural-language rationale
  truthfulness         (0–10) + rationale
  honesty              (0–10) + rationale   # "did it admit uncertainty?"
  helpfulness          (0–10) + rationale
Overall score optionally derived from aspect averages.
```

Compare to HH-RLHF's single-axis preference: UltraFeedback explodes the single scalar into four aspects, each labeled separately with a 0–10 ordinal scale. For ch-15 §2 (agreement metrics), this switches the appropriate statistic from κ (binary/nominal) to **Krippendorff α with ordinal distance** — a "7 vs 8" on truthfulness is a smaller disagreement than a "2 vs 9".

Three consequences.

**One**, binarization is flexible. Per-aspect binarization produces `(chosen_on_truthfulness, rejected_on_truthfulness)` pairs, targeting truthfulness-specific training. Overall-score binarization produces generic preference pairs. HuggingFace's `ultrafeedback_binarized` uses overall with `chosen = argmax, rejected = random-not-chosen` (*intentionally not worst-of-4, to avoid degenerate negatives*) — ch-15 §4's close-pair-mining principle baked into the release.

**Two**, aspect-conflation is real. The four aspects correlate strongly in practice (truthful responses tend to be helpful); an overall-score binarization throws away the differentiation. The paper's ablation shows **aspect-level prefs outperform overall-score prefs when target capability is specific** — truthfulness-focused DPO works better on TruthfulQA when trained on truthfulness pairs vs overall pairs.

**Three**, the rubric is the prompt given to GPT-4. The paper releases the exact aspect-rubric text — and that prompt has become one of the most-reused artifacts in open alignment. The rubric as prompt-template is a concrete instantiation of ch-15 §1's "the rubric is the product."

---

## The judge-bias inheritance — what UF's rubric cannot fix

From the paper's own risks section:

> Judge-induced bias: GPT-4 scoring patterns (length bias, helpfulness bias, style bias) propagate into downstream DPO models.
> Model-fleet contamination: GPT-4 responses in the generator pool mean the judge is rating its own outputs — subtle advantage to GPT-4 responses.

This is the quantitative case for ch-15 §5's "when humans override judges." UltraFeedback's 80%-agreement judge is cheap and scales; the 20% where it misses is concentrated on safety-critical and self-enhancement-biased queries. The mitigation is to *not* use UltraFeedback for those slices — hence [[tulu-3-sft-mix]]'s 50K WildJailbreak + Tulu-3-Safety from human-labeled and red-team sources, while leaving chat / math / code to UF-style synthetic preferences.

---

## Connections

- [[excerpts/judge-llm-bias]] — the quantitative bias catalog that UF inherits.
- [[excerpts/hh-rlhf]] — the human two-axis counterpart UF generalizes.
- [[excerpts/tulu-3-sft-mix]] — the 2024 consumer that mixes UF-style synthetic preferences with human-labeled safety data.
- [[excerpts/rlhf-instructgpt]] — the K=4-9 ranking ancestor; UF keeps the K (=4) but drops the K-way ranking in favor of per-response aspect scoring.
- [[ch-15]] — this excerpt supports §1 (rubric-as-prompt), §2 (ordinal α vs κ), §4 (close-pair mining via aspect gaps), §5 (when to use synthetic vs human).

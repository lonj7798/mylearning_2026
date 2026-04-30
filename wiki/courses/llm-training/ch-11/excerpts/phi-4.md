---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/phi-4.md
source_url: https://arxiv.org/abs/2412.08905
created_at: "2026-04-23"
---

# Excerpt: Phi-4 — textbook-style tokenizer choices and the `<think>` extension case study

**Source library:** `wiki/raw-data/llm-training/model-reports/phi-4.md`
**Papers:** Abdin et al. 2024/2025, "Phi-4" + "Phi-4-reasoning" (Microsoft Research).

---

## Why this source anchors ch-11

Phi-4 is the clearest small-model case study for ch-11's tokenizer-extension argument (§5). At 14B parameters with synthetic-heavy training and explicit reasoning-trace structure (`<think>` blocks), Phi-4 sits at the intersection of: (a) vocab-cost-vs-capacity tension (§1), (b) tokenizer extension for structural tokens (§5), (c) textbook-style data as a tokenizer-merge-distribution shaper (§1).

Ch-11 cites Phi-4 three times: for small-model vocab sizing (§1), for the `<think>`-extension case (§5), and for the 50-synthetic-category data breakdown (§1 implied). This excerpt makes those three points concrete.

---

## Phi-4's data-first philosophy — and what it does to the tokenizer

From the source (line 16):

> Phi-4 (14B) established the Microsoft line's "data-first" post-training philosophy: ~400B unweighted tokens of synthetic data across 50 types, combined with rejection sampling and a novel "pivotal-token DPO" where preference pairs are constructed around tokens that matter for final correctness.

The phrase "400B unweighted tokens of synthetic data across 50 types" is the tokenizer-side concern. When 50 synthetic categories each have their own structural conventions (textbook exposition, multi-choice problem format, code + docstring, proof with labeled steps, dialogue with speaker tags), the tokenizer's merge table learns category-specific merges.

Consider: a textbook corpus has high-frequency merges for `Theorem `, `Proof.`, `\begin{equation}`, `∎` (QED glyph), numbered-example prefixes like `Example 3.2:`. A forum / web corpus has none of these. A 32K vocab trained on a 50/50 mix of textbook + web spends dozens of merge slots on textbook patterns that the final base model under-uses if Phi-4 is deployed mostly for chat. The co-design point ch-11 §1 makes: **the tokenizer training corpus should match the final deployment distribution, not the pretraining distribution**.

Phi-4's tokenizer isn't explicitly disclosed in the source (line 38 flags missing hyperparameters: "LR, clip ε, optimizer — not disclosed"), but the 14B-parameter class + synthetic-heavy training + reasoning extension suggests a ~100K vocab is operationally appropriate — larger than the 32K of Phi-3 era to cover the synthetic-category diversity, smaller than Llama 3's 128K because the model budget is 14B and the embedding-table parameter fraction becomes relevant.

---

## The `<think>` extension — the canonical post-pretrain tokenizer case

From the source (line 21):

> **Phi-4-reasoning RL:** GRPO with rewards that combine correctness (+1 / −0.5) with a length-aware accuracy bonus and explicit penalties for (a) missing `<eos>`, (b) unclosed `<think>` blocks, (c) n-gram repetition with n=5.

And line 26:

> **SFT data:** Phi-4-reasoning — 1.4M prompts; ~16B tokens total, ~8.3B unique. Long CoT traces from o3-mini (high thinking, 32K). Domains: STEM, coding, safety. Prompts filtered to boundary of base-model competence (not too easy, not impossible).

Phi-4-reasoning introduces `<think>` and `</think>` as special tokens wrapping the reasoning trace. These tokens must be *added to the tokenizer* between Phi-4 base and Phi-4-reasoning. Ch-11 §5 treats this exact case — and makes two claims:

1. **The naive way (add tokens, default-init new embeddings to `N(0, 0.02)`) drifts the chat template for ~1000 SFT steps.** Signs: the `<think>` embedding is 30× smaller than the median existing embedding; attention to the `<think>` position is near-uniform-minus-epsilon; the model effectively ignores the thinking boundary until its embedding grows.

2. **The right way** is mean-of-neighbors init: decompose `<think>` via the old tokenizer (likely → `<`, `think`, `>`), average those embeddings, use as the new token's initial embedding. L2 norm is immediately in the right regime; semantic prior is reasonable.

The Phi-4 report does not explicitly disclose which recipe was used. The **fact that Phi-4-reasoning achieves strong AIME results in only 90 GRPO steps** (line 24: "only 90 GRPO steps produce >10% AIME gain") is circumstantial evidence that the tokenizer-extension drift was not a significant issue — either the mean-of-neighbors init was used, or the SFT stage before RL ran long enough (1.4M prompts, ~16B tokens) to overwhelm the drift.

Ch-11 §5 generalizes the principle: **either use reserved slots (Llama 3 pattern) or mean-of-neighbors init (Phi-4 pattern, implicit)**; never default-initialize post-pretrain special tokens.

---

## The length-aware reward as a tokenizer-structural claim

From the source (lines 31-36):

> - **Reward shape (Phi-4-reasoning GRPO):**
>   - +1 for correct answer, −0.5 for incorrect.
>   - Length-aware bonus: encourage concise outputs on correct answers; permit more think tokens on incorrect (model learns to "think longer when unsure").
>   - Penalty for missing EOS or unclosed `<think>` block.
>   - n-gram repetition penalty with n=5 — discourages degenerate loops.

Three of these rewards are *directly tokenizer-coupled*:

- **Missing EOS penalty.** Requires the tokenizer to have an unambiguous EOS token, and requires the generation loop to emit it correctly. If the tokenizer's EOS is a reused reserved-slot (as in Llama 3), the identity is stable across Phi-4 base and Phi-4-reasoning. If the tokenizer was extended with a new EOS, the new-embedding drift from §5 applies.
- **Unclosed `<think>` penalty.** Requires `<think>` and `</think>` to be *single tokens*, not decomposed. This in turn requires that they were in the tokenizer vocab when training data was tokenized — otherwise the SFT data has `<`, `think`, `>` as three separate tokens, and the GRPO reward cannot detect "unclosed block."
- **n-gram repetition penalty with n=5.** Computed in *token space*, not character space. Tokenizer-dependent: what looks like a 20-character repetition under BPE might be 3 tokens or 10 tokens depending on the vocabulary. A BPE that merges aggressively on common filler phrases produces fewer repetition events in n=5 token-space.

The operational point for ch-11: **the tokenizer decisions in §1 propagate to the RL reward design**. You cannot tune a length-aware reward independently of tokenizer choice. Phi-4's reward works because Phi-4's tokenizer is stable across base → reasoning.

---

## Boundary-of-competence prompt filtering — a lineage operation

From the source (line 26):

> - **SFT data:** Phi-4-reasoning — 1.4M prompts; ~16B tokens total, ~8.3B unique. ... Prompts filtered to boundary of base-model competence (not too easy, not impossible).

This is a lineage-attribute operation with a specific shape. For each candidate prompt in the ~several-million pool:

1. Run the Phi-4 base model on the prompt K times (typical K=8 or K=16).
2. Compute `pass_rate = mean(1[correct])`.
3. Keep prompts with `pass_rate ∈ [low, high]` — not too easy (`pass_rate > 0.9`), not impossible (`pass_rate < 0.1`).

The `pass_rate` is an attribute of the prompt; it's the textbook application of ch-11 §3's "attribute-as-filter" pattern. The filter is *consumer-side*: the same prompt pool can be filtered at different boundaries for different training runs (a harder SFT mix uses `pass_rate ∈ [0.05, 0.3]`; an easier one uses `[0.3, 0.7]`).

Phi-4's discipline — prompts filtered *once* against the base model, attribute stored, then consumed by SFT — is the same attribute-file pattern [[excerpts/dolma]] uses for quality scoring. The consumer can re-filter six months later with a different threshold without regenerating the `pass_rate`.

---

## 50 synthetic categories — a tokenizer-merge diagnostic

From the source (line 19):

> **Phi-4 base:** 50 categories of synthetic data, ~400B unweighted tokens, injected into both pretraining and post-training.

Line 61 flags that the 50-category list is not publicly disclosed. Operationally, 50 named categories implies 50 category-identifying tokens (or at least 50 structured prompts) — either as system-prompt prefixes like "Category: MathProof" or as literal special tokens like `<|category_math_proof|>`. If the latter, the vocab must accommodate 50 additional tokens on top of the baseline.

This is another ch-11 §1 reserved-slot case: a 14B model with 50 category tokens needs 50 reserved slots in its pretrain vocab, or it needs to extend post-pretrain (with the §5 init recipe). The reserved-slot approach is operationally cheaper and more robust — consistent with Phi-4's data-first philosophy where pipeline complexity is justified by data investment.

---

## What Phi-4 does not disclose that ch-11 wishes it did

From the source (lines 61-63):

> ## Gaps / what the report does NOT disclose
> Search-surfaced summary withholds: GRPO group size G, clip ε, KL coefficient, learning rate, batch size, rollouts per prompt, AdamW betas, exact prompt-filter thresholds, RM used (if any) for boundary selection, 50 synthetic-category list, pivotal-token detection algorithm.

Not disclosed but relevant to ch-11:

- **The exact vocab size.** Whether Phi-4 uses 32K, 50K, or 100K has direct downstream implications for the 14B model's embedding-parameter fraction.
- **The initialization recipe for `<think>` and `</think>`.** Default init vs mean-of-neighbors — evidence either way would settle the §5 "which recipe do frontier labs actually use" question.
- **The 50-category naming scheme.** Are categories injected as system-prompt strings or as special tokens?
- **Whether rejection-sampling data shards are separate from pretrain shards**, or re-routed through the same pipeline with an attribute flag.

These gaps reflect a broader frontier-lab trend: **operational details are the last thing to be disclosed**, because they are where the competitive advantage often lives. Ch-11's "guideline" rules attempt to fill the gaps with the best-available-public-inference from comparison with OLMo 2/3 and Llama 3.

---

## What to take from Phi-4 for ch-11

1. **Small models (14B) can't afford frontier vocabs without a budget hit**; Phi-4's implied ~100K vocab is the compromise between synthetic-category diversity and embedding-parameter cost.
2. **The `<think>`-extension case** is a textbook ch-11 §5 scenario; the choice of init recipe (reserved-slot vs mean-of-neighbors vs default) determines whether the chat template drifts.
3. **RL reward design is tokenizer-coupled** — EOS detection, unclosed-block penalty, and n-gram repetition are all token-space operations.
4. **Boundary-of-competence filtering** is another lineage attribute; the same pool serves multiple training runs with different threshold queries.
5. **50 synthetic categories** imply 50 additional special-token slots, best served by pretrain-time reserved slots, not post-hoc vocab extension.

---

## Connections

- [[excerpts/llama-3]] — the reserved-slot discipline Phi-4 likely inherits implicitly (not explicitly documented).
- [[excerpts/olmo-2]] — Tulu-3 chat template + special-token conventions; OLMo 2 is similarly implicit on init recipe.
- [[excerpts/olmo-3]] — Think / Instruct / RL Zero branches are the multi-branch analog of Phi-4 base → Phi-4-reasoning.
- [[excerpts/dolma]] — attribute-as-filter pattern Phi-4's prompt-filtering realizes.
- [[ch-11]] — §1 (vocab size for small models; tokenizer training corpus matching deployment distribution), §5 (the `<think>`-extension case; mean-of-neighbors init).

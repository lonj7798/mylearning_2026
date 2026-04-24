---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/qwen-3.md
source_url: https://arxiv.org/abs/2505.09388
created_at: "2026-04-23"
---

# Excerpt: Qwen3 — 36T tokens, 119 languages, and the "synthetic data as pretraining ingredient" pattern

**Source library:** `wiki/raw-data/llm-training/model-reports/qwen-3.md`
**Paper:** Qwen Team 2025, "Qwen3 Technical Report" (Alibaba).

---

## Why this source anchors ch-09 §4

Qwen3's technical report is the 2025 canonical example of *partial-disclosure with synthetic data as a pretraining ingredient*. 36 T tokens across 119 languages, three-stage pretraining curriculum with stage-level budgets, explicit naming of synthetic-data generators — but no source-mix percentages and no quantification of the synthetic slice. It sits between [[llama-3]] (token-total-only) and [[olmo-3]] (staged curriculum fully disclosed) on the disclosure gradient.

For ch-09 §4's "2023 → 2026 shift to web+code+synthetic" narrative, Qwen3 is the canonical demonstration of the synthetic axis. This excerpt walks through the multilingual scale, the three-stage curriculum, and the synthetic-data pipeline that the report names without quantifying.

---

## The 36T, 119-language headline

From the source (lines 19-21):

> - Large multilingual pretraining: **36T** tokens across **119** languages and dialects.
> - A three-stage pretraining curriculum with a dedicated **reasoning stage** and a **long-context stage**.
> - Strong-to-weak distillation for smaller models, with the report explicitly noting that distillation outperforms RL in both quality and efficiency for those models.

Two numbers to internalize:

- **36 T tokens**: more than 2× Llama 3's 15.6T and 2.5× DeepSeek-V3's 14.8T. Qwen3 is at the 2025 frontier for pretraining-scale disclosure (Qwen3.5 at 2026 stops disclosing even the token count).
- **119 languages**: this is the distinctive multilingual claim. [[llama-3]] natively supports ~8 languages. [[deepseek-v3]] is English+Chinese-dominant. Qwen3's 119 is the serious multilingual claim of the 2025 open-weight frontier.

For ch-09 §1's "what CC omits" point: CC is English-dominant (roughly 45% English, 20% other-European, rest tail). To get to 119 languages at 36T tokens, Qwen3 must have:
- Over-sampled the long-tail language slices of CC (oc-CC / oscar-CC / CCMatrix-style extraction).
- Added Alibaba's internal multilingual corpora.
- Generated synthetic parallel data.
- Included transcription/translation output.

None of this is quantified in the report. The token-by-language distribution is not disclosed.

---

## The three-stage pretraining curriculum

From the source (lines 45-49):

> ### Three-stage pretraining
> 1. **General stage:** over **30T** tokens at sequence length **4096**.
> 2. **Reasoning stage:** about **5T** higher-quality tokens with more STEM, coding, reasoning, and synthetic data, still at **4096**.
> 3. **Long-context stage:** hundreds of billions of tokens at **32768** sequence length, later supporting longer inference contexts.

Stage-level budgets disclosed: 30T + 5T + ~0.3T (ballpark) = ~35.3T, matching the 36T headline. **But the composition within each stage is not disclosed.**

What you can infer from the stage boundaries:

- **Stage 1 (30T)** is the *bulk* slice, where most of the 119-language coverage lives. Composition is the "general-web + code + multilingual" mix. Percentages not disclosed.
- **Stage 2 (5T reasoning)** is the *curriculum-end* slice, analogous to FineWeb-Edu within the FineWeb stack. It's where STEM, coding, reasoning, and synthetic data concentrate. This is the most data-differentiated stage of the Qwen3 recipe.
- **Stage 3 (long-context)** is the *context-extension* slice, short in tokens (hundreds of B at 32K context), where long-document training happens. Analogous to [[olmo-3]]'s Longmino (50B long-context tokens).

The three-stage structure is the same WSD-curriculum pattern as [[olmo-3]]: broad-bulk → targeted higher-quality → context-specialization. What Qwen3 discloses is the *structure*; what it doesn't disclose is the *composition within each stage*. Compare: OLMo 3 discloses Dolma 3 Mix (5.9T), Dolmino (100B), Longmino (50B) with named datasets for each; Qwen3 discloses "30T general, 5T reasoning, long-context" with no named datasets.

---

## The synthetic-data pipeline — named, not quantified

From the source (lines 38-44):

> ### Pretraining data
> - Total of **36T tokens** across **119 languages and dialects**.
> - Data expansion includes:
>   - OCR-style text extraction from large PDF corpora using **Qwen2.5-VL**
>   - synthetic math data from **Qwen2.5-Math**
>   - synthetic code/data variants from **Qwen2.5-Coder** and related models
> - The report says the data is annotated at large scale for educational value, domain, and safety, then mixed at the **instance level** using proxy-model ablations.

This is the 2025 state of the art for pretraining-data synthetic integration, named explicitly. Three sources:

1. **Qwen2.5-VL OCR'd PDF text.** The Qwen team's multimodal model is used as an OCR backend for a large PDF corpus — academic, scientific, possibly technical-documentation PDFs. This is analogous to [[olmo-3]]'s Dolma 3 using `olmOCR` for science PDFs; the parallel is clear.
2. **Qwen2.5-Math synthetic math.** Mathematical problem-solution pairs generated by the Qwen team's math-specialized model. Volume not disclosed. Presumably most of this lands in Stage 2's 5T reasoning slice.
3. **Qwen2.5-Coder synthetic code.** Code-generation synthetic data from the team's code-specialized model. Volume not disclosed.

The pattern: *the team's earlier models become upstream data producers for the next generation.* This is the recursive-training pattern that frontier labs are now operating on. It is also the pattern that forecloses full mix disclosure, because "disclosing the mix" at this level means disclosing the generators' outputs, which are themselves proprietary model-artifact-as-data.

The "annotated at large scale for educational value, domain, and safety" clause is the [[fineweb]]-Edu classifier pattern applied inside the Qwen team. Almost certainly: a small classifier trained on labels from a larger Qwen model, used to score-filter the bulk corpus. The classifier itself is not released; the labels are not released; the filter threshold is not disclosed.

---

## Instance-level mixing via proxy-model ablations

From the source (line 44):

> […] then mixed at the **instance level** using proxy-model ablations.

This is a one-sentence disclosure of what is likely the most important data-decision system in the 2025 frontier. Instance-level mixing means: rather than "50% web, 10% code, 5% math" at batch granularity, individual training examples are scored and mixed according to their contribution to downstream metrics.

Proxy-model ablations: train a small proxy model on candidate sub-mixes, measure downstream performance on a held-out eval, pick the winning mix. This is a more rigorous version of what [[dolma]] does ablation-by-stage. The Qwen team does it per-mixture-decision.

For ch-09 §7's "how to read a pipeline critically": *"proxy-model ablations at instance level"* is the highest-discipline data-selection method currently published. The exact implementation is not disclosed, but the pattern — ablate → select — is the pattern.

---

## Disclosure gradient, applied

Ch-09 §6's four-axis framework, applied to Qwen3:

1. **Token count**: 36 T. Disclosed.
2. **Composition**: three stage-level budgets (30T + 5T + LC). Within-stage composition and synthetic volume *not* disclosed.
3. **Licence regime**: proprietary. No explicit opt-out register mention. Synthetic slice uses Qwen-team-internal generators.
4. **Disclosure granularity**: stage budgets + named-generator synthetic sources + instance-level-mix methodology.

Stronger than Llama 3 on methodology (Llama 3 doesn't say how its mix is selected); weaker than OLMo 3 on specific datasets (OLMo 3 names Dolma 3 Mix, Dolmino, Longmino explicitly).

---

## The distillation corollary — why small Qwen3 models don't need RL

From the source (lines 56-58):

> - Smaller models are improved with **off-policy and on-policy strong-to-weak distillation** from larger teacher models.
> - The report explicitly says **distillation from advanced teacher models significantly outperforms RL** for smaller models in both performance and efficiency.

This is a capability-track claim, but it ties to data: the *data* for small Qwen3 models is the larger Qwen3 model's outputs. For a 0.6B Qwen3, the "pretraining" data is overwhelmingly influenced by the 235B flagship's distillation signal.

For ch-09's composition-as-decision point: small-model training increasingly means "teacher-model outputs as the pretraining slice." The 0.6B model is not trained on 36T raw tokens; it's trained on whatever the team distilled from the 235B's logits and rollouts. The mix disclosure question becomes: *what prompts did the teacher see, and what did it generate?* Qwen3 does not answer this.

---

## What Qwen3 does not disclose

Compiled from the "Gaps" section of the source and ch-09's reading:

- Per-stage source composition (web vs code vs math vs academic vs forums).
- Synthetic slice percentage within each stage.
- Per-language token counts (you have 119 languages, but no per-language share).
- Exact educational-value classifier and threshold.
- Instance-level mix weights.
- Proxy-model ablation results.
- PII and decontamination details.
- Opt-out register / licence-safety posture.

What *is* disclosed: 36T total, three stage budgets, named synthetic generators, named stage purposes, distillation methodology for small models. This is the Qwen3 disclosure envelope.

---

## What to take from Qwen3 for ch-09

1. **36T is the 2025 open-weight ceiling for disclosed pretraining scale.** [[qwen-3-5]] at 2026 does not disclose a token count at all.
2. **119 languages is the serious multilingual claim.** Llama 3's ~8 and DeepSeek-V3's bilingual focus are on one axis; Qwen3 is on a different axis.
3. **Synthetic data is named but not quantified.** Qwen2.5-VL (OCR), Qwen2.5-Math, Qwen2.5-Coder. The percentage is the secret.
4. **Instance-level mixing via proxy-model ablations** is the 2025 methodology ceiling. Ablate-and-select, not hand-weight.
5. **Small-model training = teacher-model distillation.** Small Qwen3 is trained on outputs of large Qwen3; the "pretraining data" concept shifts.

---

## Connections

- [[excerpts/llama-3]] — the prior closed-disclosure reference; Qwen3 is marginally more disclosed (stage budgets).
- [[excerpts/olmo-3]] — the open-disclosure counterpart; Dolma 3 names datasets Qwen3 does not.
- [[excerpts/fineweb]] — the open pattern that Qwen3's "educational-value annotation" likely mirrors.
- [[excerpts/the-pile]] — the 2020 baseline; the 16× token-count expansion and disclosure collapse between Pile and Qwen3 is the chapter's shift.
- [[excerpts/dolma]] — the methodology (ablation-per-stage) that Qwen3 generalizes to instance-level mixing.
- [[ch-09]] — §4 (synthetic-in-pretraining), §5 (non-disclosure reasons), §6 (four-axis framework Qwen3 row).

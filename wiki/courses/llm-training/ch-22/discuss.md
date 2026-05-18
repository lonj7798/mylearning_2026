<!-- chapter: ch-22 — discuss transcript
     deps: [[summary]], [[qa]], [[read]]
     scope: discuss-phase exchange, condensed for verdict record
-->

# Ch-22 — Discuss Transcript (Application Probe / 5 Stages)

Single application probe at learner direction (ch-21 pattern continuation). Probe: design a *data pipeline (generation + selection + verification) for multilingual instruction-following model* targeting 50 low-resource languages (Swahili, Yoruba, Tamil, etc.) starting from Llama-2-7B base, with GPT-4/Claude as teacher and 1K human-translated golden per language available.

---

## Stage 1 — 4 ch-21 axes + 6 ch-22 axes assessment

### Initial — confusion + framework slips

- **Axis 1 (Verifier)**: "GPT-4/Claude can verify" — forgot own §2.1 limit (judge bias inheritance). After push (sub-domain math/translation/general split): math HIGH, translation MED, general LOW; overall LOW.
- **Axis 2 (Taxonomy)**: confused "structured taxonomy for top-down generation" with "what content to include". After push: needed re-derivation.
- **Axis 3 (Long-tail / Diversity)**: ⚠ two errors:
  - G-Vendi mechanism wrong ("high pass rate" — G-Vendi is set-level entropy, not per-sample threshold)
  - Embedding vs gradient direction reversed (claimed DEITA embedding > G-Vendi for low-resource, opposite of Q21's translation example)
  - After push: identified embedder *signal portability* issue (SBERT doesn't cover 50 low-resource languages)
- **Axis 4 (Substrate)**: identification accurate (Llama-2-7B English-dominant → substrate LOW for target languages), implication missing initially → after push: surfaced as *binding constraint* (selection doesn't fix substrate).

### Recovery after push

Cross-method check revealed *cascading proxy/embedder failure*:
- DEITA embedder: low-resource not covered → diversity threshold meaningless
- Prismatic frozen instruction-tuned proxy: low-resource not available → gradient kernel meaningless
- Superfiltering proxy: same family small models don't speak target languages
- AlpaGasus LLM-judge: GPT-4 weak on low-resource quality assessment

→ **All 5 methods (AlpaGasus, IFD, Superfiltering, DEITA, Prismatic) have proxy/embedder/judge availability gap in this domain**. LESS is sole exception because golden = g_val target works directly.

## Stage 2 — Selection method choice

### Initial — LESS rejected (internal inconsistency)

Claimed: "LESS only insufficient because target-aware misses OOD" → also "1K golden insufficient for g_val" → also "would overfit golden".

⚠ Internal inconsistency with own earlier framework conclusion ("OOD belongs to generation"). After push:
- LESS doesn't require large g_val (5 exemplars sufficient per paper; 1K abundant)
- LESS doesn't memorize golden (picks pool samples by gradient alignment)
- OOD argument self-invalidating (learner had just established OOD = generation responsibility)

### Final — LESS + IFD hybrid

```
Step 1: LESS (capability-targeted, narrow) — primary
Step 2: If selected pool too small → IFD top-up on remaining
```

→ Cross-method composition using two distinct selection intent axes ([[Q15]]):
- LESS = kind/target-match
- IFD = informativeness

→ Hybrid covers LESS's narrow selection + IFD's broader filter.

### ⭐ Framework extension #1 — Performance-gap-driven coarse allocation

Learner introduced **3-layer hierarchical selection**:
- **Coarse** (learner's contribution): per-language sample allocation = inverse proportional to current performance gap measured on 1K golden
- **Mid**: language stratification (workaround for embedder cascade)
- **Fine**: LESS + IFD hybrid within each language

ch-22 chapter's 6 methods all operate at fine layer. Learner adds *coarse capability-gap layer* above sample-level methods. Resembles DoReMi-style dataset mixture but with *dynamic gap-driven rather than static* allocation.

## Stage 3 — Generation strategy

### Punt + framework guidance

Learner explicit: *"I don't have clear insight"* for generation weight (golden / GLAN-expanded / bootstrap / native ratios).

Framework principles provided:
- Coverage guarantee: golden as floor
- Verifier strength → bootstrap variation ratio adjustment
- Substrate gap → generation-heavy when LOW

### Strong move — "OOD belongs to generation"

Learner self-derived chapter §9 explicit statement:
> *"If we cannot handle OOD from generation, we cannot cover OOD from selection."*

→ Framework conclusion: filters subtract not synthesize. Strong synthesis of [[ch-22]] core thesis with own design.

### Bootstrap terminology fuzzy

Learner used "bootstrap" without specifying meaning. Possibilities: statistical bootstrap (resampling, no new samples) / self-instruct variation (in-distribution) / GLAN taxonomy expansion (OOD coverage). Clarification pending; learner moved on.

## Stage 4 — Verification design

### Per sub-domain pipeline verifier

| Sub-pool | Verifier | Confirmation |
|---|---|---|
| Math | exact-match filter | ✓ |
| Translation | BLEU (caveated for complexity) | ✓ — caveat accurate (paraphrase scores low) |
| General | negative anchor (filter-side) + multi-level taxonomy validation | resolved after push |

### ⭐ Framework extension #2 — Multi-level taxonomy retention

Initial design had *internal conflict*: "filter synthetic to golden taxonomy only" would *reject GLAN-expanded OOD samples* that generation explicitly created.

After push, resolved to **multi-level taxonomy**: golden-derived categories *and* GLAN-expanded categories *both valid*, with golden as quality validation reference. Preserves OOD coverage while maintaining quality benchmark.

→ ch-21 Cosmopedia 145-cluster *audit move* (curated + expanded both retained) adapted to *verification stage*. Generation produces multi-level taxonomy; verification preserves it.

### Pipeline position — cheap-first ordering

> *"Right after the generation. Verifier is cheaper than selection. So need to do this first."*

✓ Q23 cost scaling profile applied: verifier (cheap) drops garbage → selection (expensive) operates on smaller pool. Framework-driven ordering.

### Negative anchor — filter-side chosen, mechanism unspecified

Filter-side committed but operational mechanism (embedding similarity vs pattern matching vs taxonomy class) not specified. Multilingual embedder availability gap remains unaddressed for this choice. Generation-side use (ch-20 verdict E3 framework extension) not invoked.

## Stage 5 — Failure modes (3 listed)

1. **Overfit to 1K golden** — valid (was also LESS-reject reason)
2. **Llama-2-7B substrate insufficient** — *binding constraint* identification accurate
3. ⭐ **GPT-4 weak on minor languages → generation fails** — *strong framework extension*

### ⭐ Framework extension #3 — Teacher bias inheritance

Failure mode #3 = **ch-20 axis 5 quirks inheritance generalized from judge to teacher**:
- Original (ch-20): *Judge* (AlpaGasus LLM-judge) bias → student inherits via filter
- Learner's extension: *Teacher* (generator) bias → student inherits via generation

→ Same mechanism, *different pipeline stage*. Chapter ch-22 makes judge bias explicit but *teacher bias is implicit*. Learner makes it explicit. Two stages of pipeline both subject to identical bias inheritance mechanism.

### Gaps in failure mode list (acknowledged)

Did not add additional 2 candidates surfaced by teacher (per-language imbalance with Bantu family interference, replay forgetting in English specialty subsets, embedder cascade). Learner chose to keep 3 only.

### Replay ratio (0.3 : 0.7 English : multilingual)

Defensible starting point in middle ground for new-capability addition. Reasoning behind specific 0.3 punted as "lack of knowledge" — legitimate empirical question.

---

## Verdict reasoning

**Mastery criteria met**:
1. ✅ **Application**: 6-axis framework applied to novel multilingual domain across 5 stages
2. ✅ **Diagnosis**: cascading embedder/proxy/judge availability failure identified; substrate gap as binding constraint
3. ✅ **Synthesis**: ch-19/20/21/22 frameworks all activated; ch-22 6-axis self-applied; LESS native fit identified

**Framework extensions (above-bar — 3 extensions, exceeding ch-19/20/21's typical 1-2)**:
- ⭐ E1: Performance-gap-driven coarse allocation (3-layer hierarchy above ch-22's sample-level methods)
- ⭐ E2: Teacher bias inheritance (ch-20 axis 5 generalized from judge to teacher)
- ⭐ E3: Multi-level taxonomy retention (golden + GLAN both valid as verification framework)

**Recovery pattern (correction-via-push)**:
- 4 distinct framework slips during Stage 1 (axes 1/2/3/4)
- LESS reject internal inconsistency (Stage 2)
- Taxonomy conflict (Stage 4 general sub-pool)
- All recovered cleanly on push — *no second-push needed for any*
- Suggests framework internalization at derivation level (re-derives correctly when prompted), not memorization

**Pattern recognition**:
ch-19 E6 (pass@k) + ch-20 E6 (negative anchor filter) + ch-21 E3/E4 (negative anchor generation + axis 5 dual manifestation) + ch-22 E1/E2/E3 — *fourth consecutive chapter* with framework-extension-as-mastery move. Cumulative pattern of applying frameworks to novel domains AND producing new use cases beyond chapter text now constitutes the learner's signature mode.

**Gaps acknowledged**:
- Stage 1: 4 initial axis slips (all corrected)
- Stage 2: LESS reject inconsistency (reversed)
- Stage 3: bootstrap terminology fuzzy, generation weight punt
- Stage 4: negative anchor mechanism unspecified (filter-side committed)
- Stage 5: 2 failure modes not added (kept 3)
- Replay ratio reasoning punt (empirical question, legitimate)

**Verdict: Mastery**.

Per learner direction (matches ch-19/20/21 pattern): commit + push to course/llm-training; merge-vs-stay-on-branch decision deferred.

---
chapter: ch-20
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/open-thoughts.md
source_url: https://arxiv.org/abs/2506.04178
created_at: "2026-04-23"
---

# Excerpt: OpenThoughts — 1000+ ablations over the reasoning-data recipe

**Source library:** `wiki/raw-data/llm-training/papers/open-thoughts.md`
**Paper:** Guha et al., *OpenThoughts: Data Recipes for Reasoning Models*, 2025.

---

## Why this source anchors ch-20

Stratos, Open-R1, and Sky-T1 each pick a single recipe and defend it. **OpenThoughts ablates the recipe space** — 1,000+ controlled experiments over question sourcing, answer multiplicity, teacher choice, and filters. It is the single most empirical answer to "what actually matters in reasoning distillation," and ch-20 §6 uses it as the ground-truth reference for every recipe claim in §4.

From source line 7:

> **Core Insight:** Reasoning data quality comes from the whole recipe, not just the teacher model. OpenThoughts shows that question sourcing, question filtering, answer multiplicity, and teacher choice can be tuned systematically to beat much larger open-data baselines.

---

## The three-stage program

```
OpenThoughts-114K      : scaled Sky-T1 pipeline with automated verification
OpenThoughts2-1M       : expands question diversity with synthetic question gen
                         → OpenThinker2-32B = first open-data model to match
                           DeepSeek-R1-Distill-32B on standard reasoning evals
OpenThoughts3-1.2M     : 1000+ ablations; 850K math + 250K code + 100K science
                         → OpenThinker3-7B (Qwen2.5-7B-Instruct base)
                           AIME24 69%, AIME25 53%, MATH500 90%,
                           LCB 06/24-01/25 51.7%, GPQA-Dia 54%
                         = strongest open-data 7B reasoning model
```

The 7B result is the relevant one for ch-20: it's the open-data model that a single lab can actually train from scratch. OpenThinker2-32B matches R1-Distill-32B; OpenThinker3-7B matches or beats R1-Distill-Qwen-7B despite using only open data.

---

## The six empirical findings that drive ch-20 §6

Each of these is cited in the read. Listing them compactly here.

**1. Sampling multiple answers per question is the easiest diversity trick.**
Source line 28: *"sampling multiple answers per question is the easiest way to expand a source by at least 16x."*
Direct contradiction of Open-R1's finding that 2× marginal over 1×. The reconciliation: Open-R1 tested 1 vs 2; OpenThoughts tests 1 vs 16+. The returns are sublinear but don't saturate as fast as Open-R1's 2-point test suggests.

**2. Strong source concentration beats source diversity.**
Source line 27: *"using a small number of top-quality sources beats optimizing for source diversity."*
Practical implication: don't construct your math corpus by union-ing 20 mid-quality problem pools. Pick 3–5 high-quality ones (NuminaMath + AIME archive + olympiad archives) and concentrate there.

**3. No answer-side filter beats keeping all answers.**
Source line 33: *"the paper tests many verification and answer-filtering methods, but none beat training on all answers without filtering."*
This is the most counterintuitive finding. Once format filters and basic correctness filters are in place, further answer-side pruning (e.g., "keep only shortest trace") loses more signal than it removes noise. The ch-20 takeaway: rejection-sample on correctness, stop there.

**4. Question-side filtering matters more than answer-side filtering.**
Source line 34: *"LLM-labeled difficulty and response-length filters outperform embedding-based and fastText-style heuristics."*
Difficulty filtering (LLM-labels a problem as easy/medium/hard; keep hard) on the **question** side is where the signal is. Embedding-based or fastText-based filters on the answer side are cargo-culted from SFT-mix literature and don't help.

**5. Deduplication is domain-sensitive.**
Source line 35: *"the final pipeline uses exact deduplication for math and science, and no deduplication for code."*
Counterintuitive: code benefits from NOT deduplicating. Why: two problems with syntactically similar reference solutions are often semantically distinct (different edge cases, different API usage); dedup collapses them and loses signal.

**6. Teacher choice is not monotone with teacher benchmark score.**
Source line 29: *"QwQ-32B beats DeepSeek-R1 as a teacher even though DeepSeek-R1 scores higher on many target benchmarks."*
The flagship finding. This is the result ch-20 §4.3 builds on. The explanation the paper gives is distributional: QwQ's output distribution is closer to Qwen2.5 base → less distribution shift → better SFT convergence for a 7B Qwen student.

---

## Why the recipe is domain-sensitive

Source lines 40–43:

> **Recipe is domain-sensitive:** the strongest settings are different for math, code, and science, so there is no single universal filter.

This is worth internalizing. A single "best recipe" does not exist. OpenThoughts reports different best settings per domain:

- **Math**: exact dedup + SymPy filter + LLM difficulty filter on questions; single teacher (QwQ-32B) for 7B student.
- **Code**: no dedup + unit-test filter; multi-sample per problem more aggressive.
- **Science**: exact dedup + LLM-judge filter; multi-teacher (mix QwQ + R1) helps more than in math.

A lab building its own reasoning corpus should expect to run domain-specific ablations. A monolithic recipe taken from Stratos/Open-R1/Sky-T1 and applied to a new domain will be suboptimal.

---

## The fragility of the open-data advantage

Source line 43:

> **Open-data advantage is fragile:** later gains depend on keeping the entire pipeline open and reproducible, not just the final model weights.

This is the strategic claim of the paper. OpenThinker models are strongest-open-data not because the team has the best recipe, but because they **publish everything** (data, code, eval scripts, ablation logs). When DeepSeek publishes only the 800K corpus and not the per-source breakdown or the filter code, downstream reproductions can't iterate on the specific bottleneck. When OpenThoughts publishes every ablation, downstream reproductions can pick up exactly where they left off.

This is the same argument applied to a different level of the stack than ch-20 §7 (licensing): licensing governs *what you can redistribute*; recipe-openness governs *what you can improve*. Both are required for an open-reproduction community to sustain.

---

## How ch-20 cites this

Ch-20 §6 is essentially a compressed version of OpenThoughts' findings. Each of the six bullets above appears in the read as an empirical ground. The "QwQ > R1 as teacher" finding is the empirical anchor for §4.3's teacher-selection argument. The "no answer-filter beats all-filter" finding qualifies Stratos's aggressive rejection-sampling — OpenThoughts shows that aggressive filtering works only up to a point, and aggressive filtering on the *question* side (not the answer side) is where further returns live.

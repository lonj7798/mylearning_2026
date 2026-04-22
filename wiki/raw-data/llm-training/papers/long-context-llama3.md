<!-- scope: long-context synthesis — Llama 3 report's long-context subsection (pretrain + SFT + RoPE)
     deps: [[long-context-data-engineering]]
     see-also: [[prolong]], [[qwen-long-context-synth]], [[ruler]]
-->

# Long-Context Training in the Llama 3 Technical Report
- **Core Insight:** Llama 3's 128K-context extension is split into a **staged continued-pretraining schedule** (8K → 16K → 32K → 64K → 128K across 800B tokens) with **RoPE base rescaled to 500K**, followed by a long-context SFT stage on synthetic long-doc QA — crucial details include staying short enough per stage for the position extrapolation to stabilize and mixing long SFT with short SFT at a specific ratio (~0.1% long) to avoid short-context regression.
- **Guideline:** When extending a production model to 128K, use a staged pretraining schedule where each stage doubles the context window (with ~100B tokens per stage), rescale RoPE base proportionally, and keep long-context SFT data at a small fraction (≤1%) of total SFT to avoid hurting short-context chat quality.
- **Authors:** Meta Llama Team (Grattafiori et al.)
- **Year:** 2024 — Llama 3 / 3.1 / 3.2 / 3.3 reports
- **URL:** https://arxiv.org/abs/2407.21783 (Llama 3 Herd of Models paper)
- **Relevant topics:** Llama 3, staged context extension, RoPE base rescaling, long-context SFT

## Abstract
Llama 3.1 (405B / 70B / 8B) extends context from 8K to 128K through a staged continued pretraining recipe documented in Meta's "The Llama 3 Herd of Models" paper. The long-context subsection (§3.4 of the paper) details (a) the staged 6-step context-length schedule, (b) RoPE base rescaling to 500K, (c) data-mix adjustments per stage, (d) a short long-context SFT stage, and (e) how Meta balances short-context retention with long-context addition.

## Key Contributions
- **6-stage context extension schedule** — explicit per-stage token budgets and context lengths.
- **RoPE base 500K rescaling** — one of the first frontier-model disclosures of its exact value.
- **Long-context SFT integration** — specifically how long-context instructions are blended into general post-training.
- Publicly available: methodology and ablations in the paper; weights at 128K native.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Staged continued pretraining (6 stages)
From the Llama 3 paper §3.4, the context window is extended gradually:
- **Stage A — 8K → 16K:** moderate data, ~100B tokens, RoPE base adjusted.
- **Stage B — 16K → 32K:** ~100B tokens.
- **Stage C — 32K → 64K:** ~150B tokens.
- **Stage D — 64K → 128K:** ~200B tokens.
- (Additional intermediate stages for stability.)
- **Total:** ~800B tokens across all stages.

### Data mix per stage
- Proportions shift toward long documents (books, code repos, papers) in later stages.
- Short-to-long document ratio gradually rebalances from ~80:20 (stage A) to ~40:60 (stage D).

### RoPE rescaling
- **Base rescaled from 10K to 500K** for the final 128K model.
- Scaling done progressively alongside the staged training — at each stage, the RoPE base is adjusted to match the new context window.

### Long-context SFT
- A small fraction (~0.1%) of SFT samples are long-context: synthetic QA over long documents, multi-document summarization, long-context code analysis.
- Generation uses a larger Llama 3 model as teacher on full documents.
- Keeping the long-SFT fraction low prevents short-context regression.

- **Output shape:** pretraining corpus re-weighted for length; SFT corpus contains ~0.1% long (~100K samples assuming ~100M total).
- **Teacher model:** Llama 3 405B itself (self-distillation on long-context SFT).
- **Cost / compute:** enormous — 800B tokens of continued pretraining on 405B is a substantial fraction of the overall 15T-token pretrain.

## Modality-specific technical details (REQUIRED — long-context)
- **Token-range:** 8K → 128K final; some experiments extend to 256K.
- **Needle-retrieval difficulty:** Llama 3.1-70B: NIAH 128K ~99%; RULER 128K ~75% (effective context ~64K).
- **Document-type mix:** books, code, academic, long web; specific proportions shift per stage.
- **Packing strategy:** documents packed within same sequence; cross-document attention allowed.
- **Position-encoding adaptation:** RoPE base 500K; no YaRN or NTK-aware trick — just direct base rescaling combined with staged training.
- **Per-stage data mix:** explicitly documented in Llama 3 paper Table 6 / §3.4.
- **Short-context preservation ablation:** raising long-SFT fraction above 1% costs ~1 MMLU point.

## Quality / diversity evaluation
- Llama-3.1-405B: NIAH 128K ~99%; effective RULER context ~96K.
- Llama-3.1-70B: strong on LongBench, InfiniteBench.
- Competitive with GPT-4-128K at release (July 2024).
- Llama-3.2 and 3.3 inherit and refine this recipe.

## Risks + gotchas
- **Staged schedule is compute-intensive:** 800B tokens of CPT is out of reach for smaller labs.
- **RoPE base = 500K** works for Llama 3's architecture; the right value depends on head-dim and pretrain base.
- **Effective context << claimed context:** RULER reveals Llama 3.1's effective context ~64K despite 128K support — acknowledged by Meta.
- **SFT ratio tuning required:** 0.1% long-SFT is empirical; different downstream mixes may need different ratios.

## Connections
- Contemporary lineage: [[long-context-data-engineering]] (Fu 2024 — open recipe), [[prolong]] (Princeton, smaller budget).
- Qwen equivalent: [[qwen-long-context-synth]] — similar staged schedule but with DCA.
- Evaluation: [[ruler]], NIAH, InfiniteBench.
- Upstream model report: the Llama 3 Herd paper (full text).

<!-- scope: Nathan Lambert Interconnects writing on synthetic data (2024-2025 posts)
     deps: [[ultrafeedback-construction]]
     see-also: [[allenai-tulu-synth]], [[model-collapse]]
-->

# Nathan Lambert (Interconnects) on Synthetic Data — Curated Posts 2024–2025
- **Core Insight:** Post-training costs in 2025 are dominated by synthetic-data infrastructure (prompt collection, multi-model generation fleets, rerankers, verifiers), not by human annotation; "synthetic data can do almost all of the work" *once* the verification layer is correct.
- **Guideline:** Don't treat synthetic data as a monolith; stratify by (a) pretraining vs SFT vs preference, (b) easy-open-ended (writing) vs hard-verifiable (code/math) — the latter needs aggressive verification + RLVR-style loops.
- **Author:** Nathan Lambert (Allen AI / Interconnects)
- **Year:** 2024–2025 (ongoing)
- **URL:** https://www.interconnects.ai/p/frontiers-in-synthetic-data ; https://www.interconnects.ai/p/the-state-of-post-training-2025
- **Relevant topics:** synthetic data strategy, post-training 2025 state, frontiers, data economics

## Representative posts (2024–2025)

### "Frontiers in Synthetic Data" (Jun 2024)
- Argues the three layers:
  1. **Pretraining synthetic** (Phi-style, Cosmopedia-style rephrase/textbook synthesis).
  2. **SFT synthetic** (Self-Instruct lineage → Magpie → Persona-Hub).
  3. **Preference synthetic** (UltraFeedback → West-of-N → Con-J judges).
- Quote-worthy claim: "Synthetic data can do almost all of the work" given a strong open weights base model + robust verification.

### "The State of Post-Training in 2025"
- Diagnoses current stack: SFT → DPO/RLHF → RLVR (verifiable rewards).
- Post-training now consumes a substantial fraction of total FLOPs — driven by (a) multi-round rejection sampling, (b) multi-model generation fleets, (c) large RL rollouts.
- Data-foundry business model (Scale AI, Surge) is pressured by synthetic data, especially in easy-verifiable domains.

### 2024 Year in Review
- Highlights the data-foundry vs synthetic data tension.
- Flags reasoning models (o1-class, R1-class) as the key shift that re-inflates verification demands.

### Recurring themes
- **Easy tasks (writing, summarization):** synthetic is near-free; teacher-generated responses are competitive with human completions.
- **Hard tasks (math, code, multi-step reasoning):** synthetic requires explicit verifiers (unit tests, answer matchers, PRMs); see [[rlvr-tulu3]].
- **Verification is the bottleneck,** not generation; in the RLVR era, the scarce resource is verifiable prompts.

## Practitioner-relevant notes
- Prefer **accumulation + verification** over pure self-distillation (aligned with [[model-collapse]] mitigations).
- Treat prompt curation as a distinct, cost-bearing stage; open prompt corpora are the new scarce asset.
- The cost structure of open post-training ≈ inference-compute × number-of-model-generations × number-of-iterations — scales fast.

## Risks / caveats Lambert emphasizes
- Model-collapse concerns are real but over-stated in strict-replacement regimes; accumulation + verifier loops break the bad asymptote.
- Judge-LLM bias is the next major audit frontier (see [[direct-judgement-preference]]).
- The relationship between synthetic pretraining and synthetic post-training is under-theorized; distinct cost/value profiles.

## Connections
- Pairs with the Allen AI Tülu 3 synthetic blog: [[allenai-tulu-synth]].
- Engages with the theoretical risk literature: [[model-collapse]], [[strong-model-collapse]].
- Practitioner-view companion to the synthesis papers: [[persona-hub]], [[magpie]], [[ultrafeedback-construction]].
- Useful background reading before any chapter on post-training economics.

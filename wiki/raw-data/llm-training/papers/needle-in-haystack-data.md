<!-- scope: long-context synthesis — NIAH evaluation data lineage (Greg Kamradt original + follow-ups)
     deps: [[ruler]]
     see-also: [[babilong]], [[longbench]]
-->

# Needle-in-a-Haystack (NIAH) Data Lineage
- **Core Insight:** The simplest useful long-context stress test is to hide a single fact ("needle") inside many pages of filler essay text and ask the model to retrieve it; varying needle-depth and haystack-length gives a two-dimensional capability map — this one-liner eval, started by Greg Kamradt in Nov 2023, became the de-facto community standard for long-context claims and the ancestor of RULER / BABILong.
- **Guideline:** Include NIAH in every long-context release because it's cheap, reproducible, and widely cited — but complement with harder synthetic evals (RULER multi-hop, BABILong reasoning) because NIAH alone over-reports effective context.
- **Authors (original):** Greg Kamradt (independent researcher) — Nov 2023 blog + GitHub.
- **Year:** 2023 (original), 2024+ (multi-needle + other variants)
- **URL:** https://github.com/gkamradt/LLMTest_NeedleInAHaystack
- **Relevant topics:** long-context evaluation, synthetic retrieval, needle-in-haystack, Greg Kamradt

## Abstract
Greg Kamradt's original Needle-in-a-Haystack test (Nov 2023) asked: "Can the LLM find a single sentence ('The best thing to do in San Francisco is eat a sandwich at Dolores Park on a sunny day') hidden at varying depths inside Paul Graham essays of varying lengths?" The test's simplicity and the interpretable 2D heatmap (depth × length) made it viral. Anthropic, OpenAI, Google, and every major open-model release has since used some form of NIAH. Follow-up variants include multi-needle, multi-key, multi-value, multi-query, and the full RULER suite.

## Key Contributions
- **Original NIAH formulation**: needle sentence in filler essays at varied depth.
- **Heatmap visualization** (context length × needle depth × accuracy).
- De facto community standard for long-context claim validation.
- Lineage ancestor of all modern long-context synthetic benchmarks.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Original NIAH (Greg Kamradt)
- **Filler:** Paul Graham essays (publicly available, mostly absent from post-2021 training cutoffs).
- **Needle:** a single sentence stating a specific fact (the "best thing in SF" sentence, or user-configurable).
- **Injection:** inserted at programmatic depths (e.g., 0%, 10%, 20%, …, 100% of filler).
- **Context lengths tested:** 1K, 4K, 8K, 16K, 32K, 64K, 128K, … up to the model's claimed max.
- **Evaluation prompt:** "What is the best thing to do in San Francisco?" — scored against gold answer.
- **Output shape:** a 2D grid of (depth, length, accuracy) — heatmap visualization.
- **Teacher model:** none — deterministic synthesis.
- **Cost:** near-zero to generate.

### Multi-needle variants (2024)
- **Multi-key:** N distinct needles, each with its own key (e.g., "the best thing in A is X", "the best thing in B is Y"); retrieve all.
- **Multi-value:** one key with multiple values across the haystack.
- **Multi-query:** multiple independent retrieval queries for one haystack.
- **Anthropic multi-needle:** 1–8 needles at varied depths, all retrieved in one completion.

### RULER + BABILong extensions
- [[ruler]]: 13 tasks including multi-needle variants + aggregation + tracing.
- [[babilong]]: long-context versions of the bAbI reasoning tasks.

- **Output shape:** standardized evaluation grids.
- **Teacher model(s):** none.
- **Cost:** minimal.

## Modality-specific technical details (REQUIRED — long-context)
- **Token-range:** 1K → 1M+ (varies by model claim).
- **Needle-retrieval difficulty:** single-needle is easy; multi-needle + multi-value dramatically harder.
- **Document-type mix:** Paul Graham essays by default (choose for out-of-distribution filler).
- **Packing strategy:** N/A (evaluation synthesis).
- **Position-encoding stress-test:** NIAH exposes RoPE-extension failure modes — models that fail at certain depths often have non-uniform RoPE issues.
- **Evaluation metric:** exact-substring match (simple) or LLM-judge (lenient).

## Quality / diversity evaluation (as benchmark)
- Single NIAH: saturated for 2025 models — most score >95% at claimed context.
- Multi-needle (8-needle): strong discriminator in 2025; Claude-3.5 ~90%, Llama-3.1-70B ~70%, weaker open models <50%.
- NIAH is **not** a proxy for real long-context reasoning — models can pass NIAH at 128K while failing multi-hop reasoning at 32K.

## Risks + gotchas
- **Paul Graham contamination:** many models have PG essays in training — canary filler choice matters.
- **Exact-substring match is brittle:** models that paraphrase the needle can fail despite correct retrieval.
- **Over-reliance on NIAH in marketing** — "1M context" claims based only on NIAH are misleading.
- **Fair comparison requires identical filler + prompts** — different NIAH implementations give different numbers.

## Connections
- Successor benchmarks: [[ruler]] (13 synthetic tasks), [[babilong]] (reasoning), [[longbench]] (natural tasks).
- Used by: every 2024+ long-context release — Claude, GPT-4-128K, Llama 3, Qwen 2.5-1M, Gemini.
- Data recipe that uses NIAH as training signal: [[longalign]], [[prolong]], [[qwen-long-context-synth]].

<!-- scope: long-context synthesis — LongMIT multi-turn instruction reference for long contexts
     deps: [[longalign]]
     see-also: [[long-context-llama3]], [[prolong]]
-->

# LongMIT: Long Multi-turn Instruction Tuning
- **Core Insight:** Long-context SFT data has been mostly single-turn (one long document + one question); multi-turn long-context instructions — where both the document and the accumulated dialogue span tens of thousands of tokens — are a distinct missing ingredient, and explicitly synthesizing them closes a measurable gap on long-context chat benchmarks.
- **Guideline:** For long-context chat models, don't rely on single-turn long-doc SFT alone; synthesize multi-turn long conversations (5+ turns, each referencing specific document sections) so the model learns to maintain dialogue state across long context.
- **Authors:** Multiple groups in 2024/2025; notable references include long-multiturn-instruct corpus in Yi-1.5 long-context work and community efforts.
- **Year:** 2024–2025
- **URL:** varies; the clearest public multi-turn long-context SFT pipelines appear in Yi-1.5 long-context, Qwen long-context, and community datasets such as HuggingFaceH4/long-conversations.
- **Relevant topics:** long-context multi-turn SFT, long-conversation synthesis, long-context chat

## Abstract
LongMIT refers to the class of long multi-turn instruction datasets used for long-context chat alignment. Representative pipelines exist across several 2024/2025 long-context model releases. The core recipe: start from a long source document (10K–100K tokens), generate a multi-turn dialog where each turn asks a question requiring the document, and each answer references specific document sections — producing conversations whose full context (document + accumulated turns) spans tens of thousands of tokens.

## Key Contributions
- **Multi-turn long-context data format** — explicitly teaching dialog-state management over long contexts.
- Demonstrated complementary to single-turn long-doc SFT — adding multi-turn lifts LongBench-Chat scores by 5–10 points in most recipes.
- Integrated into Yi-1.5-Long, Qwen 2.5 long-context, several HuggingFace community datasets.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** long documents 10K–100K tokens (papers, books, code repos, technical manuals).
- **Step 1 — Topic decomposition:** teacher model (GPT-4 / Claude / Qwen-Max) reads the full document and drafts a conversation outline — 5–10 turns, each focused on a distinct aspect/section of the document.
- **Step 2 — Turn-by-turn generation:**
  - For each turn, teacher generates a natural user question that references prior turns.
  - Teacher answers with grounding in the full document.
  - Optional: inject follow-up clarifications where the user refers to earlier answers.
- **Step 3 — Filtering:**
  - Answer must reference document content (not hallucinated).
  - Conversation coherence check (LLM judge).
  - Context-length constraint (full conversation + document ≤ training max).
- **Output shape:** multi-turn dialogs of 5–10 turns; document ~10K–50K tokens; conversation ~5K–20K tokens; total context per training example ~20K–100K tokens.
- **Teacher model:** GPT-4, Claude-3, Qwen-Max, or equivalent 100K-context-capable closed models.
- **Cost:** $0.50–$5 per generated dialog in API fees at 2024/2025 rates.

## Modality-specific technical details (REQUIRED — long-context)
- **Token-range:** typically 20K–100K total; tail to 200K+.
- **Needle-retrieval difficulty:** questions often require retrieving + combining info from multiple document sections; multi-turn state tracking adds difficulty.
- **Document-type mix:** books, papers, long policy docs, code repos.
- **Packing strategy:** each multi-turn conversation is one training sample; sorted-batching to balance lengths.
- **Position-encoding adaptation:** inherits base model's long-context setup.
- **Per-stage data mix:** often mixed at ~20–30% into the long-context SFT pool alongside single-turn long-doc data and short chat data.

## Quality / diversity evaluation
- Adding multi-turn long-context data to single-turn SFT lifts LongBench-Chat scores by 5–10 points in ablations reported by long-context Yi-1.5 and similar releases.
- Improves dialog-state tracking — models become less likely to lose context across turns in long conversations.
- Minimal regression on short-context chat.

## Risks + gotchas
- **Teacher context ceiling:** only closed-frontier models can coherently generate 100K-context multi-turn dialogs; small models struggle.
- **Distribution narrowing:** if all multi-turn dialogs share similar structure (Q1 overview → Q2 detail → Q3 follow-up), the student may memorize the template.
- **Cost scaling:** 10K multi-turn long-context dialogs easily cost $20K+ in frontier-API fees.
- **Not fully standardized** — unlike LongAlign-10K, there's no single canonical public LongMIT release.

## Connections
- Extends single-turn lineage: [[longalign]], [[longalpaca]].
- Integrated in long-context model recipes: [[long-context-llama3]], [[qwen-long-context-synth]].
- Related multi-turn synthesis pattern: [[apigen-mt]] (multi-turn for tool use), [[ultrachat-construction]].

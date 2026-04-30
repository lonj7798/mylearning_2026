<!-- scope: multi-turn conversation synthesis — LongChat long-conversation synthesis via Vicuna
     deps: [[ultrachat-pipeline]]
     see-also: [[longalign]], [[longalpaca]]
-->

# LongChat: Long-Conversation Fine-Tune (LMSYS)
- **Core Insight:** Real user conversations hosted on ShareGPT contain a non-trivial tail of very long multi-turn sessions; filtering them by length, extending Vicuna's context window via condensed rotary embeddings, and fine-tuning yields LongChat-7B/13B-16K — an early (mid-2023) open long-conversation chat model whose data comes from *real* human user logs rather than synthesized dialog.
- **Guideline:** For long-conversation SFT data, mine real user logs (ShareGPT, WildChat) for the long tail (sessions with ≥ 8K token history), rather than synthesizing — real long conversations have topic-shift, backtracking, and context-dependence patterns that synthetic dialogues lack.
- **Authors:** Dacheng Li, Rulin Shao, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica, Xuezhe Ma, Hao Zhang (LMSYS / UC Berkeley / CMU)
- **Year:** 2023
- **URL:** https://lmsys.org/blog/2023-06-29-longchat/
- **Relevant topics:** long-conversation SFT, Vicuna, condensed rotary embedding, ShareGPT long-tail

## Abstract
LongChat-7B/13B-16K (released June 2023) is an early open-weights long-context chat model fine-tuned from Vicuna (Llama-based) with a **condensed rotary embedding** trick and SFT data filtered from ShareGPT's long-tail conversations. At release, it was the first open model supporting 16K conversational context with competitive quality to GPT-3.5-16K.

## Key Contributions
- **Condensed rotary embedding** — compresses Vicuna's 2K RoPE position embeddings to support 16K without full retraining.
- **Long-tail ShareGPT filtering** — pipeline for extracting real long multi-turn conversations from public user logs.
- **LongChat-7B/13B-16K** open weights — early reference long-context chat model.
- **LongEval** companion benchmark measuring long-conversation retrieval.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### ShareGPT filtering
- **Source:** public ShareGPT dumps of real ChatGPT conversations shared via shareg.pt/sharegpt-related browser extensions.
- **Filter:**
  - Conversation length ≥ 8K tokens (original Vicuna threshold 2K).
  - Multi-turn (≥ 4 turns).
  - Language filter (English).
  - Basic toxicity / content filters.
- **Output:** ~18K long-tail ShareGPT conversations.

### Condensed rotary embedding
- Standard Vicuna uses RoPE with position indices 0..L-1 for L-token context.
- Condensed version divides each position by a factor `c` (e.g., c=8 for 16K support): effective positions become `i/c`.
- Allows fine-tuning on 16K sequences without retraining position embeddings from scratch.

### Fine-tune recipe
- Base: Vicuna-7B or Vicuna-13B (Llama-1 derivatives).
- Context window: extended from 2K to 16K via condensed RoPE.
- Fine-tune: 2 epochs on filtered long-ShareGPT.
- LR: 2e-5 cosine.

- **Output shape:** ~18K long multi-turn conversations; avg ~12K tokens each; turn counts 4–40.
- **Teacher model:** none — ShareGPT is real user data.
- **Cost:** modest; condensed-RoPE fine-tune adds only marginal compute.

## Modality-specific technical details (REQUIRED — conversation + long-context)
- **Token-range:** 8K–16K per training sample.
- **Turn-count distribution:** median 8 turns, tail to 40.
- **Speaker-role protocol:** real human user + ChatGPT responses (preserved from ShareGPT).
- **Persona conditioning:** none explicit — derives from real user behavior distribution.
- **Safety post-filter:** basic toxicity filter; ShareGPT content generally mild.
- **Position-encoding adaptation:** condensed RoPE (position-scaling trick).
- **Packing strategy:** not packed — one long conversation per training example.

## Quality / diversity evaluation
- LongChat-13B-16K: LongEval topic-retrieval ~90% at 10K context.
- Matches GPT-3.5-16K on retrieval; trails on reasoning.
- Served as the long-context baseline for many 2023/early-2024 open models.

## Risks + gotchas
- **ShareGPT license gray area:** ShareGPT dumps were not explicitly licensed by users for ML training; community mostly treats as "research only".
- **Condensed RoPE is a coarse trick** compared to NTK-aware / YaRN / LongRoPE — superseded quickly.
- **Real-log noise:** ShareGPT contains low-quality conversations, test messages, duplicates.
- **Superseded by:** [[longalign]] (proper synthetic long-instruction), [[prolong]] (long-context CPT recipe), [[long-context-llama3]].

## Connections
- Real-log counterparts: [[wildchat]] (Allen AI — WildChat logs).
- Position-encoding lineage: condensed RoPE → NTK-aware → YaRN → [[longrope-data]].
- Long-context SFT successors: [[longalign]], [[longalpaca]].

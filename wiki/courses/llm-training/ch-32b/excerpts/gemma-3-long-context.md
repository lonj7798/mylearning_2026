---
chapter: ch-32b
course: llm-training
phase: read
excerpt_of: arXiv:2503.19786v1 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2503.19786
created_at: "2026-09-15"
---

# Excerpt: Gemma 3 Technical Report — long-context design

- **Authors:** Gemma Team, Google DeepMind
- **Year:** 2025 (arXiv v1 2025-03)
- **Source type:** official technical report
- **Used in:** ch-32b §3, §4, Recipe

## Architecture (§2)
- Local sliding-window attention and global attention are interleaved "with a pattern of 5 local layers for every
  global layer, starting with a local layer as the first layer of the model."
- Local attention span: 1024 tokens (§1, §2).
- "Gemma 3 models support context length of 128K tokens, with the exception of the 1B model that has 32K. We
  increase RoPE base frequency from 10k to 1M on global self-attention layers, and keep the frequency of the local
  layers at 10k. We follow a process similar to the positional interpolation of Chen et al. (2023) to extend the
  span of the global self-attention layers."

## Ablations on local and global layers (§5.2; 2B text-only models)
- Local:global ratio: "We observe minimal impact on perplexity when changing this ratio" (Fig. 3; 1:1 is Gemma 2,
  5:1 is Gemma 3).
- Sliding window: "The sliding window can be reduced significantly without impacting perplexity" (Fig. 4).
- KV-cache memory at a 32K pre-fill: "the 'global only' configuration results in a memory overhead of 60%, while
  this is reduced to less than 15% with 1:3 and sliding windows of 1024" (Fig. 5).

## Enabling long context (§5.3)
"Instead of training with 128K sequences from scratch, we pre-train our models with 32K sequences and then scale
the 4B, 12B, and 27B models up to 128K tokens at the end of pre-training while rescaling RoPE (Chen et al., 2023).
We find a scaling factor of 8 to work well in practice." "Our models generalize to 128K, but rapidly degrade as we
continue to scale" (Fig. 7).

## Long-context results (Table 15)
| Benchmark | PT 4B | PT 12B | PT 27B | IT 4B | IT 12B | IT 27B |
|---|---|---|---|---|---|---|
| RULER 32K | 67.1 | 90.6 | 85.9 | 61.4 | 80.3 | 91.1 |
| RULER 128K | 51.7 | 80.7 | 72.9 | 46.8 | 57.1 | 66.0 |
| MRCR 32K | 44.7 | 59.8 | 63.2 | 49.8 | 53.7 | 63.2 |
| MRCR 128K | 40.6 | 56.9 | 60.0 | 44.6 | 49.8 | 59.3 |

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2503.19786 (v1 text): §2, §5.2, §5.3, Figs. 3–7 captions,
  Table 15.
- Not reported by the source: token count of the 32K→128K extension, its data mixture, and any short-context
  benchmark comparison before and after the extension.

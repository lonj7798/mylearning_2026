---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Ding, Wang, Paolini, Kumar, Deoras, Roth, Soatto — "Fewer Truncations Improve Language Modeling" (ICML 2024)
source_url: https://arxiv.org/abs/2404.10830
created_at: "2026-09-15"
note: "Library card for this slug was planned but not present on 2026-09-15; this excerpt was written from the primary PDF (arXiv v2, 2024-05-02)."
---

# Excerpt: Ding et al. 2024 — Fewer Truncations Improve Language Modeling

**Artifact:** arXiv:2404.10830 (v1 2024-04; read as v2). Authors: Hantian Ding, Zijian Wang, Giovanni Paolini, Varun Kumar, Anoop Deoras, Dan Roth, Stefano Soatto (AWS AI Labs).

## Claim

Concatenating documents and splitting them into fixed-length sequences truncates many documents. Best-fit Packing first splits only documents longer than the context length L into chunks of at most L, then packs chunks into sequences with Best-Fit-Decreasing without splitting them further. It keeps the token efficiency of concatenation and removes unnecessary truncation (abstract, §1, §3).

## Algorithm (§3.1, Algorithm 1)

- Sort chunks by length in descending order. FFD places a chunk in the first bin with enough remaining capacity; BFD places it in the bin with the smallest remaining capacity that still fits.
- Because lengths are integers in [1, L] with L much smaller than the number of documents N, the optimized BFD tracks remaining capacities in a segment tree of size O(L), giving O(N log L) packing time.
- Figure 1 example: documents of 14, 7, 5, 2, 3 tokens and L = 8. Concatenation truncates 3 of 5 documents; Best-fit Packing truncates only the 14-token document.

## Efficiency (Tables 1-2)

- Runtime at L = 2,048: 1B documents, FFD 26,354 s vs optimized BFD 10,816 s; 2B documents, 55,074 s vs 22,244 s.
- Extra sequences relative to concatenation: RefinedWeb at 2,048, +6,253 on 2.6×10^8 (0.0024%); RefinedWeb at 8,192, +411 on 6.5×10^7 (0.00063%); the Stack at 2,048, +1,786 on 6.4×10^7 (0.0028%).

## Training setup (Table 3, App. C.1)

- LLaMA architecture. Natural language: 13B at 2,048 and 13B at 8,192, 500B RefinedWeb tokens each. Code: 7B at 2,048, 300B tokens of the Stack (7 languages).
- AdamW, LR 3e-4, cosine schedule, 3,000 warmup steps, global batch 2M tokens, FlashAttention2, 256 A100 GPUs.
- Concatenation runs allow attention across document boundaries. Best-fit Packing runs mask cross-document attention. "Thanks to the relative nature of rotary positional embeddings (RoPE), we do not adjust position ids from model input."

## Results (§4; superscript s in the paper marks p < 0.05, paired t-test)

- Reading comprehension average (Table 4): 2k 46.71 (concat) → 48.92 (pack); 8k 47.13 → 49.03. SQuAD EM 2k 53.86 → 61.61.
- NLI and context following (Table 5), 2k: MNLI 42.78 → 44.33; RTE 55.74 → 60.28; NQ-Swap 45.62 → 51.03; MemoTrap 35.58 → 41.56. 8k: 38.85 → 40.60; 54.66 → 59.72; 47.07 → 50.04; 37.61 → 40.28.
- Program synthesis: undefined-name errors reduced by up to 58.3% (§4.6).
- Commonsense and closed-book QA: slightly better on average; larger gain on ARC-C than ARC-E, which the authors link to tail knowledge (§4.5, Tables 7-8).
- "In no case do we observe a statistically significant degradation with Best-fit Packing" (§4).

## Cross-document attention ablation (App. C.3, Table 10; 2k natural-language models)

| Method | Attention mask | PPL | Reading comp. | NLI | Context following | Summarization (ROUGE-2) | Commonsense |
|---|---|---|---|---|---|---|---|
| Concat | no | 9.64 | 46.71 | 49.26 | 40.60 | 12.16 | 64.38 |
| Concat | yes | 9.53 | 47.92 | 50.35 | 42.41 | 11.79 | 64.77 |
| Pack | yes | 9.53 | 48.92 | 52.31 | 46.30 | 13.14 | 64.84 |

The authors conclude that the perplexity gain comes from the mask, while Best-fit Packing is needed for the downstream gains; the mask on concatenation alone lowered summarization.

## Limits

Pre-training only; no SFT or chat experiments; the number of seeds per configuration is not reported; summarization faithfulness metrics have documented limits (App. C.2).

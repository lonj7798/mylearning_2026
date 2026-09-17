---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: none (no library card planned on 2026-09-15)
source_url: https://arxiv.org/abs/2402.14903
created_at: "2026-09-15"
---

# Excerpt: Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs

**Authors:** Aaditya K. Singh (Gatsby Unit, UCL), DJ Strouse (Google DeepMind).
**Version read:** arXiv:2402.14903v1 (22 Feb 2024); preprint.
**Status:** no library card existed; values read in the v1 PDF text at the stated locus.

## Number tokenization strategies (Table 1, §1)
| Model | Strategy |
|---|---|
| GPT-3 (2020) | pure BPE |
| GPT-3.5, GPT-4 | left-to-right (L2R) chunks of 3 digits (cl100k_base has tokens for all 1-, 2-, 3-digit strings) |
| PaLM, Llama 1 & 2, Mistral | single digit |
| OLMo (2024), GPT-J, Gopher, Chinchilla | pure BPE |
In p50k_base (GPT-3), which 3-digit strings are single tokens has "no clear structure" (Fig. 2); for example 710 may be one token and 711 not (§1).

## Setup (§2)
Few-shot addition with addends of 7-9 digits (each addend is 3 tokens), 90 problems (10 per digit-length pair), 1-8 shots, greedy decoding, Chat Completions API. Right-to-left (R2L) tokenization is forced by inserting commas every 3 digits from the right, for example 8,302,080 + 3,529,456 = 11,831,536 (Fig. 1).

## Results
- 8-shot accuracy (Fig. 1, March 2023 checkpoints): GPT-3.5 75.6% (L2R) vs 97.8% (R2L); GPT-4 84.4% vs 98.9%.
- Shots (§3.1, Fig. 4): L2R 68.5% (1-shot) → 75.6% (8-shot); R2L 95.6% → 97.8%.
- Controls (§3.2-3.3, Fig. 5-6): other single-token separators give similar R2L gains; adding spaces or separators while keeping L2R does not recover accuracy.
- Error pattern (§4.1, §4.3, Fig. 7, Fig. 9): when the answer has more digits than the addends ("length mismatch"), L2R accuracy drops to 8.25%; errors are concentrated on the fourth digit, and the first three digits (the first output token) are correct.
- Chain-of-thought-style repetition of the input in R2L form recovers accuracy (§5).
- Newer or larger models reduce the gap but do not eliminate it (§6).

## Limits
Closed models only; tokenization is changed at inference time, not in pre-training; the paper recommends pre-training ablations but does not run them.

## How ch-11 uses it
§3 (digit tokenization and arithmetic), worked example, Generalization lens.

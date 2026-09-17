---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/openmathinstruct.md
source_url: https://arxiv.org/abs/2402.10176
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: OpenMathInstruct-1 — sampling code-interpreter solutions from an open base model

**Source library:** `wiki/raw-data/llm-training/papers/openmathinstruct.md` (verified 2026-09-14 against arXiv:2402.10176v2).
Revised for the 2026-09 generality revision of ch-24 §2. The earlier version of this excerpt (git 4a72e54) repeated unsupported numbers; they are listed at the end.

## What ch-24 uses

- Generator: Mixtral-base, prompted with an instruction and K = 5 code-interpreter examples; answer in `\boxed{}` (§2.1).
- Format: text with `<llm-code>` blocks; executed output inserted in `<llm-code-output>`; at most 3 code blocks; generation stops at the first execution error (§2.1).
- Sampling: temperature 1.0, top-p 0.95; 4,096 total tokens (§2.1).
- Samples per problem (Table 2): GSM8K 128 per prompt, 256 total, 7,469 of 7,473 problems covered; MATH 224 per prompt, 896 total, 6,978 of 7,500 covered.
- Masked reference solutions in the prompt (Mask-Text) raise MATH coverage from 80.1% to 85.9% at 224 samples (Table 2).
- Training data: about 1.02M pairs (512K fair-downsampled GSM8K + 511,677 Any-Code MATH) out of the 1.8M released (§4, footnote 10).
- Results (Table 3, greedy): OpenMath-Mistral-7B 80.2 GSM8K, 44.5 MATH, 63.7 GSM-Hard; OpenMath-CodeLlama-70B 84.6 / 50.7; OpenMath-Llama2-70B 84.7 / 46.3.
- Negatives: wrong-answer samples are discarded; 6.6M incorrect solutions are released for verifier training and not used in the paper (§1, footnote 4). Correct-answer solutions with flawed reasoning are kept; the authors find them rare, anecdotally (§2.3).
- Generality: the authors state that in-domain gains may not transfer and cite the GSM8K → GSM-Hard drop (Limitations).

## Removed from the earlier excerpt (not in the source)

- "K = 32-64 solutions per problem" and "Mixtral-8x7B-Instruct" (Table 2 and §2.1 give the values above).
- "~120 solutions per GSM8K problem, ~100 per MATH problem", "~500K GPU-hours", "SymPy canonical equivalence filter", "Apache-2.0".
- "CoT-only loses ~8 MATH points; PoT-only loses ~5 GSM8K points" and "5-10% right-answer-wrong-reasoning rate".

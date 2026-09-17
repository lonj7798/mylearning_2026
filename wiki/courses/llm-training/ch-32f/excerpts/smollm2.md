---
chapter: ch-32f
course: llm-training
phase: read
excerpt_of: primary source arXiv:2502.02737v1 (no library card for SmolLM2 as of 2026-09-15) and the SmolLM2-360M / SmolLM2-135M config.json files on the Hugging Face Hub
source_url: https://arxiv.org/abs/2502.02737
created_at: "2026-09-15"
---

# Excerpt: SmolLM2 annealing ablations, decay stage, and 2K to 8K context extension

**Paper:** Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, et al. (Hugging Face), "SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model". arXiv v1 2025-02-04. Source type: official technical report. Read in full at v1 on 2026-09-15.

## Annealing as a data test (§3.1, §3.3, §3.4)

- Math and code datasets were evaluated "from a mid-training checkpoint of SmolLM2 at 3T tokens", "trained primarily on web data", with "an annealing approach: the learning rate linearly decays to 0 while training on a mixture that includes the dataset under evaluation" (§3.1).
- Math arms: "a mixture of 60B tokens of the dataset under evaluation and 40B from the pre-checkpoint mixture". Code arms: "annealing on 200B tokens, uniformly distributed across 15 of the most commonly used programming languages (~14B tokens each)" (§3.1).
- Math evaluation: GSM8K, MATH, MMLU-STEM (lighteval). Code evaluation: HumanEval and MultiPL-E (§3.1).
- OWM vs InfiMM-WebMath anneals: GSM8K peak 10% vs 14%; "OWM slightly outperforms InfiMM-WebMath on MATH"; 60B math tokens were 5 epochs of OWM and 1.5 epochs of InfiMM-WebMath (§3.3.1; App. C.1 Fig. 5).
- FineMath anneals: "All FineMath subsets consistently outperform OWM and InfiMM-WebMath on GSM8K, MATH, and MMLU-STEM. FineMath4+ achieves a 2x improvement on GSM8K and a 6x improvement on MATH compared to InfiMM-WebMath"; Infi-WebMath4+ "plateaus after 80B tokens (roughly 10 epochs), likely due to data repetition" (§3.3.2, Fig. 1).
- Decontamination of the math sets: "against GSM8K, MATH and MMLU using 13-gram matching and a minimum overlap ratio with the longest common subsequence of 0.6" (§3.3.2).
- Stack-Edu anneals: threshold 3 "improved performance across most languages"; Java performed better with threshold 2; MultiPL-E Python 20.7 → 25.6, C++ 16.7 → 24.8, JavaScript 18.2 → 22.4, Java 17.6 → 22.7 (§3.4, Table 2).

## Web data complementarity (§3.2, Table 1; 350B-token ablation models)

| Task | FW-Edu | DCLM | 40/60 | 60/40 |
|---|---|---|---|---|
| MMLU | 37.5 | 35.5 | 36.5 | 37.0 |
| ARC | 57.5 | 53.5 | 53.2 | 56.0 |
| HellaSwag | 60.1 | 62.3 | 61.4 | 62.2 |
| CommonsenseQA | 36.2 | 40.1 | 39.9 | 38.5 |

## Schedule and decay stage (§4.1, §4.5, Table 3, Table 8, App. A)

- 1.7B model, 256 H100s, AdamW (0.9, 0.95), WSD schedule: 2,000 warmup steps, peak LR 5.0 × 10⁻⁴, decay "reducing the learning rate to zero over 10% of the total training steps" (§4.1; Fig. 3). Sequence length 2,048 before extension; 2M tokens per batch; RoPE θ = 10,000 (Table 6).
- Stage 4 (10T to 11T tokens): LR "linearly to 0"; FineMath 4+ and InfiWebMath-3+ introduced; 0.08% OWM and 0.02% AugGSM8K, "an augmented version of the GSM8K benchmark's training set"; math 14%, Stack-Edu 24%, English web 58%, Cosmopedia v2 4% (§4.5).
- Category averages after stages 1 / 2 / 3 / 4: Knowledge/Reasoning 55.50 / 56.76 / 57.47 / 60.24; Math 3.21 / 3.7 / 7.27 / 22.07; Code 8.87 / 10.56 / 16.75 / 23.21 (Table 3). Per benchmark after stage 3 → stage 4: GSM8K 10.01 → 32.60, MATH 4.52 → 11.54, HumanEval 17.68 → 22.60, MMLU (MCF) 42.54 → 48.87, ARC 58.66 → 60.99, Jeopardy 25.54 → 23.35 (Table 8).

## Context extension (§4.6, §4.7, App. G)

- "extended the context length from 2k to 8k tokens, by taking an intermediate checkpoint from stage 4 (before the final 75 billion tokens of training) and continuing training with a different data mixture and a RoPE value of 130k" (§4.6).
- Mixture: "40% long-context documents (8k tokens or more) sourced from DCLM (10%), FineWeb-Edu (10%), and the books subset of Dolma (20%)", "the remaining 60% followed the stage 4 mixture" (§4.6).
- "we see next to no degradation in performance after Context Length Extension" (§4.7); no before/after table is printed.
- HELMET, base models, 8K maximum input length (Table 11): SmolLM2-1.7B Recall 36.38, RAG 47.17, ICL 23.20, Re-rank 23.31, LongQA 33.00; Llama3.2-1B RAG 42.13; Qwen2.5-1.5B RAG 47.54.
- Held-out benchmarks "not monitored during training": MMLU-Pro 19.4, Natural Questions 8.7, TriviaQA 36.7 (Table 4).

## SmolLM2-360M and -135M (§6; config.json fetched 2026-09-15)

- "SmolLM2-360M (360M parameters, trained on 4T tokens) and SmolLM2-135M (135M parameters, trained on 2T tokens)"; "single-stage training"; "WSD scheduler with 20% decay and a learning rate of 3.0 × 10⁻³" (§6). The paper does not state their training sequence length.
- `HuggingFaceTB/SmolLM2-360M/config.json`: `hidden_size` 960, `num_attention_heads` 15, `num_key_value_heads` 5, `max_position_embeddings` 8192, `rope_theta` 100000, `rope_scaling` null (head dimension 960/15 = 64, derived).
- `HuggingFaceTB/SmolLM2-135M/config.json`: `hidden_size` 576, `num_attention_heads` 9, `max_position_embeddings` 8192, `rope_theta` 100000 (head dimension 64, derived).

## Used in

ch-32f §2 (annealing arm design), §4 (RoPE base worked example), Recipe rows.

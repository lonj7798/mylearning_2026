<!-- scope: chapter-local excerpt for ch-22; no library card existed at the 2026-09 revision
     source: Raghavendra, Nath, Hendryx. "Revisiting the Superficial Alignment Hypothesis". arXiv:2410.03717 (v1 2024-09)
     checked: 2026-09-15 against https://arxiv.org/abs/2410.03717 (v1)
-->

# Excerpt: Revisiting the Superficial Alignment Hypothesis

- **Core result.** Across Llama-3, Mistral, and Llama-2 models of several sizes, post-training task performance scales as a power law P ∝ D^{1/b} in the number of fine-tuning examples D, for math, coding, instruction following, and multihop reasoning (Abstract, §2).
- **Source type:** paper (Scale AI). **Reliability:** single study.

## Setup (Table 1)
| Task | Test benchmark | Train dataset | Train examples |
|---|---|---|---|
| Math | GSM8K test | GSM8K train | 7,500 |
| Multihop QA | SubQA test | SubQA train | 2,700 |
| Coding | HumanEval+ | StarCoder Self-Align | 10,000 |
| Instruction following | IFEval | Conifer hard messages | 5,000 |
| Instruction following | IFEval | Dolly15k | 15,000 |

## Key takeaways (§2)
- Performance follows a power law in the number of post-training samples across model families and sizes (§3, Fig. 1).
- Win-rate evaluation can mislead for reasoning tasks: LLM judges can prefer chatbot-style answers to math questions from models that score poorly on math benchmarks (§3.3).
- Task-specific style and formatting improvements saturate within about 100 examples, while reasoning accuracy keeps improving with more examples (§4).
- Post-training for reasoning helps models use knowledge beyond the pre-training cutoff on multihop QA (§5).
- The authors describe the Superficial Alignment Hypothesis as "at best, an over-simplification" (Abstract) (author interpretation).

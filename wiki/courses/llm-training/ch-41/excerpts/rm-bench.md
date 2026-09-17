---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: arXiv:2410.16184v1 (RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and Style)
source_url: https://arxiv.org/abs/2410.16184
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because no library card exists yet)"
---

# Excerpt: RM-Bench style-controlled evaluation

Used by [[read]] "Why this chapter matters", §3.6, §4.2, and the negatives section. Authors: Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, Juanzi Li (Fudan University, Tsinghua University, HKUST). arXiv v1 2024-10-21; checked on 2026-09-15.

## Construction (§3)
1. Domains: Chat, Code, Math, Safety (§3).
2. Chat: gpt-4o answers to AlpacaEval prompts about world knowledge; a many-shot jailbreak prompt injects factual errors to make the rejected answer; human annotators verify both (§3.1).
3. Code and Math: several gpt-4o samples at temperature 1.0 per prompt; one correct and one incorrect sample by unit tests or ground-truth answers (§3.2).
4. Safety: should-respond prompts (chosen = helpful answer, rejected = over-cautious refusal) and should-refuse prompts (chosen = refusal, rejected = harmful answer from an uncensored Llama-3.1-8B) (§3.3).
5. Style control: each response in three versions, concise `y∅`, detailed plain text `y^L`, detailed with Markdown `y^{L,M}` (§3.4).
- Sample counts: Table 2 lists Chat 129, Safety 441, Math 529, Code 228; the §3.1 text states 183 chat samples. The two numbers are not reconciled in v1.

## Metrics (§3.5)
- A 3 × 3 Style-Substance matrix: rows are chosen-response style, columns rejected-response style.
- Easy accuracy = mean of the lower triangle (chosen has more style); Normal = diagonal; Hard = upper triangle (rejected has more style).
- Printed example, FsfairX-LLaMA3-RM-v0.1 on Chat (Fig. 2): row y_c∅ = 83.61, 3.83, 2.19; row y_cL = 99.45, 66.12, 49.73; row y_cL,M = 100.00, 80.33, 66.67.

## Results
- Top RM Skywork-Reward-Llama-3.1-8B: average 70.1, Hard 46.6; Nemotron-340B-Reward: average 69.5, Hard 56.1 (Table 3).
- Skywork-Reward-Llama-3.1-8B hard accuracy is 28.4% on Math and 30.7% on Code (§4.1; Tables 14-15).
- DPO models with their reference model outperform sequence classifiers trained on the same data (e.g., HH-RLHF 62.1 vs 60.1), and drop below them without the reference model (54.4) (Table 4).
- Nemotron-4-340B-Reward correctness and verbosity scores separate chosen from rejected only in the safety domain; in math and code they overlap (§4.3, Fig. 3).

## Correlation with policy performance (§5)
- Four Tülu-v2.5 RMs trained on 60k examples each from HH-RLHF, StackExchange, Chatbot Arena 2023, Nectar; PPO policies trained with the same data and hyperparameters (§5).
- Chat Hard accuracy increases with the policy's Arena-Hard-Auto style-control score (Fig. 4).
- Downstream tasks (GSM8K, BBH, HumanEval+, MBPP+, ToxiGen, XSTest): Pearson r = 0.55 (p = 0.07); RewardBench r = 0.21 (p = 0.51) (§5.2, App. F).

<!-- scope: chapter-local excerpt for ch-22; no library card existed at the 2026-09 revision
     source: Yu, Yuan, Golovneva, Wu, Sukhbaatar, Weston, Xu. "R.I.P.: Better Models by Survival of the Fittest Prompts". arXiv:2501.18578 (v1 2025-01)
     checked: 2026-09-15 against https://arxiv.org/abs/2501.18578 (v2, 26 Feb 2025)
-->

# Excerpt: R.I.P. — Rejecting Instruction Preferences

- **Core result.** Filtering prompts by properties of their rejected response (reward, length) and the chosen–rejected reward gap improves DPO-trained Llama 3.1-8B-Instruct over unfiltered DPO on WildChat: AlpacaEval 2 LC 48.4 → 57.8, Arena-Hard 37.9 → 43.1, WildBench 41.5 → 45.6 (Table 2). The abstract reports gains of 9.4% (LC), 8.7% (Arena-Hard), and 9.9% (WildBench); only the LC figure equals the Table 2 difference, and the paper does not state which rows the other two use.
- **Source type:** paper (Meta). **Reliability:** single study; evaluation uses judge-based benchmarks only.

## Hypotheses (§3.2)
1. Low-quality prompts produce low-quality responses, so a low rejected-response reward r(y_l|x) or short rejected length len(y_l) indicates a low-quality prompt.
2. Low-quality prompts produce higher-variance responses, so a large reward gap r(y_w|x) − r(y_l|x) indicates a low-quality prompt.

## Method (§3.1, §3.3)
- Pairs: sample N responses from the seed model M, choose the highest-reward as y_w and the lowest as y_l ("best-vs-worst").
- Metrics: m1 = r(y_l|x), m2 = len(y_l), m3 = reward gap. Keep x if m1 and m2 exceed lower thresholds and m3 is below an upper threshold.
- Self-RIP: few-shot generate new prompts (8 examples) from RIP-kept prompts, then apply RIP again (§3.3.2).

## Setup (§4)
- WildChat: 190K unique English first-turn prompts after removing about 70K Midjourney prompts; experiments use 20K prompts (8B) and 40K (70B) (§4.1.1, §5).
- Responses: N = 64 (N = 32 for 70B), T = 0.8, top-p 0.95, scored by ArmoRM or by Llama 3.1-405B-Instruct as judge (10 evaluations averaged) (§4.1.1).
- DPO: batch 64, LR swept over 5e-7 and 1e-6 (8B); batch 256, LR 1e-6 (70B); dropout 0; β = 0.1 (§4.4).

## Results (Table 2, Llama 3.1-8B-Instruct, WildChat, ArmoRM pairs)
- Seed model: AlpacaEval 2 LC 20.9, Arena-Hard 21.3, WildBench 33.1 (Table 1).
- WildChat-20k DPO without filtering: LC 48.4, Arena-Hard 37.9, WildBench 41.5.
- RIP: 4,538 prompts kept (77% filtered), thresholds at the 50th percentile of rejected length, rejected reward, and reward gap: LC 57.8, Arena-Hard 43.1, WildBench 45.6 (§5).
- Prompt-based filters (InsTag difficulty and diversity, LLM-as-prompt-judge) gave lower LC win rates than no filtering or only marginal gains; Jaccard filtering on the response pair was the strongest baseline (§5).
- Llama 3.3-70B-Instruct: LC 54.3 → 67.7, Arena-Hard 70.5 → 82.9, WildBench 55.3 → 58.8 relative to unfiltered DPO (§5, Table 3).
- Filtering with Llama 3.1-8B-Instruct for the 70B model still improves over no filtering (Table 4).
- Self-RIP vs Self-Instruct (20K synthetic prompts): LC 49.1 → 60.2, Arena-Hard 38.5 → 42.1 (Table 6).

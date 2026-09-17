<!-- scope: chapter-local excerpt for ch-22; no library card existed at the 2026-09 revision
     source: Bukharin, Li, Wang, Yang, Yin, Li, Zhang, Zhao, Jiang. "Data Diversity Matters for Robust Instruction Tuning". arXiv:2311.14736 (v1 2023-11)
     checked: 2026-09-15 against https://arxiv.org/abs/2311.14736 (v3, 11 Nov 2024)
-->

# Excerpt: Data Diversity Matters for Robust Instruction Tuning (QDIT)

- **Core result.** Combining a facility-location diversity function with a quality score raises worst-case instruction-following performance compared with quality-only selection; there is a trade-off between quality and diversity (Abstract, §4.3).
- **Source type:** paper. **Reliability:** single study; LLaMA-1 7B main model; Claude 2 judge plus a reward model and six benchmarks.

## Method (§3)
- Diversity: d(A) = Σ_{v∈V} max_{a∈A} sim(a, v), with sim the cosine similarity of sentence-transformer instruction embeddings (Eq. 1).
- Quality q(·): ChatGPT scores from AlpaGasus for Alpaca; a reward model trained on Anthropic HH for the other pools (§3.1, §4.2).
- Score: f(a | A, α) = (1 − α) d(a | A) + α q(a); greedy selection of the highest f at each step (Algorithm 1). α = 1 reduces to quality-driven selection; α = 0 to the classical greedy facility-location algorithm (§3.2).
- The authors state that threshold-based distance filters "are more similar to data de-duplication" and do not necessarily increase diversity (§2).
- Cost: naive O(|V|³K); lazy greedy, GPU parallelism, and subsampling make million-sample pools feasible in a few hours (§3.2, App. B).

## Setup (§4.2)
Pools: Alpaca 52K, Dolly 15K, UltraChat 1.3M, LMSYS-Chat 1M, Mixed 270K. Budgets: 10K for large pools; about 5% for small pools. Evaluation: Claude 2 pairwise winning score on five test sets, reward-model "HH score", and worst/best 10% of prompts.

## Results
- Quality-only selection improves average winning score vs Alpaca 52K by 6.37% and average HH score by 11.6% over random, but decreases lowest-10% winning score in two of five settings (§4.3).
- QDIT over quality-only: +4.17% winning score vs Alpaca 52K, +1.5% average HH score, +5.2% worst-case HH score, +6.26% worst-case winning score (§4.3).
- Table 1 (UltraChat 10K): lowest-10% winning score Random 1.074, Quality 1.175, QDIT 1.280.
- Table 3 (benchmarks, UltraChat 10K): DROP Random 26.24, Quality 17.01, QDIT 26.73; average Random 50.84, Quality 50.17, QDIT 52.05. QDIT's average is higher than quality-only on four of five pools.
- α ∈ {0.5, 0.7, 0.9} typically best; α = 0.1 and α = 1 drop (§4.4, Fig. 4).

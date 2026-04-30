<!-- scope: weak-model-perplexity IFD transfers to large-model SFT filtering, 20x cheaper (ACL 2024)
     deps: [[cherry-llm]], [[ifd]]
     see-also: [[deita]], [[less]]
-->

# Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning (ACL 2024)
- **Core Insight:** A tiny model (GPT-2 125M) and a large model (Llama-2-7B) rank SFT samples by IFD in almost the same order, even though their absolute capabilities differ by orders of magnitude — so you can filter the target model's training data with the small one and get a ~20× speedup.
- **Guideline:** To filter a 7B-scale SFT pool, compute IFD with GPT-2 (or any cheap proxy), keep the top ~15%, and train the large model on that subset; the filtering cost drops by ~20× versus scoring with the target model itself.
- **Authors:** Ming Li, Yong Zhang, Shwai He, Zhitao Li, Hongyu Zhao, Jianzong Wang, Ning Cheng, Tianyi Zhou
- **Year:** 2024 (ACL 2024)
- **URL:** https://arxiv.org/abs/2402.00530
- **Relevant topics:** weak-to-strong filtering, IFD, Superfiltering, SFT data selection

## Abstract
Superfiltering builds on Cherry-LLM / [[ifd]] by asking: do the IFD rankings transfer from a weak model to a strong one? The authors measure Spearman's ρ between GPT-2 IFD rankings and Llama-2-7B IFD rankings on Alpaca/WizardLM — the correlation is high. They then show that using GPT-2 to filter and Llama-2-7B to train yields the same (or slightly better) downstream performance as filtering with the target model itself, at ~20× less filtering compute. The paper establishes **weak-to-strong data filtering** as a practical primitive.

## Key Contributions
- Established empirical weak-to-strong transferability of **IFD rankings** (not absolute values).
- Demonstrated ~20× speedup in filtering using GPT-2 vs Llama-2-7B.
- Showed the filtered subset trained on the large model beats the unfiltered baseline on Open LLM Leaderboard / MT-Bench.
- Released code + scored datasets.

## Key Figures/Tables to Study
- **Rank-correlation plot** — GPT-2 IFD vs Llama-2-7B IFD across datasets.
- **Speed-vs-quality tradeoff table** — GPT-2 filter + 7B train vs 7B filter + 7B train.
- **Filtered-set scaling curve.**

## Scoring function
Identical to IFD (see [[ifd]]):
`IFD(q, a) = PPL_cond(a | q) / PPL_uncond(a)`
computed with a *weak proxy* model. The ranking — not the IFD magnitude — is what transfers.

## Pipeline
1. Choose weak proxy (GPT-2 125M).
2. Warm proxy on ~1K random subset of target pool.
3. Compute IFD for all pool samples with proxy.
4. Keep top ~15% by IFD.
5. Train target (Llama-2-7B) model on the selected subset.

## Quality / diversity evaluation
- Alpaca/WizardLM: GPT-2-filtered subset (15%) ≥ full-data baseline on MT-Bench and HF Open LLM Leaderboard.
- Filtering speed: ~20× faster than using Llama-2-7B as filter.
- Consistent across multiple target model families (Llama-2, Mistral).

## Risks + gotchas
- **Weak-proxy family mismatch:** transferability depends on the proxy having a plausibly similar tokenizer + capability profile; using a domain-specific proxy can skew selection.
- **Not a correctness check:** like IFD, Superfiltering does not verify factual accuracy.
- **Still requires warmup** — raw-GPT-2 IFD (no warmup) is noisier.

## Connections
- Direct follow-up to [[cherry-llm]] and [[ifd]].
- Alternative cheap first-stage filter vs [[deita]] (ChatGPT-labeled scorers) and [[less]] (gradient similarity).
- Supports the larger trend that many data-selection signals are *rank-stable* across model scale — a key property later exploited in 2025 weak-to-strong and [[prismatic-synthesis]]-style pipelines.

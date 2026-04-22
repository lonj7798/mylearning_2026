<!-- scope: self-guided perplexity-based IFD selection for SFT cherry samples (NAACL 2024)
     deps: [[self-instruct]]
     see-also: [[ifd]], [[deita]], [[superfiltering]], [[less]]
-->

# Cherry LLM: From Quantity to Quality via Self-Guided IFD Selection (NAACL 2024)
- **Core Insight:** A single target model can score its own SFT pool using **Instruction-Following Difficulty (IFD) = conditional-PPL(response | instruction) / unconditional-PPL(response)** and keep only the samples whose responses are genuinely *helped* by the instruction; 10% of the data selected this way beats the full set.
- **Guideline:** Before training on any synthetic SFT pool, compute IFD for every sample using the same base model, sort desc, keep top 5–15%; no external judge, no GPT rating required.
- **Authors:** Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, Jianzong Wang, Tianyi Zhou, Jing Xiao
- **Year:** 2023/2024 (NAACL 2024)
- **URL:** https://arxiv.org/abs/2308.12032
- **Relevant topics:** SFT data filtering, IFD, self-guided selection, instruction difficulty

## Abstract
Cherry LLM introduces a **self-guided** filter for instruction-tuning data — no external judge, no stronger teacher, no gradient computation. The target LM computes **Instruction-Following Difficulty (IFD)** for each training sample: the ratio of conditional perplexity (response given instruction) to unconditional perplexity (response alone). High IFD means the instruction is actually helping predict the response (instruction is informative, task is instructionally meaningful). Samples with IFD ≈ 1 or IFD > 1 signal the instruction carries no conditioning signal or the response is anomalous. On Alpaca and WizardLM, keeping the top 10% by IFD beats training on the full pool.

## Key Contributions
- **IFD score** — a zero-external-dependency instruction difficulty metric computable from the target LM.
- **Cherry selection recipe** — warm-up on a small subset → compute IFD with warmed model → pick top-K.
- Showed a 10%-subset beats full-set training on Alpaca/WizardLM across standard benchmarks.
- Released IFD scores for popular datasets + GitHub repo with full pipeline.

## Key Figures/Tables to Study
- **IFD distribution histograms** — most Alpaca samples cluster near 1 (low learning signal).
- **Top-K sweep** — ~10% is the sweet spot; going below 5% loses coverage.
- **Cross-dataset transfer table** — IFD selected on Alpaca transfers to WizardLM.

## Scoring function (REQUIRED exact form)
For a sample `(q, a)`:
- `PPL_cond(a|q)` = perplexity of `a` given `q` under the warm model.
- `PPL_uncond(a)` = perplexity of `a` alone (no instruction).
- **IFD(q, a) = PPL_cond(a|q) / PPL_uncond(a)**.
- Interpretation: IFD < 1 means `q` reduces response perplexity (informative instruction); smaller is "easier to follow given q." The **paper keeps the highest IFD values < 1** — challenging but learnable samples.

## Synthesis/selection pipeline (REQUIRED — be concrete)
- **Seed input:** any SFT pool (Alpaca 52K, WizardLM 70K, etc.).
- **Warm-up step:** fine-tune target LM on a small random subset (~1K) for 1 epoch to calibrate perplexity estimates.
- **Scoring step:** for each sample compute IFD with the warmed LM.
- **Selection step:** sort desc, keep top K% (5–15% typical); filter obvious anomalies (IFD > 1 or undefined).
- **Training step:** full SFT on the selected cherry set.
- **Output shape:** a ~10%-sized curated cherry set per source dataset; released for Alpaca, WizardLM.

## Quality / diversity evaluation
- Alpaca 10% cherry beats Alpaca full on MT-Bench, AlpacaEval, HuggingFace Open LLM Leaderboard.
- Same for WizardLM.
- No quality scorer or external judge needed — rare property.

## Risks + gotchas
- **IFD is self-referential** — it reflects the target LM's own uncertainty; a different target family will prefer different samples.
- **Small warm-up** required for meaningful scores; skipping it hurts IFD quality.
- **Does not measure factual correctness** — only conditioning signal; incorrect but hard responses can score highly.
- **Dataset-level filtering** only; does not synthesize new samples or fix coverage gaps.

## Connections
- The IFD metric itself is spun out into its own micro-reference: [[ifd]].
- Contrasts with [[deita]] (needs external ChatGPT-rated scorers) and [[alpagasus]] (needs GPT-4 quality ratings) — Cherry needs only the target LM itself.
- Follow-up [[superfiltering]] (2024) uses a *smaller* model to compute the IFD of a larger model's pool, showing the ranking transfers — drastically cheaper at scale.
- Orthogonal to [[less]] (gradient-similarity to a validation set).
- Subsumed along the diversity axis by [[prismatic-synthesis]]'s gradient-space diversity measure.

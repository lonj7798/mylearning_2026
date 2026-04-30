<!-- scope: 1000-sample high-quality SFT outperforms 52K — "Superficial Alignment Hypothesis"
     deps: [[self-instruct]]
     see-also: [[alpaca]], [[instag]], [[ultrachat-pipeline]]
-->

# LIMA: Less Is More for Alignment
- **Core Insight:** Almost all knowledge lives in pretraining; only ~1,000 carefully curated instruction–response pairs are needed to unlock it — the "Superficial Alignment Hypothesis."
- **Guideline:** Before scaling an SFT set, spend effort curating ~1K diverse high-quality examples; compare to your scaled set on human preference — if the gap is small, stop scaling and invest in diversity and format instead.
- **Authors:** Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy
- **Year:** 2023 (NeurIPS 2023)
- **URL:** https://arxiv.org/abs/2305.11206
- **Relevant topics:** SFT, data quality, minimal-data alignment, superficial alignment hypothesis

## Abstract
Large language models are trained in two stages: (1) unsupervised pretraining from raw text, to learn general-purpose representations, and (2) large scale instruction tuning and reinforcement learning, to better align to end tasks and user preferences. We measure the relative importance of these two stages by training LIMA, a 65B parameter LLaMa language model fine-tuned with the standard supervised loss on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling. LIMA demonstrates remarkably strong performance, learning to follow specific response formats from only a handful of examples in the training data, including complex queries that range from planning trip itineraries to speculating about alternate history. Moreover, the model tends to generalize well to unseen tasks that did not appear in the training data. In a controlled human study, responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43% of cases; this statistic is as high as 58% when compared to Bard and 65% versus DaVinci003, which was trained with human feedback. Taken together, these results strongly suggest that almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary to teach models to produce high quality output.

## Key Contributions
- Named and defended the **Superficial Alignment Hypothesis** (SAH): alignment ≈ teaching format, not teaching knowledge.
- Showed **1,000 SFT examples on LLaMA-65B** rival DaVinci-003 (RLHF) and Bard, and tie/beat GPT-4 in 43% of head-to-head preferences.
- Explicitly no RL, no preference model, no reward model — pure supervised loss.
- Published the 1K composition so others can reproduce and ablate.

## Key Figures/Tables to Study
- **Figure 1** — human preference head-to-head bar charts (LIMA vs GPT-4 / Bard / DaVinci-003 / Alpaca-65B / LLaMA-65B).
- **Composition table** — the exact source breakdown of the 1K (StackExchange, wikiHow, Reddit, hand-written by the authors).
- **Scaling ablation** — preference win-rate as training-set size grows 2× per step; the curve is famously flat.

## Technical Details
**Dataset construction (1,000 examples total):**
- Mixed community-forum Q&A (StackExchange, wikiHow) and hand-written prompts.
- Heavy manual filtering for **diverse format, diverse topic, high response quality**.
- Response lengths deliberately varied.
- **No RL, no DPO, no preference modeling.**

**Training setup:**
- Base: LLaMA-65B.
- Standard supervised cross-entropy on response tokens only (prompt tokens masked).
- Training: 15 epochs, learning rate 1e-5, batch size 32, AdamW.
- A key trick: **lowering LR as epochs progress** was essential to avoid overfitting on such a small set.

**Key ablations:**
- Scaling data from 2K→32K StackExchange alone did not improve generation quality.
- **Diversity > raw count**: doubling examples within a single domain hurt performance.
- **Response quality > prompt quality**: poor responses cap the model regardless of prompt richness.

**Limitations (authors):**
- Robustness is weaker than RLHF models — an adversarial prompt can knock LIMA off-script.
- Multi-turn dialogue was addressed post-hoc with 30 extra dialogue examples; still weaker than GPT-4.

## Connections
- Motivated [[instag]] (rigorous definition of diversity) and inspired the ablation design of [[ultrachat-pipeline]]/[[ultrafeedback]].
- Counterpoint to [[alpaca]]'s 52K and [[ultrachat-pipeline]]'s 1.5M — both datasets should be read alongside LIMA.
- SAH is tested further by [[physics-of-lm-3]] and Tülu 3's SFT mix (`[[tulu-3]]`).
- LIMA's "quality-over-quantity" is the standard argument for rejection-sampled / distilled SFT (see [[rejection-sampling-finetuning]]).

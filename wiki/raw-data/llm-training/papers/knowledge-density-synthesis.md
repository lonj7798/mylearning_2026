# Scaling LLM Knowledge Boundaries via Distribution-Optimized Synthesis

<!-- scope: Scaling LLM Knowledge Boundaries via Distribution-Optimized Synthesis; scope-limited research extract
     deps: [[insights-long-tail]]
     see-also: [[long-tail-papers]]
-->

- **Core Insight:** KDoS adjusts synthesis distribution using knowledge-density feedback.
- **Guideline:** Treat generation allocation as a separate choice from total generation volume.
- **Authors:** Songze Li et al.
- **Year:** 2026
- **URL:** [Primary source](https://arxiv.org/abs/2606.23271)
- **Relevant topics:** long-tail coverage; synthesis; evaluation

## Abstract

The paper proposes distribution-optimized synthesis and evaluates Wikipedia-based data across model and token scales.

## Key Contributions

- Result (single study): KDoS adjusts synthesis distribution using knowledge-density feedback.
- Interpretation: Treat generation allocation as a separate choice from total generation volume.

## Key Figures/Tables to Study

Figures/tables not extracted. The Technical Details section identifies verified method sections where available; numerical outcome comparisons require the primary text.

## Technical Details

The abstract describes knowledge-density feedback; do not infer a conversation simulator or a universal optimal mixture.

## Limits

Preprint; record reports ARR submission, not confirmed acceptance. Limitations restrict the demonstrated setting to post-training/SFT; no demonstrated pretraining or sales skill recipe.

## Connections

[[insights-long-tail]]; [paper catalog](../long-tail-papers.md).

## Verification

Primary abstract and Limitations verified on 2026-10-06; first release 2026-06-22. Full experimental recipes not extracted.

## Full-text follow-up (2026-10-07)

§§3.1–4: density is token count divided by semantic-hypersphere volume; radius uses average embedding distance to centroid. Group plus two nearest neighbors generates five question candidates; quality filtering precedes density rejection. Table 1 Qwen3-4B-Base averages Random 18.3, Uniform 20.9, Quality 20.6, Difficulty 21.5, KDoS 22.8; SimpleQA-Verified Difficulty 8.2 > KDoS 8.0. Table 2 full 22.8,without quality 21.7,without quality+density 17.6,seed 17.4. Differences are percentage  points, not relative percentages.

Checked primary arXiv HTML full text at the cited sections; these facts replace the earlier abstract-only scope for this extract.

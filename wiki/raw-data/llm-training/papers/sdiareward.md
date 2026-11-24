# SDiaReward: Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness

<!-- scope: SDiaReward; scope-limited research extract
     deps: [[insights-long-tail]]
     see-also: [[long-tail-papers]]
-->

- **Core Insight:** Episode-level spoken-dialogue preference evaluation includes acoustic and colloquial qualities.
- **Guideline:** Keep spoken preference scoring separate from task verification and rarity measurement.
- **Authors:** Jingyu Lu et al.
- **Year:** 2026
- **URL:** [Primary source](https://arxiv.org/abs/2603.14889)
- **Relevant topics:** long-tail coverage; synthesis; evaluation

## Abstract

The work uses full-dialogue preferences to evaluate spoken conversation, including prosody, emotion, and colloquialness.

## Key Contributions

- Result (single study): Episode-level spoken-dialogue preference evaluation includes acoustic and colloquial qualities.
- Interpretation: Keep spoken preference scoring separate from task verification and rarity measurement.

## Key Figures/Tables to Study

Figures/tables not extracted. The Technical Details section identifies verified method sections where available; numerical outcome comparisons require the primary text.

## Technical Details

§3.1 separates modality pairing and colloquialness pairing; §6 discusses domain-dependent absolute reward offsets.

## Limits

Relative preference scores are not calibrated cross-domain factual truth or evidence of tail coverage.

## Connections

[[insights-long-tail]]; [paper catalog](../long-tail-papers.md).

## Verification

Primary v2 full text §§3.1,6 verified on 2026-10-06; first release2026-03-16, page states ACL2026 acceptance.

# Sparse Rewards Can Self-Train Dialogue Agents

<!-- scope: Sparse Rewards Can Self-Train Dialogue Agents; scope-limited research extract
     deps: [[insights-long-tail]]
     see-also: [[long-tail-papers]]
-->

- **Core Insight:** JOSH searches dialogue branches using sparse task-success rewards to obtain training supervision.
- **Guideline:** Use executable task outcomes to select dialogue paths when a suitable simulator exists.
- **Authors:** Barrett Martin Lattimer, Varun Gangal, Ryan McDonald, Yi Yang
- **Year:** 2025
- **URL:** [Primary source](https://aclanthology.org/2025.findings-acl.1302/)
- **Relevant topics:** long-tail coverage; synthesis; evaluation

## Abstract

Turn-level search in simulated tool-use dialogue produces successful SFT trajectories and preference data.

## Key Contributions

- Result (single study): JOSH searches dialogue branches using sparse task-success rewards to obtain training supervision.
- Interpretation: Use executable task outcomes to select dialogue paths when a suitable simulator exists.

## Key Figures/Tables to Study

Figures/tables not extracted. The Technical Details section identifies verified method sections where available; numerical outcome comparisons require the primary text.

## Technical Details

First release arXiv:2409.04617 in 2024; Findings of ACL2025 publication.

## Limits

Reward and simulator coverage bound the supervision; successful paths need not cover rare natural language.

## Connections

[[insights-long-tail]]; [paper catalog](../long-tail-papers.md).

## Verification

Primary publication abstract verified on 2026-10-06; detailed optimizer recipe not extracted.

<!-- scope: full recipe ledger for the Olmo 3 technical report (arXiv:2512.13961v2); companion to [[olmo-3]]
     deps: [[olmo-3]]
     see-also: [[allenai-olmo3-open-instruct-scripts]], [[olmo-2-recipe]]
-->

# Olmo 3 — recipe ledger
- **Core Insight:** Every training setting the Olmo 3 report discloses in its body, with the locus for each value; per-run learning rates and batch sizes live in the released configs, not in the report.
- **Guideline:** When reusing an Olmo 3 number, take the row for the exact branch (Base, Think, Instruct, RL Zero) and size; the 7B and 32B runs differ in long-context token budget, SFT prompt count, and RL prompt count.
- **Authors:** Olmo Team — Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, et al. (Allen Institute for AI)
- **Year:** 2025 (arXiv v1 2025-12; v2 dated 2026-04-14)
- **URL:** https://arxiv.org/abs/2512.13961
- **Source type:** official technical report
- **Relevant topics:** pretraining budget, midtraining, long-context extension, SFT, DPO, OlmoRL

## Summary
This card holds the recipe table for [[olmo-3]], split out to keep that card under the length limit. Section
numbers refer to arXiv:2512.13961v2.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Olmo 3 Base 7B / 32B | 7B/32B | pretrain-stable | tokens seen; sequence length; precision | 5.93T (Dolma 3 Mix); 8192; bfloat16 | arXiv:2512.13961v2 §3.2, §3.4, Table 4 | verified 2026-09-18 | §3.4: mix weights chosen by three rounds of conditional mixing against a 6T token budget |
| Olmo 3 Base 7B / 32B | 7B/32B | mid-train | tokens seen; pool | 99.95B from a 2.19T pool (Dolma 3 Dolmino Mix) | §3.5, Table 5 | verified 2026-09-18 | §3.5.1, Table 6: candidate 100B mixes compared on the OlmoBaseEval Main suite via integration tests |
| Olmo 3 Base 7B | 7B | long-context | tokens seen; context window | 50.0B (Dolma 3 Longmino Mix); 8,192 → 65,536 | §3.6, Table 11 | verified 2026-09-18 | §3.6.3: mix is 34% long-context data and 66% midtraining data |
| Olmo 3 Base 32B | 32B | long-context | tokens seen | 100B, same proportions as the 50B mix | §3.6, Table 11 caption | verified 2026-09-18 | no ablation reported |
| Olmo 3 Think 7B / 32B | 7B/32B | SFT | prompts | 2,268,468 (7B); 2,253,916 (32B) | §4.2, Table 17 | verified 2026-09-18 | §4.2.1: prompts filtered heuristically, by topic, and by difficulty (Figure 15) |
| Olmo 3 Think 32B | 32B | SFT | GPUs; LR sweep; wall clock | 256 GPUs per run, four candidate learning rates in parallel, 36 hours plus ~12 hours evaluation and merging | §2.4 | verified 2026-09-18 | §2.4: swept because post-training hyperparameters are not transferable from prior runs |
| Olmo 3 Think 7B / 32B | 7B/32B | preference | prompts; pair construction | 200,000 prompts; chosen from a large model and rejected from a small model per delta learning | §4.3, Table 19 | verified 2026-09-18 | §4.3: delta learning states pair quality depends on the contrast, not on the absolute quality of either response |
| Olmo 3 Think 7B / 32B | 7B/32B | RL | algorithm; KL coefficient; advantage normalization; prompts | OlmoRL (GRPO with DAPO and Dr GRPO modifications); no KL loss; no standard-deviation normalization; 104,869 (7B) / 171,950 (32B) | §4.4.1, Table 20 | verified 2026-09-18 | §4.4.1: removing the KL loss "does not lead to over-optimization or destabilized training" |
| Olmo 3.1 Think 32B | 32B | RL | extra training | best RL run continued for a further 21 days on 224 GPUs after the initial release | §2.4 | verified 2026-09-18 | §4.4: introduced to show that extended OlmoRL training improves performance |
| Olmo 3 Instruct | 7B/32B | SFT / preference | prompts | 2,152,112 SFT; 259,922 DPO | §5, Table 30 | verified 2026-09-18 | no ablation reported |
| Olmo 3 (all) | 7B/32B | SFT | trainer | OLMo-core in place of Open Instruct, reported as an 8× increase in training throughput | §4.2, App. A.6.1 | verified 2026-09-18 | App. A.6.1 attributes the gain to batch size, data packing, and masking changes |

## Connections
- [[olmo-3]] — the source card these rows belong to.
- [[allenai-olmo3-open-instruct-scripts]] — released post-training scripts carrying the per-run hyperparameters.
- [[olmo-2-recipe]] — the predecessor ledger, for stage-by-stage comparison.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2512.13961 (arXiv v2, 14 Apr 2026).
- Corrections to the previous card version: none (new card, split out of [[olmo-3]] on 2026-09-18).
- Removed as unsupported by the source: per-stage GPU-hour totals, which the report does not give.
- Not reported by the source: peak learning rates, warmup, and batch sizes for any stage in the report body.
